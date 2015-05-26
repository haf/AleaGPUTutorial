(*** hide ***)
module Tutorial.Fs.advancedTechniques.GenericMatrixMult

#if SCRIPT_REFS
#r "..\\..\\..\\packages\\Alea.IL.\\lib\\net40\\Alea.dll"
#r "..\\..\\..\\packages\\Alea.CUDA\\lib\\net40\\Alea.CUDA.dll"
#r "..\\..\\..\\packages\\Alea.CUDA.IL\\lib\\net40\\Alea.CUDA.IL.dll"
#r "..\\..\\..\\packages\\NUnit\\lib\\nunit.framework.dll"
#endif

open FSharp.Charting
open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.IL
open NUnit.Framework

(**
# Workflow Technique 

The instance based technique is already very powerful but has limitations in generic 
programming with generic arithmetic operations. 
The workflow technique in combination with F# inline functions overcomes this limitation.
Note that this approach does not work in C# because it is less powerful generic programming capabilities.  
 
To illustrate the workflow technique we consider a generic matrix multiplication algorithm. 

The `multiply` kernel implements the tiled matrix multiplication algorithm using shared memory for matrices
in row major storage format. The following picture illustrates the algorithm. The matrices are decomposed into 
square blocks of size `blockSize`. The tile `Csub` of `C` is calculated by a thread block from a the row tiles of 
`A` and column tiles of `B`, where one thread calculates one element of `Csub`. The thread block iterates `wA / blockSize` 
times to load a tile from the row tiles of `A` and the column tiles of `B` into shared memory with coalesing
memory loads once. Then the data in shared memory is used from multiple threads to update `Csub`. For this reason
we have to use `__syncthreads()` after loading the data to shared memory. 

<img src="../content/images/matrixMultTiling.png" width="700" alt="matrix mult">

The function `multiply` takes the block size as argument and returns the matrix multiplication algorithm as a function 
expression. Note that it is important to declare `multiply` as `inline` for the generic math operations to work.
*)
let inline multiply (blockSize:int) =
    <@ fun (wA:int) (wB:int) (A:deviceptr<'T>) (B:deviceptr<'T>) (C:deviceptr<'T>) ->
        let bx = blockIdx.x
        let by = blockIdx.y
        let tx = threadIdx.x
        let ty = threadIdx.y

        // offset to first element of the first sub-matrix of A processed by the block
        let aBegin = wA * blockSize * by

        // index of the last sub-matrix of A processed by the block
        let aEnd = aBegin + wA - 1

        // step size used to iterate through the sub-matrices of A
        let aStep = blockSize

        // offset to first element of the first sub-matrix of B processed by the block
        let bBegin = blockSize * bx

        // step size used to iterate through the sub-matrices of B
        let bStep = blockSize * wB

        // Csub is used to store the element of the block sub-matrix that is computed by the thread
        let mutable Csub = 0G

        // loop over all the sub-matrices of A and B required to compute the block sub-matrix
        let mutable a = aBegin
        let mutable b = bBegin
        while a <= aEnd do
            let As = __shared__.Array2D(blockSize, blockSize)
            let Bs = __shared__.Array2D(blockSize, blockSize)

            // load the matrices from device memory to shared memory; each thread loads one element of each matrix 
            As.[ty, tx] <- A.[a + wA * ty + tx]
            Bs.[ty, tx] <- B.[b + wB * ty + tx]

            __syncthreads()

            // multiply the two matrices together; each thread computes one element of the block sub-matrix
            for k = 0 to blockSize - 1 do
                Csub <- Csub + As.[ty, k] * Bs.[k, tx]

            __syncthreads()

            a <- a + aStep
            b <- b + bStep

        // write the block sub-matrix to device memory; each thread writes one element
        let c = wB * blockSize * by + blockSize * bx
        C.[c + wB * ty + tx] <- Csub @>

(**
We use a generic structure `MatrixMultiplyImpl<'T>` for the selected block size and the
generated matrix multiplication kernel. 
*)
type MatrixMultiplyImpl<'T> =
    { BlockSize : int
      Kernel : Kernel<int -> int -> deviceptr<'T> -> deviceptr<'T> -> deviceptr<'T> -> unit> }

(**
We create a generic base GPU module type constructed from a `GPUModuleTarget` and a `Template`
which provides a generic matrix multiplication kernel as an instance of `MatrixMultiplyImpl<'T>`.
We extract the implementation and create a entry function with `implTemplate |> this.GPUDefineEntry`.

We then define several overloads `Mult` to execute the matrix multiplication. 
The first one assumes that all the data is already on the GPU. The second copies 
the data to the GPU, allocates storage for the matrix product and gathers the final 
result from the GPU. Both forward to the generic `mult` function, which is 
responsible to calculate the launch parameters and executing the kernel through
the entry function defined previously. 
*)
type MatrixMultiplyModule<'T>(target, implTemplate:Template<Entry<MatrixMultiplyImpl<'T>>>) as this =
    inherit ILGPUModule(target)

    let impl = implTemplate |> this.GPUDefineEntry

    let mult (wA:int) (wB:int) (hC:int) (A:deviceptr<'T>) (B:deviceptr<'T>) (C:deviceptr<'T>) =  
        let blockSize = impl.Runtime.BlockSize           
        let block = dim3(blockSize, blockSize)
        let grid = dim3(wB / block.x, hC / block.y)
        let lp = LaunchParam(grid, block)
        impl.Runtime.Kernel.Launch lp wA wB A B C

    override this.Dispose(disposing) =
        base.Dispose(disposing)

    // device version
    member this.Mult(wA:int, wB:int, hC:int, A:deviceptr<'T>, B:deviceptr<'T>, C:deviceptr<'T>) =
        mult wA wB hC A B C

    // host version with rows and columns a multiple of the block size
    member this.Mult(wA:int, wB:int, A:'T[], B:'T[]) =
        let blockSize = impl.Runtime.BlockSize
        let wC = wB
        let hC = A.Length / wA
        let hB = B.Length / wB
        if wA % blockSize <> 0 then failwith "width of A must be a multiple of %d" blockSize
        if wB % blockSize <> 0 then failwith "width of B must be a multiple of %d" blockSize
        if hC % blockSize <> 0 then failwith "height of A must be a multiple of %d" blockSize
        if hB <> wA then failwith "height of B %d must be equal to width of A %d" hB wA
        use dA = this.GPUWorker.Malloc(A)
        use dB = this.GPUWorker.Malloc(B)
        use dC = this.GPUWorker.Malloc(wC*hC)
        this.Mult(wA, wB, hC, dA.Ptr, dB.Ptr, dC.Ptr)
        dC.Gather()

    // general host version with padding 
    member this.Mult(A:'T[,], B:'T[,]) =
        let blockSize = impl.Runtime.BlockSize
        let pad (a:'T[,]) =
            let h = a.GetLength(0)
            let w = a.GetLength(1)
            let h' = blockSize * divup h blockSize
            let w' = blockSize * divup w blockSize
            let a' = Array.zeroCreate (h'*w')
            for i = 0 to h-1 do
                for j = 0 to w-1 do
                   a'.[i*w' + j] <- a.[i, j] 
            h, w, h', w', a'

        let hA, _, hA', wA', A' = pad A
        let _, wB, hB', wB', B' = pad B

        let C' = this.Mult(wA', wB', A', B')

        let C = Array2D.zeroCreate hA wB
        for i = 0 to hA-1 do
            for j = 0 to wB do
                C.[i, j] <- C'.[i*wB' + j]
        C

(**
We specialize the generic `MatrixMultiplyModule` implementation for the concrete types `float` and `float32`  
but keep the block size variable. The static member function `ImplTemplate` uses the `cuda` computational workflow
to define a GPU kernel resource with `Compiler.DefineKernel`.

The computational workflow returns an entry function, which retrieves the
compiled kernel function from the compiled program with `kernel |> program.Apply`,
and returns it together with the selected block size. This `Template` instance can 
now be used in the constructor of the specialized `MatrixMultiplyModule<float32>` type. 
*)
type MatrixMultiplyModuleF32(target, blockSize:int) =
    inherit MatrixMultiplyModule<float32>(target, MatrixMultiplyModuleF32.ImplTemplate blockSize)

    static member ImplTemplate blockSize =
        cuda {
            let! kernel = multiply blockSize |> Compiler.DefineKernel
            return Entry(fun program ->
                { BlockSize = blockSize
                  Kernel = kernel |> program.Apply } ) }

(**
We also create a specialization for type `float`. 
*)
type MatrixMultiplyModuleF64(target, blockSize:int) =
    inherit MatrixMultiplyModule<float>(target, MatrixMultiplyModuleF64.ImplTemplate blockSize)

    static member ImplTemplate blockSize =
        cuda {
            let! kernel = multiply blockSize |> Compiler.DefineKernel
            return Entry(fun program ->
                { BlockSize = blockSize
                  Kernel = kernel |> program.Apply } ) }

(**
So far the block size is still variable. It must become a compile time constant
in order to apply AOT compilation. For both `float` and `float32` we create default types 
with a block size of 32.  
*)
[<AOTCompile>]
type DefaultMatrixMultiplyModuleF32(target) =
    inherit MatrixMultiplyModuleF32(target, 32)
    static let defaultInstance = lazy new DefaultMatrixMultiplyModuleF32(GPUModuleTarget.DefaultWorker)
    static member Default = defaultInstance.Value

[<AOTCompile>]
type DefaultMatrixMultiplyModuleF64(target) =
    inherit MatrixMultiplyModuleF64(target, 32)
    static let defaultInstance = lazy new DefaultMatrixMultiplyModuleF64(GPUModuleTarget.DefaultWorker)
    static member Default = defaultInstance.Value

(**
To validate our result we provide a very simple serial matrix multiplication implementation.
*)
let multiplyCPU (wA:int) (wB:int) (A:float[]) (B:float[]) =
    let hA = A.Length / wA
    let C = Array.zeroCreate<float> (hA * wB)
    for i = 0 to hA - 1 do
        for j = 0 to wB - 1 do
            let mutable sum = 0.0
            for k = 0 to wA - 1 do
                let a = A.[i * wA + k]
                let b = B.[k * wB + j]
                sum <- sum + a * b
            C.[i * wB + j] <- sum
    C

(**
We validate the implementation for several differnt matrix dimensions. 
*)
[<Test>]
let matrixMultTest () =
    let validate (dimA:int*int) (dimB:int*int) =
        let wA, hA = dimA
        let wB, hB = dimB
        let sizeA = wA * hA
        let sizeB = wB * hB
        let rng = Random()
        let n = 1000
        let A = Array.init sizeA (fun _ -> rng.NextDouble())    
        let B = Array.init sizeB (fun _ -> rng.NextDouble())
        let dAB = DefaultMatrixMultiplyModuleF64.Default.Mult(wA, wB, A, B)
        let hAB = multiplyCPU wA wB A B
        let err = Array.map2 (fun d h -> abs (d - h)) dAB hAB |> Array.max 
        printfn "dimA %A, dimB %A, error = %A" dimA dimB err

    let dimensions = [(128, 128); (512, 512); (1024, 1024); (2048, 2048)]
    List.iter2 validate dimensions dimensions


