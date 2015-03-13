(*** hide ***)
module Tutorial.Fs.examples.matrixMultiplication

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.IL
open NUnit.Framework

(**
# Matrix Multiplication

Matrix multiplication using tiling and shared memory to reduce multiple reads of the same data
in multiple threads. 
*)

(*** define:matrixMultiplyModule ***) 
type MatrixMultiplyModule(target, blockSize) =
    inherit ILGPUModule(target)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (wA:int) (wB:int) (A:deviceptr<float>) (B:deviceptr<float>) (C:deviceptr<float>) =
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
        let mutable Csub = 0.0

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
        C.[c + wB * ty + tx] <- Csub 

    // device version
    member this.Mult(wA:int, wB:int, hC:int, A:deviceptr<float>, B:deviceptr<float>, C:deviceptr<float>) =  
        let block = dim3(blockSize, blockSize)
        let grid = dim3(wB / block.x, hC / block.y)
        let lp = LaunchParam(grid, block)
        this.GPULaunch <@ this.Kernel @> lp wA wB A B C

    override this.Dispose(disposing) =
        base.Dispose(disposing)

    // host version with rows and columns a multiple of the block size
    member this.Mult(wA:int, wB:int, A:float[], B:float[]) =
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
    member this.Mult(A:float[,], B:float[,]) =
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
So far the block size is still variable. It must become a compile time constant
in order to apply AOT compilation. We create a default type with a block size of 32.  
*)

(*** define:defaultMatrixMultiplyModule ***)
[<AOTCompile>]
type DefaultMatrixMultiplyModule(target) =
    inherit MatrixMultiplyModule(target, 32)
    static let defaultInstance = lazy new DefaultMatrixMultiplyModule(GPUModuleTarget.DefaultWorker)
    static member DefaultInstance = defaultInstance.Value

(**
To validate our result we provide a very simple serial matrix multiplication implementation
and validate the implementation for several differnt matrix dimensions. 
*)

(*** define:matrixMultiplyTest ***)
let matrixMultiplyCPU (wA:int) (wB:int) (A:float[]) (B:float[]) =
    let hA = A.Length / wA
    let C = Array.zeroCreate<float> (hA * wB)
    for i = 0 to hA - 1 do
        for j = 0 to wB - 1 do
            let mutable sum = 0.0
            for k = 0 to wA - 1 do 
                sum <- sum + A.[i * wA + k] * B.[k * wB + j]
            C.[i * wB + j] <- sum
    C

[<Test>]
let matrixMultiplyTest () =
    let validate (dimA:int*int) (dimB:int*int) =
        let wA, hA = dimA
        let wB, hB = dimB
        let sizeA = wA * hA
        let sizeB = wB * hB
        let rng = Random()
        let A = Array.init sizeA (fun _ -> rng.NextDouble())    
        let B = Array.init sizeB (fun _ -> rng.NextDouble())
        let dAB = DefaultMatrixMultiplyModule.DefaultInstance.Mult(wA, wB, A, B)
        let hAB = matrixMultiplyCPU wA wB A B
        let err = Array.map2 (fun d h -> abs (d - h)) dAB hAB |> Array.max 
        printfn "dimA %A, dimB %A, error = %A" dimA dimB err

    let dimensions = [(128, 128); (512, 512); (1024, 1024); (2048, 2048)]
    List.iter2 validate dimensions dimensions


