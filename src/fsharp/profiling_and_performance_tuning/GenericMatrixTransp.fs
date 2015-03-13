(*** hide ***)
module Tutorial.Fs.performanceTuning.GenericMatrixTransp

#if SCRIPT_REFS
#r "..\\..\\..\\packages\\Alea.IL\\lib\\net40\\Alea.IL.dll"
#r "..\\..\\..\\packages\\Alea.CUDA\\lib\\net40\\Alea.CUDA.dll"
#r "..\\..\\..\\packages\\Alea.CUDA.IL\\lib\\net40\\Alea.CUDA.IL.dll"
#endif

open System

(*** hide ***) 
open System
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.IL

(**
# Tuning Generic Matrix Transpose

We optimize the transpose of a matrix that operates out-of-place in multiple steps. As a reference benchmark we 
take the matrix copy operation. The relevant performance metric is the effective bandwidth, calculated in GB/s.
For simplicity we only consider square matrices with dimension an integral multiple of the tile dimension. 

The examples are based on a [blog article](http://devblogs.nvidia.com/parallelforall/efficient-matrix-transpose-cuda-cc/) of Mark Harris. 

The copy and various transpose kernels map threads to matrix elements using Cartesian (x,y) coordinates, rather than a rows and columns to simplify the meaning of the 
components of the `threadId`. We use `threadIdx.x` for the horizontal, e.g. columns, and `threadIdx.y` for vertical coordinate, e.g. rows.
This mapping of threads to matrix elements is up to the programmer. However, the mapping should be designed to ensure coalescing memory access.
To achieve this we want to map the fastets varying component to contiguous elements in memory. For arrays in Fortran storage order (column major), 
this is the first index of a multidimensional array, hence `threadIdx.x` and `blockIdx.x` vary quickest within blocks and grids.

All kernels launch blocks of `tileDim * blockRows` threads with `tileDim` a multiple of `blockRows` and each thread block transposes or copies a 
tile of size `tileDim * tileDim`. Using a thread block with fewer threads than elements in a tile is advantageous for the matrix transpose 
because each thread transposes  `tileDim / blockRows` matrix elements, so much of the index calculation cost is amortized over these elements. 
*)

(**
The matrix copy kernel indicates the best possible performance that we can expect from the matrix transpose operation and serves as our benchmark.

Each thread copies four elements of the matrix in a loop at the end of this routine because the number of threads in a block is smaller by a factor of `tileDim / blockRows` than 
the number of elements in a tile. Note also that `tileDim` must be used in the calculation of the matrix index `y` rather than `blockRows` or `blockDim.y`. 
The loop iterates over the second dimension and not the first so that contiguous threads load and store contiguous data, and all reads from `idata` and writes to `odata` are coalesced.
*)

[<ReflectedDefinition>]
let copy tileDim blockRows (width:int) (height:int) (idata:deviceptr<'T>) (odata:deviceptr<'T>) =
    __static_assert(__is_static_constant(tileDim))
    __static_assert(__is_static_constant(blockRows))
    let xIndex = blockIdx.x * tileDim + threadIdx.x
    let yIndex = blockIdx.y * tileDim + threadIdx.y

    let index  = xIndex + width*yIndex
    let mutable i = 0
    while i < tileDim do
        odata.[index + i*width] <- idata.[index + i*width]
        i <- i + blockRows

(** The CPU implementation of the matrix transpose is straigthforward. 
    Note that we use a flat array to store the data in row major storage format. 
*)

let transpose sizeX sizeY (idata:'T[]) =
    let odata = Array.zeroCreate (sizeX*sizeY)
    for y = 0 to sizeY - 1 do
        for x = 0 to sizeX - 1 do
            odata.[x * sizeY + y] <- idata.[y * sizeX + x]
    odata


(**
## Naive Transpose with Noncoalescing Memory Access Pattern

The first version of the transpose kernel is very similar to the copy kernel, the only difference is that the indices for `odata` are swapped.

The reads from `idata` are coalesced as in the copy kernel, but the writes to `odata` have a strided access pattern. We expect the performance of 
this kernel to suffer accordingly.
*)

[<ReflectedDefinition>]
let transposeNaive tileDim blockRows (width:int) (height:int) (idata:deviceptr<'T>) (odata:deviceptr<'T>) =
    __static_assert(__is_static_constant(tileDim))
    __static_assert(__is_static_constant(blockRows))
    let xIndex = blockIdx.x * tileDim + threadIdx.x
    let yIndex = blockIdx.y * tileDim + threadIdx.y
    let index_in  = xIndex + width * yIndex
    let index_out = yIndex + height * xIndex

    let mutable i = 0
    while i < tileDim do
        odata.[index_out + i] <- idata.[index_in + i*width]
        i <- i + blockRows


(** 
## Coalesced Transpose with Bank Conflicts

The remedy for the poor transpose performance is to use shared memory to avoid the large strides through global memory. The following figure 
explains how shared memory is used in the transpose.

<img src="../content/images/sharedTranspose.jpg" width="800" alt="shared transpose">


In the first while loop, a warp of threads reads contiguous data from `idata` into rows of the shared memory tile. 
After recalculating the array indices, a column of the shared memory tile is written to contiguous addresses in `odata`. 
Because threads write different data to `odata` than they read from `idata`, we must execute a block-wise barrier `__syncthreads()`. 
*)

[<ReflectedDefinition>]
let transposeCoalesced tileDim blockRows (width:int) (height:int) (idata:deviceptr<'T>) (odata:deviceptr<'T>) =
    __static_assert(__is_static_constant(tileDim))
    __static_assert(__is_static_constant(blockRows))
    let tile = __shared__.Array<'T>(tileDim*tileDim)

    let mutable xIndex = blockIdx.x * tileDim + threadIdx.x
    let mutable yIndex = blockIdx.y * tileDim + threadIdx.y
    let index_in = xIndex + (yIndex)*width

    xIndex <- blockIdx.y * tileDim + threadIdx.x
    yIndex <- blockIdx.x * tileDim + threadIdx.y
    let index_out = xIndex + yIndex*height

    let mutable i = 0
    while i < tileDim do
        tile.[(threadIdx.y + i)*tileDim + threadIdx.x] <- idata.[index_in + i*width]
        i <- i + blockRows
    
    __syncthreads()

    i <- 0
    while i < tileDim do
        odata.[index_out + i*height] <- tile.[threadIdx.x*tileDim + threadIdx.y + i]
        i <- i + blockRows


(**
## Coalesced Transpose without Bank Conflicts

The kernel `transposeCoalesced` already improves the performance significantly but is still far from the performance of the copy kernel. 
We may guess that the performance gap is the overhead associated with using shared memory and the required synchronization barrier `__syncthreads()`.
In fact this is not the case and could be test using a copy kernel that uses shared memory in the same way as the `transposeCoalesced` kernel
and would have approximately the same performance as the copy kernel.

For a shared memory tile of 32 * 32 elements, all elements in a column of data map to the same shared memory bank, resulting in a worst-case scenario for 
memory bank conflicts: reading a column of data results in a 32-way bank conflict. The access is serialized into 32 transactions. The simple solution for 
this problem is to pad the width in the declaration of the shared memory tile, making the tile 33 elements wide rather than 32.

    let tile = __shared__.Array<'T>(tileDim*(tileDim+1))

Removing the bank conflicts in this way improves the performance as close as about 95% of the copy throughput.
*)

[<ReflectedDefinition>]
let transposeNoBankConflicts tileDim blockRows (width:int) (height:int) (idata:deviceptr<'T>) (odata:deviceptr<'T>) =
    __static_assert(__is_static_constant(tileDim))
    __static_assert(__is_static_constant(blockRows))
    let tile = __shared__.Array<'T>(tileDim*(tileDim+1))

    let mutable xIndex = blockIdx.x * tileDim + threadIdx.x
    let mutable yIndex = blockIdx.y * tileDim + threadIdx.y
    let index_in = xIndex + (yIndex)*width

    xIndex <- blockIdx.y * tileDim + threadIdx.x
    yIndex <- blockIdx.x * tileDim + threadIdx.y
    let index_out = xIndex + yIndex*height

    let mutable i = 0
    while i < tileDim do
        tile.[(threadIdx.y + i)*(tileDim + 1) + threadIdx.x] <- idata.[index_in + i*width]
        i <- i + blockRows
    
    __syncthreads()

    i <- 0
    while i < tileDim do
        odata.[index_out + i*height] <- tile.[threadIdx.x*(tileDim + 1) + threadIdx.y + i]
        i <- i + blockRows

(** 
The proper performance measure to compare the kernels is the memory bandwidth in Gb per second.
The factor 2 comes from the fact that we load and store the data.
*)
let memoryBandwidth memSize kernelTimeMs = 
    2.0 * 1000.0 * float(memSize) / (1024.0*1024.0*1024.0) / kernelTimeMs

(**
We create a GPU module to call the different kernels and a function to execute the performance measurements. 
*)
type MatrixTranspose<'T>(target:GPUModuleTarget, tileDim:int, blockRows:int) =
    inherit GPUModule(target)

    [<Kernel;ReflectedDefinition>]
    member this.CopyKernel (width:int) (height:int) (idata:deviceptr<'T>) (odata:deviceptr<'T>) =
        copy tileDim blockRows width height idata odata

    [<Kernel;ReflectedDefinition>]
    member this.TransposeNaiveKernel (width:int) (height:int) (idata:deviceptr<'T>) (odata:deviceptr<'T>) =
        transposeNaive tileDim blockRows width height idata odata

    [<Kernel;ReflectedDefinition>]
    member this.TransposeCoalescedKernel (width:int) (height:int) (idata:deviceptr<'T>) (odata:deviceptr<'T>) =
        transposeCoalesced tileDim blockRows width height idata odata

    [<Kernel;ReflectedDefinition>]
    member this.TransposeNoBankConflictsKernel (width:int) (height:int) (idata:deviceptr<'T>) (odata:deviceptr<'T>) =
        transposeNoBankConflicts tileDim blockRows width height idata odata

    member this.LaunchParams width height =
        let threads = dim3(tileDim, blockRows)
        let grid = dim3(width / tileDim, height / tileDim)
        LaunchParam(grid, threads)

    member this.Copy (width:int) (height:int) (idata:deviceptr<'T>) (odata:deviceptr<'T>) =
        let lp = this.LaunchParams width height
        this.GPULaunch <@ this.CopyKernel @> lp width height idata odata

    member this.TransposeNaive (width:int) (height:int) (idata:deviceptr<'T>) (odata:deviceptr<'T>) =
        let lp = this.LaunchParams width height
        this.GPULaunch <@ this.TransposeNaiveKernel @> lp width height idata odata

    member this.TransposeCoalesced (width:int) (height:int) (idata:deviceptr<'T>) (odata:deviceptr<'T>) =
        let lp = this.LaunchParams width height
        this.GPULaunch <@ this.TransposeCoalescedKernel @> lp width height idata odata

    member this.TransposeNoBankConflicts (width:int) (height:int) (idata:deviceptr<'T>) (odata:deviceptr<'T>) =
        let lp = this.LaunchParams width height
        this.GPULaunch <@ this.TransposeNoBankConflictsKernel @> lp width height idata odata

    member this.Profile(sizeX:int, sizeY:int, generateTestData:int -> 'T[], validate:'T[] -> 'T[] -> unit) =

        if sizeX % tileDim <> 0 || sizeY % tileDim <> 0 || sizeX <> sizeY then 
            failwith "matrix sizeX and sizeY must be equal and a multiple of tile dimension" 

        let size = sizeX*sizeY
        let A = generateTestData size 

        use dA = this.GPUWorker.Malloc(A)
        use dAt = this.GPUWorker.Malloc(size)

        this.GPUWorker.ProfilerStart()

        this.Copy sizeX sizeY dA.Ptr dAt.Ptr

        this.TransposeNaive sizeX sizeY dA.Ptr dAt.Ptr

        this.TransposeCoalesced sizeX sizeY dA.Ptr dAt.Ptr

        this.TransposeNoBankConflicts sizeX sizeY dA.Ptr dAt.Ptr

        this.GPUWorker.Synchronize()
        this.GPUWorker.ProfilerStop()

    member this.MeasurePerformance(nIter:int, sizeX:int, sizeY:int, generateTestData:int -> 'T[], validate:'T[] -> 'T[] -> unit) =

        if sizeX % tileDim <> 0 || sizeY % tileDim <> 0 || sizeX <> sizeY then 
            failwith "matrix sizeX and sizeY must be equal and a multiple of tile dimension" 

        printfn "Matrix Transpose Using CUDA - starting..."
        printfn "GPU Device %d: %A with compute capability %d.%d\n"
                this.GPUWorker.Device.ID this.GPUWorker.Device.Name
                this.GPUWorker.Device.Arch.Major this.GPUWorker.Device.Arch.Minor
        printfn "Matrix(%d,%d)\n" sizeY sizeX 

        let size = sizeX*sizeY
        let esize = sizeof<'T>
        let memSize = esize*size
        let A = generateTestData size 
        let At = transpose sizeX sizeY A

        use dA = this.GPUWorker.Malloc(A)
        use dAt = this.GPUWorker.Malloc(size)

        this.GPUWorker.ProfilerStart()

        this.Copy sizeX sizeY dA.Ptr dAt.Ptr

        this.TransposeNaive sizeX sizeY dA.Ptr dAt.Ptr
        validate (dAt.Gather()) At 

        this.TransposeCoalesced sizeX sizeY dA.Ptr dAt.Ptr
        validate (dAt.Gather()) At 

        this.TransposeNoBankConflicts sizeX sizeY dA.Ptr dAt.Ptr
        validate (dAt.Gather()) At 

        this.GPUWorker.Synchronize()
        this.GPUWorker.ProfilerStop()

        use start = this.GPUWorker.CreateEvent()
        use stop = this.GPUWorker.CreateEvent()

        start.Record()
        for i = 1 to nIter do
            this.Copy sizeX sizeY dA.Ptr dAt.Ptr
        stop.Record()
        stop.Synchronize()
        let time = Event.ElapsedMilliseconds(start, stop) / (float nIter)
        printfn "copy\nthroughput = %f Gb/s\nkernel time = %f ms\nnum elements = %d\nelement size = %d\n" (memoryBandwidth memSize time) time (memSize/esize) esize
        
        start.Record()
        for i = 1 to nIter do
            this.TransposeNaive sizeX sizeY dA.Ptr dAt.Ptr
        stop.Record()
        stop.Synchronize()
        let time = Event.ElapsedMilliseconds(start, stop) / (float nIter)
        printfn "naive transpose\nthroughput = %f Gb/s\nkernel time = %f ms\nnum elements = %d\nelement size = %d\n" (memoryBandwidth memSize time) time (memSize/esize) esize

        start.Record()
        for i = 1 to nIter do
            this.TransposeCoalesced sizeX sizeY dA.Ptr dAt.Ptr
        stop.Record()
        stop.Synchronize()

        let time = Event.ElapsedMilliseconds(start, stop) / (float nIter)
        printfn "coalesced transpose\nthroughput = %f Gb/s\nkernel time = %f ms\nnum elements = %d\nelement size = %d\n" (memoryBandwidth memSize time) time (memSize/esize) esize
                
        start.Record()
        for i = 1 to nIter do
            this.TransposeNoBankConflicts sizeX sizeY dA.Ptr dAt.Ptr
        stop.Record()
        stop.Synchronize()
        let time = Event.ElapsedMilliseconds(start, stop) / (float nIter)
        printfn "coalesced no bank conflict transpose\nthroughput = %f Gb/s\nkernel time = %f ms\nnum elements = %d\nelement size = %d\n" (memoryBandwidth memSize time) time (memSize/esize) esize

(*** hide ***)
let createF32 n = Array.init n (fun i -> float32(i))

let validateF32 dA A = 
    let err = Array.map2 (fun d h -> abs (d - h)) dA A |> Array.max
    if err <> 0.0f then failwithf "failed with error %A" err

let createF64 n = Array.init n (fun i -> float(i))

let validateF64 dA A = 
    let err = Array.map2 (fun d h -> abs (d - h)) dA A |> Array.max
    if err <> 0.0 then failwithf "failed with error %A" err

(**
For profiling and performance measurement we choose `tileDim = 32`, `blockRows = 8` and select a matrix size of 2560 by 2560.
Note that `tileDim` must be a multiple of `blockRows`.
*)

let tileDim = 32
let blockRows = 8

let sizeX = 2560
let sizeY = 2560

let nIter = 100

(**
For profiling we analzye single and double precision separately. 
*)
let matrixTransposeProfileF32 () =
    printfn "Profiling single precision"
    printfn "=========================="

    let matrixTransposeF32 = new MatrixTranspose<float32>(GPUModuleTarget.DefaultWorker, tileDim, blockRows)
    matrixTransposeF32.Profile(sizeX, sizeY, createF32, validateF32)

let matrixTransposeProfileF64 () =
    printfn "Profiling double precision"
    printfn "=========================="

    let matrixTransposeF64 = new MatrixTranspose<float>(GPUModuleTarget.DefaultWorker, tileDim, blockRows)
    matrixTransposeF64.Profile(sizeX, sizeY, createF64, validateF64)

(**
Performance measurement for single and double precision and JIT compilation. 
*)
let matrixTransposePerformanceJIT () =
    printfn "Performance single precision"
    printfn "============================"

    let matrixTransposeF32 = new MatrixTranspose<float32>(GPUModuleTarget.DefaultWorker, tileDim, blockRows)
    matrixTransposeF32.MeasurePerformance(nIter, sizeX, sizeY, createF32, validateF32)

    printfn ""
    printfn "Performance double precision"
    printfn "============================"

    let matrixTransposeF64 = new MatrixTranspose<float>(GPUModuleTarget.DefaultWorker, tileDim, blockRows)
    matrixTransposeF64.MeasurePerformance(nIter, sizeX, sizeY, createF64, validateF64)

(**
For ahead of time compilation we create a concrete type derived from `MatrixTranspose<'T>`.
*)

[<AOTCompile(AOTOnly=true)>]
type MatrixTransposeF32(target:GPUModuleTarget) =
    inherit MatrixTranspose<float32>(target, tileDim, blockRows) 

    member this.MeasurePerformance(nIter:int, sizeX:int, sizeY:int) =
        (this :> MatrixTranspose<float32>).MeasurePerformance(nIter, sizeX, sizeY, createF32, validateF32)

[<AOTCompile(AOTOnly=true)>]
type MatrixTransposeF64(target:GPUModuleTarget) =
    inherit MatrixTranspose<float>(target, tileDim, blockRows) 

    member this.MeasurePerformance(nIter:int, sizeX:int, sizeY:int) =
        (this :> MatrixTranspose<float>).MeasurePerformance(nIter, sizeX, sizeY, createF64, validateF64)

(**
To measure the performance of AOT compiled code make sure that `FodyWeavers.xml` file in the 
project holds the proper configuration so that optimized code is generated.
    
    [lang=text]
    <?xml version="1.0" encoding="utf-8"?>
    <Weavers>
      <Alea.CUDA Level="Optimized" SMs="sm20" Bits="32;64" />
    </Weavers>

Then we call the corresponding performane measure function for single and double precision.
*)
let matrixTransposePerformanceAOT () =
    printfn "Performance single precision"
    printfn "============================"

    let matrixTransposeF32 = new MatrixTransposeF32(GPUModuleTarget.DefaultWorker)
    matrixTransposeF32.MeasurePerformance(nIter, sizeX, sizeY)

    printfn ""
    printfn "Performance double precision"
    printfn "============================"

    let matrixTransposeF64 = new MatrixTransposeF64(GPUModuleTarget.DefaultWorker)
    matrixTransposeF64.MeasurePerformance(nIter, sizeX, sizeY)