(*** hide ***)
module Tutorial.Fs.examples.matrixTranspose

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.IL

(**
Matrix transpose module. 
*)

(*** define:matrixTransposeModule ***)
type MatrixTransposeModule<'T>(target:GPUModuleTarget, tileDim:int, blockRows:int) =
    inherit GPUModule(target)

    member this.Transpose sizeX sizeY (idata:'T[]) =
        let odata = Array.zeroCreate (sizeX*sizeY)
        for y = 0 to sizeY - 1 do
            for x = 0 to sizeX - 1 do
                odata.[x * sizeY + y] <- idata.[y * sizeX + x]
        odata

    [<Kernel;ReflectedDefinition>]
    member this.CopyKernel (width:int) (height:int) (idata:deviceptr<'T>) (odata:deviceptr<'T>) =
        __static_assert(__is_static_constant(tileDim))
        __static_assert(__is_static_constant(blockRows))
        let xIndex = blockIdx.x * tileDim + threadIdx.x
        let yIndex = blockIdx.y * tileDim + threadIdx.y

        let index  = xIndex + width*yIndex
        let mutable i = 0
        while i < tileDim do
            odata.[index + i*width] <- idata.[index + i*width]
            i <- i + blockRows

    [<Kernel;ReflectedDefinition>]
    member this.TransposeNaiveKernel (width:int) (height:int) (idata:deviceptr<'T>) (odata:deviceptr<'T>) =
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

    [<Kernel;ReflectedDefinition>]
    member this.TransposeCoalescedKernel (width:int) (height:int) (idata:deviceptr<'T>) (odata:deviceptr<'T>) =
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

    [<Kernel;ReflectedDefinition>]
    member this.TransposeNoBankConflictsKernel (width:int) (height:int) (idata:deviceptr<'T>) (odata:deviceptr<'T>) =
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

    member this.MemoryBandwidth memSize kernelTimeMs = 
        2.0 * 1000.0 * float(memSize) / (1024.0*1024.0*1024.0) / kernelTimeMs

    member this.Profile(sizeX:int, sizeY:int, generateTestData:int -> 'T[]) =

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
        let At = this.Transpose sizeX sizeY A

        use dA = this.GPUWorker.Malloc(A)
        use dAt = this.GPUWorker.Malloc(size)

        // warm up and validate kernels
        this.Copy sizeX sizeY dA.Ptr dAt.Ptr
        this.TransposeNaive sizeX sizeY dA.Ptr dAt.Ptr
        validate (dAt.Gather()) At 
        this.TransposeCoalesced sizeX sizeY dA.Ptr dAt.Ptr
        validate (dAt.Gather()) At 
        this.TransposeNoBankConflicts sizeX sizeY dA.Ptr dAt.Ptr
        validate (dAt.Gather()) At 

        use startEvent = this.GPUWorker.CreateEvent()
        use stopEvent = this.GPUWorker.CreateEvent()

        startEvent.Record()
        for i = 1 to nIter do
            this.Copy sizeX sizeY dA.Ptr dAt.Ptr
        stopEvent.Record()
        stopEvent.Synchronize()
        let time = Event.ElapsedMilliseconds(startEvent, stopEvent) / (float nIter)
        printfn "copy\nthroughput = %f Gb/s\nkernel time = %f ms\nnum elements = %d\nelement size = %d\n" (this.MemoryBandwidth memSize time) time (memSize/esize) esize
        
        startEvent.Record()
        for i = 1 to nIter do
            this.TransposeNaive sizeX sizeY dA.Ptr dAt.Ptr
        stopEvent.Record()
        stopEvent.Synchronize()
        let time = Event.ElapsedMilliseconds(startEvent, stopEvent) / (float nIter)
        printfn "naive transpose\nthroughput = %f Gb/s\nkernel time = %f ms\nnum elements = %d\nelement size = %d\n" (this.MemoryBandwidth memSize time) time (memSize/esize) esize

        startEvent.Record()
        for i = 1 to nIter do
            this.TransposeCoalesced sizeX sizeY dA.Ptr dAt.Ptr
        stopEvent.Record()
        stopEvent.Synchronize()

        let time = Event.ElapsedMilliseconds(startEvent, stopEvent) / (float nIter)
        printfn "coalesced transpose\nthroughput = %f Gb/s\nkernel time = %f ms\nnum elements = %d\nelement size = %d\n" (this.MemoryBandwidth memSize time) time (memSize/esize) esize
                
        startEvent.Record()
        for i = 1 to nIter do
            this.TransposeNoBankConflicts sizeX sizeY dA.Ptr dAt.Ptr
        stopEvent.Record()
        stopEvent.Synchronize()
        let time = Event.ElapsedMilliseconds(startEvent, stopEvent) / (float nIter)
        printfn "coalesced no bank conflict transpose\nthroughput = %f Gb/s\nkernel time = %f ms\nnum elements = %d\nelement size = %d\n" (this.MemoryBandwidth memSize time) time (memSize/esize) esize

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

(**
For ahead of time compilation we create a concrete type derived from `MatrixTransposeModule<'T>`.
*)

(*** define:matrixTransposeAOT ***)
[<AOTCompile>]
type MatrixTransposeF32(target:GPUModuleTarget) =
    inherit MatrixTransposeModule<float32>(target, tileDim, blockRows) 

    member this.MeasurePerformance(nIter:int, sizeX:int, sizeY:int) =
        (this :> MatrixTransposeModule<float32>).MeasurePerformance(nIter, sizeX, sizeY, createF32, validateF32)

[<AOTCompile>]
type MatrixTransposeF64(target:GPUModuleTarget) =
    inherit MatrixTransposeModule<float>(target, tileDim, blockRows) 

    member this.MeasurePerformance(nIter:int, sizeX:int, sizeY:int) =
        (this :> MatrixTransposeModule<float>).MeasurePerformance(nIter, sizeX, sizeY, createF64, validateF64)

(**
We call the corresponding performane measure function for single and double precision.
*)
(*** define:matrixTransposePerformance ***)
let matrixTransposePerformance () =
    let sizeX = 2560
    let sizeY = 2560
    let nIter = 100

    printfn "Performance single precision"
    printfn "============================"

    let matrixTransposeF32 = new MatrixTransposeF32(GPUModuleTarget.DefaultWorker)
    matrixTransposeF32.MeasurePerformance(nIter, sizeX, sizeY)

    printfn ""
    printfn "Performance double precision"
    printfn "============================"

    let matrixTransposeF64 = new MatrixTransposeF64(GPUModuleTarget.DefaultWorker)
    matrixTransposeF64.MeasurePerformance(nIter, sizeX, sizeY)




