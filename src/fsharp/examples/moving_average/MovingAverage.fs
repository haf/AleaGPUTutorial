(*** hide ***)
module Tutorial.Fs.examples.movingAverage

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework
open Tutorial.Fs.examples.genericScan
  
let assertArrayEqual (tol:float option) (h:'T[]) (d:'T[]) =
    (h,d)||>Array.iter2(fun h d -> tol |> function | Some tol -> Assert.That(d, Is.EqualTo(h).Within(tol)) | None -> Assert.AreEqual(h,d))

(**
Windows differnce kernel to produce moving average from scan.
Note that we assume that values have an additonal zero at the end so `n - windowSize`
is the right length of the moving average vector, see comment below as well.
*)
(*** define:MovingAvgWinDiff ***)
let inline windowDifference () = cuda {
    let! windowDifference = 
        <@ fun n (windowSize:int) (x:deviceptr<'T>) (y:deviceptr<'T>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start + windowSize
            let normalizer = __gconv windowSize
            while i < n do
                y.[i - windowSize] <- (x.[i] - x.[i - windowSize]) / normalizer
                i <- i + stride
        @> |> Compiler.DefineKernel 

    return fun (program:Program) ->
        let worker = program.Worker
        let windowDifference = program.Apply(windowDifference)
        let numSm = program.Worker.Device.Attributes.MULTIPROCESSOR_COUNT
        let blockSize = 256

        fun n windowSize (input:deviceptr<'T>) (output:deviceptr<'T>) ->
            
            let gridSize = min numSm (divup n blockSize)
            let lp = LaunchParam(gridSize, blockSize)
         
            windowDifference.Launch lp n windowSize input output
    }  

(**

Moving average kernel.

Slightly different version than above scan based implementation as it produces a vector of same length as the initial data, e.g.
`values = [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0|]`.

The scan based moving averag prduces `[|2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0|]`.

This version produces `[|1.0; 1.5; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0|]`.

*)
(*** define:MovingAvgKernel ***)
let inline movingAverage () = cuda {
    let! kernel =
        <@ fun windowSize (n:int) (values:deviceptr<'T>) (results:deviceptr<'T>) ->
            let blockSize = blockDim.x
            let idx = threadIdx.x
            let iGlobal = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x
            let shared = __shared__.ExternArray<'T>()    

            let mutable idxShared = idx + windowSize - 1
            let mutable idxGlobal = iGlobal
            while idxShared >= 0 do
                let value = if idxGlobal >=0 && idxGlobal < n then values.[idxGlobal] else 0G
                shared.[idxShared] <- value
                idxShared <- idxShared - blockSize
                idxGlobal <- idxGlobal - blockSize

            __syncthreads()

            if iGlobal < n then
                let mutable temp = 0G
                let mutable k = 0
                while k <= min (windowSize - 1) iGlobal do
                    temp <- (temp * __gconv k + shared.[idx - k + windowSize - 1]) / (__gconv k + 1G)
                    k <- k + 1
                results.[iGlobal] <- temp
        @> |> Compiler.DefineKernel 
                
    return fun (program:Program) n -> 
        let worker = program.Worker
        let kernel = program.Apply(kernel)
        let maxBlockSize = 256
        let blockSize = min n maxBlockSize
        let gridSizeX = (n - 1) / blockSize + 1

        fun windowSize (values:deviceptr<'T>) (results:deviceptr<'T>) ->
            let sharedMem = (blockSize + windowSize - 1) * __sizeof<'T>()
            let lp = 
                if gridSizeX <= 65535 then 
                    LaunchParam(gridSizeX, blockSize, sharedMem)
                else
                    let gridSizeY = 1 + (n - 1) / (blockSize * 65535)
                    let gridSizeX = 1 + (n - 1) / (blockSize * gridSizeY)
                    LaunchParam(dim3(gridSizeX, gridSizeY), dim3(blockSize), sharedMem)

            kernel.Launch lp windowSize n values results
    }

(**
CPU version to calculate moving average based on `Seq.windowed`.
*)
let inline movingAverageSeq windowSize (series:seq<'T>) =
    series    
    |> Seq.windowed windowSize
    |> Seq.map Array.sum
    |> Seq.map (fun a -> a / (__gconv windowSize))  

(**
Fast CPU version based on arrays to calculate moving average
*)
(*** define:MovingAvgArray ***)
let inline movingAverageArray windowSize (series:'T[]) =
    let sums = Array.scan (fun s x -> s + x) 0G series
    let ma = Array.zeroCreate (sums.Length - windowSize)
    for i = windowSize to sums.Length - 1 do
        ma.[i - windowSize] <- (sums.[i] - sums.[i - windowSize]) / (__gconv windowSize)   
    ma   

(**
Creating a moving average work flow with data in CPU memory
by first performing a scan and then a window difference.
Note that the values must have a zero appended.
*)
(*** define:MovingAvgScan ***)
let inline movingAverageScan () = cuda {      
    let! scanner = ScanApi.rawSum Plan.Planner.Default  
    let! windowDifference = windowDifference ()  
    return Entry(fun program windowSize (values:'T[]) ->
        let worker = program.Worker
        let n = values.Length
        let scanner = scanner program n
        let windowDifference = windowDifference program n
        use ranges = worker.Malloc(scanner.Ranges)
        use rangeTotals = worker.Malloc<'T>(scanner.NumRanges)
        use values = worker.Malloc(values)
        use sums = worker.Malloc(n)
        use results = worker.Malloc(n - windowSize)
        scanner.Scan ranges.Ptr rangeTotals.Ptr values.Ptr sums.Ptr false
        windowDifference windowSize sums.Ptr results.Ptr
        results.Gather()
    ) }

(**
Creating a moving average work flow with data in CPU memory
by using a direct implementation.
*)
let inline movingAverageDirect () = cuda {      
    let! movingAverage = movingAverage ()  
    return Entry(fun program windowSize (values:'T[]) ->
        let worker = program.Worker
        let n = values.Length
        let movingAverage = movingAverage program n
        use values = worker.Malloc(values)
        use results = worker.Malloc(n)
        movingAverage windowSize values.Ptr results.Ptr
        results.Gather()
    ) }

(*** define:MovingAvgTestFunc ***)
let inline test (real:RealTraits<'T>) sizes (movingAverageGPU:Program<int -> 'T[] -> 'T[]>) (tol:float) direct =
    let movingAverageGold = movingAverageArray
    let values n = Array.init n (fun _ -> TestUtil.genRandomDouble -5.0 5.0 0.0 |> real.Of)
    //let values n = Array.init n float
    let windowSizes = [| 2; 3; 10 |]
    //let windowSizes = [| 1024 |]

    let compare n windowSize =
        let v = values n |> Array.append [|0G|]
        let d = movingAverageGPU.Run windowSize v |> fun d -> 
            if   direct 
            then d.[windowSize-1..] 
            else d
        let h = 
            let h = movingAverageGold windowSize v in
            if   direct
            then h
            else h.[..h.Length-2]

        printfn "window %d" windowSize
        printfn "gpu size: %A" d.Length
        printfn "cpu size: %A" h.Length
        printfn "gpu : %A" d
        printfn "cpu : %A" h

        assertArrayEqual (Some tol) h d
        
    sizes |> Array.iter (fun n -> let compare = compare n in windowSizes |> Array.iter compare)

(*** define:MovingAvgTest ***)
[<Test>]
let movingAverageTest() =

    let sizes = [| 12; 15; 20; 32; 64; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193; 9000; 10000; 2097152; 8388608 |]
    let real = RealTraits.Real64
    use movingAverageScan = movingAverageScan() |> Compiler.load Worker.Default
    test real sizes movingAverageScan 1e-8 false

(*** define:MovingAvgDirectTest ***)
[<Test>]
let movingAverageDirectTest() = 

    let sizes = [| 12; 15; 20; 32; 64; 128; 512; 1024; 1200; 4096; 5000; 8191; 8192; 8193; 9000; 10000; 2097152; 8388608 |]
    let real = RealTraits.Real64
    use movingAverageDirect = movingAverageDirect() |> Compiler.load Worker.Default   
    test real sizes movingAverageDirect 1e-8 true

