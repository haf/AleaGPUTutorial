module Tutorial.Fs.examples.curand.EstimatePi

open System
open Alea.CUDA
open Alea.CUDA.CULib
open Alea.CUDA.Utilities.Timing
open NUnit.Framework
open CURANDInterop

// Target value
let PI = 3.14159265359

(*** define:cuRANDReduceSum ***)
let [<ReflectedDefinition>] reduceSum in' = 
    let sdata = __shared__.ExternArray<int>()

    // Perform first level of reduction:
    // - Write to shared memory
    let ltid = threadIdx.x

    sdata.[ltid] <- in'
    __syncthreads()

    // Do reduction in shared mem
    let mutable s = blockDim.x / 2
    while s > 0 do
        if ltid < s then sdata.[ltid] <- sdata.[ltid] + sdata.[ltid + s]
        __syncthreads()
        s <- s >>> 1
    
    sdata.[0]

(*** define:cuRANDComputeValue ***)
let computeValue =
    <@ fun (results:deviceptr<float>) (points:deviceptr<float>) (numSims:int) ->
        // Determine thread ID
        let bid = blockIdx.x
        let tid = blockIdx.x * blockDim.x + threadIdx.x
        let step = gridDim.x * blockDim.x

        // Shift the input/output pointers
        let mutable pointx = points + tid
        let mutable pointy = pointx + numSims
        
        // Count the number of points which lie inside the unit quarter-circle
        let mutable pointsInside = 0

        let mutable i = tid
        while i < numSims do
            let x = pointx.[0]
            let y = pointy.[0]
            let l2norm2 = x * x + y * y

            if l2norm2 < 1.0 then pointsInside <- pointsInside + 1

            i <- i + step; pointx <- pointx + step; pointy <- pointy + step

        // Reduce within the block
        pointsInside <- reduceSum pointsInside

        // Store the result
        if threadIdx.x = 0 then results.[bid] <- float pointsInside
    @>

(*** define:cuRANDPiEstimator ***)
let piEstimator (worker:Worker) numSims threadBlockSize =

    // Aim to launch around ten or more times as many blocks as there
    // are multiprocessors on the target device.
    let blocksPerSM = 10
    let numSMs = worker.Device.Attributes.MULTIPROCESSOR_COUNT

    // Determine how to divide the work between cores
    let block = dim3 threadBlockSize
    let grid = 
        let mutable x = (numSims + threadBlockSize - 1) / threadBlockSize
        while x > 2 * blocksPerSM * numSMs do x <- x >>> 1
        dim3(x)    

    let program = cuda {
        let! kernel = computeValue |> Compiler.DefineKernel
        
        return Entry(fun program ->
            let kernel = program.Apply kernel
            
            // Allocate memory for points
            // Each simulation has two random numbers to give X and Y coordinate
            let n = 2 * numSims
            let dPoints = worker.Malloc<float>(n)

            // Allocate memory for result
            // Each thread block will produce one result
            let dResults = worker.Malloc<float>(grid.x)

            // Generate random points in unit square
            let curand = new CURAND(worker, curandRngType_t.CURAND_RNG_QUASI_SOBOL64)
            curand.SetQuasiRandomGeneratorDimensions(2u)
            curand.SetGeneratorOrdering(curandOrdering_t.CURAND_ORDERING_QUASI_DEFAULT)
            curand.GenerateUniformDouble(dPoints.Ptr, size_t n)

            fun () ->
                let lp = LaunchParam(grid, block, block.x * sizeof<uint32>)
                kernel.Launch lp dResults.Ptr dPoints.Ptr numSims
                // Return partial results
                dResults.Gather()
            
                    )} |> worker.LoadProgram

    worker.Eval <| fun _ ->
        // Count the points inside unit quarter-circle
        let results = program.Run()

        // Complete sum-reduction on host
        let value = results |> Array.sum

        // Determine the propertion of points inside the quarter-circle,
        // i.e. the area of the unit quarter-circle
        let value = value / float numSims

        // Value is currently an estimate of the area of a unit quarter-circle, so we can
        // scale to a full circle by multiplying by four.  Now since the area of a circle
        // is pi * r^2, and r is one, the value will be an estimate for the value of pi.
        value * 4.0

(*** define:cuRANDEstimatePiTest ***)
let [<Test>] estimatePi() =
    let numSims = 100000
    let threadBlockSize = 128
    
    // Evaluate on GPU
    let worker = Worker.Default
    printfn "Estimating Pi on GPU (%s)\n" (worker.Device.Name)
    let estimator = piEstimator worker numSims threadBlockSize
    let result, t = tictoc(fun _ -> piEstimator worker numSims threadBlockSize)
    let elapsedTime = t.TotalMilliseconds

    // Tolerance to compare result with expected
    // This is just to check that nothing has gone very wrong with the
    // test, the actual accuracy of the result depends on the number of
    // Monte Carlo trials
    let tol = 0.01

    // Display results
    let abserror = abs(result - PI)
    let relerror = abserror / PI
    printfn "Precision:         %s" "double"
    printfn "Number of sims:    %d" numSims
    printfn "Tolerance:         %e" tol
    printfn "GPU result:        %e" result
    printfn "Expected:          %e" PI
    printfn "Absolute error:    %e" abserror
    printfn "Relative error:    %e" relerror

    // Check result
    if relerror > tol then printfn "computed result (%e) does not match expected result (%e)." result PI
    
    // Print results
    printfn "MonteCarloEstimatePiQ, Performance = %.2f sims/s, Time = %.2f(ms), NumDevsUsed = %u, Blocksize = %u"
        (float numSims / (elapsedTime/1000.0)) elapsedTime 1 threadBlockSize
    
    Assert.AreEqual(PI, result, tol)