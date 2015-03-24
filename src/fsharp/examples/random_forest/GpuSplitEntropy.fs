module Tutorial.Fs.examples.RandomForest.GpuSplitEntropy

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound

open Tutorial.Fs.examples.RandomForest.Cuda
open Tutorial.Fs.examples.RandomForest.DataModel

[<Literal>]
let private DEBUG = false

(**
Matrix row scan resource.

Using block range scan from Alea Unbound. 
*)
type MatrixRowScanPrimitive(arch:DeviceArch, addressSize, blockThreads) =

    let blockRangeScan = DeviceScanPolicy.Create(arch, addressSize, blockThreads).BlockRangeScan

    [<ReflectedDefinition>]
    member this.BlockRangeScan blockOffset blockEnd (inputs:deviceptr<int>) =
        let tempStorage = blockRangeScan.TempStorage.AllocateShared()
        
        blockRangeScan.ConsumeRangeConsecutiveInclusive
            tempStorage (Iterator inputs) (Iterator inputs) (+) blockOffset blockEnd

    member this.BlockThreads = blockRangeScan.BlockThreads

type EntropyOptimizationOptions =
    {
        AbsMinWeight : int
        RelMinDivisor : int
        RelMinBound : int
        Decimals : int
        FeatureSelector : int -> bool[] // computes a subset of n features, i.e. maps n to a boolean vector of length n
    }

    static member Default = 
        {
            AbsMinWeight = 1
            RelMinDivisor = 10
            RelMinBound = 25
            Decimals = 6
            FeatureSelector = fun n -> Array.create n true
        }

    member this.MinWeight numClasses total =
        let relativeMinWeight = min (total / (numClasses * this.RelMinDivisor)) this.RelMinBound
        max this.AbsMinWeight relativeMinWeight

    static member SquareRootFeatureSelector (rnd : System.Random) (n : int) =
        let k = float n |> sqrt |> int
        let idcs = Array.randomSubIndices rnd n k
        let mask = Array.create n false
        idcs |> Array.iter (fun idx -> mask.[idx] <- true)
        mask

type EntropyOptimizingProblem =
    { labelMatrix : Cublas.Matrix<Label>
      indexMatrix : Cublas.Matrix<int>
      gpuWeights : Cublas.Matrix<int>
      weightMatrix : Cublas.Matrix<int>
      nonZeroIdcsPerFeature : Cublas.Matrix<int>
      weightsPerFeatureAndClass : Cublas.Matrix<int>
      entropyMatrix : Cublas.Matrix<float>
      gpuMask : Cublas.Matrix<int>
      numClasses : int
      numFeatures : int
      numSamples : int }

    member this.Dispose() =
        this.labelMatrix.Dispose()
        this.indexMatrix.Dispose()
        this.gpuWeights.Dispose()
        this.weightMatrix.Dispose()
        this.nonZeroIdcsPerFeature.Dispose()
        this.weightsPerFeatureAndClass.Dispose()
        this.entropyMatrix.Dispose()
        this.gpuMask.Dispose()
        ()


[<AOTCompile>]
type EntropyOptimizer(target) as this =
    inherit GPUModule(target)

    let primitive = 
        fun (options:CompileOptions) ->
            cuda { return MatrixRowScanPrimitive(options.MinimalArch, options.AddressSize, None) }
        |> this.GPUDefineResource

    let minimzer = new MultiChannelReduce.MatrixRowOptimizer(target)

//    let mutable labelMatrix : Cublas.Matrix<_> option = None
//    let mutable indexMatrix : Cublas.Matrix<_> option = None
//    let mutable gpuWeights : Cublas.Matrix<_> option = None
//    let mutable weightMatrix  : Cublas.Matrix<_> option = None
//    let mutable nonZeroIdcsPerFeature : Cublas.Matrix<_> option = None
//    let mutable weightsPerFeatureAndClass : Cublas.Matrix<_> option = None
//    let mutable entropyMatrix : Cublas.Matrix<_> option = None
//    let mutable gpuMask : Cublas.Matrix<_> option = None
//    let mutable numClasses = 0
//    let mutable numFeatures = 0
//    let mutable numSamples = 0

//    let disposeOf (matrix : Cublas.Matrix<_> option) =
//        match matrix with
//        | None -> ()
//        | Some m -> m.Dispose()

    let ptrOf (matrix : Cublas.Matrix<_>) =
        matrix.DeviceData.Ptr

    let summarizeWeights (weights : Weights) =
        let mutable sum = 0
        let mutable count = 0
        for weight in weights do
            sum <- sum + weight
            count <- count + (min weight 1)
        sum, count

    [<Literal>]
    let BLOCK_SIZE = 512

    static let instance = Lazy.Create <| fun _ -> new EntropyOptimizer(GPUModuleTarget.DefaultWorker)

    static member Default = instance.Value

    member this.BlockAndGridDim (problem:EntropyOptimizingProblem) numCols =
        let blockDim = dim3(BLOCK_SIZE)
        let gridDim = dim3(divup numCols blockDim.x, problem.numFeatures)
        blockDim, gridDim        

    member this.Init(numberOfClasses, labelsPerFeature : Labels[], indicesPerFeature : Indices[]) =
        let worker = this.GPUWorker
        let labelMatrix = new Cublas.Matrix<_>(worker, labelsPerFeature)
        let indexMatrix = new Cublas.Matrix<_>(worker, indicesPerFeature)
        let numClasses = numberOfClasses
        let numFeatures = labelMatrix.NumRows
        let numSamples = labelMatrix.NumCols
        if numFeatures <> indexMatrix.NumRows || numSamples <> indexMatrix.NumCols then
            failwith "Dimensions of labels and indices per feature must agree"
        // temporary space
        let gpuWeights = new Cublas.Matrix<_>(worker, 1, numSamples)
        let weightMatrix = new Cublas.Matrix<_>(worker, numFeatures, numSamples)
        let nonZeroIdcsPerFeature = new Cublas.Matrix<_>(worker, numFeatures, numSamples)
        let weightsPerFeatureAndClass = new Cublas.Matrix<_>(worker, numFeatures * numClasses, numSamples)
        let entropyMatrix = new Cublas.Matrix<_>(worker, numFeatures, numSamples)
        let gpuMask = new Cublas.Matrix<_>(worker, 1, numFeatures, Array.create numFeatures 1)

        { labelMatrix = labelMatrix
          indexMatrix = indexMatrix
          gpuWeights = gpuWeights
          weightMatrix = weightMatrix
          nonZeroIdcsPerFeature = nonZeroIdcsPerFeature
          weightsPerFeatureAndClass = weightsPerFeatureAndClass
          entropyMatrix = entropyMatrix
          gpuMask = gpuMask
          numClasses = numClasses
          numFeatures = numFeatures
          numSamples = numSamples }

    [<Kernel;ReflectedDefinition>]
    member private this.CumSumKernel (numCols:int) (numValid:int) (inputs:deviceptr<int>) =
        let blockOffset = blockIdx.x * numCols
        primitive.Resource.BlockRangeScan blockOffset (blockOffset + numValid) inputs
        ()

    member this.CumSums(matrix : Cublas.Matrix<_>, numValid) =
        let lp = LaunchParam(matrix.NumRows, primitive.Resource.BlockThreads)
        this.GPULaunch <@ this.CumSumKernel @> lp matrix.NumCols numValid (matrix.DeviceData.Ptr)

    [<Kernel;ReflectedDefinition>]
    member private this.LogicalWeightExpansionKernel 
        (weightMatrix:deviceptr<_>) (indexMatrix:deviceptr<_>) (weights:deviceptr<_>) numSamples =
        let sampleIdx = blockIdx.x * blockDim.x + threadIdx.x
        let featureIdx = blockIdx.y
        if sampleIdx < numSamples then
            let matrixIdx = featureIdx * numSamples + sampleIdx
            let weight = weights.[indexMatrix.[matrixIdx]]
            weightMatrix.[matrixIdx] <- __nv_min weight 1

    [<Kernel;ReflectedDefinition>]
    member private this.findNonZeroIndicesKernel (indexMatrix:deviceptr<_>) (cumWeightMatrix:deviceptr<_>) numSamples =
        let sampleIdx = blockIdx.x * blockDim.x + threadIdx.x
        let featureIdx = blockIdx.y
        if sampleIdx < numSamples then
            let weightIdx = featureIdx * numSamples + sampleIdx
            let weight = cumWeightMatrix.[weightIdx]
            let prevWeight = if sampleIdx = 0 then 0 else cumWeightMatrix.[weightIdx - 1]
            if weight > prevWeight then
                indexMatrix.[featureIdx * numSamples + weight - 1] <- sampleIdx

    [<Kernel;ReflectedDefinition>]
    member private this.WeightExpansionKernel 
        (weightMatrix:deviceptr<_>) (labelMatrix:deviceptr<Label>) (indexMatrix:deviceptr<int>)
            (weights:deviceptr<int>) numSamples numClasses numValid (nonZeroIdcsMatrix:deviceptr<_>) =
        let nonZeroIdx = blockIdx.x * blockDim.x + threadIdx.x
        let featureIdx = blockIdx.y
        let numFeatures = gridDim.y
        let rowOffset = featureIdx * numSamples
        if nonZeroIdx < numValid then
            let smallMatrixIdx = rowOffset + nonZeroIdx
            let sampleIdx = nonZeroIdcsMatrix.[smallMatrixIdx]
            let largeMatrixIdx = rowOffset + sampleIdx
            //let weight = weights.[indexMatrix.[largeMatrixIdx]]
            //let label = labelMatrix.[largeMatrixIdx]
            for classIdx = 0 to numClasses - 1 do
                let classOffset = numFeatures * numSamples * classIdx
                weightMatrix.[classOffset + smallMatrixIdx] <- indexMatrix.[largeMatrixIdx]
                //weightMatrix.[classOffset + smallMatrixIdx] <- if label = classIdx then weight else 0

    member this.FindNonZeroIndices (problem:EntropyOptimizingProblem) (weights:Weights) =
        if weights.Length <> problem.numSamples then failwith "InumValid number of weights"

        let blockDim, gridDim = this.BlockAndGridDim problem problem.numSamples
        let lp = LaunchParam(gridDim, blockDim)

        problem.gpuWeights.Scatter(weights)
//        this.GPUWorker.Synchronize()
        this.GPULaunch <@ this.LogicalWeightExpansionKernel @> lp (problem.weightMatrix.DeviceData.Ptr) 
            (problem.indexMatrix.DeviceData.Ptr) (problem.gpuWeights.DeviceData.Ptr) problem.numSamples
        this.GPUWorker.Eval <| fun _ ->

//        let weightsPtr = problem.gpuWeights |> ptrOf
//        let weightMatrix = problem.weightMatrix
//        let nonZeroIdcsPerFeature = problem.nonZeroIdcsPerFeature
//        this.GPULaunch <@ this.LogicalWeightExpansionKernel @> lp (weightMatrix.DeviceData.Ptr) 
//            (problem.indexMatrix |> ptrOf) weightsPtr problem.numSamples
        
        // cum sums over the weight matrix        
//        this.CumSums(problem.weightMatrix, problem.numSamples)
            let lp1 = LaunchParam(problem.weightMatrix.NumRows, primitive.Resource.BlockThreads)
            this.GPULaunch <@ this.CumSumKernel @> lp1 problem.weightMatrix.NumCols problem.numSamples (problem.weightMatrix.DeviceData.Ptr)
//        this.GPUWorker.Synchronize()
            this.GPULaunch <@ this.findNonZeroIndicesKernel @> lp (problem.nonZeroIdcsPerFeature.DeviceData.Ptr) 
                (problem.weightMatrix.DeviceData.Ptr) problem.numSamples 

//        if DEBUG then
//            printfn "weight matrix:\n%A" (weightMatrix.ToArray2D())
//            printfn "nonZeroIdcsPerFeature:\n%A" (nonZeroIdcsPerFeature.ToArray2D())

    member this.ExpandWeights (problem:EntropyOptimizingProblem) numValid =
        let blockDim, gridDim = this.BlockAndGridDim problem numValid
        let lp = LaunchParam(gridDim, blockDim)
        this.GPULaunch <@ this.WeightExpansionKernel @> lp (problem.weightsPerFeatureAndClass |> ptrOf) 
            (problem.labelMatrix |> ptrOf) (problem.indexMatrix |> ptrOf) (problem.gpuWeights |> ptrOf)
            problem.numSamples problem.numClasses numValid
            (problem.nonZeroIdcsPerFeature |> ptrOf)

        if DEBUG then
            printfn "weightsPerFeatureAndClass:\n%A" (problem.weightsPerFeatureAndClass.ToArray2D())

    [<Kernel;ReflectedDefinition>]
    member private this.EntropyKernel (entropyMatrix:deviceptr<_>) (cumSumsMatrix:deviceptr<_>) numValid 
        numSamples numClasses minWeight roundingFactor (mask : deviceptr<int>) =

        let totals = __shared__.ExternArray()
        let featureIdx = blockIdx.y
        let mutable classIdx = threadIdx.x
        while classIdx < numClasses do
            // last entry of this feature's row in the submatrix corresponding to class classIdx
            let classTotal = cumSumsMatrix.[numSamples * (gridDim.y * classIdx + featureIdx) + numValid - 1]
            totals.[classIdx] <- classTotal
            classIdx <- classIdx + blockDim.x 
        __syncthreads()

        let entropyTerm (x : int) = 
            if x > 0 then
                let xf = float x
                xf * (__nv_log2 xf)
            elif x = 0 then
                0.0
            else
                __nan()

        let sampleIdx = blockIdx.x * blockDim.x + threadIdx.x
        let upperBound = numValid - 1
        if sampleIdx <= upperBound then
            let matrixIdx = featureIdx * numSamples + sampleIdx
            entropyMatrix.[matrixIdx] <-
                if mask.[featureIdx] <> 0 then
                    let mutable leftEntropy = 0.0
                    let mutable rightEntropy = 0.0
                    let mutable leftTotal = 0
                    let mutable rightTotal = 0

                    for classIdx = 0 to numClasses - 1 do
                        let matrixOffset = gridDim.y * numSamples * classIdx + matrixIdx
                        let leftCount = cumSumsMatrix.[matrixOffset]
                        let rightCount = totals.[classIdx] - leftCount
                        leftEntropy  <- leftEntropy  - (entropyTerm leftCount)
                        rightEntropy <- rightEntropy - (entropyTerm rightCount)
                        leftTotal  <- leftTotal  + leftCount
                        rightTotal <- rightTotal + rightCount

                    let leftEntropy  = leftEntropy  + (entropyTerm leftTotal)
                    let rightEntropy = rightEntropy + (entropyTerm rightTotal)
                    let rawEntropy = (leftEntropy + rightEntropy) / (float (leftTotal + rightTotal))
                    let condition = sampleIdx = upperBound || (minWeight <= leftTotal && minWeight <= rightTotal)
                
                    if condition then (__nv_round (rawEntropy * roundingFactor)) / roundingFactor else infinity
                else infinity
    

    /// Returns a GPU matrix that the caller needs to dispose of
    member this.Entropy (problem:EntropyOptimizingProblem) (options : EntropyOptimizationOptions) totals numValid =
        let roundingFactor = 10.0 ** (float options.Decimals)
        let minWeight = options.MinWeight problem.numClasses totals

        let cumSumsMatrix = problem.weightsPerFeatureAndClass
        let entropyMatrix = problem.entropyMatrix
        let blockDim, gridDim = this.BlockAndGridDim problem numValid
        let lp = LaunchParam(gridDim, blockDim, problem.numClasses * sizeof<int>)
        this.GPULaunch <@ this.EntropyKernel @> lp (entropyMatrix.DeviceData.Ptr) (cumSumsMatrix.DeviceData.Ptr) 
            numValid problem.numSamples problem.numClasses minWeight roundingFactor (problem.gpuMask |> ptrOf)

    member this.MinimumEntropy (problem:EntropyOptimizingProblem) numValid =
        // start kernel immediately
        let optima = minimzer.MinAndArgMin(problem.entropyMatrix, problem.nonZeroIdcsPerFeature, numValid)
        let fullEntropy = problem.entropyMatrix.Gather(0, numValid - 1)
         
        if fullEntropy > 0.0 then
            optima
        else
            Array.create problem.numFeatures (0.0, problem.numSamples - 1)

    member this.Optimize (problem:EntropyOptimizingProblem) options (weights:Weights) = 
        problem.gpuMask.Scatter(options.FeatureSelector problem.numFeatures |> Array.map (fun x -> if x then 1 else 0))
        this.FindNonZeroIndices problem weights
        let sum, count = summarizeWeights weights
        this.ExpandWeights problem count

        this.CumSums(problem.weightsPerFeatureAndClass, count)
        this.Entropy problem options sum count
        this.MinimumEntropy problem count
//        this.GPUWorker.Eval <| fun _ ->

    //        printfn "count=(%d)" count

//    member this.LabelMatrix = labelMatrix.Value
//    member this.IndexMatrix = indexMatrix.Value
//    member this.NonZeroIndices = nonZeroIdcsPerFeature.Value
//    member this.WeightsPerFeatureAndClass = weightsPerFeatureAndClass.Value
//    member this.EntropyMatrix = entropyMatrix.Value
//
//    override this.Dispose(disposing) =
//        if disposing then
//            labelMatrix |> disposeOf
//            indexMatrix |> disposeOf
//            gpuWeights |> disposeOf
//            weightMatrix |> disposeOf
//            nonZeroIdcsPerFeature |> disposeOf
//            weightsPerFeatureAndClass |> disposeOf
//            entropyMatrix |> disposeOf
//            gpuMask |> disposeOf
//            minimzer.Dispose(disposing)
//
//        base.Dispose(disposing)

