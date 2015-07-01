(*** hide ***)
module Tutorial.Fs.examples.RandomForest.GpuSplitEntropy

#nowarn "9"
#nowarn "51"

open System.Runtime.InteropServices
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound
open Tutorial.Fs.examples.RandomForest.DataModel

(**
# Gpu Split Entropy

This file contains the function `entropyTerm` shared between CPU and GPU implementation, as well as all the functionality for the GPU implementation of `Optimizer`:

- Code in order to calculate minAndArgmin using the `BlockReduce` algorithms from Alea CUDA.
- Code to calculate cumulative sums using the `Alea.CUDA.Unbound` framework.
- Class `EntropyOptimizationOptions` containing the parameters for the entropy optimization.
- Records `EntropyOptimizationProblem` and `EntropyOptimizationMemories` entropy optimization related matrices.
- Class `EntropyOptimizationModule` with the two methods called `Optimize` (one of them using [CUDA-streams](http://devblogs.nvidia.com/parallelforall/how-overlap-data-transfers-cuda-cc/) one of them not) which are the GPU methods used for training the a decision tree.
*)
[<Literal>]
let private DEBUG = false

let DEFAULT_BLOCK_SIZE = 128
let DEFAULT_REDUCE_BLOCK_SIZE = 512

(**
## Entropy Calculation Function

Function calculating the entropy term for the count of a bin of a histogram.
Used in CPU and GPU implementation.
*)
[<ReflectedDefinition>]
let inline entropyTerm (x : int) =
    if x > 0 then
        let xf = float x
        xf * __nv_log2 xf
    elif x = 0 then 0.0
    else __nan()

(**
## Functionality to Calculate Min and Argmin, Resp. Max and Argmax

Type `ValueAndIndex` used to get back `min` and `argmin` resp. `max` and `argmax`.
*)
[<Struct; Align(8)>]
type ValueAndIndex =
    val Value : float
    val Index : int

    [<ReflectedDefinition>]
    new(value, index) =
        { Value = value
          Index = index }

    override this.ToString() = sprintf "Value: %f Index: %d" this.Value this.Index

(**
Returns the maximum/minimum of the two values `a` and `b`
If `sign` is 1 it computes the maximum, if -1 the minimum.
In case `a` and `b` have the same value, the `ValueAndIndex` with
the largest index will be returned. This is important to get the same results for the different implementation (CPU & GPU).
*)
    [<ReflectedDefinition>]
    static member inline Opt sign (a : ValueAndIndex) (b : ValueAndIndex) =
        if a.Value = b.Value then
            if a.Index > b.Index then a
            else b
        else
            let comparison =
                if (a.Value > b.Value) then sign
                else -sign
            if comparison > 0 then a
            else b

    [<ReflectedDefinition>]
    static member inline OPT_INFINITY sign = ValueAndIndex(System.Double.NegativeInfinity * (float sign), -1)

    [<ReflectedDefinition>]
    static member ofDouble (value : float) index = new ValueAndIndex(value, index)

    [<ReflectedDefinition>]
    static member ofSingle (value : float32) index = new ValueAndIndex(float value, index)

(**
Type to distinguish between minimum and maximum in `ValueAndIndex`.
*)
type MinOrMax =
    | Min
    | Max

    member this.Sign =
        match this with
        | Min -> -1
        | Max -> 1

    member this.OptFun a b = ValueAndIndex.Opt this.Sign a b
    member this.DefaultValue = ValueAndIndex.OPT_INFINITY this.Sign

(**
Matrix row scan resource, using block range scan from Alea-Unbound.
We will need it to find value and index of the minimal entropy.
*)
type MultiChannelReducePrimitive(arch : DeviceArch, reduceBlockSize) =
    let blockReduce = BlockReduce.WarpReductions<ValueAndIndex>(dim3 reduceBlockSize, arch)
    [<ReflectedDefinition>]
    member inline internal this.OptAndArgOpt (output : deviceptr<ValueAndIndex>) numOutputCols sign
                  (matrix : deviceptr<_>) numCols numValid (indices : deviceptr<int>) hasIndices =
        __static_assert (__is_static_constant hasIndices)
        let optFun = ValueAndIndex.Opt sign
        let defaultValue = ValueAndIndex.OPT_INFINITY sign
        let tempStorage = blockReduce.TempStorage.AllocateShared()
        let tid = threadIdx.x
        let colIdx = blockIdx.y * blockDim.x + tid

        let reduceInput =
            if colIdx < numValid then
                let matrixIdx = blockIdx.x * numCols + colIdx

                let index =
                    if hasIndices then indices.[matrixIdx]
                    else colIdx
                ValueAndIndex.ofDouble (__gfloat matrix.[matrixIdx]) index
            else defaultValue

        let aggregate = blockReduce.Reduce(tempStorage, reduceInput, optFun)
        if tid = 0 then output.[blockIdx.x * numOutputCols + blockIdx.y] <- aggregate

(**
## Functionality to Calculate the Cummulative Sum

Matrix row scan resource using block range scan from Alea-Unbound.
Here we use it to calculate a cumulative sum.
*)
type MatrixRowScanPrimitive(arch : DeviceArch, addressSize, blockThreads) =
    let blockRangeScan = DeviceScanPolicy.Create(arch, addressSize, blockThreads).BlockRangeScan

    [<ReflectedDefinition>]
    member this.BlockRangeScan blockOffset blockEnd (inputs : deviceptr<int>) =
        let tempStorage = blockRangeScan.TempStorage.AllocateShared()
        blockRangeScan.ConsumeRangeConsecutiveInclusive tempStorage (Iterator inputs) (Iterator inputs) (+) blockOffset
            blockEnd

    member this.BlockThreads = blockRangeScan.BlockThreads

(**
## Required Classes and Records

Entropy options.
See `MinWeight` function for purpose of `AbsMinWeight`, `RelMinDivisor` and `RelMinBound`.
`Decimals` is used to round to the given decimals in the entropy split.
`FeatureSelector` selects a (random) subset of features. Next to the random weights this is the second way to bring some randomness into the tree building.
*)
type EntropyOptimizationOptions =
    { AbsMinWeight : int
      RelMinDivisor : int
      RelMinBound : int
      Decimals : int
      // computes a subset of n features, i.e. maps n to a boolean vector of length n
      // returns integer instead of bool, for bool is not blittable type
      FeatureSelector : int -> int[] }

    static member Default =
        { AbsMinWeight = 1
          RelMinDivisor = 10
          RelMinBound = 25
          Decimals = 6
          FeatureSelector = fun n -> Array.create n 1 }

    static member DefaultWithSquareRootFeatureSelector = 
        { EntropyOptimizationOptions.Default 
          with FeatureSelector = EntropyOptimizationOptions.SquareRootFeatureSelector (getRngFunction 42) }

    member this.MinWeight numClasses total =
        let relativeMinWeight = min (total / (numClasses * this.RelMinDivisor)) this.RelMinBound
        max this.AbsMinWeight relativeMinWeight

    /// Returns array of length `n` of booleans where randomly $\sqrt$ `n` are true.
    static member SquareRootFeatureSelector (rnd : int -> int) (n : int) =
        let k = float n |> sqrt |> int

        let idcs = Array.randomSubIndices rnd n k
        let mask = Array.create n 0
        idcs |> Array.iter (fun idx -> mask.[idx] <- 1)
        mask

(**
Record containing the optimization parameters.
*)
type EntropyOptimizationProblem =
    { NumClasses : int
      NumFeatures : int
      NumSamples : int
      LabelMatrix : Cublas.Matrix<Label>
      IndexMatrix : Cublas.Matrix<int>
      IndicesPerFeature : Indices[]
      LabelsPerFeature : Labels[] }
    member this.Dispose() =
        this.LabelMatrix.Dispose()
        this.IndexMatrix.Dispose()

(**
Record containing matrices for entropy optimization.
*)
type EntropyOptimizationMemories =
    { GpuWeights : Cublas.Matrix<int>
      WeightMatrix : Cublas.Matrix<int>
      NonZeroIdcsPerFeature : Cublas.Matrix<int>
      WeightsPerFeatureAndClass : Cublas.Matrix<int>
      EntropyMatrix : Cublas.Matrix<float>
      ReducedOutput : Cublas.Matrix<ValueAndIndex>
      GpuMask : Cublas.Matrix<int>
      ValuesAndIndices : ValueAndIndex[]
      mutable ValuesAndIndicesHandle : GCHandle option }
    member this.Dispose() =
        if this.ValuesAndIndicesHandle.IsSome then failwith "BUG"
        this.GpuWeights.Dispose()
        this.WeightMatrix.Dispose()
        this.NonZeroIdcsPerFeature.Dispose()
        this.WeightsPerFeatureAndClass.Dispose()
        this.EntropyMatrix.Dispose()
        this.ReducedOutput.Dispose()
        this.GpuMask.Dispose()

(**
## Entropy Optimization Module

Entropy optimization module for GPU optimization with resp. without CUDA-streams.
*)
[<AOTCompile(SpecificArchs = "sm20;sm30;sm35")>]
type EntropyOptimizationModule(target) as this =
    inherit GPUModule(target)
    let primitiveScan =
        fun (options : CompileOptions) ->
            cuda { return MatrixRowScanPrimitive(options.MinimalArch, options.AddressSize, None) }
        |> this.GPUDefineResource
    let primitiveReduce =
        fun (options : CompileOptions) -> cuda { return MultiChannelReducePrimitive(options.MinimalArch, DEFAULT_REDUCE_BLOCK_SIZE) }
        |> this.GPUDefineResource

    let summarizeWeights (weights : Weights) =
        let mutable sum = 0
        let mutable count = 0
        for weight in weights do
            sum <- sum + weight
            count <- count + (min weight 1)
        sum, count

    let lp numFeatures numCols =
        let blockDim = dim3 (DEFAULT_BLOCK_SIZE)
        let gridDim = dim3 (divup numCols DEFAULT_BLOCK_SIZE, numFeatures)
        LaunchParam(gridDim, blockDim)

    let streams = System.Collections.Generic.Queue<Stream>()

(**
We create a factory method returning a default instance of this class.
*)
    static let instance DEFAULT_BLOCK_SIZE DEFAULT_REDUCE_BLOCK_SIZE = Lazy.Create <| fun _ -> new EntropyOptimizationModule(GPUModuleTarget.DefaultWorker)
    static member Default = (instance DEFAULT_BLOCK_SIZE DEFAULT_REDUCE_BLOCK_SIZE).Value

(**
Launching functions does take some time. As we have many small function calls this time adds up.
Hence we use a function cache using `Lazy` in order to minimize this unnecessary expenditure of time.
*)
    [<DefaultValue>]
    val mutable LaunchOptAndArgOptKernelDoubleWithIdcs : Lazy<LaunchParam -> deviceptr<ValueAndIndex> -> int -> int -> deviceptr<float> -> int -> int -> deviceptr<int> -> unit>
    [<DefaultValue>]
    val mutable LaunchCumSumKernel : Lazy<LaunchParam -> int -> int -> deviceptr<int> -> unit>
    [<DefaultValue>]
    val mutable LaunchLogicalWeightExpansionKernel : Lazy<LaunchParam -> deviceptr<int> -> deviceptr<int> -> deviceptr<int> -> int -> unit>
    [<DefaultValue>]
    val mutable LaunchFindNonZeroIndicesKernel : Lazy<LaunchParam -> deviceptr<int> -> deviceptr<int> -> int -> unit>
    [<DefaultValue>]
    val mutable LaunchWeightExpansionKernel : Lazy<LaunchParam -> deviceptr<int> -> deviceptr<Label> -> deviceptr<int> -> deviceptr<int> -> int -> int -> int -> deviceptr<int> -> unit>
    [<DefaultValue>]
    val mutable LaunchEntropyKernel : Lazy<LaunchParam -> deviceptr<float> -> deviceptr<int> -> int -> int -> int -> int -> float -> deviceptr<int> -> unit>

    override this.OnLoad(_) =
        this.LaunchOptAndArgOptKernelDoubleWithIdcs <- lazy this.GPULaunch<deviceptr<ValueAndIndex>, int, int, deviceptr<float>, int, int, deviceptr<int>> <@ this.OptAndArgOptKernelDoubleWithIdcs @>
        this.LaunchCumSumKernel <- lazy this.GPULaunch<int, int, deviceptr<int>> <@ this.CumSumKernel @>
        this.LaunchLogicalWeightExpansionKernel <- lazy this.GPULaunch<deviceptr<int>, deviceptr<int>, deviceptr<int>, int> <@ this.LogicalWeightExpansionKernel @>
        this.LaunchFindNonZeroIndicesKernel <- lazy this.GPULaunch<deviceptr<int>, deviceptr<int>, int> <@ this.FindNonZeroIndicesKernel @>
        this.LaunchWeightExpansionKernel <- lazy this.GPULaunch<deviceptr<int>, deviceptr<Label>, deviceptr<int>, deviceptr<int>, int, int, int, deviceptr<int>> <@ this.WeightExpansionKernel @>
        this.LaunchEntropyKernel <- lazy this.GPULaunch<deviceptr<float>, deviceptr<int>, int, int, int, int, float, deviceptr<int>> <@ this.EntropyKernel @>

(**
We need to dispose the created CUDA-streams.
*)
    override this.Dispose(disposing) =
        if disposing then streams |> Seq.iter (fun stream -> stream.Dispose())
        base.Dispose(disposing)

    // NOTE: non-thread-safe
    member this.BorrowStream() =
        if streams.Count > 0 then streams.Dequeue()
        else
            let stream = this.GPUWorker.CreateStream()
            stream

    member this.ReturnStream(stream) = streams.Enqueue(stream)

(**
Creates an object of class `EntropyOptimizationProblem` containing the optimization parameters. The record must be disposed of.
*)
    member this.CreateProblem(numberOfClasses : int, labelsPerFeature : Labels[], indicesPerFeature : Indices[]) =
        let worker = this.GPUWorker
        let labelMatrix = new Cublas.Matrix<_>(worker, labelsPerFeature)
        let indexMatrix = new Cublas.Matrix<_>(worker, indicesPerFeature)
        let numClasses = numberOfClasses
        let numFeatures = labelMatrix.NumRows
        let numSamples = labelMatrix.NumCols
        if numFeatures <> indexMatrix.NumRows || numSamples <> indexMatrix.NumCols then
            failwith "Dimensions of labels and indices per feature must agree"
        { NumClasses = numClasses
          NumFeatures = numFeatures
          NumSamples = numSamples
          LabelMatrix = labelMatrix
          IndexMatrix = indexMatrix
          LabelsPerFeature = labelsPerFeature
          IndicesPerFeature = indicesPerFeature }

(**
Allocates needed memory on GPU using the `Cublas.Matrix` class. The record must be disposed of.
*)
    member this.CreateMemories(problem : EntropyOptimizationProblem) =
        let worker = this.GPUWorker
        let numSamples = problem.NumSamples
        let numFeatures = problem.NumFeatures
        let numClasses = problem.NumClasses
        // temporary space
        let gpuWeights = new Cublas.Matrix<_>(worker, 1, numSamples)
        let weightMatrix = new Cublas.Matrix<_>(worker, numFeatures, numSamples)
        let nonZeroIdcsPerFeature = new Cublas.Matrix<_>(worker, numFeatures, numSamples)
        let weightsPerFeatureAndClass = new Cublas.Matrix<_>(worker, numFeatures * numClasses, numSamples)
        let entropyMatrix = new Cublas.Matrix<_>(worker, numFeatures, numSamples)
        let reducedOutput = new Cublas.Matrix<_>(worker, numFeatures, divup numSamples DEFAULT_REDUCE_BLOCK_SIZE)
        let gpuMask = new Cublas.Matrix<_>(worker, 1, numFeatures)
        let valuesAndIndices = Array.zeroCreate (numFeatures * (divup numSamples DEFAULT_REDUCE_BLOCK_SIZE))
        { GpuWeights = gpuWeights
          WeightMatrix = weightMatrix
          NonZeroIdcsPerFeature = nonZeroIdcsPerFeature
          WeightsPerFeatureAndClass = weightsPerFeatureAndClass
          EntropyMatrix = entropyMatrix
          ReducedOutput = reducedOutput
          GpuMask = gpuMask
          ValuesAndIndices = valuesAndIndices
          ValuesAndIndicesHandle = None }

(**
GPU kernel finding optimum and index of optimum using the `MultiChannelReducePrimitive` type.
The `sign` decides if maximum or minimum is chosen.
*)
    [<Kernel; ReflectedDefinition>]
    member private this.OptAndArgOptKernelDoubleWithIdcs output numOutCols sign (matrix : deviceptr<float>) numCols
           numValid indices =
        primitiveReduce.Resource.OptAndArgOpt output numOutCols sign matrix numCols numValid indices true

(**
GPU kernel & launching method, calculating the cumulative sum of values in a vector using the `MatrixRowScanPrimitive` type.
*)
    [<Kernel; ReflectedDefinition>]
    member private this.CumSumKernel (numCols : int) (numValid : int) (inputs : deviceptr<int>) =
        let blockOffset = blockIdx.x * numCols
        primitiveScan.Resource.BlockRangeScan blockOffset (blockOffset + numValid) inputs

    member this.CumSum(matrix : Cublas.Matrix<int>, numValid) =
        let lp = LaunchParam(matrix.NumRows, primitiveScan.Resource.BlockThreads)
        this.LaunchCumSumKernel.Value lp matrix.NumCols numValid (matrix.DeviceData.Ptr)

(**
Returns for every initial weight (before sorting samples) if it has been non-zero.
*)
    [<Kernel; ReflectedDefinition>]
    member private this.LogicalWeightExpansionKernel (weightMatrix : deviceptr<int>) (indexMatrix : deviceptr<int>)
           (weights : deviceptr<int>) (numSamples : int) =
        let sampleIdx = blockIdx.x * blockDim.x + threadIdx.x
        let featureIdx = blockIdx.y
        if sampleIdx < numSamples then
            let matrixIdx = featureIdx * numSamples + sampleIdx
            let weight = weights.[indexMatrix.[matrixIdx]]
            weightMatrix.[matrixIdx] <- min weight 1

(**
Writes indices where initial weight is non-zero into `indexMatrix`. Note weights which make up `cumWeightMatrix` have been $\in \{0,1\}$.
*)
    [<Kernel; ReflectedDefinition>]
    member private this.FindNonZeroIndicesKernel (indexMatrix : deviceptr<int>) (cumWeightMatrix : deviceptr<int>) (numSamples : int) =
        let sampleIdx = blockIdx.x * blockDim.x + threadIdx.x
        let featureIdx = blockIdx.y
        if sampleIdx < numSamples then
            let weightIdx = featureIdx * numSamples + sampleIdx
            let weight = cumWeightMatrix.[weightIdx]

            let prevWeight =
                if sampleIdx = 0 then 0
                else cumWeightMatrix.[weightIdx - 1]
            if weight > prevWeight then indexMatrix.[featureIdx*numSamples + weight - 1] <- sampleIdx

(**
Launches three kernels in order to find indices where the initial weights were non-zero.
As for every feature the samples are sorted increasingli according to their feature values and hence not anymore in the same order as the weights.
Weight-expansion finds to every sample the initial weight.
*)
    member this.FindNonZeroIndices(problem : EntropyOptimizationProblem, memories : EntropyOptimizationMemories) =
        let lp = lp problem.NumFeatures problem.NumSamples
        this.LaunchLogicalWeightExpansionKernel.Value lp memories.WeightMatrix.DeviceData.Ptr
            problem.IndexMatrix.DeviceData.Ptr memories.GpuWeights.DeviceData.Ptr problem.NumSamples
        // cum sums over the weight matrix
        this.CumSum(memories.WeightMatrix, problem.NumSamples)
        this.LaunchFindNonZeroIndicesKernel.Value lp memories.NonZeroIdcsPerFeature.DeviceData.Ptr
            memories.WeightMatrix.DeviceData.Ptr problem.NumSamples
        if DEBUG then
            printfn "weight matrix:\n%A" (memories.WeightMatrix.ToArray2D())
            printfn "nonZeroIdcsPerFeature:\n%A" (memories.NonZeroIdcsPerFeature.ToArray2D())

(**
Launches three kernels in order to find indices where the initial weights were non-zero. Uses CUDA-streams.
*)
    member this.FindNonZeroIndices(problem : EntropyOptimizationProblem, param : (Stream * EntropyOptimizationMemories)[]) =
        let lp = lp problem.NumFeatures problem.NumSamples
        let launch = this.LaunchLogicalWeightExpansionKernel.Value
        param
        |> Array.iter
               (fun (stream, memories) ->
               launch (lp.NewWithStream(stream)) memories.WeightMatrix.DeviceData.Ptr problem.IndexMatrix.DeviceData.Ptr
                   memories.GpuWeights.DeviceData.Ptr problem.NumSamples)
        let launch = this.LaunchCumSumKernel.Value
        param |> Array.iter (fun (stream, memories) ->
                     // cum sums over the weight matrix
                     let matrix = memories.WeightMatrix
                     let numValid = problem.NumSamples
                     let lp = LaunchParam(matrix.NumRows, primitiveScan.Resource.BlockThreads, 0, stream)
                     launch lp matrix.NumCols numValid matrix.DeviceData.Ptr)
        let launch = this.LaunchFindNonZeroIndicesKernel.Value
        param
        |> Array.iter
               (fun (stream, memories) ->
               launch (lp.NewWithStream(stream)) memories.NonZeroIdcsPerFeature.DeviceData.Ptr
                   memories.WeightMatrix.DeviceData.Ptr problem.NumSamples)

(**
Writes initial weights (before sorting samples) into `weightMatrix` and sorted in seperate groups for different classes.
Given the following table consisting of weights and labels (assumed to be ordered according to the feature-value):

    Weight: Label:
    1       0
    2       1
    1       1
    2       0

We get the following expanded weights (0 if feature has different class): 

    1 0 0 2 0 2 1 0
*)
    [<Kernel; ReflectedDefinition>]
    member private this.WeightExpansionKernel (weightMatrix : deviceptr<int>) (labelMatrix : deviceptr<Label>)
           (indexMatrix : deviceptr<int>) (weights : deviceptr<int>) (numSamples : int) (numClasses : int)
           (numValid : int) (nonZeroIdcsMatrix : deviceptr<int>) =
        let nonZeroIdx = blockIdx.x * blockDim.x + threadIdx.x
        let featureIdx = blockIdx.y
        let numFeatures = gridDim.y
        let rowOffset = featureIdx * numSamples
        if nonZeroIdx < numValid then
            let smallMatrixIdx = rowOffset + nonZeroIdx
            let sampleIdx = nonZeroIdcsMatrix.[smallMatrixIdx]
            let largeMatrixIdx = rowOffset + sampleIdx
            let weight = weights.[indexMatrix.[largeMatrixIdx]]
            let label = labelMatrix.[largeMatrixIdx]
            for classIdx = 0 to numClasses - 1 do
                let classOffset = numFeatures * numSamples * classIdx
                weightMatrix.[classOffset + smallMatrixIdx] <- if label = classIdx then weight
                                                               else 0

(**
Launches `ExpandWeights` in order to get initial weights (before sorting samples) into `WeightsPerFeatureAndClass`, seperate groups for different classes. See example in `WeightExpansionKernel`.
*)
    member this.ExpandWeights(problem : EntropyOptimizationProblem, memories : EntropyOptimizationMemories, numValid : int) =
        let lp = lp problem.NumFeatures numValid
        this.LaunchWeightExpansionKernel.Value lp memories.WeightsPerFeatureAndClass.DeviceData.Ptr
            problem.LabelMatrix.DeviceData.Ptr problem.IndexMatrix.DeviceData.Ptr memories.GpuWeights.DeviceData.Ptr
            problem.NumSamples problem.NumClasses numValid memories.NonZeroIdcsPerFeature.DeviceData.Ptr
        if DEBUG then printfn "weightsPerFeatureAndClass:\n%A" (memories.WeightsPerFeatureAndClass.ToArray2D())

(**
Launches `ExpandWeights` in order to get initial weights (before sorting samples) into `WeightsPerFeatureAndClass`, seperate groups for different classes. See example in `WeightExpansionKernel`. Uses CUDA-streams.
*)
    member this.ExpandWeights(problem : EntropyOptimizationProblem, param : (Stream * EntropyOptimizationMemories)[], numValids : int[]) =
        let launch = this.LaunchWeightExpansionKernel.Value
        param
        |> Array.iteri
               (fun i (stream, memories) ->
               let numValid = numValids.[i]
               let lp = lp problem.NumFeatures numValid
               launch (lp.NewWithStream(stream)) memories.WeightsPerFeatureAndClass.DeviceData.Ptr
                   problem.LabelMatrix.DeviceData.Ptr problem.IndexMatrix.DeviceData.Ptr
                   memories.GpuWeights.DeviceData.Ptr problem.NumSamples problem.NumClasses numValid
                   memories.NonZeroIdcsPerFeature.DeviceData.Ptr)

(**
Kernel calculating entropy for all features and samples, where samples are indexed using the x-dimension and features are indexed in the y-dimension.
*)
    [<Kernel; ReflectedDefinition>]
    member private this.EntropyKernel (entropyMatrix : deviceptr<float>) (cumSumsMatrix : deviceptr<int>)
           (numValid : int) (numSamples : int) (numClasses : int) (minWeight : int) (roundingFactor : float)
           (mask : deviceptr<int>) =
        let totals = __shared__.ExternArray()
        let featureIdx = blockIdx.y
        let mutable classIdx = threadIdx.x
        while classIdx < numClasses do
            // last entry of this feature's row in the submatrix corresponding to class classIdx
            let classTotal = cumSumsMatrix.[numSamples*(gridDim.y*classIdx + featureIdx) + numValid - 1]
            totals.[classIdx] <- classTotal
            classIdx <- classIdx + blockDim.x
        __syncthreads()
        let sampleIdx = blockIdx.x*blockDim.x + threadIdx.x
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
                        leftEntropy <- leftEntropy - (entropyTerm leftCount)
                        rightEntropy <- rightEntropy - (entropyTerm rightCount)
                        leftTotal <- leftTotal + leftCount
                        rightTotal <- rightTotal + rightCount
                    let leftEntropy = leftEntropy + (entropyTerm leftTotal)
                    let rightEntropy = rightEntropy + (entropyTerm rightTotal)
                    let rawEntropy =
                        (leftEntropy + rightEntropy) / (float (leftTotal + rightTotal))
                    let condition =
                        sampleIdx = upperBound
                        || (minWeight <= leftTotal && minWeight <= rightTotal)
                    if condition then
                        (__nv_nearbyint (rawEntropy * roundingFactor)) / roundingFactor
                    else infinity
                else infinity

(**
Method launching of the entropy kernel without CUDA-streams.
*)
    member this.Entropy(problem : EntropyOptimizationProblem, memories : EntropyOptimizationMemories, options : EntropyOptimizationOptions, totals : int, numValid : int) =
        let roundingFactor = 10.0 ** (float options.Decimals)
        let minWeight = options.MinWeight problem.NumClasses totals
        let cumSumsMatrix = memories.WeightsPerFeatureAndClass
        let lp = (lp problem.NumFeatures numValid).NewWithSharedMemorySize(problem.NumClasses * sizeof<int>)
        this.LaunchEntropyKernel.Value lp memories.EntropyMatrix.DeviceData.Ptr cumSumsMatrix.DeviceData.Ptr numValid
            problem.NumSamples problem.NumClasses minWeight roundingFactor memories.GpuMask.DeviceData.Ptr

(**
Method launching of the entropy kernel with CUDA-streams.
*)
    member this.Entropy(problem : EntropyOptimizationProblem, param : (Stream * EntropyOptimizationMemories)[], options : EntropyOptimizationOptions, totals : int[], numValids : int[]) =
        let roundingFactor = 10.0 ** (float options.Decimals)
        let launch = this.LaunchEntropyKernel.Value
        param
        |> Array.iteri
               (fun i (stream, memories) ->
               let totals = totals.[i]
               let numValid = numValids.[i]
               let minWeight = options.MinWeight problem.NumClasses totals
               let cumSumsMatrix = memories.WeightsPerFeatureAndClass
               let lp = (lp problem.NumFeatures numValid).NewWithSharedMemorySize(problem.NumClasses * sizeof<int>)
               launch (lp.NewWithStream(stream)) memories.EntropyMatrix.DeviceData.Ptr cumSumsMatrix.DeviceData.Ptr
                   numValid problem.NumSamples problem.NumClasses minWeight roundingFactor
                   memories.GpuMask.DeviceData.Ptr)

(**
Method searching for every feature the best split according to the already calculated entropies.
*)
    member this.MinimumEntropy(problem : EntropyOptimizationProblem, memories : EntropyOptimizationMemories, numValid : int, masks : int[]) =
        let optima() =
            let minOrMax = MinOrMax.Min
            let numRows = problem.NumFeatures
            let numCols = problem.NumSamples
            let numOutputCols = divup numCols DEFAULT_REDUCE_BLOCK_SIZE
            let sign = minOrMax.Sign
            let numBlocks = divup numValid DEFAULT_REDUCE_BLOCK_SIZE
            let gridSize = dim3 (numRows, numBlocks)
            let lp = LaunchParam(gridSize, dim3 DEFAULT_REDUCE_BLOCK_SIZE)
            let matrix = memories.EntropyMatrix
            let idcs = memories.NonZeroIdcsPerFeature
            let reducedOut = memories.ReducedOutput
            this.LaunchOptAndArgOptKernelDoubleWithIdcs.Value lp reducedOut.DeviceData.Ptr numOutputCols sign
                matrix.DeviceData.Ptr numCols numValid idcs.DeviceData.Ptr
            let valuesAndIndices = memories.ValuesAndIndices
            this.GPUWorker.Gather(reducedOut.DeviceData.Ptr, valuesAndIndices)
            let optima = Array.create numRows (ValueAndIndex.ofDouble System.Double.PositiveInfinity (problem.NumSamples - 1))
            for rowIdx = 0 to numRows - 1 do
                let rowOffset = rowIdx * numOutputCols
                for colIdx = 0 to numBlocks - 1 do
                    optima.[rowIdx] <- minOrMax.OptFun optima.[rowIdx] valuesAndIndices.[rowOffset + colIdx]
            optima

        let optima = optima()

        let r = optima |> Array.mapi (fun i x ->
                        let fullEntropy = memories.EntropyMatrix.Gather(i, numValid - 1)
                        if fullEntropy > 0.0 then x.Value, x.Index
                        else if masks.[i] <> 0 then 0.0, problem.NumSamples - 1
                        else System.Double.PositiveInfinity, problem.NumSamples - 1)
        if DEBUG then printfn "%A" r
        r

(**
Method searching for every feature the best split according to the already calculated entropies. This variant uses CUDA-streams.
*)
    member this.MinimumEntropy(problem : EntropyOptimizationProblem, param : (Stream * EntropyOptimizationMemories)[], numValids : int[]) =
        let minOrMax = MinOrMax.Min
        let numRows = problem.NumFeatures
        let numCols = problem.NumSamples
        let numOutputCols = divup numCols DEFAULT_REDUCE_BLOCK_SIZE
        let sign = minOrMax.Sign
        let size = sizeof<float> |> nativeint
        let launch = this.LaunchOptAndArgOptKernelDoubleWithIdcs.Value

        let results =
            param |> Array.mapi (fun i (stream, memories) ->
                         let numValid = numValids.[i]
                         let offset = nativeint ((numValid - 1) * sizeof<float>)
                         let mutable fullEntropy = nan
                         cuSafeCall
                             (cuMemcpyDtoHAsync
                                  (&&fullEntropy |> NativeInterop.NativePtr.toNativeInt,
                                   memories.EntropyMatrix.DeviceData.Ptr.Handle + offset, size, stream.Handle))
                         cuSafeCall (cuStreamSynchronize (stream.Handle))
                         if fullEntropy > 0.0 then
                             let numBlocks = divup numValid DEFAULT_REDUCE_BLOCK_SIZE
                             let gridSize = dim3 (numRows, numBlocks)
                             let lp = LaunchParam(gridSize, dim3 DEFAULT_REDUCE_BLOCK_SIZE, 0, stream)
                             let matrix = memories.EntropyMatrix
                             let idcs = memories.NonZeroIdcsPerFeature
                             let reducedOut = memories.ReducedOutput
                             launch lp reducedOut.DeviceData.Ptr numOutputCols sign matrix.DeviceData.Ptr numCols
                                 numValid idcs.DeviceData.Ptr
                             None
                         else Some(Array.create problem.NumFeatures (0.0, problem.NumSamples - 1)))

        let size = numRows * numOutputCols * sizeof<ValueAndIndex> |> nativeint
        param |> Array.iteri (fun i (stream, memories) ->
                     if results.[i].IsNone then
                         let handle = GCHandle.Alloc(memories.ValuesAndIndices, GCHandleType.Pinned)
                         cuSafeCall
                             (cuMemcpyDtoHAsync
                                  (handle.AddrOfPinnedObject(), memories.ReducedOutput.DeviceData.Ptr.Handle, size,
                                   stream.Handle))
                         memories.ValuesAndIndicesHandle <- Some handle)
        param |> Array.iteri (fun i (stream, memories) ->
                     if results.[i].IsNone then
                         let numValid = numValids.[i]
                         let numBlocks = divup numValid DEFAULT_REDUCE_BLOCK_SIZE
                         let optima = Array.create numRows minOrMax.DefaultValue
                         cuSafeCall (cuStreamSynchronize (stream.Handle))
                         memories.ValuesAndIndicesHandle.Value.Free()
                         memories.ValuesAndIndicesHandle <- None
                         let valuesAndIndices = memories.ValuesAndIndices
                         for rowIdx = 0 to numRows - 1 do
                             let rowOffset = rowIdx * numOutputCols
                             for colIdx = 0 to numBlocks - 1 do
                                 optima.[rowIdx] <- minOrMax.OptFun optima.[rowIdx]
                                                        valuesAndIndices.[rowOffset + colIdx]
                         results.[i] <- Some(optima |> Array.map (fun x -> (x.Value, x.Index))))
        results |> Array.choose id

(**
Optimization not using CUDA-streams.
Returns an array consisting of entropy and split-index by:

- Masking features using the `FeatureSelector`.
- Scattering memory to GPU.
- Running `FindNonZeroIndices`:
    - Launches the `LogicalWeightExpansionKernel`: returning if the initial weights of a given sample has been zero before sorting.
    - Make a cumulative sum.
    - Launch the `NonZeroIndicesKernel`.
- Summing up weights.
- Expanding weights using the `ExpandWeights`, i.e. returning the index of a given sample before sorting and ordering them according to their labels.
- Makeing a cumulative sum using the `CumSumKernel` kernel.
- Calculating the Entropy for all features and samples using the `Entropy` kernel.
- Calculating the minimum entropy using the `OptAndArgOptKernelDoubleWithIdcs` kernel.
*)
    member this.Optimize(problem : EntropyOptimizationProblem, memories : EntropyOptimizationMemories, options : EntropyOptimizationOptions, weights : Weights) =
        this.GPUWorker.Eval <| fun _ ->
            let masks = options.FeatureSelector problem.NumFeatures
            memories.GpuMask.Scatter(masks)
            memories.GpuWeights.Scatter(weights)
            this.FindNonZeroIndices(problem, memories)
            let sum, count = summarizeWeights weights
            this.ExpandWeights(problem, memories, count)
            this.CumSum(memories.WeightsPerFeatureAndClass, count)
            this.Entropy(problem, memories, options, sum, count)
            this.MinimumEntropy(problem, memories, count, masks)

(**
Optimization using CUDA-streams.
Returns an array consisting of entropy and split-index in a similar way as the method above but using CUDA-streams.
*)
    member this.Optimize(problem : EntropyOptimizationProblem, param : (Stream * EntropyOptimizationMemories)[], options : EntropyOptimizationOptions, weights : Weights[]) =
        this.GPUWorker.Eval <| fun _ ->
            let size = weights.[0].Length * sizeof<int> |> nativeint

            let handleWeights =
                param |> Array.mapi (fun i (stream, memories) ->
                             let weights = weights.[i]
                             let handle = GCHandle.Alloc(weights, GCHandleType.Pinned)
                             cuSafeCall
                                 (cuMemcpyHtoDAsync
                                      (memories.GpuWeights.DeviceData.Ptr.Handle, handle.AddrOfPinnedObject(), size,
                                       stream.Handle))
                             handle)

            let size = problem.NumFeatures * sizeof<int> |> nativeint

            let handleMasks =
                param |> Array.map (fun (stream, memories) ->
                             let mask = options.FeatureSelector problem.NumFeatures
                             let handle = GCHandle.Alloc(mask, GCHandleType.Pinned)
                             cuSafeCall
                                 (cuMemcpyHtoDAsync
                                      (memories.GpuMask.DeviceData.Ptr.Handle, handle.AddrOfPinnedObject(), size,
                                       stream.Handle))
                             mask, handle)
            this.FindNonZeroIndices(problem, param)
            let sums, counts =
                weights
                |> Array.Parallel.map summarizeWeights
                |> Array.unzip
            this.ExpandWeights(problem, param, counts)
            let launch = this.LaunchCumSumKernel.Value
            param |> Array.iteri (fun i (stream, memories) ->
                         // cum sums over the weight matrix
                         let matrix = memories.WeightsPerFeatureAndClass
                         let numValid = counts.[i]
                         let lp = LaunchParam(matrix.NumRows, primitiveScan.Resource.BlockThreads, 0, stream)
                         launch lp matrix.NumCols numValid matrix.DeviceData.Ptr)
            this.Entropy(problem, param, options, sums, counts)
            let results = this.MinimumEntropy(problem, param, counts)
            handleWeights |> Array.iter (fun handle -> handle.Free())
            handleMasks |> Array.iter (fun (_, handle) -> handle.Free())
            results