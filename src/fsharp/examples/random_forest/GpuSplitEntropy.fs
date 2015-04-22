(**
GPU functionality to train a random forest.
*)
module Tutorial.Fs.examples.RandomForest.GpuSplitEntropy

#nowarn "9"
#nowarn "51"

open System.Runtime.InteropServices
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound
open Tutorial.Fs.examples.RandomForest.DataModel

[<Literal>]
let private DEBUG = false

let DEFAULT_BLOCK_SIZE = 128
let DEFAULT_REDUCE_BLOCK_SIZE = 512

[<Struct; Align(8)>]
type ValueAndIndex =
    val Value : float
    val Index : int

    [<ReflectedDefinition>]
    new(value, index) =
        { Value = value
          Index = index }

    override this.ToString() = sprintf "Value: %f Index: %d" this.Value this.Index

    [<ReflectedDefinition>]
    /// Returns the maximum/minimum of the two values `a` and `b`
    /// If `sign` is 1 it computes the maximum, if -1 the minimum.
    /// In case `a` and `b` have the same value, the ValueAndIndex with
    /// largest index will be returned.
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

type MinOrMax =
    | Min
    | Max

    member this.Sign =
        match this with
        | Min -> -1
        | Max -> 1

    member this.OptFun a b = ValueAndIndex.Opt this.Sign a b
    member this.DefaultValue = ValueAndIndex.OPT_INFINITY this.Sign

[<ReflectedDefinition>]
let entropyTerm (x : int) =
    if x > 0 then
        let xf = float x
        xf * __nv_log2 xf
    elif x = 0 then 0.0
    else __nan()

(**
Matrix row scan resource.

Using block range scan from Alea Unbound.
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
Matrix row scan resource.

Using block range scan from Alea Unbound.
*)
type MatrixRowScanPrimitive(arch : DeviceArch, addressSize, blockThreads) =
    let blockRangeScan = DeviceScanPolicy.Create(arch, addressSize, blockThreads).BlockRangeScan

    [<ReflectedDefinition>]
    member this.BlockRangeScan blockOffset blockEnd (inputs : deviceptr<int>) =
        let tempStorage = blockRangeScan.TempStorage.AllocateShared()
        blockRangeScan.ConsumeRangeConsecutiveInclusive tempStorage (Iterator inputs) (Iterator inputs) (+) blockOffset
            blockEnd

    member this.BlockThreads = blockRangeScan.BlockThreads

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

    /// Returns array of length `n` of booleans where randomly sqrt `n` are true.
    static member SquareRootFeatureSelector (rnd : int -> int) (n : int) =
        let k = float n |> sqrt |> int

        let idcs = Array.randomSubIndices rnd n k
        let mask = Array.create n 0
        idcs |> Array.iter (fun idx -> mask.[idx] <- 1)
        mask

type EntropyOptimizationProblem =
    { numClasses : int
      numFeatures : int
      numSamples : int
      labelMatrix : Cublas.Matrix<Label>
      indexMatrix : Cublas.Matrix<int>
      indicesPerFeature : Indices[]
      labelsPerFeature : Labels[] }
    member this.Dispose() =
        this.labelMatrix.Dispose()
        this.indexMatrix.Dispose()

type EntropyOptimizationMemories =
    { gpuWeights : Cublas.Matrix<int>
      weightMatrix : Cublas.Matrix<int>
      nonZeroIdcsPerFeature : Cublas.Matrix<int>
      weightsPerFeatureAndClass : Cublas.Matrix<int>
      entropyMatrix : Cublas.Matrix<float>
      reducedOutput : Cublas.Matrix<ValueAndIndex>
      gpuMask : Cublas.Matrix<int>
      valuesAndIndices : ValueAndIndex[]
      mutable valuesAndIndicesHandle : GCHandle option }
    member this.Dispose() =
        if this.valuesAndIndicesHandle.IsSome then failwith "BUG"
        this.gpuWeights.Dispose()
        this.weightMatrix.Dispose()
        this.nonZeroIdcsPerFeature.Dispose()
        this.weightsPerFeatureAndClass.Dispose()
        this.entropyMatrix.Dispose()
        this.reducedOutput.Dispose()
        this.gpuMask.Dispose()

[<AOTCompile(SpecificArchs = "sm20;sm30;sm35")>]
type EntropyOptimizationModule(target, blockSize, reduceBlockSize) as this =
    inherit GPUModule(target)
    let primitiveScan =
        fun (options : CompileOptions) ->
            cuda { return MatrixRowScanPrimitive(options.MinimalArch, options.AddressSize, None) }
        |> this.GPUDefineResource
    let primitiveReduce =
        fun (options : CompileOptions) -> cuda { return MultiChannelReducePrimitive(options.MinimalArch, reduceBlockSize) }
        |> this.GPUDefineResource

    let summarizeWeights (weights : Weights) =
        let mutable sum = 0
        let mutable count = 0
        for weight in weights do
            sum <- sum + weight
            count <- count + (min weight 1)
        sum, count

    let lp numFeatures numCols =
        let blockDim = dim3 (blockSize)
        let gridDim = dim3 (divup numCols blockSize, numFeatures)
        LaunchParam(gridDim, blockDim)

    let streams = System.Collections.Generic.Queue<Stream>()
    static let instance blockSize reduceBlockSize = Lazy.Create <| fun _ -> new EntropyOptimizationModule(GPUModuleTarget.DefaultWorker, blockSize, reduceBlockSize)
    static member Default = (instance DEFAULT_BLOCK_SIZE DEFAULT_REDUCE_BLOCK_SIZE).Value
    // launching function cache, because we have many small kernel, and will
    // launch them many times, so it is good to do a lazy cache, set then in OnLoad() function.
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

    member this.CreateProblem(numberOfClasses : int, labelsPerFeature : Labels[], indicesPerFeature : Indices[]) =
        let worker = this.GPUWorker
        let labelMatrix = new Cublas.Matrix<_>(worker, labelsPerFeature)
        let indexMatrix = new Cublas.Matrix<_>(worker, indicesPerFeature)
        let numClasses = numberOfClasses
        let numFeatures = labelMatrix.NumRows
        let numSamples = labelMatrix.NumCols
        if numFeatures <> indexMatrix.NumRows || numSamples <> indexMatrix.NumCols then
            failwith "Dimensions of labels and indices per feature must agree"
        { numClasses = numClasses
          numFeatures = numFeatures
          numSamples = numSamples
          labelMatrix = labelMatrix
          indexMatrix = indexMatrix
          labelsPerFeature = labelsPerFeature
          indicesPerFeature = indicesPerFeature }

    member this.CreateMemories(problem : EntropyOptimizationProblem) =
        let worker = this.GPUWorker
        let numSamples = problem.numSamples
        let numFeatures = problem.numFeatures
        let numClasses = problem.numClasses
        // temporary space
        let gpuWeights = new Cublas.Matrix<_>(worker, 1, numSamples)
        let weightMatrix = new Cublas.Matrix<_>(worker, numFeatures, numSamples)
        let nonZeroIdcsPerFeature = new Cublas.Matrix<_>(worker, numFeatures, numSamples)
        let weightsPerFeatureAndClass = new Cublas.Matrix<_>(worker, numFeatures * numClasses, numSamples)
        let entropyMatrix = new Cublas.Matrix<_>(worker, numFeatures, numSamples)
        let reducedOutput = new Cublas.Matrix<_>(worker, numFeatures, divup numSamples reduceBlockSize)
        let gpuMask = new Cublas.Matrix<_>(worker, 1, numFeatures)
        let valuesAndIndices = Array.zeroCreate (numFeatures * (divup numSamples reduceBlockSize))
        { gpuWeights = gpuWeights
          weightMatrix = weightMatrix
          nonZeroIdcsPerFeature = nonZeroIdcsPerFeature
          weightsPerFeatureAndClass = weightsPerFeatureAndClass
          entropyMatrix = entropyMatrix
          reducedOutput = reducedOutput
          gpuMask = gpuMask
          valuesAndIndices = valuesAndIndices
          valuesAndIndicesHandle = None }

    [<ReflectedDefinition; Kernel>]
    member private this.OptAndArgOptKernelDoubleWithIdcs output numOutCols sign (matrix : deviceptr<float>) numCols
           numValid indices =
        primitiveReduce.Resource.OptAndArgOpt output numOutCols sign matrix numCols numValid indices true

    [<Kernel; ReflectedDefinition>]
    member private this.CumSumKernel (numCols : int) (numValid : int) (inputs : deviceptr<int>) =
        let blockOffset = blockIdx.x * numCols
        primitiveScan.Resource.BlockRangeScan blockOffset (blockOffset + numValid) inputs

    member this.CumSum(matrix : Cublas.Matrix<int>, numValid) =
        let lp = LaunchParam(matrix.NumRows, primitiveScan.Resource.BlockThreads)
        this.LaunchCumSumKernel.Value lp matrix.NumCols numValid (matrix.DeviceData.Ptr)

    [<Kernel; ReflectedDefinition>]
    member private this.LogicalWeightExpansionKernel (weightMatrix : deviceptr<int>) (indexMatrix : deviceptr<int>)
           (weights : deviceptr<int>) (numSamples : int) =
        let sampleIdx = blockIdx.x * blockDim.x + threadIdx.x
        let featureIdx = blockIdx.y
        if sampleIdx < numSamples then
            let matrixIdx = featureIdx * numSamples + sampleIdx
            let weight = weights.[indexMatrix.[matrixIdx]]
            weightMatrix.[matrixIdx] <- min weight 1

    [<Kernel; ReflectedDefinition>]
    member private this.FindNonZeroIndicesKernel (indexMatrix : deviceptr<int>) (cumWeightMatrix : deviceptr<int>)
           (numSamples : int) =
        let sampleIdx = blockIdx.x * blockDim.x + threadIdx.x
        let featureIdx = blockIdx.y
        if sampleIdx < numSamples then
            let weightIdx = featureIdx * numSamples + sampleIdx
            let weight = cumWeightMatrix.[weightIdx]

            let prevWeight =
                if sampleIdx = 0 then 0
                else cumWeightMatrix.[weightIdx - 1]
            if weight > prevWeight then indexMatrix.[featureIdx*numSamples + weight - 1] <- sampleIdx

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

    member this.FindNonZeroIndices(problem : EntropyOptimizationProblem, memories : EntropyOptimizationMemories) =
        let lp = lp problem.numFeatures problem.numSamples
        this.LaunchLogicalWeightExpansionKernel.Value lp memories.weightMatrix.DeviceData.Ptr
            problem.indexMatrix.DeviceData.Ptr memories.gpuWeights.DeviceData.Ptr problem.numSamples
        // cum sums over the weight matrix
        this.CumSum(memories.weightMatrix, problem.numSamples)
        this.LaunchFindNonZeroIndicesKernel.Value lp memories.nonZeroIdcsPerFeature.DeviceData.Ptr
            memories.weightMatrix.DeviceData.Ptr problem.numSamples
        if DEBUG then
            printfn "weight matrix:\n%A" (memories.weightMatrix.ToArray2D())
            printfn "nonZeroIdcsPerFeature:\n%A" (memories.nonZeroIdcsPerFeature.ToArray2D())

    member this.FindNonZeroIndices(problem : EntropyOptimizationProblem,
                                   param : (Stream * EntropyOptimizationMemories)[]) =
        let lp = lp problem.numFeatures problem.numSamples
        let launch = this.LaunchLogicalWeightExpansionKernel.Value
        param
        |> Array.iter
               (fun (stream, memories) ->
               launch (lp.NewWithStream(stream)) memories.weightMatrix.DeviceData.Ptr problem.indexMatrix.DeviceData.Ptr
                   memories.gpuWeights.DeviceData.Ptr problem.numSamples)
        let launch = this.LaunchCumSumKernel.Value
        param |> Array.iter (fun (stream, memories) ->
                     // cum sums over the weight matrix
                     let matrix = memories.weightMatrix
                     let numValid = problem.numSamples
                     let lp = LaunchParam(matrix.NumRows, primitiveScan.Resource.BlockThreads, 0, stream)
                     launch lp matrix.NumCols numValid matrix.DeviceData.Ptr)
        let launch = this.LaunchFindNonZeroIndicesKernel.Value
        param
        |> Array.iter
               (fun (stream, memories) ->
               launch (lp.NewWithStream(stream)) memories.nonZeroIdcsPerFeature.DeviceData.Ptr
                   memories.weightMatrix.DeviceData.Ptr problem.numSamples)

    member this.ExpandWeights(problem : EntropyOptimizationProblem, memories : EntropyOptimizationMemories, numValid : int) =
        let lp = lp problem.numFeatures numValid
        this.LaunchWeightExpansionKernel.Value lp memories.weightsPerFeatureAndClass.DeviceData.Ptr
            problem.labelMatrix.DeviceData.Ptr problem.indexMatrix.DeviceData.Ptr memories.gpuWeights.DeviceData.Ptr
            problem.numSamples problem.numClasses numValid memories.nonZeroIdcsPerFeature.DeviceData.Ptr
        if DEBUG then printfn "weightsPerFeatureAndClass:\n%A" (memories.weightsPerFeatureAndClass.ToArray2D())

    member this.ExpandWeights(problem : EntropyOptimizationProblem, param : (Stream * EntropyOptimizationMemories)[], numValids : int[]) =
        let launch = this.LaunchWeightExpansionKernel.Value
        param
        |> Array.iteri
               (fun i (stream, memories) ->
               let numValid = numValids.[i]
               let lp = lp problem.numFeatures numValid
               launch (lp.NewWithStream(stream)) memories.weightsPerFeatureAndClass.DeviceData.Ptr
                   problem.labelMatrix.DeviceData.Ptr problem.indexMatrix.DeviceData.Ptr
                   memories.gpuWeights.DeviceData.Ptr problem.numSamples problem.numClasses numValid
                   memories.nonZeroIdcsPerFeature.DeviceData.Ptr)

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

    /// Returns a GPU matrix that the caller needs to dispose of.
    /// Calculates Entropy includeing heuristic for avoiding small splits.
    member this.Entropy(problem : EntropyOptimizationProblem, memories : EntropyOptimizationMemories, options : EntropyOptimizationOptions, totals : int, numValid : int) =
        let roundingFactor = 10.0 ** (float options.Decimals)
        let minWeight = options.MinWeight problem.numClasses totals
        let cumSumsMatrix = memories.weightsPerFeatureAndClass
        let lp = (lp problem.numFeatures numValid).NewWithSharedMemorySize(problem.numClasses * sizeof<int>)
        this.LaunchEntropyKernel.Value lp memories.entropyMatrix.DeviceData.Ptr cumSumsMatrix.DeviceData.Ptr numValid
            problem.numSamples problem.numClasses minWeight roundingFactor memories.gpuMask.DeviceData.Ptr

    member this.Entropy(problem : EntropyOptimizationProblem, param : (Stream * EntropyOptimizationMemories)[], options : EntropyOptimizationOptions, totals : int[], numValids : int[]) =
        let roundingFactor = 10.0 ** (float options.Decimals)
        let launch = this.LaunchEntropyKernel.Value
        param
        |> Array.iteri
               (fun i (stream, memories) ->
               let totals = totals.[i]
               let numValid = numValids.[i]
               let minWeight = options.MinWeight problem.numClasses totals
               let cumSumsMatrix = memories.weightsPerFeatureAndClass
               let lp = (lp problem.numFeatures numValid).NewWithSharedMemorySize(problem.numClasses * sizeof<int>)
               launch (lp.NewWithStream(stream)) memories.entropyMatrix.DeviceData.Ptr cumSumsMatrix.DeviceData.Ptr
                   numValid problem.numSamples problem.numClasses minWeight roundingFactor
                   memories.gpuMask.DeviceData.Ptr)

    member this.MinimumEntropy(problem : EntropyOptimizationProblem, memories : EntropyOptimizationMemories, numValid : int, masks : int[]) =
        let optima() =
            let minOrMax = MinOrMax.Min
            let numRows = problem.numFeatures
            let numCols = problem.numSamples
            let numOutputCols = divup numCols reduceBlockSize
            let sign = minOrMax.Sign
            let numBlocks = divup numValid reduceBlockSize
            let gridSize = dim3 (numRows, numBlocks)
            let lp = LaunchParam(gridSize, dim3 reduceBlockSize)
            let matrix = memories.entropyMatrix
            let idcs = memories.nonZeroIdcsPerFeature
            let reducedOut = memories.reducedOutput
            this.LaunchOptAndArgOptKernelDoubleWithIdcs.Value lp reducedOut.DeviceData.Ptr numOutputCols sign
                matrix.DeviceData.Ptr numCols numValid idcs.DeviceData.Ptr
            let valuesAndIndices = memories.valuesAndIndices
            this.GPUWorker.Gather(reducedOut.DeviceData.Ptr, valuesAndIndices)
            let optima = Array.create numRows (ValueAndIndex.ofDouble System.Double.PositiveInfinity (problem.numSamples - 1))
            for rowIdx = 0 to numRows - 1 do
                let rowOffset = rowIdx * numOutputCols
                for colIdx = 0 to numBlocks - 1 do
                    optima.[rowIdx] <- minOrMax.OptFun optima.[rowIdx] valuesAndIndices.[rowOffset + colIdx]
            optima

        let optima = optima()

        let r = optima |> Array.mapi (fun i x ->
                        let fullEntropy = memories.entropyMatrix.Gather(i, numValid - 1)
                        if fullEntropy > 0.0 then x.Value, x.Index
                        else if masks.[i] <> 0 then 0.0, problem.numSamples - 1
                        else System.Double.PositiveInfinity, problem.numSamples - 1)
        if DEBUG then printfn "%A" r
        r

    member this.MinimumEntropy(problem : EntropyOptimizationProblem, param : (Stream * EntropyOptimizationMemories)[], numValids : int[]) =
        let minOrMax = MinOrMax.Min
        let numRows = problem.numFeatures
        let numCols = problem.numSamples
        let numOutputCols = divup numCols reduceBlockSize
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
                                   memories.entropyMatrix.DeviceData.Ptr.Handle + offset, size, stream.Handle))
                         cuSafeCall (cuStreamSynchronize (stream.Handle))
                         if fullEntropy > 0.0 then
                             let numBlocks = divup numValid reduceBlockSize
                             let gridSize = dim3 (numRows, numBlocks)
                             let lp = LaunchParam(gridSize, dim3 reduceBlockSize, 0, stream)
                             let matrix = memories.entropyMatrix
                             let idcs = memories.nonZeroIdcsPerFeature
                             let reducedOut = memories.reducedOutput
                             launch lp reducedOut.DeviceData.Ptr numOutputCols sign matrix.DeviceData.Ptr numCols
                                 numValid idcs.DeviceData.Ptr
                             None
                         else Some(Array.create problem.numFeatures (0.0, problem.numSamples - 1)))

        let size = numRows * numOutputCols * sizeof<ValueAndIndex> |> nativeint
        param |> Array.iteri (fun i (stream, memories) ->
                     if results.[i].IsNone then
                         let handle = GCHandle.Alloc(memories.valuesAndIndices, GCHandleType.Pinned)
                         cuSafeCall
                             (cuMemcpyDtoHAsync
                                  (handle.AddrOfPinnedObject(), memories.reducedOutput.DeviceData.Ptr.Handle, size,
                                   stream.Handle))
                         memories.valuesAndIndicesHandle <- Some handle)
        param |> Array.iteri (fun i (stream, memories) ->
                     if results.[i].IsNone then
                         let numValid = numValids.[i]
                         let numBlocks = divup numValid reduceBlockSize
                         let optima = Array.create numRows minOrMax.DefaultValue
                         cuSafeCall (cuStreamSynchronize (stream.Handle))
                         memories.valuesAndIndicesHandle.Value.Free()
                         memories.valuesAndIndicesHandle <- None
                         let valuesAndIndices = memories.valuesAndIndices
                         for rowIdx = 0 to numRows - 1 do
                             let rowOffset = rowIdx * numOutputCols
                             for colIdx = 0 to numBlocks - 1 do
                                 optima.[rowIdx] <- minOrMax.OptFun optima.[rowIdx]
                                                        valuesAndIndices.[rowOffset + colIdx]
                         results.[i] <- Some(optima |> Array.map (fun x -> (x.Value, x.Index))))
        results |> Array.choose id

    (**
    Optimization not using Streams, optimizing each element of the `weights`-array for itself.
    *)
    member this.Optimize(problem : EntropyOptimizationProblem, memories : EntropyOptimizationMemories, options : EntropyOptimizationOptions, weights : Weights) =
        this.GPUWorker.Eval <| fun _ ->
            let masks = options.FeatureSelector problem.numFeatures
            memories.gpuMask.Scatter(masks)
            memories.gpuWeights.Scatter(weights)
            this.FindNonZeroIndices(problem, memories)
            let sum, count = summarizeWeights weights
            this.ExpandWeights(problem, memories, count)
            this.CumSum(memories.weightsPerFeatureAndClass, count)
            this.Entropy(problem, memories, options, sum, count)
            this.MinimumEntropy(problem, memories, count, masks)

    (**
    Optimization using Streams. Optimizes the `weights`-array in parrallel.
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
                                      (memories.gpuWeights.DeviceData.Ptr.Handle, handle.AddrOfPinnedObject(), size,
                                       stream.Handle))
                             handle)

            let size = problem.numFeatures * sizeof<int> |> nativeint

            let handleMasks =
                param |> Array.map (fun (stream, memories) ->
                             let mask = options.FeatureSelector problem.numFeatures
                             let handle = GCHandle.Alloc(mask, GCHandleType.Pinned)
                             cuSafeCall
                                 (cuMemcpyHtoDAsync
                                      (memories.gpuMask.DeviceData.Ptr.Handle, handle.AddrOfPinnedObject(), size,
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
                         let matrix = memories.weightsPerFeatureAndClass
                         let numValid = counts.[i]
                         let lp = LaunchParam(matrix.NumRows, primitiveScan.Resource.BlockThreads, 0, stream)
                         launch lp matrix.NumCols numValid matrix.DeviceData.Ptr)
            this.Entropy(problem, param, options, sums, counts)
            let results = this.MinimumEntropy(problem, param, counts)
            handleWeights |> Array.iter (fun handle -> handle.Free())
            handleMasks |> Array.iter (fun (_, handle) -> handle.Free())
            results