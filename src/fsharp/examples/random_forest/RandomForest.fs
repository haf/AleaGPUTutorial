(*** hide ***)
module Tutorial.Fs.examples.RandomForest.RandomForest

open Alea.CUDA.Utilities
open Tutorial.Fs.examples.RandomForest.DataModel
open Tutorial.Fs.examples.RandomForest.GpuSplitEntropy
open Tutorial.Fs.examples.RandomForest.Array

[<Literal>]
let private DEBUG = false

(**
# Random Forest

This file contains the CPU tree-building and evaluating algorithms as well as infrastructure code integrating the GPU code:

- Types for labels.
- Evaluation function: `forecastTree` and `forecast`.
- Several functions to calculate histograms & entropy needed in the function `optimizeFeatures` the CPU implementation of `Optimizer`.
- The recursive function `trainTrees` training trees using a `sortedTrainingSet` as well as `weights` and a `IEntropyOptimizer` with the method `Optimizer`.
    
- Infrastructure code such as the type `EntropyDevice` creating an instance of `IEntropyOptimizer`.
- The end-user functions `randomForestClassifier` and `randomStumpsClassifier` training a random forest.

Training data is expected in the `LabeledFeaterSet` form (see type below). It helps to transform `LabeledSamples` (input format) into `SortedFeatures` (working format).
The input format has the form:

    [|([| sepalLength; sepalWidth; petalLength; petalWidth |], 1)|]

i.e. an array of tuples consisting of a array with features (floats) and a label (int).
*)

let sortFeature (labels : Labels) (featureValues : FeatureArray) =
    let tupleArray = featureValues |> Array.mapi (fun i value -> (value, labels.[i], i))
    tupleArray |> Array.sortInPlace
    tupleArray |> Array.unzip3

let sortAllFeatures labels domains = domains |> Array.Parallel.map (sortFeature labels)

(**
The `LabeledFeatureSet` contains the training data which can be saved in three different ways, where only the first is important to the end user:

1. As `LabeledSamples`, i.e. an array containing tuples of feature vectors and a label and is the most canonical way for entering a dataset.
2. As `LabeledFeatures`, i.e. a tuple of a `FeatureArray` (where instead of having features of a sample together, all features of different samples are saved in an array) and an array of Labels.
3. As `SortedFeatures`, where for every feature all the values are sorted in ascending order as well as labelled and completed with the index before sorting. It is mainly used for finding the best split. The indices are needed in order to find the weights before ordering the features.

The canonical form of the input data is represented by `LabeledSamples`, which is an array of tuples `Sample * Label`. Recall that a Sample is just an array of float vectors and `Label` is an integer value representing the class of the sample. For the Iris data set a `LabeledSample` is a tuple of a float vector of four values and an integer `([|sepalLength; sepalWidth; petalLength; petalWidth|], 1)`.

The type `SortedFeatures` is used to represent the sorted features during the implementation of the algorithm. For each feature, it holds the feature values from all samples in a `FeatureArray`, the corresponding labels in `Labels` and the indices representing the old position of the features.

Note that both the `Sample` and `FeatureArray` are both float arrays. The reason why we distinguish them is because a `Sample` is an observation of one feature value per feature, whereas a `FeatureArray` holds the value of a single feature for a collection of samples. If we stack the samples row by row in a matrix, the rows would correspond to samples, whereas the columns would correspond to feature arrays.
*)
type LabeledFeatureSet =
    | LabeledSamples of LabeledSample[]
    | LabeledFeatures of FeatureArrays*Labels
    | SortedFeatures of (FeatureArray*Labels*Indices)[]

    member this.Labels =
        this |> function
        | LabeledSamples trainingSet ->
            trainingSet
            |> Array.unzip
            |> snd
        | LabeledFeatures(_, labels) -> labels
        | SortedFeatures features -> features.[0] |> (fun (_, x, _) -> x)

    member this.Length = this.Labels |> Array.length
    member this.NumClasses = Array.max this.Labels + 1
    member this.Sorted =
        this |> function
        | LabeledSamples trainingSet ->
            let numSamples = this.Length
            let samples, labels = trainingSet |> Array.unzip
            let domains = Array.init samples.[0].Length (fun i -> Array.init numSamples (fun j -> samples.[j].[i]))
            SortedFeatures(sortAllFeatures labels domains)
        | LabeledFeatures(domains, labels) -> SortedFeatures(sortAllFeatures labels domains)
        | SortedFeatures _ -> this

(**
Forecasts the label of a `sample` by traversing the `tree`.
*)
let rec forecastTree tree (sample : Sample) =
    match tree with
    | Tree.Leaf label -> label
    | Tree.Node(low, split, high) ->
        if sample.[split.Feature] <= split.Threshold then forecastTree low sample 
        else forecastTree high sample

(**
Forecasts the label of a `sample` by majority voting on labels received by trees in `model`.
*)
let forecast model sample : Label =
    match model with
    | RandomForest(trees, numClasses) ->
        (Array.zeroCreate numClasses, trees)
        ||> Seq.fold (fun hist tree ->
                let label = forecastTree tree sample
                hist.[label] <- hist.[label] + 1
                hist)
        |> maxAndArgMax |> snd

(**
Function returning an array of histograms on `labels` weighted by `weights`.
*)
let cumHistograms numClasses (labels : Labels) (weights : Weights) : LabelHistogram seq =
    ((Array.zeroCreate numClasses, 0), seq { 0..weights.GetUpperBound(0) })
    ||> Seq.scan (fun (hist, sum) i ->
            let weight = weights.[i]
            let label = labels.[i]
            hist.[label] <- hist.[label] + weight
            (hist, sum + weight))
    |> Seq.skip 1

(**
Function returning a histogram on `labels` weighted by `weights`.
*)
let countTotals numClasses labels weights = Seq.last <| cumHistograms numClasses labels weights

(**
Function calculating the entropy for a given branch, summing entropies for all bins in the histogram.
*)
let private entropyTermSum node =
    let hist, total = node
    if (total = 0) then 0.0
    else
        let sumLogs = (0.0, hist) ||> Seq.fold (fun e c -> e - (entropyTerm c))
        (sumLogs + entropyTerm total)

(**
Sums up the entropy for all new branches after this split (only two for the continuous case).
*)
let entropy total (nodes : LabelHistogram seq) =
    if (total = 0) then 0.0
    else
        let entropySum = (0.0, nodes) ||> Seq.fold (fun e node -> e + (entropyTermSum node))
        entropySum / (float total)

(**
Returns the complementary histogram, i.e. the difference between the `total`- and the `left`-histogram.
*)
let complementHist (total : LabelHistogram) (left : LabelHistogram) : LabelHistogram =
    let totalHist, totalCount = total
    let hist, count = left
    (totalHist, hist) ||> Array.map2 (-), totalCount - count

(**
Calculates the entropy for all the possible splits in an ordered feature. Returning them in a sequence.
*)
let splitEntropies (mask : bool seq) (countsPerSplit : LabelHistogram seq) (totals : LabelHistogram) =
    let complement = complementHist totals
    let entropy = entropy (totals |> snd)
    (mask, countsPerSplit) ||> Seq.map2 (fun isValid low ->
        if isValid then
            let histograms = seq { yield low; yield complement low }
            entropy histograms
        else System.Double.PositiveInfinity)

(**
Returns the value in the middle between `splidIdx` and next non-empty index if it exists, else returns `None`.
*)
let featureArrayThreshold weights (featureArray : _[]) splitIdx =
    let nextIdx = Array.findNextNonZeroIdx weights (splitIdx + 1)
    match nextIdx with
    | None -> None
    | Some nextSplitIdx -> Some((featureArray.[splitIdx] + featureArray.[nextSplitIdx]) * 0.5)

(**
Returns a histogram from `weights` on `labels`.
*)
let countSamples (weights : Weights) numClasses (labels : Labels) =
    let hist = Array.zeroCreate numClasses
    for i = 0 to weights.GetUpperBound(0) do
        let weight = weights.[i]
        let label = labels.[i]
        hist.[label] <- hist.[label] + weight
    hist

(**
Returns the `class`/`label` with the highest `weight`.
*)
let findMajorityClass weights numClasses labels =
    countSamples weights numClasses labels
    |> maxAndArgMax |> snd

(**
Masks out splits which are known to have non optimal entropy.
It is not necessary, but increases the performance of the CPU implementation.
*)
let entropyMask (weights : _[]) (labels : _[]) totalWeight absMinWeight =
    let findNextWeight = Array.findNextNonZeroIdx weights
    ((false, findNextWeight 0, 0), seq { 0..weights.GetUpperBound(0) })
    ||> Seq.scan (fun (_, nextNonZero, lowWeight) i ->
            match nextNonZero with
            | Some idx ->
                if i < idx then // i has zero weight
                    (false, nextNonZero, lowWeight)
                else // i = idx and i has non-zero weight
                    let nextNonZero = findNextWeight (i + 1)
                    let lowWeight = lowWeight + weights.[i]
                    match nextNonZero with
                    | Some idx ->
                        let labelChange = labels.[i] <> labels.[idx] // test whether label changes at next valid element
                        let highWeight = totalWeight - lowWeight
                        let weightChange = lowWeight = absMinWeight || highWeight = absMinWeight
                        let enoughWeight = lowWeight >= absMinWeight && highWeight >= absMinWeight
                        ((labelChange || weightChange) && enoughWeight, nextNonZero, lowWeight)
                    | None -> (true, nextNonZero, lowWeight) // i is the last valid element
            | None -> (false, nextNonZero, lowWeight) // no valid elements left
            )
    |> Seq.skip 1
    |> Seq.map (fun (x, _, _) -> x)

(**
Discriminate between `Sequential` and `Parallel` CPU mode.
*)
type CpuMode =
    | Sequential
    | Parallel

(**
Abstract class for the entropy optimizer granting the same interface for CPU and GPU implementation.
*)
type IEntropyOptimizer =
    inherit System.IDisposable
    abstract Optimize : Weights[] -> (float * int)[][]

(**
CPU implementation of the optimizer function. Uses `Array.Parallel.mapi` for parallelization.
Returns for every feature the best split entropy to and its corresponding index.
The following steps are taken:

- Reorder the weights such that every weight is stored at the same index as the corresponding sample, which is done for every feature independently.
- Project all features and weights which are non-zero to the beginning of the array.
- Implement the helper functions
    - mapper which, depending on the mode, is either Array.mapi or Array.Parallel.mapi.
    - translator which translates the index after projection to the post projection index, or the last index in case for the last non-zero index.
    - A mapping function, the argument for the mapper, which for all non-masked features performs the following:
        - Get the weights.
        - Create the histogram for every split possibility for the given feature.
        - Mask splits which are clearly not optimal. This is not necessary, but helps speeding up the CPU implementation.
        - Calculate the entropy for remaining split possibilities.
        - Return the lowest entropy and its index and translates the index to its value before the projection using the translator function.

- Select a subset of the features using a FeatureSelector.
- Calculate the entropy for the remaining split possibilities using the helper functions.
*)
let optimizeFeatures (mode : CpuMode) (options : EntropyOptimizationOptions) numClasses (labelsPerFeature : Labels[])
    (indicesPerFeature : Indices[]) weights =
    // remove zero weights
    let weightsPerFeature = Array.expandWeights indicesPerFeature weights
    let nonZeroIdcsPerFeature = Array.findNonZeroWeights weightsPerFeature

    let mapper f =
        match mode with
        | Sequential -> Array.mapi f
        | Parallel -> Array.Parallel.mapi f

    let projector (sources : _[][]) =
        sources |> mapper (fun i source -> source |> Array.projectIdcs nonZeroIdcsPerFeature.[i])
    let weightsPerFeature = projector weightsPerFeature
    let labelsPerFeature = projector labelsPerFeature
    let upperBoundNonZero = nonZeroIdcsPerFeature.[0].GetUpperBound(0)
    let upperBoundWeights = weights.GetUpperBound(0)

    let translator featureIdx (x, i) =
        if i = upperBoundNonZero then x, upperBoundWeights
        else x, nonZeroIdcsPerFeature.[featureIdx].[i]

    let totals = countTotals numClasses labelsPerFeature.[0] weightsPerFeature.[0]
    let total = (totals |> snd)
    // heuristic for avoiding small splits
    let combinedMinWeight = options.MinWeight numClasses total
    let rounding (value : float) = System.Math.Round(value, options.Decimals)
    let featureSelectingMask = options.FeatureSelector(labelsPerFeature |> Array.length)

    // for each feature find minimum entropy and corresponding index
    let mapping =
        (fun featureIdx labels ->
        if (featureSelectingMask.[featureIdx]) <> 0 then
            let featureWeights = weightsPerFeature.[featureIdx]
            let countsPerSplit = cumHistograms numClasses labels featureWeights
            let mask = entropyMask featureWeights labels total combinedMinWeight
            let entropyPerSplit = splitEntropies mask countsPerSplit totals
            entropyPerSplit
            |> Seq.map rounding
            |> minAndArgMin
            |> translator featureIdx
        else (infinity, upperBoundWeights))

    let r = labelsPerFeature |> mapper mapping
    if DEBUG then printfn "%A" r
    r

(**
The methods `restrict`, `restrictBelow` and `restrictAbove` are used in order to set weights for samples to zero, if they do not belong the current branch.
*)
let restrict startIdx count (source : _[]) =
    let target = Array.zeroCreate source.Length
    Array.blit source startIdx target startIdx count
    target

let restrictBelow idx (source : _[]) = restrict 0 (idx + 1) source
let restrictAbove idx (source : _[]) = restrict idx (source.Length - idx) source

(**
Main function to train a decision tree.

Using recursion the following steps are done:

- Calculate triples: entropy, split index and feature index with the minimal split entropy.
- Find middle between the values to split using the function `featureArrayThreshold`. If no non-zero index exists, stop branching and create a leaf.
- Set weights for samples with value below / above the split-value to zero for the lower / upper branch of the next iteration.
- If depth = 0 then return the tree, else train new trees on lower and upper branch.

*Note:* `FeatureArrays` need to be sorted in ascending order.
*)
let rec trainTrees depth (optimizer : IEntropyOptimizer) numClasses
        (sortedTrainingSet : (FeatureArray*Labels*Indices)[]) (weights : Weights[]) =
    let triples =
        if depth = 0 then weights |> Array.map (fun weights -> nan, weights.GetUpperBound(0), 0)
        else
            optimizer.Optimize(weights)
                |> Array.map (fun results ->
                    results
                    |> Array.mapi (fun i (entropy, splitIdx) -> entropy, splitIdx, i)
                    |> Array.minBy (fun (entropy, _, _) -> entropy))

    let trees0 =
        triples |> Array.mapi (fun i (entropy, splitIdx, featureIdx) ->
            let weights = weights.[i]
            let featureArray, labels, indices = sortedTrainingSet.[featureIdx]
            let sortedWeights = weights |> Array.projectIdcs indices
            let threshold = featureArrayThreshold sortedWeights featureArray splitIdx
            if DEBUG then
                printfn "depth: %A, entropy: %A, splitIdx: %A, featureIdx: %A" depth entropy splitIdx
                    featureIdx
                printf "Labels:\n["
                (sortedWeights, labels) ||> Array.iteri2 (fun i weight label ->
                                                if weight <> 0 then printf " (%A * %A) " label weight
                                                if (i = splitIdx) then printf "|")
                printfn "]"
            match threshold with
            | Some num ->
                // set weights to 0 for elements which aren't included in left and right branches respectively
                let lowWeights = restrictBelow splitIdx sortedWeights
                let highWeights = restrictAbove (splitIdx + 1) sortedWeights
                if DEBUG then
                    printfn "Low  counts: %A" (countSamples lowWeights numClasses labels)
                    printfn "High counts: %A" (countSamples highWeights numClasses labels)
                let lowWeights = Array.permByIdcs indices lowWeights
                let highWeights = Array.permByIdcs indices highWeights

                let set (lowTree : Tree) (highTree : Tree) =
                    match lowTree, highTree with
                    | Leaf lowLabel, Leaf highLabel when lowLabel = highLabel -> i, Tree.Leaf lowLabel
                    | _ ->
                        i,
                        Tree.Node(lowTree,
                                  { Feature = featureIdx
                                    Threshold = num }, highTree)
                None, Some lowWeights, Some highWeights, Some set
            | None ->
                let label = findMajorityClass sortedWeights numClasses labels
                Some(Tree.Leaf label), None, None, None)

    let trees() = trees0 |> Array.choose (fun (tree, _, _, _) -> tree)
    let lowWeights = trees0 |> Array.choose (fun (_, lowWeights, _, _) -> lowWeights)
    let highWeights = trees0 |> Array.choose (fun (_, _, highWeights, _) -> highWeights)
    if lowWeights.Length = 0 then trees()
    else
        let lowTrees = trainTrees (depth - 1) optimizer numClasses sortedTrainingSet lowWeights
        let highTrees = trainTrees (depth - 1) optimizer numClasses sortedTrainingSet highWeights
        let setFuncs = trees0 |> Array.choose (fun (_, _, _, set) -> set)
        for i = 0 to lowTrees.Length - 1 do
            let set = setFuncs.[i]
            let lowTree = lowTrees.[i]
            let highTree = highTrees.[i]
            let originIdx, tree = set lowTree highTree
            trees0.[originIdx] <- Some tree, None, None, None
        trees()

let trainStump optimizer numClasses sortedTrainingSet weights =
    trainTrees 1 optimizer numClasses sortedTrainingSet weights

(**
Types for deciding which computational device should run in what mode and method to create an entropy device.
*)
type GpuModuleProvider =
    | DefaultModule
    | Specified of gpuModule : GpuSplitEntropy.EntropyOptimizationModule
    member this.GetModule() =
        match this with
        | DefaultModule -> GpuSplitEntropy.EntropyOptimizationModule.Default
        | Specified gpuModule -> gpuModule

type GpuMode =
    | SingleWeight
    | SingleWeightWithStream of numStreams : int

type PoolMode =
    | EqualPartition

type EntropyDevice =
    // GPU mode
    | GPU of mode : GpuMode * provider : GpuModuleProvider
    /// CPU mode
    | CPU of mode : CpuMode
    /// Problem will be split in n partitions and is run on a pool of devices.
    | Pool of mode : PoolMode * devices : EntropyDevice seq

    member this.Create options numClasses (sortedTrainingSet : (_*Labels*Indices)[]) : IEntropyOptimizer =
        let _, labelsPerFeature, indicesPerFeature = sortedTrainingSet |> Array.unzip3
        match this with
        | GPU(mode, provider) ->
            let gpuModule = provider.GetModule()
            gpuModule.GPUForceLoad()
            let worker = gpuModule.GPUWorker
            match mode with
            | SingleWeight ->
                let problem, memories =
                    worker.Eval <| fun _ ->
                        let problem = gpuModule.CreateProblem(numClasses, labelsPerFeature, indicesPerFeature)
                        let memories = gpuModule.CreateMemories(problem)
                        problem, memories
                { new IEntropyOptimizer with
                      member this.Optimize(weights) =
                          gpuModule.GPUWorker.Eval
                          <| fun _ ->
                              weights
                              |> Array.map (fun weights -> gpuModule.Optimize(problem, memories, options, weights))
                  interface System.IDisposable with
                      member this.Dispose() =
                          memories.Dispose()
                          problem.Dispose() }
            | SingleWeightWithStream numStreams ->
                let problem, param =
                    worker.Eval <| fun _ ->
                        let problem = gpuModule.CreateProblem(numClasses, labelsPerFeature, indicesPerFeature)

                        let param =
                            Array.init numStreams (fun _ ->
                                let stream = gpuModule.BorrowStream()
                                let memories = gpuModule.CreateMemories(problem)
                                stream, memories)
                        problem, param
                { new IEntropyOptimizer with
                      member this.Optimize(weights) =
                          let div = weights.Length / numStreams
                          let rem = weights.Length % numStreams
                          seq {
                              for i = 0 to div - 1 do
                                  let weights = Array.sub weights (i * numStreams) numStreams
                                  yield gpuModule.Optimize(problem, param, options, weights)
                              if rem > 0 then
                                  let weights = Array.sub weights (div * numStreams) rem
                                  let param = Array.sub param 0 rem
                                  yield gpuModule.Optimize(problem, param, options, weights)
                          }
                          |> Seq.toArray
                          |> Array.concat
                  interface System.IDisposable with
                      member this.Dispose() =
                          problem.Dispose()
                          param |> Array.iter (fun (stream, memories) ->
                                       gpuModule.ReturnStream(stream)
                                       memories.Dispose()) }
        | CPU mode ->
            { new IEntropyOptimizer with
                  member this.Optimize(weights) =
                      weights
                      |> Array.map
                             (fun weights ->
                             optimizeFeatures mode options numClasses labelsPerFeature indicesPerFeature weights)
              interface System.IDisposable with
                  member this.Dispose() = () }
        | Pool(mode, devices) ->
            let optimizers =
                devices
                |> Seq.map (fun dev -> dev.Create options numClasses sortedTrainingSet)
                |> Array.ofSeq
            match mode with
            | PoolMode.EqualPartition ->
                { new IEntropyOptimizer with
                      member this.Optimize(weights) =
                          let batchSize = divup weights.Length optimizers.Length
                          optimizers
                          |> Array.mapi (fun i optimizer ->
                                 let offset = i * batchSize
                                 if offset < weights.Length then
                                     let length = min batchSize (weights.Length - offset)
                                     let weights = Array.sub weights offset length
                                     async { return optimizer.Optimize(weights) }
                                 else async { return Array.empty })
                          |> Async.Parallel
                          |> Async.RunSynchronously
                          |> Array.concat
                  interface System.IDisposable with
                      member this.Dispose() = optimizers |> Array.iter (fun opt -> opt.Dispose()) }

    member this.CreateDefaultOptions numClasses (sortedTrainingSet : (_ * Labels * Indices)[]) =
        this.Create EntropyOptimizationOptions.Default numClasses sortedTrainingSet

(**
Options for decision-trees:

- `MaxDepth`: maximal number of tree-layers.
- `Device`: Decide between CPU and GPU implementation.
- `EntropyOptions`: EntropyOptimizationOptions, `DefaultWithSquareRootFeatureSelector` is a good choice. different feature-selectors might be choosen.
*)
type TreeOptions =
    { MaxDepth : int
      Device : EntropyDevice
      EntropyOptions : EntropyOptimizationOptions }

    static member Default =
        { MaxDepth = System.Int32.MaxValue
          Device = GPU(GpuMode.SingleWeight, GpuModuleProvider.DefaultModule)
          EntropyOptions = EntropyOptimizationOptions.Default }

let bootstrappedForestClassifier (options : TreeOptions) (weightsPerBootstrap : Weights[])
    (trainingSet : LabeledFeatureSet) : Model =
    let numClasses = trainingSet.NumClasses

    let sortedFeatures =
        match trainingSet.Sorted with
        | SortedFeatures features -> features
        | _ -> failwith "features are unsorted"

    use optimizer = options.Device.Create options.EntropyOptions numClasses sortedFeatures
    let trainer = trainTrees options.MaxDepth optimizer numClasses sortedFeatures
    let trees = trainer weightsPerBootstrap
    if trees.Length <> weightsPerBootstrap.Length then failwith "length not match!"
    RandomForest(trees, numClasses)

let bootstrappedStumpsClassifier weights =
    let options = { TreeOptions.Default with MaxDepth = 1 }
    bootstrappedForestClassifier options weights

let weightedTreeClassifier (options : TreeOptions) (trainingSet : LabeledFeatureSet) (weights : Weights) =
    let model = bootstrappedForestClassifier options [| weights |] trainingSet
    match model with
    | RandomForest(trees, _) -> trees |> Seq.head

let treeClassfier options (trainingSet : LabeledFeatureSet) =
    Array.create trainingSet.Length 1 |> weightedTreeClassifier options trainingSet

let randomWeights (rnd : int -> int) length : Weights =
    let hist = Array.zeroCreate length
    for i = 1 to length do
        let index = rnd(length)
        hist.[index] <- hist.[index] + 1
    hist

(**
Create a random forest from a `trainingSet`:

- `options`: The default options using GPU is a fair choice.
- `rnd`: A random number generating function in order to create different weights for the trees. Default choice is a function created by `getRngFunction`.
- `numTrees`: The number of trees to be grown in the random forest.
- `trainingSet`: Data to train the random forest.

The method returns a random forest consisting of an array of trees and the number of classes (i.e. number of possible labels).
*)
let randomForestClassifier options rnd numTrees (trainingSet : LabeledFeatureSet) =
    let numSamples = trainingSet.Length
    let weights = Array.init numTrees (fun _ -> randomWeights rnd numSamples)
    bootstrappedForestClassifier options weights trainingSet

(**
Trains only stumps, i.e. a random forest with trees of depth 1.

- `rnd`: A random number generating function in order to create different weights for the trees. Default choice is a function created by `getRngFunction`.
- `numTrees`: The number of trees to be grown in the random forest.
- `trainingSet`: Data to train the random forest.
*)
let randomStumpsClassifier : (int -> int) -> int -> LabeledFeatureSet -> Model =
    randomForestClassifier { TreeOptions.Default with MaxDepth = 1 }