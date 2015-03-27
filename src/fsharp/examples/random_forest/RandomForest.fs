module Tutorial.Fs.examples.RandomForest.RandomForest

open FSharp.Collections.ParallelSeq

open Alea.CUDA
open Alea.CUDA.Utilities

open Tutorial.Fs.examples.RandomForest.DataModel
open Tutorial.Fs.examples.RandomForest.Common
open Tutorial.Fs.examples.RandomForest.GpuSplitEntropy

[<Literal>]
let private DEBUG = false

let sortFeature (labels : Labels) (featureValues : Domain) =
    let tupleArray = featureValues |> Array.mapi (fun i value -> (value, labels.[i], i))
    tupleArray |> Array.sortInPlace
    tupleArray |> Array.unzip3

let sortAllFeatures labels domains =
    domains |> Array.Parallel.map (sortFeature labels)

type LabeledFeatureSet =
    /// Array LabeledSamples, LabeledSample: Array of features & a label.
    | LabeledSamples of LabeledSample[]
    /// LabeledDomains might be seen as transposition of LabeledSamples.
    | LabeledDomains of Domains * Labels
    /// Similar to Labeled domains, but every domain is sorted and has its own attached Label and an index in order to find back to initial ordering 
    /// (elements in different index belong to the same sample).
    | SortedFeatures of (Domain * Labels * Indices)[]

    member this.Labels = 
        this |> function
        | LabeledSamples trainingSet -> trainingSet |> Array.unzip |> snd
        | LabeledDomains (_, labels) -> labels
        | SortedFeatures features -> features.[0] |> (fun (_, x, _) -> x)

    member this.Length = 
        this.Labels |> Array.length

    member this.NumClasses =
        Array.max this.Labels + 1

    member this.Sorted =
        this |> function
        | LabeledSamples trainingSet -> 
            let numSamples = this.Length
            let samples, labels = trainingSet |> Array.unzip
            let domains = Array.init samples.[0].Length (fun i -> Array.init numSamples (fun j -> samples.[j].[i]))
            SortedFeatures (sortAllFeatures labels domains)
        | LabeledDomains (domains, labels) -> 
            SortedFeatures (sortAllFeatures labels domains)
        | SortedFeatures _ -> this
        
/// Forcasts `label` of a given `sample` and decision-`tree`.
let rec forecastTree (sample:Sample) (tree : Tree) =
    match tree with
    | Tree.Leaf label -> label
    | Tree.Node (low, split, high) ->
        if sample.[split.Feature] <= split.Threshold then 
            forecastTree sample low 
        else 
            forecastTree sample high

/// Forcasts `label` of `sample` by majority voting in random forest `model`.
let forecast (model : Model) (sample : Sample) : Label =
    match model with
    | RandomForest (trees, numClasses) ->
        (Array.zeroCreate numClasses, trees)
        ||> Seq.fold (fun hist tree ->
            let label = forecastTree sample tree
            hist.[label] <- hist.[label] + 1
            hist
        )
        |> MinMax.maxAndArgMax |> snd

/// Returns an array of histograms on `labels` weighted by `weights`.
let cumHistograms (numClasses : int) (labels:Labels) (weights : Weights) : LabelHistogram seq =
    ((Array.zeroCreate numClasses, 0), seq { 0 .. weights.GetUpperBound(0) })
    ||> Seq.scan (fun (hist, sum) i ->
            let weight = weights.[i]
            let label = labels.[i]
            hist.[label] <- hist.[label] + weight
            (hist, sum + weight)
    )
    |> Seq.skip 1

/// Returns a histogram on `labels` weighted by `weights`.
let countTotals numClasses labels weights =
    Seq.last <| cumHistograms numClasses labels weights

let private log_2 = log 2.0

let private entropyTerm (x : int) = 
    if x > 0 then
        let xf = float x
        xf * (log xf)
    elif x = 0 then
        0.0
    else
        failwith "undefined"

let private entropyTermSum (node : LabelHistogram) =
    let hist, total = node
    if (total = 0) then
        0.0
    else
        let sumLogs = (0.0, hist) ||> Seq.fold (fun e c -> e - (entropyTerm c))
        (sumLogs + entropyTerm total)

/// Returns entropy as $\frac{1}{\rm{total} log 2} \sum_i - f_i * log(f_i/F)$.
let entropy total (nodes : LabelHistogram seq) =
    if (total = 0) then
        0.0
    else
        let entropySum = (0.0, nodes) ||> Seq.fold (fun e node -> e + (node |> entropyTermSum))
        entropySum / (float total * log_2)

/// Returns the difference between `total`- and `left`-Histogram.
let complementHist (total : LabelHistogram) (left : LabelHistogram) : LabelHistogram =
    let totalHist, totalCount = total
    let hist, count = left
    (totalHist, hist) ||> Array.map2 (-), totalCount - count

/// Calculates the entropy for all the possible splits in an ordered feature. Returning them in a sequence.
let splitEntropies (mask : bool seq) (countsPerSplit : LabelHistogram seq) (totals : LabelHistogram) =
    let complement = complementHist totals
    let entropy = entropy (totals |> snd)
    (mask, countsPerSplit) ||> Seq.map2 (fun isValid low ->
        if isValid then
            let histograms = seq { yield low; yield complement low }
            entropy histograms
        else
            System.Double.PositiveInfinity
    )

/// Returns Some(0.5*(`domain`[`splitIdx`] + `domain`[`splitIdx`+1])) if `domain`[`splitIdx`+1] exists, 
/// else returns None.
let domainThreshold weights (domain : _[]) splitIdx =
    let nextIdx = Array.findNextNonZeroIdx weights (splitIdx + 1)
    match nextIdx with
    | None ->
        None
    | Some nextSplitIdx ->
        Some ((domain.[splitIdx] + domain.[nextSplitIdx]) * 0.5)

/// Returns a histogram from `weights` on `labels`.
let countSamples (weights : Weights) numClasses (labels : Labels) =
    let hist = Array.zeroCreate numClasses
    for i = 0 to weights.GetUpperBound(0) do
        let weight = weights.[i]
        let label = labels.[i]
        hist.[label] <- hist.[label] + weight
    hist

/// Returns the `class`/`label` with the most `weight`.
let findMajorityClass weights numClasses labels =
    countSamples weights numClasses labels
    |> MinMax.maxAndArgMax |> snd

let entropyMask (weights : _[]) (labels : _[]) totalWeight absMinWeight =
    let findNextWeight = Array.findNextNonZeroIdx weights
    ((false, findNextWeight 0, 0), seq { 0 .. weights.GetUpperBound(0) })
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

type CpuMode = 
    | Sequential
    | Parallel

type GpuModuleProvider =
    | DefaultModule
    | Specified of gpuModule:GpuSplitEntropy.EntropyOptimizationModule

    member this.GetModule() =
        match this with
        | DefaultModule -> GpuSplitEntropy.EntropyOptimizationModule.Default
        | Specified gpuModule -> gpuModule

type GpuMode =
    | SingleWeight
    | SingleWeightWithStream of numStreams:int

type PoolMode =
    | EqualPartition

type IEntropyOptimizer =
    inherit System.IDisposable
    abstract Optimize : Weights[] -> (float * int)[][]

/// Returns best entropy to split and its corresponding index for every feature.   
let optimizeFeatures (mode : CpuMode) (options : EntropyOptimizationOptions) numClasses (labelsPerFeature : Labels[]) (indicesPerFeature : Indices[]) weights =
    // remove zero weights
    let weightsPerFeature = Array.expandWeights indicesPerFeature weights
    let nonZeroIdcsPerFeature = Array.findNonZeroWeights weightsPerFeature
    
    let mapper f =
        match mode with
        | Sequential -> Array.mapi f
        | Parallel   -> Array.Parallel.mapi f

    let projector (sources : _[][]) =
        sources |> mapper (fun i source -> source |> Array.projectIdcs nonZeroIdcsPerFeature.[i])
    let weightsPerFeature = projector weightsPerFeature
    let labelsPerFeature  = projector labelsPerFeature
    let upperBoundNonZero = nonZeroIdcsPerFeature.[0].GetUpperBound(0)
    let upperBoundWeights = weights.GetUpperBound(0)
    let translator featureIdx (x, i) = 
        if i = upperBoundNonZero then x, upperBoundWeights else x, nonZeroIdcsPerFeature.[featureIdx].[i]

    let totals = countTotals numClasses labelsPerFeature.[0] weightsPerFeature.[0]
    let total = (totals |> snd)

    // heuristic for avoiding small splits
    let combinedMinWeight = options.MinWeight numClasses total
    let rounding (value : float) = System.Math.Round(value, options.Decimals)
    let mask = options.FeatureSelector (labelsPerFeature |> Array.length)

    // for each feature find minimum entropy and corresponding index
    let mapping = (fun featureIdx labels ->
        if (mask.[featureIdx]) <> 0 then
            let featureWeights = weightsPerFeature.[featureIdx]
            let countsPerSplit = cumHistograms numClasses labels featureWeights
            let mask = entropyMask featureWeights labels total combinedMinWeight
            let entropyPerSplit = splitEntropies mask countsPerSplit totals
            entropyPerSplit |> Seq.map rounding |> MinMax.minAndArgMin |> translator featureIdx
        else
            (infinity, upperBoundWeights)
    )

    let result = labelsPerFeature |> mapper mapping

    let fullEntropy = 
        (result, mask)
        ||> Seq.zip
        |> Seq.filter (fun (_, mask) -> mask <> 0)
        |> Seq.sumBy (fun ((entropy,_),_) -> entropy)

    if fullEntropy > 0.0 then result
    else result |> Array.map (fun (x, _) -> x, upperBoundWeights)
//    result

let restrict startIdx count (source : _[]) = 
    let target  = Array.zeroCreate source.Length
    Array.blit source startIdx target startIdx count
    target

let restrictBelow idx (source : _[]) = 
    source |> restrict 0 (idx + 1)

let restrictAbove idx (source : _[]) = 
    source |> restrict idx (source.Length - idx)

// [Old] Domains need to be sorted in ascending order
let trainTree depth (optimizer:IEntropyOptimizer) numClasses sortedTrainingSet (weights:Weights[]) =
    let rec trainTree depth (optimizer : Weights -> (float * int)[]) 
        numClasses (sortedTrainingSet : (Domain * Labels * Indices)[]) (weights : Weights) =
        let entropy, splitIdx, featureIdx =
            if depth = 0 then
                nan, weights.GetUpperBound(0), 0
            else
                optimizer weights
                |> Array.mapi (fun i (entropy, splitIdx) -> (entropy, splitIdx, i))
                |> Array.minBy (fun (entropy, _, _) -> entropy)

        let domain, labels, indices = sortedTrainingSet.[featureIdx]
        let sortedWeights = weights |> Array.projectIdcs indices
        let threshold = domainThreshold sortedWeights domain splitIdx

        if DEBUG then
            printfn "depth: %A, entropy: %A, splitIdx: %A, featureIdx: %A" depth entropy splitIdx featureIdx
            printf "Labels:\n["
            (sortedWeights, labels) ||> Array.iteri2 (fun i weight label ->
                if weight <> 0 then
                    printf " (%A * %A) " label weight
                if (i = splitIdx) then printf "|"
            )
            printfn "]"

        match threshold with
        | Some num ->
            // set weights to 0 for elements which aren't included in left and right branches respectively
            let lowWeights  = sortedWeights |> restrictBelow splitIdx
            let highWeights = sortedWeights |> restrictAbove (splitIdx + 1)
            if DEBUG then
                printfn "Low  counts: %A" (countSamples lowWeights numClasses labels)
                printfn "High counts: %A" (countSamples highWeights numClasses labels)
            let trainTree weights = 
                trainTree (depth - 1) optimizer numClasses sortedTrainingSet (weights |> Array.permByIdcs indices)
            let low  = trainTree lowWeights
            let high = trainTree highWeights
            match (low, high) with
            | (Leaf lowLabel, Leaf highLabel) when lowLabel = highLabel -> 
                Tree.Leaf lowLabel
            | (_, _) ->
                Tree.Node (low, {Feature = featureIdx; Threshold = num}, high)
        | None -> 
            let label = findMajorityClass sortedWeights numClasses labels
            Tree.Leaf label

    let optimizer weights = optimizer.Optimize([|weights|]).[0]

    weights |> Array.map (fun weights -> trainTree depth optimizer numClasses sortedTrainingSet weights)
        
/// Domains need to be sorted in ascending order
let rec trainTree' depth (optimizer:IEntropyOptimizer) numClasses (sortedTrainingSet:(Domain * Labels * Indices)[]) (weights:Weights[]) =
    let triples = 
        if depth = 0 then
            weights |> Array.map (fun weights -> nan, weights.GetUpperBound(0), 0)
        else
            optimizer.Optimize(weights)
            |> Array.map (fun results ->
                results
                |> Array.mapi (fun i (entropy, splitIdx) -> entropy, splitIdx, i)
                |> Array.minBy (fun (entropy, _, _) -> entropy))

    let trees0 = triples |> Array.mapi (fun i (entropy, splitIdx, featureIdx) ->
        let weights = weights.[i]

        let domain, labels, indices = sortedTrainingSet.[featureIdx]
        let sortedWeights = weights |> Array.projectIdcs indices
        let threshold = domainThreshold sortedWeights domain splitIdx

        if DEBUG then
            printfn "depth: %A, entropy: %A, splitIdx: %A, featureIdx: %A" depth entropy splitIdx featureIdx
            printf "Labels:\n["
            (sortedWeights, labels) ||> Array.iteri2 (fun i weight label ->
                if weight <> 0 then
                    printf " (%A * %A) " label weight
                if (i = splitIdx) then printf "|"
            )
            printfn "]"

        match threshold with
        | Some num ->
            // set weights to 0 for elements which aren't included in left and right branches respectively
            let lowWeights  = sortedWeights |> restrictBelow splitIdx
            let highWeights = sortedWeights |> restrictAbove (splitIdx + 1)
            if DEBUG then
                printfn "Low  counts: %A" (countSamples lowWeights numClasses labels)
                printfn "High counts: %A" (countSamples highWeights numClasses labels)
            let lowWeights = lowWeights |> Array.permByIdcs indices
            let highWeights = highWeights |> Array.permByIdcs indices

            let set (lowTree:Tree) (highTree:Tree) =
                match lowTree, highTree with
                | Leaf lowLabel, Leaf highLabel when lowLabel = highLabel ->
                    i, Tree.Leaf lowLabel
                | _ ->
                    i, Tree.Node(lowTree, {Feature = featureIdx; Threshold = num}, highTree)

            None, Some lowWeights, Some highWeights, Some set

        | None ->
            let label = findMajorityClass sortedWeights numClasses labels
            Some (Tree.Leaf label), None, None, None )

    let trees() = trees0 |> Array.choose (fun (tree, _, _, _) -> tree)

    let lowWeights = trees0 |> Array.choose (fun (_, lowWeights, _, _) -> lowWeights)
    let highWeights = trees0 |> Array.choose (fun (_, _, highWeights, _) -> highWeights)

    if lowWeights.Length = 0 then trees()
    else
        let lowTrees = trainTree (depth - 1) optimizer numClasses sortedTrainingSet lowWeights
        let highTrees = trainTree (depth - 1) optimizer numClasses sortedTrainingSet highWeights
        let setFuncs = trees0 |> Array.choose (fun (_, _, _, set) -> set)
        for i = 0 to lowTrees.Length - 1 do
            let set = setFuncs.[i]
            let lowTree = lowTrees.[i]
            let highTree = highTrees.[i]
            let originIdx, tree = set lowTree highTree
            trees0.[originIdx] <- Some tree, None, None, None
        trees()

let trainStump optimizer numClasses sortedTrainingSet weights =
    trainTree 1 optimizer numClasses sortedTrainingSet weights

type EntropyDevice = 
    | GPU of mode:GpuMode * provider:GpuModuleProvider
    | CPU of mode:CpuMode
    | Pool of mode:PoolMode * devices:EntropyDevice seq

    member this.Create options numClasses (sortedTrainingSet:(_ * Labels * Indices)[]) : IEntropyOptimizer =
        let _, labelsPerFeature, indicesPerFeature = sortedTrainingSet |> Array.unzip3
        match this with
        | GPU (mode, provider) ->
            let gpuModule = provider.GetModule()
            gpuModule.GPUForceLoad()
            let worker = gpuModule.GPUWorker
            match mode with
            | SingleWeight ->
                let problem, memories = worker.Eval <| fun _ ->
                    let problem = gpuModule.CreateProblem(numClasses, labelsPerFeature, indicesPerFeature)
                    let memories = gpuModule.CreateMemories(problem)
                    problem, memories

                { new IEntropyOptimizer with
                    member this.Optimize(weights) =
                        gpuModule.GPUWorker.Eval <| fun _ ->
                            weights |> Array.map (fun weights ->
                                gpuModule.Optimize(problem, memories, options, weights))

                  interface System.IDisposable with
                    member this.Dispose() =
                        memories.Dispose()
                        problem.Dispose() }

            | SingleWeightWithStream numStreams ->
                let problem, param = worker.Eval <| fun _ ->
                    let problem = gpuModule.CreateProblem(numClasses, labelsPerFeature, indicesPerFeature)
                    let param = Array.init numStreams (fun _ ->
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
                                let weights = Array.sub weights (i*numStreams) numStreams
                                yield gpuModule.Optimize(problem, param, options, weights)

                            if rem > 0 then
                                let weights = Array.sub weights (div*numStreams) rem
                                let param = Array.sub param 0 rem
                                yield gpuModule.Optimize(problem, param, options, weights) }
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
                    weights |> Array.map (fun weights -> optimizeFeatures mode options numClasses labelsPerFeature indicesPerFeature weights)
              interface System.IDisposable with
                member this.Dispose() = () }

        | Pool (mode, devices) ->
            let optimizers = devices |> Seq.map (fun dev -> dev.Create options numClasses sortedTrainingSet) |> Array.ofSeq
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
                            else async { return Array.empty } )
                        |> Async.Parallel
                        |> Async.RunSynchronously
                        |> Array.concat
                                        
                  interface System.IDisposable with
                    member this.Dispose() =
                        optimizers |> Array.iter (fun opt -> opt.Dispose()) }

    member this.CreateDefaultOptions numClasses (sortedTrainingSet : (_ * Labels * Indices)[]) = 
        this.Create EntropyOptimizationOptions.Default numClasses sortedTrainingSet

type TreeOptions = 
    { MaxDepth : int
      Device : EntropyDevice
      EntropyOptions : EntropyOptimizationOptions }

    static member Default = 
        { MaxDepth = System.Int32.MaxValue
          Device = GPU(GpuMode.SingleWeight, GpuModuleProvider.DefaultModule)
          EntropyOptions = EntropyOptimizationOptions.Default }

let bootstrappedForestClassifier (options:TreeOptions) (weightsPerBootstrap:Weights[]) (trainingSet:LabeledFeatureSet) : Model =
    let numSamples = trainingSet.Length
    let numClasses = trainingSet.NumClasses
    let sortedFeatures = 
        match trainingSet.Sorted with
        | SortedFeatures features -> features
        | _ -> failwith "features are unsorted"
    use optimizer = options.Device.Create options.EntropyOptions numClasses sortedFeatures
    let trainer = trainTree options.MaxDepth optimizer numClasses sortedFeatures
    let trees = trainer weightsPerBootstrap
    if trees.Length <> weightsPerBootstrap.Length then failwith "length not match!"
    RandomForest(trees, numClasses)

let bootstrappedStumpsClassifier weights = 
    let options = { TreeOptions.Default with MaxDepth = 1}
    bootstrappedForestClassifier options weights

let weightedTreeClassifier (options : TreeOptions) (trainingSet : LabeledFeatureSet) (weights : Weights) =
    let model = bootstrappedForestClassifier options [| weights |] trainingSet
    match model with
    | RandomForest(trees, numClasses) -> trees |> Seq.head

let treeClassfier options (trainingSet : LabeledFeatureSet) = 
    Array.create trainingSet.Length 1 |> weightedTreeClassifier options trainingSet

let randomWeights (rnd:System.Random) (length:int) : Weights =
    let hist = Array.zeroCreate length
    for i = 1 to length do
        let index = rnd.Next(length)
        hist.[index] <- hist.[index] + 1
    hist

/// Trains a random forest
let randomForestClassifier (rnd : System.Random) options numTrees (trainingSet : LabeledFeatureSet) =
    let numSamples = trainingSet.Length
    //let weights = (Seq.init numTrees <| fun _ -> randomWeights rnd numSamples) |> Seq.bufferedAsync 10
    let weights = Array.init numTrees (fun _ -> randomWeights rnd numSamples)
    bootstrappedForestClassifier options weights trainingSet

/// Trains a random forest of stumps
let randomStumpsClassifier (rnd : System.Random) numStumps =
    let options = { TreeOptions.Default with MaxDepth = 1}
    randomForestClassifier (rnd : System.Random) options numStumps 