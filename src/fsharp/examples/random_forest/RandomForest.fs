module Tutorial.Fs.examples.RandomForest.RandomForest

open Tutorial.Fs.examples.RandomForest.DataModel
open Tutorial.Fs.examples.RandomForest.Common
open Tutorial.Fs.examples.RandomForest.Cuda
open Tutorial.Fs.examples.RandomForest.GpuSplitEntropy
open FSharp.Collections.ParallelSeq

[<Literal>]
let private DEBUG = false

let sortFeature (labels : Labels) (featureValues : Domain) =
    let tupleArray = featureValues |> Array.mapi (fun i value -> (value, labels.[i], i))
    tupleArray |> Array.sortInPlace
    tupleArray |> Array.unzip3

let sortAllFeatures labels domains =
    domains |> Array.Parallel.map (sortFeature labels)

type LabeledFeatureSet =
    | LabeledSamples of LabeledSample[]
    | LabeledDomains of Domains * Labels
    | SortedFeatures of (Domain * Labels * int[])[]

    member this.Labels = 
        this |> function
        | LabeledSamples trainingSet -> 
            trainingSet |> Array.unzip |> snd
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
            let domains = Array.init samples.[0].Length (fun i ->
                Array.init numSamples (fun j -> samples.[j].[i])
            )
            SortedFeatures (domains |> sortAllFeatures labels)
        | LabeledDomains (domains, labels) -> 
            SortedFeatures (domains |> sortAllFeatures labels)
        | SortedFeatures _ -> this
        

let rec forecastTree (sample:Sample) (tree : Tree) =
    match tree with
    | Tree.Leaf label -> label
    | Tree.Node (low, split, high) ->
        if sample.[split.Feature] <= split.Threshold then 
            forecastTree sample low 
        else 
            forecastTree sample high


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

let cumHistograms (numClasses : int) (labels:Labels) (weights : Weights) : LabelHistogram seq =
    ((Array.zeroCreate numClasses, 0), seq { 0 .. weights.GetUpperBound(0) })
    ||> Seq.scan (fun (hist, sum) i -> 
            let weight = weights.[i]
            let label = labels.[i]
            hist.[label] <- hist.[label] + weight
            (hist, sum + weight)
    )
    |> Seq.skip 1

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

let entropy total (nodes : LabelHistogram seq) =
    if (total = 0) then
        0.0
    else
        let entropySum = (0.0, nodes) ||> Seq.fold (fun e node -> e + (node |> entropyTermSum))
        entropySum / (float total * log_2)

let complementHist (total : LabelHistogram) (left : LabelHistogram) : LabelHistogram =
    let totalHist, totalCount = total
    let hist, count = left
    (totalHist, hist) ||> Array.map2 (-), totalCount - count

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

let domainThreshold weights (domain : _[]) splitIdx =
    let nextIdx = Array.findNextNonZeroIdx weights (splitIdx + 1)
    match nextIdx with
    | None ->
        None
    | Some nextSplitIdx -> 
        Some ((domain.[splitIdx] + domain.[nextSplitIdx]) * 0.5)

let countSamples (weights : Weights) numClasses (labels : Labels) =
    let hist = Array.zeroCreate numClasses
    for i = 0 to weights.GetUpperBound(0) do
        let weight = weights.[i]
        let label = labels.[i]
        hist.[label] <- hist.[label] + weight
    hist 

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

let optimizeFeatures (mode : CpuMode) (options : EntropyOptimizationOptions) numClasses (labelsPerFeature : Labels[]) 
    (indicesPerFeature : Indices[]) weights =
    // Remove zero weights
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

    // For each feature find minimum entropy and corresponding index
    let mapping = (fun featureIdx labels ->
        if (mask.[featureIdx]) then
            let featureWeights = weightsPerFeature.[featureIdx]
            let countsPerSplit = cumHistograms numClasses labels featureWeights
            let mask = entropyMask featureWeights labels total combinedMinWeight
            let entropyPerSplit = splitEntropies mask countsPerSplit totals
            entropyPerSplit |> Seq.map rounding |> MinMax.minAndArgMin |> translator featureIdx
        else
            (infinity, upperBoundWeights)
    )

    labelsPerFeature |> mapper mapping

let restrict startIdx count (source : _[]) = 
    let target  = Array.zeroCreate source.Length
    Array.blit source startIdx target startIdx count
    target

let restrictBelow idx (source : _[]) = 
    source |> restrict 0 (idx + 1)

let restrictAbove idx (source : _[]) = 
    source |> restrict idx (source.Length - idx)

/// Domains need to be sorted in ascending order
let rec trainTree depth (optimizer : Weights -> (float * int)[]) 
    numClasses (sortedTrainingSet : (Domain * Labels * Indices)[]) (weights : Weights) =
    let entropy, splitIdx, featureIdx =
        if depth = 0 then
            nan, weights.GetUpperBound(0), 0
        else
            optimizer weights
            |> Array.mapi (fun i (entropy, splitIdx) -> (entropy, splitIdx, i))
            |> Array.minBy (fun (entropy, _, _) -> entropy)

    let (domain, labels, indices) = sortedTrainingSet.[featureIdx]
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
        //Set weights to 0 for elements which aren't included in left and right branches respectively
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

let trainStump = trainTree 1

type OptimizerSignature = (Weights -> (float * int)[])
type DisposerSignature = unit -> unit

type EntropyDevice = 
    | GPU
    | CPU of mode : CpuMode
    | Cached of origin: EntropyDevice * optimizer : OptimizerSignature
    | Pooled of origin: EntropyDevice * optimizers : Collections.BlockingObjectPool<OptimizerSignature>

    member this.Create options numClasses (sortedTrainingSet : (_ * Labels * Indices)[]) : 
        OptimizerSignature * DisposerSignature =
        let _, labelsPerFeature, indicesPerFeature = sortedTrainingSet |> Array.unzip3
        let emptyDisposer = (fun () -> ())
        match this with
        | GPU -> 
            let worker = Alea.CUDA.Worker.Create(Alea.CUDA.Device.Default)
            use matrix = worker.Malloc<MultiChannelReduce.ValueAndIndex>(1) // work-around for bug in Alea.CUDA
            let entropyOptimizer = new EntropyOptimizer(Alea.CUDA.GPUModuleTarget.Worker(worker))
            entropyOptimizer.Init(numClasses, labelsPerFeature, indicesPerFeature)
            entropyOptimizer.Optimize options, (fun () -> 
                entropyOptimizer.Dispose()
                worker.Dispose()
            )
        | CPU mode -> 
            optimizeFeatures mode options numClasses labelsPerFeature indicesPerFeature, emptyDisposer
        | Cached (_, optimizer) -> optimizer, emptyDisposer
        | Pooled (_, optimizers) -> 
            let optimize weights = 
                let optimizer = optimizers.Acquire()
                let result = optimizer weights
                optimizers.Release(optimizer)
                result
            optimize, emptyDisposer

    member this.ToCached options numClasses sortedTrainingSet =
        match this with
        | Cached _ -> failwith "already cached"
        | Pooled _ -> failwith "cannot cache a pool"
        | _ ->
            let (optimizer, disposer) = this.Create options numClasses sortedTrainingSet
            Cached (this, optimizer), disposer

    member this.CreatePool poolSize options numClasses sortedTrainingSet =
        match this with
        | Cached _ -> failwith "cannot pool a cached optimizer"
        | Pooled _ -> failwith "already pooled"
        | _ ->
            let (devices, disposers) = 
                Array.init poolSize (fun _ -> this.ToCached options numClasses sortedTrainingSet) |> Array.unzip
            let optimizers = devices |> Array.map (function 
                | Cached (_, optimizer) -> optimizer
                | _ -> failwith "not cached"
            )
            Pooled (this, Collections.BlockingObjectPool<_>(optimizers)), disposers

    member this.CreateDefaultOptions numClasses (sortedTrainingSet : (_ * Labels * Indices)[]) = 
        this.Create EntropyOptimizationOptions.Default numClasses sortedTrainingSet

type TreeOptions = 
    {
        MaxDepth : int
        Device : EntropyDevice
        EntropyOptions : EntropyOptimizationOptions
    }

    static member Default = 
        {
            MaxDepth = System.Int32.MaxValue
            Device = GPU
            EntropyOptions = EntropyOptimizationOptions.Default
        }

let bootstrappedForestClassifier (options : TreeOptions) 
    (weightsPerBootstrap : Weights seq) (trainingSet : LabeledFeatureSet) : Model =
    let numSamples = trainingSet.Length
    let numClasses = trainingSet.NumClasses
    let sortedFeatures = 
        match trainingSet.Sorted with
        | SortedFeatures features -> features
        | _ -> failwith "features are unsorted"
    let optimizer, disposer = options.Device.Create options.EntropyOptions numClasses sortedFeatures
    let trainer = trainTree options.MaxDepth optimizer numClasses sortedFeatures
    let mapper f =         
        match options.Device with
        | Pooled (_, optimizers) -> 
            PSeq.mapi (fun i x -> (i, f x))
            >> PSeq.withDegreeOfParallelism (optimizers.Size) 
            >> Array.ofSeq
            >> Array.sortBy fst
            >> Array.map snd
        | _ -> Seq.map f >> Array.ofSeq

    try
        let trees = 
            weightsPerBootstrap 
            |> mapper (fun weights ->
                if (Array.length weights) <> numSamples then
                    failwith "Invalid number of weights"
                weights |> trainer
            )
        RandomForest(trees, numClasses)
    finally
        disposer()

let bootstrappedStumpsClassifier weights = 
    let options = { TreeOptions.Default with MaxDepth = 1}
    bootstrappedForestClassifier options weights

let weightedTreeClassifier (options : TreeOptions) (trainingSet : LabeledFeatureSet) (weights : Weights) =
    let model = bootstrappedForestClassifier options (seq {yield weights;}) trainingSet
    match model with
    | RandomForest(trees, numClasses) -> trees |> Seq.head

let treeClassfier options (trainingSet : LabeledFeatureSet) = 
    Array.create trainingSet.Length 1 |> weightedTreeClassifier options trainingSet

let randomWeights (rnd : System.Random) (length : int) : Weights =
    let hist = Array.zeroCreate length
    for i = 1 to length do
        let index = rnd.Next(length)
        hist.[index] <- hist.[index] + 1
    hist

/// Trains a random forest
let randomForestClassifier (rnd : System.Random) options numTrees (trainingSet : LabeledFeatureSet) =
    let numSamples = trainingSet.Length
    let weights = (Seq.init numTrees <| fun _ -> randomWeights rnd numSamples) |> Seq.bufferedAsync 10
    bootstrappedForestClassifier options weights trainingSet

/// Trains a random forest of stumps
let randomStumpsClassifier (rnd : System.Random) numStumps =
    let options = { TreeOptions.Default with MaxDepth = 1}
    randomForestClassifier (rnd : System.Random) options numStumps 