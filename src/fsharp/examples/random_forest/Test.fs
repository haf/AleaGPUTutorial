module Tutorial.Fs.examples.RandomForest.Tests

open NUnit.Framework
open FsUnit
open Alea.CUDA
open Tutorial.Fs.examples.RandomForest.Array
open Tutorial.Fs.examples.RandomForest.DataModel
open Tutorial.Fs.examples.RandomForest.GpuSplitEntropy
open Tutorial.Fs.examples.RandomForest.RandomForest

let blockSize = 128
let reduceBlockSize = 512

[<Test>]
let ``Transform training set``() =
    let trainingSamples =
        LabeledSamples [| [| 0.4; 0.5; 0.1 |], 0
                          [| 0.8; 0.1; 0.4 |], 1
                          [| 0.2; 0.7; 0.5 |], 1
                          [| 0.6; 0.5; 0.9 |], 0 |]

    let expected =
        [| [| 0.2; 0.4; 0.6; 0.8 |], [| 1; 0; 0; 1 |], [| 2; 0; 3; 1 |]
           [| 0.1; 0.5; 0.5; 0.7 |], [| 1; 0; 0; 1 |], [| 1; 0; 3; 2 |]
           [| 0.1; 0.4; 0.5; 0.9 |], [| 0; 1; 1; 0 |], [| 0; 1; 2; 3 |] |]

    let expectedDomains, expectedLabels, expectedIndices = expected |> Array.unzip3
    match trainingSamples.Sorted with
    | SortedFeatures features ->
        let domains, labels, indices = features |> Array.unzip3
        domains |> should equal expectedDomains
        labels |> should equal expectedLabels
        indices |> should equal expectedIndices
    | _ -> failwith "expected SortedFeatures"

[<Test>]
let ``Sort features``() =
    let featureValues = [| 10.0; 5.0; 9.0; 7.0; 3.0; 2.0 |]
    let labels = [| 1; 1; 0; 0; 1; 0 |]
    let expectedFeatures = [| 2.0; 3.0; 5.0; 7.0; 9.0; 10.0 |]
    let expectedLabels = [| 0; 1; 1; 0; 0; 1 |]
    let expectedIdxs = [| 5; 4; 1; 3; 2; 0 |]
    let sortedFeatures, sortedLabels, idxs = (labels, featureValues) ||> sortFeature
    sortedFeatures |> should equal expectedFeatures
    sortedLabels |> should equal expectedLabels
    idxs |> should equal expectedIdxs

[<Test>]
let ``Min and max functions``() =
    let testData = [| 2.0; 1.0; 3.0; 10.0; 9.0; 5.0 |]
    testData |> minAndArgMin |> should equal (1.0, 1)
    testData |> maxAndArgMax |> should equal (10.0, 3)

[<Test>]
let ``Threshold and labels``() =
    let numClasses = 2
    let domain = [| 4.0; 5.0; 10.0; 12.0 |]
    let labels = [| 1; 0; 0; 1 |]
    let weights = Array.init labels.Length (fun _ -> 1)

    let thresholdAndLabels splitIdx =
        let threshold = featureArrayThreshold weights domain splitIdx
        let lowWeights = weights |> restrictBelow splitIdx
        let highWeights = weights |> restrictAbove (splitIdx + 1)
        let low = findMajorityClass lowWeights numClasses labels
        let high = findMajorityClass highWeights numClasses labels
        threshold, low, high
    thresholdAndLabels 0 |> should equal (Some 4.5, 1, 0)
    thresholdAndLabels 1 |> should equal (Some 7.5, 1, 1)
    thresholdAndLabels 2 |> should equal (Some 11.0, 0, 1)
    let threshold, _, _ = thresholdAndLabels 3
    threshold |> should equal None

[<Test>]
let ``Forecast random forest of stumps``() =
    let stump1 = Node(Leaf 0, Split.Create 0 5.0, Leaf 1)
    let stump2 = Node(Leaf 0, Split.Create 0 7.0, Leaf 1)

    let stumps =
        Array.init 100 (fun i ->
            if i % 3 = 0 then stump2
            else stump1)

    let model = RandomForest(stumps, 2)
    forecast model [| 6.0 |] |> should equal 1
    forecast model [| 4.0 |] |> should equal 0

let checkStump feature threshold low high (stump : Tree) =
    match stump with
    | Leaf label -> label |> should equal low
    | Node(left, split, right) ->
        split.Feature |> should equal feature
        split.Threshold |> should equal threshold
        left |> should equal (Tree.Leaf low)
        right |> should equal (Tree.Leaf high)

[<Test>]
let ``Train simple stump``() =
    let optimizeStump numClasses sortedTrainingSet weights =
        let device = EntropyDevice.GPU(GpuMode.MultiWeightWithStream 15, GpuModuleProvider.DefaultModule)
        use optimizer = device.CreateDefaultOptions numClasses sortedTrainingSet
        let trainer = trainStump optimizer numClasses sortedTrainingSet
        (trainer [| weights |]).[0]

    let numSamples = 10
    let splitIdx = 5
    let weights = Array.init numSamples (fun _ -> 1)

    let labels =
        Array.init numSamples (fun i ->
            if i < splitIdx then 0
            else 1)

    let domain = Array.init numSamples (fun i -> float i)
    let indices = Array.init numSamples (fun i -> i)
    optimizeStump 2 [| domain, labels, indices |] weights |> checkStump 0 4.5f 0 1
    let mixedLabels = Array.init numSamples (fun i -> i % 2)
    optimizeStump 2 [| domain, mixedLabels, indices
                       domain, labels, indices |] weights
    |> checkStump 1 4.5f 0 1
    optimizeStump 2 [| domain, mixedLabels, indices |] weights |> checkStump 0 8.5f 0 1
    let biasedWeights = Array.zeroCreate numSamples
    biasedWeights.[numSamples - 1] <- numSamples / 2
    biasedWeights.[numSamples - 2] <- numSamples / 2
    optimizeStump 2 [| domain, mixedLabels, indices |] biasedWeights |> checkStump 0 8.5f 0 1

[<Test>]
let ``Sort domains labels and weights``() =
    let numSamples = 10
    let reverseDomain = Array.init numSamples (fun i -> float (numSamples - i - 1))

    let skewedWeights =
        Array.init numSamples (fun i ->
            if i < 2 then numSamples / 2
            else 0)

    let singleElementClass =
        Array.init numSamples (fun i ->
            if i = 0 then 1
            else 0)

    let trainigSet = LabeledFeatures([| reverseDomain |], singleElementClass)
    let model = bootstrappedStumpsClassifier [| skewedWeights |] trainigSet
    match model with
    | RandomForest(stumps, _) ->
        stumps
        |> Seq.head
        |> checkStump 0 8.5f 0 1

[<Test>]
let ``Train random stumps``() =
    let numStumps = 3500
    let numSamples = 32
    let splitIdx = numSamples / 2
    let samples = Array.init numSamples (fun i -> [| float i |])

    let labels =
        Array.init numSamples (fun i ->
            if i < splitIdx then 0
            else 1)

    let labeledSamples =
        (samples, labels)
        ||> Array.zip
        |> LabeledSamples

    let (RandomForest(stumps, _)) = randomStumpsClassifier (getRngFunction 0) numStumps labeledSamples

    let num, sum =
        Seq.fold (fun (num, sum) s ->
            match s with
            | Node(low, split, high) -> (num + 1, sum + float split.Threshold)
            | _ -> (num, sum)) (0, 0.0) stumps
    sum / float num |> should (equalWithin 1e-2) (float splitIdx - 0.5)

[<Test>]
let ``Random weights``() =
    let numElements = 100
    let weights = randomWeights (getRngFunction 0) numElements
    weights
    |> Array.sum
    |> should equal numElements
    weights
    |> Seq.forall (fun x -> x >= 0)
    |> should be True

let singleEntropy node =
    let sum = node |> Array.sum
    seq { yield node, sum } |> entropy sum

[<Test>]
let ``Entropy of single nodes``() =
    [| 0; 0 |] |> singleEntropy |> should equal 0.0
    [| 5; 0 |] |> singleEntropy |> should equal 0.0
    [| 0; 5 |] |> singleEntropy |> should equal 0.0
    [| 5; 5 |] |> singleEntropy |> should (equalWithin 1e-8) 1.0
    [| 2; 3 |] |> singleEntropy |> should (equalWithin 1e-3) 0.971
    [| 3; 2 |] |> singleEntropy |> should (equalWithin 1e-3) 0.971
    [| 9; 5 |] |> singleEntropy |> should (equalWithin 1e-3) 0.940
    [| 5; 4; 5 |] |> singleEntropy |> should (equalWithin 1e-3) 1.577

[<Test>]
let ``Multistage property of entropy``() =
    [| 2; 3; 4 |] |> singleEntropy |> should (equalWithin 1e-8) <| (singleEntropy [| 2; 7 |]) + 7.0 / 9.0 * (singleEntropy [| 3; 4 |])

[<Test>]
let ``Entropy of splits``() =
    let entropyWithoutTotals (hist : LabelHistogram seq) =
        let total =
            hist
            |> Seq.map snd
            |> Seq.sum
        entropy total hist

    let seqEntropy =
        Seq.ofList
        >> Seq.map (fun hist -> hist, hist |> Array.sum)
        >> entropyWithoutTotals

    [ [| 2; 3 |]; [| 4; 0 |]; [| 3; 2 |] ] |> seqEntropy |> should (equalWithin 1e-3) 0.693
    [ [| 4; 2 |]; [| 5; 3 |] ] |> seqEntropy |> should (equalWithin 1e-3) 0.939
    [ [| 4; 2 |]; [| 0; 0 |] ] |> seqEntropy |> should (equalWithin 1e-3) (singleEntropy [| 4; 2 |])
    [ [| 4; 2 |]; [| 6; 0 |] ] |> seqEntropy |> should (equalWithin 1e-3) (0.5 * singleEntropy [| 4; 2 |])

[<Test>]
let ``Entropy mask``() =
    let weights = [| 1; 0; 2; 1; 1; 2; 1; 0 |]
    let labels = [| 1; 0; 0; 1; 1; 0; 1; 0 |]
    entropyMask weights labels (weights |> Array.sum) 1 |> should equal [| true; false; true; false; true; true; true; false |]
    entropyMask weights labels (weights |> Array.sum) 2 |> should equal [| false; false; true; false; true; false; true; false |]

[<Test>]
let ``Restrict array``() =
    let numElems = 100
    let arr = Array.init numElems id
    let startIdx = 10
    let count = 20
    let restricted = arr |> restrict startIdx count
    let splitIdx = numElems / 2
    let restrictedBelow = arr |> restrictBelow splitIdx
    let restrictedAbove = arr |> restrictAbove (splitIdx + 1)
    let stopIdx = startIdx + count
    for i = 0 to numElems - 1 do
        if startIdx <= i && i < stopIdx then restricted.[i] |> should equal arr.[i]
        else restricted.[i] |> should equal 0
        if i <= splitIdx then
            restrictedBelow.[i] |> should equal arr.[i]
            restrictedAbove.[i] |> should equal 0
        else
            restrictedBelow.[i] |> should equal 0
            restrictedAbove.[i] |> should equal arr.[i]

[<Test>]
let ``CPU vs GPU optimizer``() =
    let rnd = System.Random 0
    let numSamples = 10000
    let numFeatures = 20
    let numClasses = 2
    let numTrees = 10
    let domains = Array.init numFeatures (fun _ -> Array.init numSamples (fun _ -> rnd.NextDouble()))
    let labels = Array.init numSamples (fun i -> i % numClasses)
    let trainingSet = LabeledFeatures(domains, labels)
    let cpuDevice = EntropyDevice.CPU(CpuMode.Parallel)

    let test (gpuDevice : EntropyDevice) =
        match trainingSet.Sorted with
        | SortedFeatures sortedTrainingSet ->
            use gpuOptimizer = gpuDevice.CreateDefaultOptions numClasses sortedTrainingSet
            use cpuOptimizer = cpuDevice.CreateDefaultOptions numClasses sortedTrainingSet
            let weights = Array.init numTrees (fun _ -> randomWeights (getRngFunction 0) numSamples)
            let gpuResults = gpuOptimizer.Optimize weights
            let cpuResults = cpuOptimizer.Optimize weights
            gpuResults.Length |> should equal cpuResults.Length
            (gpuResults, cpuResults) ||> Array.iter2 (fun gpuResult cpuResult ->
                                             let gpuMinEnt, gpuMinEntArg = gpuResult |> Array.unzip
                                             let cpuMinEnt, cpuMinEntArg = cpuResult |> Array.unzip
                                             gpuMinEnt |> should (equalWithin 1e-10) cpuMinEnt
                                             gpuMinEntArg |> should equal cpuMinEntArg)
        | _ -> failwith "expected sorted features"

    use worker2 = Worker.Create(Device.Default)
    use gpuModule2 = new GpuSplitEntropy.EntropyOptimizationModule(GPUModuleTarget.Worker(worker2), blockSize, reduceBlockSize)
    let gpuDevice1 = GPU(GpuMode.SingleWeight, GpuModuleProvider.DefaultModule)
    let gpuDevice2 = GPU(GpuMode.MultiWeightWithStream 10, GpuModuleProvider.Specified(gpuModule2))
    let gpuDevicePool = Pool(PoolMode.EqualPartition, [ gpuDevice1; gpuDevice2 ])
    gpuDevice1 |> test
    gpuDevice2 |> test
    gpuDevicePool |> test

[<Test>]
let ``Tree with one feature``() =
    let device = GPU(GpuMode.MultiWeightWithStream 10, GpuModuleProvider.DefaultModule)
    let labels = [| 0; 0; 1; 1; 0; 0; 1; 1 |]
    let domain = Array.init labels.Length (fun i -> float i)
    let trainingSet = LabeledFeatureSet.LabeledFeatures([| domain |], labels)
    [| (1,
        Node(Leaf 0,
             { Feature = 0
               Threshold = 5.5 }, Leaf 1))
       (2,
        Node(Node(Leaf 1,
                  { Feature = 0
                    Threshold = 3.5 }, Leaf 0),
             { Feature = 0
               Threshold = 5.5 }, Leaf 1))
       (3,
        Node(Node(Node(Leaf 0,
                       { Feature = 0
                         Threshold = 1.5 }, Leaf 1),
                  { Feature = 0
                    Threshold = 3.5 }, Leaf 0),
             { Feature = 0
               Threshold = 5.5 }, Leaf 1))
       (4,
        Node(Node(Node(Leaf 0,
                       { Feature = 0
                         Threshold = 1.5 }, Leaf 1),
                  { Feature = 0
                    Threshold = 3.5 }, Leaf 0),
             { Feature = 0
               Threshold = 5.5 }, Leaf 1)) |]
    |> Array.iter (fun (depth, expectedTree) ->
           let options =
               { TreeOptions.Default with MaxDepth = depth
                                          Device = device }

           let tree = treeClassfier options trainingSet
           tree |> should equal expectedTree)

[<Test>]
let ``Tree with two features``() =
    let device = GPU(GpuMode.MultiWeightWithStream 10, GpuModuleProvider.DefaultModule)
    let labels = [| 1; 0; 1; 0 |]

    let options =
        { TreeOptions.Default with MaxDepth = 4
                                   Device = device }
    [| [| 0.0; 1.0; 2.0; 3.0 |], [| 0.0; 1.0; 2.0; 3.0 |],
       Node(Node(Node(Leaf 1,
                      { Feature = 0
                        Threshold = 0.5 }, Leaf 0),
                 { Feature = 0
                   Threshold = 1.5 }, Leaf 1),
            { Feature = 0
              Threshold = 2.5 }, Leaf 0)
       [| 0.0; 1.0; 2.0; 3.0 |], [| 2.0; 1.0; 0.0; 3.0 |],
       Node(Node(Node(Leaf 1,
                      { Feature = 0
                        Threshold = 0.5 }, Leaf 0),
                 { Feature = 0
                   Threshold = 1.5 }, Leaf 1),
            { Feature = 0
              Threshold = 2.5 }, Leaf 0)
       [| 0.0; 1.0; 3.0; 2.0 |], [| 0.0; 3.0; 2.0; 1.0 |],
       Node(Node(Leaf 1,
                 { Feature = 0
                   Threshold = 0.5 }, Leaf 0),
            { Feature = 0
              Threshold = 2.5 }, Leaf 1)
       [| 0.0; 2.0; 1.0; 3.0 |], [| 0.0; 2.0; 1.0; 3.0 |],
       Node(Leaf 1,
            { Feature = 0
              Threshold = 1.5 }, Leaf 0) |]
    |> Array.iter (fun (domainA, domainB, expectedTree) ->
           let trainingSet = LabeledFeatureSet.LabeledFeatures([| domainA; domainB |], labels)
           let tree = treeClassfier options trainingSet
           tree |> should equal expectedTree)

[<Test>]
let ``Tree with weights``() =
    let device = GPU(GpuMode.MultiWeightWithStream 10, GpuModuleProvider.DefaultModule)
    let labels = [| 0; 1; 0 |]
    let domain = [| 0.0; 1.0; 2.0 |]
    let weights = [| 1; 0; 2 |]
    let options = { TreeOptions.Default with Device = device }
    let expectedTree = Leaf 0
    let trainingSet = LabeledFeatureSet.LabeledFeatures([| domain |], labels)
    let tree = weightedTreeClassifier options trainingSet weights
    tree |> should equal expectedTree

let randomTrainingData (rnd : System.Random) numSamples numFeatures numClasses =
    let domains = Array.init numFeatures (fun _ -> Array.init numSamples (fun _ -> rnd.NextDouble()))
    let labels = Array.init numSamples (fun _ -> rnd.Next(numClasses))
    LabeledFeatures(domains, labels)

let defaultTrainingData =
    let numSamples = 200
    let numFeatures = 10
    let numClasses = 2
    let rnd = System.Random(0)
    let trainingData = randomTrainingData rnd numSamples numFeatures numClasses
    trainingData.Sorted

let compareForests gpuOptions cpuOptions =
    let trainingData = defaultTrainingData
    let numTrees = 1
    let weights = Array.init numTrees (fun _ -> randomWeights (getRngFunction 0) trainingData.Length)
    let classifier options = bootstrappedForestClassifier options weights
    printfn "GPU--------------"
    let model1 = classifier gpuOptions defaultTrainingData
    let (RandomForest(trees1, _)) = model1
    printfn "CPU--------------"
    let model2 = classifier cpuOptions trainingData
    let (RandomForest(trees2, _)) = model2
    Array.iteri2 (fun i tree1 tree2 ->
        try
            tree1 |> should equal tree2
        with _ ->
            printfn "Tree Nr. %d" i
            reraise()) trees1 trees2

[<Test>]
let ``Random forest on CPU Parallel vs CPU Sequential``() =
    let options = { TreeOptions.Default with MaxDepth = 4 }
    compareForests { options with Device = CPU Parallel } { options with Device = CPU Sequential }

[<Test>]
let ``Random forest on CPU Parallel vs GPU single threaded``() =
    let options = { TreeOptions.Default with MaxDepth = 4 }
    compareForests { options with Device = GPU(GpuMode.SingleWeight, GpuModuleProvider.DefaultModule) }
        { options with Device = CPU Parallel }

[<Test>]
let ``Random forest on CPU thread pool vs GPU thread pool``() =
    use worker2 = Worker.Create(Device.Default)
    use gpuModule2 = new GpuSplitEntropy.EntropyOptimizationModule(GPUModuleTarget.Worker(worker2), blockSize, reduceBlockSize)
    let gpuDevice1 = GPU(GpuMode.MultiWeightWithStream 10, GpuModuleProvider.DefaultModule)
    let gpuDevice2 = GPU(GpuMode.MultiWeightWithStream 10, GpuModuleProvider.Specified(gpuModule2))
    let gpuDevice = Pool(PoolMode.EqualPartition, [ gpuDevice1; gpuDevice2 ])
    let cpuDevice = Pool(PoolMode.EqualPartition, List.init 5 (fun _ -> CPU(CpuMode.Sequential)))
    let options = { TreeOptions.Default with MaxDepth = 4 }
    compareForests { options with Device = cpuDevice } { options with Device = gpuDevice }

let addSquareRootFeatureSelector seed options =
    let featureSelector = EntropyOptimizationOptions.SquareRootFeatureSelector (getRngFunction 0)
    let entropyOptions = { options.EntropyOptions with FeatureSelector = featureSelector }
    { options with EntropyOptions = entropyOptions }

[<Test>]
let ``Features subselection``() =
    use worker2 = Worker.Create(Device.Default)
    use gpuModule2 = new GpuSplitEntropy.EntropyOptimizationModule(GPUModuleTarget.Worker(worker2), blockSize, reduceBlockSize)
    let gpuDevice1 = GPU(GpuMode.SingleWeight, GpuModuleProvider.DefaultModule)
    let gpuDevice2 = GPU(GpuMode.MultiWeightWithStream 10, GpuModuleProvider.Specified(gpuModule2))
    let gpuDevices = Pool(PoolMode.EqualPartition, [ gpuDevice1; gpuDevice2 ])
    //    let cpuDevice = Pool(PoolMode.EqualPartition, List.init 2 (fun i -> CPU(CpuMode.Sequential)))
    let cpuDevice = CPU(CpuMode.Sequential)
    let options = { TreeOptions.Default with MaxDepth = 6 }
    compareForests 
        ({ options with Device = gpuDevice1 } |> addSquareRootFeatureSelector 50)
        ({ options with Device = cpuDevice } |> addSquareRootFeatureSelector 50)
    compareForests 
        ({ options with Device = gpuDevice2 } |> addSquareRootFeatureSelector 50)
        ({ options with Device = cpuDevice } |> addSquareRootFeatureSelector 50)
    compareForests 
        ({ options with Device = gpuDevices } |> addSquareRootFeatureSelector 50)
        ({ options with Device = cpuDevice } |> addSquareRootFeatureSelector 50)

[<Test>]
let ``Random forest on CPU Parallel vs GPU thread pool``() =
    use worker2 = Worker.Create(Device.Default)
    use gpuModule2 = new GpuSplitEntropy.EntropyOptimizationModule(GPUModuleTarget.Worker(worker2), blockSize, reduceBlockSize)
    let gpuDevice1 = GPU(GpuMode.SingleWeight, GpuModuleProvider.DefaultModule)
    let gpuDevice2 = GPU(GpuMode.MultiWeightWithStream 10, GpuModuleProvider.Specified(gpuModule2))
    let gpuDevice = Pool(PoolMode.EqualPartition, [ gpuDevice1; gpuDevice2 ])
    let cpuDevice = Pool(PoolMode.EqualPartition, List.init 2 (fun _ -> CPU(CpuMode.Sequential)))
    let options = { TreeOptions.Default with MaxDepth = 4 }
    compareForests { options with Device = cpuDevice } { options with Device = gpuDevice }

[<Test>]
let ``Speed of training random forests``() =
    let measureRandomForestTraining options numTrees trainingData =
        printfn "Options:\n%A" options
        let watch = System.Diagnostics.Stopwatch.StartNew()
        randomForestClassifier options (getRngFunction 0) numTrees trainingData |> ignore
        watch.Stop()
        let elapsed = watch.Elapsed
        printfn "Total time elapsed: %A" elapsed
        elapsed

    let SEPERATOR = "-------------------------------------------------------------------------------------------"

    let measureDevice numWarmups numTrees (options : TreeOptions) (trainingData : LabeledFeatureSet)
        (entropyDevice : EntropyDevice) =
        let deviceOptions = { options with Device = entropyDevice }
        printfn "%s\n%A warm-up with %d trees" SEPERATOR entropyDevice numWarmups
        measureRandomForestTraining deviceOptions numWarmups trainingData |> ignore
        printfn "%s\n%A measurement with %d trees" SEPERATOR entropyDevice numTrees
        let time = measureRandomForestTraining deviceOptions numTrees trainingData
        time

    let measureGpuVsCpu numSamples numFeatures numClasses numTrees maxDepth poolSize =
        let numWarmups = numTrees / 10 |> min 10 |> max 3
        printf "Generating training data ... "
        let rnd = System.Random(0)
        let trainingData = randomTrainingData rnd numSamples numFeatures numClasses
        printf "sorting ... "
        let trainingData = trainingData.Sorted
        printfn "done."
        let options = { TreeOptions.Default with MaxDepth = maxDepth; EntropyOptions = { TreeOptions.Default.EntropyOptions with AbsMinWeight = 2 } }
        printfn "Starting measurements with %A samples, %A features, %A classes, %A trees, %A threads" numSamples
            numFeatures numClasses numTrees poolSize
        let runner = measureDevice numWarmups numTrees options trainingData
        use worker2 = Worker.Create(Device.Default)
        use gpuModule2 = new GpuSplitEntropy.EntropyOptimizationModule(GPUModuleTarget.Worker(worker2), blockSize, reduceBlockSize)
        let gpuDevice1 = GPU(GpuMode.SingleWeight, GpuModuleProvider.DefaultModule)
        let gpuDevice2 = GPU(GpuMode.MultiWeightWithStream 10, GpuModuleProvider.Specified(gpuModule2))
        let gpuDevice3 = GPU(GpuMode.MultiWeightWithStream 10, GpuModuleProvider.DefaultModule)
        let gpuDevicePool = Pool(PoolMode.EqualPartition, [ gpuDevice1; gpuDevice3 ])
        let cpuDevicePool = Pool(PoolMode.EqualPartition, List.init 2 (fun i -> CPU(CpuMode.Parallel)))
        let gpuNoStreamTime = gpuDevice1 |> runner
        let gpuWithStreamTime = gpuDevice2 |> runner
        let gpuPool2Time = gpuDevicePool |> runner
        let cpuPool2Time = cpuDevicePool |> runner
        printfn "GPU without stream     : %A" gpuNoStreamTime
        printfn "GPU with stream        : %A" gpuWithStreamTime
        printfn "GPU pool 2 with stream : %A" gpuPool2Time
        printfn "CPU pool 2 parallel    : %A" cpuPool2Time
        let cpuTime = cpuPool2Time
        let gpuTime = gpuWithStreamTime
        let speedUp = cpuTime.TotalMilliseconds / gpuTime.TotalMilliseconds
        printfn "Speed up: %.1f" speedUp
        gpuTime.TotalMilliseconds, cpuTime.TotalMilliseconds

    let numSamples = 20000
    let numFeatures = 20
    let numClasses = 2
    let numTrees = 1000
    let maxDepth = 1
    let gpuTime, cpuTime = measureGpuVsCpu numSamples numFeatures numClasses numTrees maxDepth 2
    printf "cpu time: %f, gpu time: %f" cpuTime gpuTime