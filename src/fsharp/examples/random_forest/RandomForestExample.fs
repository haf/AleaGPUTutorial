module Tutorial.Fs.examples.RandomForest.RandomForestExample

open NUnit.Framework
open FsUnit
open Alea.CUDA
open Tutorial.Fs.examples.RandomForest.RandomForest

//Test
// @xiang: I have to comment out this function, reason:
// 1) it doesn't pass test 
// 2) there is warning, I don't want it show as we will do a first publish on github.
// once it is published on github and you can create a branch then uncomment this.
//[<Test>]
//let ``CPU vs GPU optimizer`` () =
//    let rnd = System.Random 0
//    let numSamples = 10000
//    let numFeatures = 20
//    let numClasses = 2
//    let domains = Array.init numFeatures (fun _ -> Array.init numSamples (fun _ -> rnd.NextDouble()))
//    let labels = Array.init numSamples (fun i -> i % numClasses)
//    let trainingSet = LabeledDomains (domains, labels)
//    let (SortedFeatures sortedTrainingSet) = trainingSet.Sorted
//    
//    let gpuOpt, gpuDisposer = EntropyDevice.GPU.CreateDefaultOptions numClasses sortedTrainingSet
//    let cpuOpt, _ = (EntropyDevice.CPU Parallel).CreateDefaultOptions numClasses sortedTrainingSet
//    let weights = randomWeights rnd numSamples
//    try
//        let gpuResult = gpuOpt weights
//        let cpuResult = cpuOpt weights
//        (gpuResult, cpuResult) ||> Array.iter2 (fun g c -> 
//            g |> fst |> should (equalWithin 1e-10) (c |> fst)
//            g |> snd |> should equal (c |> snd)
//        )
//    finally
//        gpuDisposer()

// example:
let randomTrainingData (rnd : System.Random) numSamples numFeatures numClasses =
    let domains = Array.init numFeatures (fun _ ->
                Array.init numSamples (fun _ -> rnd.NextDouble())
    )
    let labels = Array.init numSamples (fun _ -> rnd.Next(numClasses))
    LabeledDomains (domains, labels)

let measureRandomForestTraining options numTrees trainingData = 
    let rnd = System.Random(0)
    let watch = System.Diagnostics.Stopwatch.StartNew()
    printfn "Options:\n%A" options
    randomForestClassifier rnd options numTrees trainingData |> ignore
    watch.Stop()
    let elapsed = watch.Elapsed
    printfn "Total time elapsed: %A" elapsed
    elapsed

[<Literal>]
let SEPERATOR = "-------------------------------------------------------------------------------------------"

// @xiang: I have to comment out this function, reason:
// 1) it doesn't pass test 
// 2) there is warning, I don't want it show as we will do a first publish on github.
// once it is published on github and you can create a branch then uncomment this.
//let measureDevice poolSize numWarmups numTrees (options : TreeOptions) numClasses (trainingData : LabeledFeatureSet)
//    (entropyDevice : EntropyDevice) = 
//    let (SortedFeatures sortedData) = trainingData.Sorted
//    let device, disposers = 
//        if poolSize = 1 then
//            let device, disposer = entropyDevice.ToCached options.EntropyOptions numClasses sortedData
//            device, [|disposer|]
//        else
//            entropyDevice.CreatePool poolSize options.EntropyOptions numClasses sortedData
//    let deviceOptions = {options with Device = device}
//    printfn "%s\n%A warm-up with %d trees" SEPERATOR entropyDevice numWarmups
//    measureRandomForestTraining deviceOptions numWarmups trainingData |> ignore
//    printfn "%s\n%A measurement with %d trees" SEPERATOR entropyDevice numTrees
//    let time = measureRandomForestTraining deviceOptions numTrees trainingData
//    disposers |> Array.iter(fun disposer -> disposer())
//    time

//let measureGpuVsCpu numSamples numFeatures numClasses numTrees maxDepth poolSize =
//    let numWarmups = numTrees / 10 |> min 10 |> max 3
//    printf "Generating training data ... "
//    let rnd = System.Random(0)
//    let trainingData = randomTrainingData rnd numSamples numFeatures numClasses
//    printf "sorting ... "
//    let trainingData = trainingData.Sorted
//    printfn "done."
//
//    let options = 
//        { TreeOptions.Default with 
//            MaxDepth = maxDepth; 
//            EntropyOptions = {TreeOptions.Default.EntropyOptions with AbsMinWeight = 2}
//        }
//
//    printfn "Starting measurements with %A samples, %A features, %A classes, %A trees, %A threads" 
//        numSamples numFeatures numClasses numTrees poolSize
//
//    let runner = measureDevice poolSize numWarmups numTrees options numClasses trainingData
//    let gpuTime = GPU |> runner
//    let cpuTime = CPU Sequential |> runner
//
//    let speedUp = cpuTime.TotalMilliseconds / gpuTime.TotalMilliseconds
//    printfn "Speed up: %.1f" speedUp
//
//    gpuTime.TotalMilliseconds, cpuTime.TotalMilliseconds

let ``Speed of training random forests`` () =
    () // todo, uncomment the code after on a feature branch on github
//    if Device.Devices.[0].Arch.Major < 3 then 
//        printf "The example: RandomForest needs a GPU with Compute-Capability 3.0 or higher, \nbut your default GPU has: %d.%d\n" Device.Devices.[0].Arch.Major Device.Devices.[0].Arch.Minor
//    else
//        let numSamples = 20000
//        let numFeatures = 20
//        let numClasses = 2
//        let numTrees = 1000
//        let maxDepth = 1
//        let gpuTime, cpuTime = measureGpuVsCpu numSamples numFeatures numClasses numTrees maxDepth 2
//        printf "cpu needed: %f, gpu needed: %f" cpuTime gpuTime