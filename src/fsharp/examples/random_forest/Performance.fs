(**
Performance measurement CPU vs GPU implementations.

System has been a Intel Core [i7-4771 CPU @ 3.50GHz](http://ark.intel.com/products/77656/Intel-Core-i7-4771-Processor-8M-Cache-up-to-3_90-GHz) and 16 GB RAM.
The GPU is a [Titan Black](http://www.nvidia.com/gtx-700-graphics-cards/gtx-titan-black/).

The Following performance test consists of 20000 samples, 20 features, 2 classes and 1000 trees with maximal depth of 1.
CPU version runs parallel on two cores, GPU implementation with and without streams.

on Windows we get the following performance (max depth = 1):

    GPU without stream     : 00:00:03.9269281
    GPU with stream        : 00:00:02.2141418
    GPU pool 2 with stream : 00:00:03.7811637
    CPU pool 2 parallel    : 00:00:58.9696802
    Speed up: 26.6
    cpu time: 58969.680200, gpu time: 2214.141800

on Linux (max depth = 1):

    GPU without stream     : 00:00:03.0450348
    GPU with stream        : 00:00:01.9684565
    GPU pool 2 with stream : 00:00:03.4957762
    CPU pool 2 parallel    : 00:01:49.9881960
    Speed up: 55.9
    cpu time: 109988.196000, gpu time: 1968.456500

We get a nice speedup of over 26 on Windows. On linux the the CPU version is much slower ([a known problem with mono in the past](http://flyingfrogblog.blogspot.ch/2009/01/mono-22.html),
now probably mainly due to the weaker garbage collector of mono) but the GPU version is still slightly faster than on Windows.

In order to get a feeling for the time, we build a random forest using the same feature set with RandomForestClassifier from Pythons sklearn library:

    import csv
    from sklearn.ensemble import RandomForestClassifier
    import time

    numTrees = 1000
    maxDepth = 1
    nJobs = -1  # If -1, then the number of jobs is set to the number of cores.

    features = []
    labels = []
    with open('Test.csv', 'rb') as csvfile:
        sampleReader = csv.reader(csvfile, delimiter=',')
        for row in sampleReader:
            d = map(float, row)
            features.append(d[0:19])
            labels.append(d[20])

    # creating random forest
    start = time.clock()
    forest = RandomForestClassifier(n_estimators=numTrees, criterion="entropy", max_features=None,
                                    max_depth=maxDepth, n_jobs=nJobs)

    forestFit = forest.fit(features, labels)
    end = time.clock()

    print("elapsed time: " + str(end - start))

which builds the 1000 trees in between 43 s (single CPU core) and 9.6 s (all CPU cores). This is much faster than our CPU implementation, which was expected
as our code though reasonably well optimized is not expected to catch up with the higly optimized sklearn library. Our GPU implementation however still gets 
a nice speed up of over 4 against the most parallel python.

The same test can be done with deeper trees:

on Windows we get the following performance (max depth = 5):

    GPU without stream     : 00:01:02.9673374
    GPU with stream        : 00:00:33.9749665
    GPU pool 2 with stream : 00:00:48.6911049
    CPU pool 2 parallel    : 00:05:24.8222738
    Speed up: 9.6
    cpu time: 324822.273800, gpu time: 33974.966500

on Linux (max depth = 5):

    GPU without stream     : 00:00:36.0482717
    GPU with stream        : 00:00:24.6314113
    GPU pool 2 with stream : 00:00:30.7857099
    CPU pool 2 parallel    : 00:17:49.6928148
    Speed up: 43.4
    cpu time: 1069692.814800, gpu time: 24631.411300

Here the speedup is smaller. This was expected as the GPU implementation is parallel on the number of samples and they shrink after every split.

Python needs for this parameters between 3min 40s (single CPU core) and 44 s (all CPU cores).

Performance test implementation:
*)
module Tutorial.Fs.examples.RandomForest.Performance

open Alea.CUDA
open Tutorial.Fs.examples.RandomForest.DataModel
open Tutorial.Fs.examples.RandomForest.GpuSplitEntropy
open Tutorial.Fs.examples.RandomForest.RandomForest
open Tutorial.Fs.examples.RandomForest.Tests

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

    let savetrainingData (data:LabeledFeatureSet) =
        let features, labels = match data with
                               | LabeledFeatures(x, y) -> x, y
                               | _ -> failwithf "Only Labeled Features are allowed."

        use outFile = new System.IO.StreamWriter("PerformanceTestData.csv")

        for j = 0 to labels.Length - 1 do
            let mutable row = ""
            for i = 0 to features.Length - 1 do
                row <- row + features.[i].[j].ToString(System.Globalization.CultureInfo.InvariantCulture) + ","
            row <- row + labels.[j].ToString()
            outFile.WriteLine(row)
        outFile.Flush()

    let measureGpuVsCpu numSamples numFeatures numClasses numTrees maxDepth poolSize =
        let numWarmups = numTrees / 10 |> min 10 |> max 3
        printf "Generating training data ... "
        let rnd = System.Random(0)
        let trainingData = randomTrainingData rnd numSamples numFeatures numClasses
        printf "saving ... "
        savetrainingData trainingData
        printf "sorting ... "
        let trainingData = trainingData.Sorted
        printfn "done."
        let options = { TreeOptions.Default with MaxDepth = maxDepth; EntropyOptions = { TreeOptions.Default.EntropyOptions with AbsMinWeight = 1 } }
        printfn "Starting measurements with %A samples, %A features, %A classes, %A trees, %A threads" numSamples
            numFeatures numClasses numTrees poolSize
        let runner = measureDevice numWarmups numTrees options trainingData
        use worker2 = Worker.Create(Device.Default)
        use gpuModule2 = new GpuSplitEntropy.EntropyOptimizationModule(GPUModuleTarget.Worker(worker2), blockSize, reduceBlockSize)
        let gpuDevice1 = GPU(GpuMode.SingleWeight, GpuModuleProvider.DefaultModule)
        let gpuDevice2 = GPU(GpuMode.SingleWeightWithStream 10, GpuModuleProvider.Specified(gpuModule2))
        let gpuDevice3 = GPU(GpuMode.SingleWeightWithStream 10, GpuModuleProvider.DefaultModule)
        let gpuDevicePool = Pool(PoolMode.EqualPartition, [ gpuDevice1; gpuDevice3 ])
        let cpuDevicePool = Pool(PoolMode.EqualPartition, List.init poolSize (fun _ -> CPU(CpuMode.Parallel)))
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