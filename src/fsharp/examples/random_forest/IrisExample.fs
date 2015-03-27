module Tutorial.Fs.examples.RandomForest.IrisExample

open FSharp.Data
open FSharp.Charting

open Tutorial.Fs.examples.RandomForest.GpuSplitEntropy
open Tutorial.Fs.examples.RandomForest.RandomForest
open Tutorial.Fs.examples.RandomForest.Array

(***
Make scatter-plots of the iris data. 

<img src="../content/images/Sepal-length_Sepal-width.png" width="500" alt="parallel square result"> 
<img src="../content/images/Sepal-Width_Petal-length.png" width="500" alt="parallel square result"> 
<img src="../content/images/Petal-length_Petal-width.png" width="500" alt="parallel square result"> 
*)

let irisScatterPlot (trainingData : DataModel.LabeledSample[]) =

    let createPoints (trainingData : DataModel.LabeledSample[]) filter choose =
        trainingData |> Array.filter filter 
                     |> Array.map fst
                     |> Array.map choose

    let setosaFilter = (fun e -> snd e = 0)
    let versicolorFilter = (fun e -> snd e = 1)
    let virginicaFilter = (fun e -> snd e = 2)

    let chooseSepalLengthWidth (x:_[]) = x.[0], x.[1] // (fun x -> x.[0], x.[1])
    let chooseSepalWidthPetalLenght (x:_[]) = x.[1], x.[2]
    let choosePetalLenghtPetalWidth (x:_[]) = x.[2], x.[3]

    let createChart (choose : float[] -> float*float) filename =
        let chart =  Chart.Combine( [  Chart.Point(createPoints trainingData setosaFilter choose, "setosa")
                                       Chart.Point(createPoints trainingData versicolorFilter choose, "versicolor")
                                       Chart.Point(createPoints trainingData virginicaFilter choose, "virginica") ])
//                                       |> Chart.WithYAxis(Min=1.5, Max=4.5)
//                                       |> Chart.WithXAxis(Min=4.0, Max=8.0)
                                       |> Chart.WithLegend(true)
                                       |> Chart.WithTitle(filename)
//                                       |> Chart.WithXAxis(Enabled=false)
//                                       |> Chart.WithYAxis(Enabled=false)
        chart.ShowChart() |> ignore
        chart.SaveChartAs(filename + ".png", ChartTypes.ChartImageFormat.Png)

    createChart chooseSepalLengthWidth "Sepal-length_Sepal-width"
    createChart chooseSepalWidthPetalLenght "Sepal-Width_Petal-length"
    createChart choosePetalLenghtPetalWidth "Petal-length_Petal-width"

    System.Windows.Forms.Application.Run()

let printFractionOfCorrectForcasts trainingData device =

    // split up data in training and test data:
    let trainingData, testData = randomlySplitUpArray trainingData (System.Random()) (70 * Array.length trainingData / 100)

//    let options = GpuSplitEntropy.EntropyOptimizationOptions.Default
    let options = { GpuSplitEntropy.EntropyOptimizationOptions.Default with
                        FeatureSelector = GpuSplitEntropy.EntropyOptimizationOptions.SquareRootFeatureSelector (System.Random()) }

    let options = { TreeOptions.Default with
                        Device = device
                        EntropyOptions = options }

    printfn "%A" options

    // train model
    let trainingData = LabeledSamples trainingData
    let model = randomForestClassifier (System.Random()) options 100 trainingData
    
    // predict labels
    let features, expectedLabels = Array.unzip testData
    let forecastedLabels = Array.map (forecast model) features
    let fraction = (forecastedLabels, expectedLabels) ||> Array.map2  (fun x y -> if x=y then 1.0 else 0.0)
                                                       |> Array.average
    printfn "%f  of forecasts were correct (out of sample)" (fraction*100.0)


// see: http://archive.ics.uci.edu/ml/ for more machine learning datasets
let irisExample () =
    // read in data
    let path = @"..\src\fsharp\examples\random_forest\irisExample.csv"
    let data =  CsvFile.Load(path).Cache()
 
    let trainingData = 
        [| for row in data.Rows ->
            [|
                  row.GetColumn "Sepal length" |> float
                  row.GetColumn "Sepal width" |> float
                  row.GetColumn "Petal length" |> float
                  row.GetColumn "Petal width" |> float
            |],
            row.GetColumn "Species" |> (fun x -> match x with
                                                 | "I. setosa" -> 0
                                                 | "I. versicolor" -> 1
                                                 | "I. virginica" -> 2
                                                 | x -> failwithf "should not happen %A" x)
        |]

    plotData trainingData

    let cpuDevice = CPU(CpuMode.Parallel)
    let gpuDevice = GPU(GpuMode.SingleWeightWithStream 10, GpuModuleProvider.DefaultModule)

    printFractionOfCorrectForcasts trainingData cpuDevice
    printFractionOfCorrectForcasts trainingData gpuDevice

let titanicExample () =
    // read in data
    let path = @"..\src\fsharp\examples\random_forest\titanicExample.csv"
    let data =  CsvFile.Load(path).Cache()
 
    let trainingData = 
        [| for row in data.Rows ->
            [|
                  row.GetColumn "PassengerId" |> float
                  row.GetColumn "Pclass" |> float
                  row.GetColumn "Sex" |> (fun x -> if x = "male" then 1.0 else 0.0)
                  row.GetColumn "Age" |> (fun x -> if x = "" then -1.0 else float x)
            |],
            row.GetColumn "Survived" |> int
        |]

    printFractionOfCorrectForcasts trainingData 