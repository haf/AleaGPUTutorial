(**
The [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set) contains four features (length and the width of both: 
[sepals and petals](http://en.wikipedia.org/wiki/Sepal#/media/File:Petal-sepal.jpg)) of three species:

 - [Iris setosa](http://en.wikipedia.org/wiki/Iris_flower_data_set#/media/File:Kosaciec_szczecinkowaty_Iris_setosa.jpg), 
 - [Iris virginica](http://en.wikipedia.org/wiki/Iris_flower_data_set#/media/File:Iris_virginica.jpg) and  
 - [Iris versicolor](http://en.wikipedia.org/wiki/Iris_flower_data_set#/media/File:Iris_versicolor_3.jpg). 

It is not the typical dataset for random forests, but as it only has few features it gives small trees ideal as an example.
*)

module Tutorial.Fs.examples.RandomForest.IrisExample

open FSharp.Data
open FSharp.Charting

open Tutorial.Fs.examples.RandomForest.GpuSplitEntropy
open Tutorial.Fs.examples.RandomForest.RandomForest
open Tutorial.Fs.examples.RandomForest.Array

(**
In order to get a feeling for the data, we do three scatterplots of the different features:

<img src="../../content/images/Sepal-length_Sepal-width.png" width="500" alt="scatter plot: sepal length vs. sepal width"> 
<img src="../../content/images/Sepal-Width_Petal-length.png" width="500" alt="scatter plot: sepal width vs. petal length"> 
<img src="../../content/images/Petal-length_Petal-width.png" width="500" alt="scatter plot: petal length vs. petal widht"> 
*)
let irisScatterPlot (trainingData : DataModel.LabeledSample[]) =

    let setosaFilter = (fun e -> snd e = 0)
    let versicolorFilter = (fun e -> snd e = 1)
    let virginicaFilter = (fun e -> snd e = 2)

    let chooseSepalLengthWidth (x:_[]) = x.[0], x.[1]
    let chooseSepalWidthPetalLenght (x:_[]) = x.[1], x.[2]
    let choosePetalLenghtPetalWidth (x:_[]) = x.[2], x.[3]

    let createChart (chooseFeatures : float[] -> float*float) filename =
        let extractPoints (trainingData : DataModel.LabeledSample[]) filterLabel chooseFeature =
            trainingData |> Array.filter filterLabel 
                         |> Array.map fst
                         |> Array.map chooseFeature
        let chart =  Chart.Combine([  Chart.Point(extractPoints trainingData setosaFilter chooseFeatures, "setosa")
                                      Chart.Point(extractPoints trainingData versicolorFilter chooseFeatures, "versicolor")
                                      Chart.Point(extractPoints trainingData virginicaFilter chooseFeatures, "virginica") 
                                   ])
                        |> Chart.WithLegend(true)
                        |> Chart.WithTitle(filename)
        chart.ShowChart() |> ignore
        chart.SaveChartAs(filename + ".png", ChartTypes.ChartImageFormat.Png)

    createChart chooseSepalLengthWidth "Sepal-length Sepal-width"
    createChart chooseSepalWidthPetalLenght "Sepal-Width Petal-length"
    createChart choosePetalLenghtPetalWidth "Petal-length Petal-width"

    System.Windows.Forms.Application.Run()
//    ()

(**
Train random forest and to an out of sample test.

1. Randomly split up dat in `training` and `test` set.
2. Create set of Parameters for random forest.
3. Train the model using the `training` - set.
4. predict labels of `test` - set and calculate the fraction of correct predictions.
*)
let printFractionOfCorrectForcasts trainingData device =
    // split up data in training and test data:
    let trainingData, testData = randomlySplitUpArray trainingData (System.Random(42)) (70 * Array.length trainingData / 100)

//    let options = GpuSplitEntropy.EntropyOptimizationOptions.Default
    let options = { GpuSplitEntropy.EntropyOptimizationOptions.Default with
                        FeatureSelector = GpuSplitEntropy.EntropyOptimizationOptions.SquareRootFeatureSelector (System.Random()) }

    let options = { TreeOptions.Default with
                        MaxDepth = 3
                        Device = device
                        EntropyOptions = options }

    printfn "%A" options

    // train model
    let trainingData = LabeledSamples trainingData
    let model = randomForestClassifier options (System.Random(42)) 100 trainingData
    
    // predict labels
    let features, expectedLabels = Array.unzip testData
    let forecastedLabels = Array.map (forecast model) features
    let fraction = (forecastedLabels, expectedLabels) ||> Array.map2  (fun x y -> if x=y then 1.0 else 0.0)
                                                       |> Array.average
    printfn "%f %% of forecasts were correct (out of sample)" (fraction*100.0)


(**
Read in the Iris data set from csv-file using csv-type-provider
and call the above functions for CPU as well as for GPU.
Prediction-accuracy is between 95% and 100%.
*)
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

    irisScatterPlot trainingData

    let cpuDevice = CPU(CpuMode.Parallel)
    let gpuDevice = GPU(GpuMode.MultiWeightWithStream 10, GpuModuleProvider.DefaultModule)

    printFractionOfCorrectForcasts trainingData cpuDevice
    printFractionOfCorrectForcasts trainingData gpuDevice