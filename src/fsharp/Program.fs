[<EntryPoint>]
let main argv =
    let help() =
        printfn "Tutorial.Fs.exe [name]"
        printfn "    name = ["
        printfn "        QuickStartSquareChart                |"
        printfn "        QuickStartSquareTest                 |"
        printfn "        AdvancedSinCosTest                   |"
        printfn "        AdvancedMatrixMultTest               |"
        printfn "        PerfTuningMatrixTranspProfileF32     |"
        printfn "        PerfTuningMatrixTranspProfileF64     |"
        printfn "        PerfTuningMatrixTranspPerformanceJIT |"
        printfn "        PerfTuningMatrixTranspPerformanceAOT |"
        printfn "        ExamplesBasicSinTest                 |"
        printfn "        ExamplesDeviceQuery                  |"
        printfn "        ExamplesSimpleScan                   |"
        printfn "        ExamplesReduceTest                   |"
        printfn "        ExamplesScanTest                     |"
        printfn "        ExamplesMatrixMultTest               |"
        printfn "        ExamplesMatrixTranspPerformance      |"
        printfn "        ExamplesMovingAverage                |"
        printfn "        ExamplesMovingAverageDirect          |"
        printfn "        ExamplesTriDiagSolverTest            |"
        printfn "        ExamplesHeatPdeTest                  |"
        printfn "        ExamplesHeatPdeDirect3d              |"
        printfn "        ExamplesCublasAxpyTest               |"
        printfn "        ExamplesCublasGemmTest               |"
        printfn "        ExamplesCublasGemmBatchedTest        |"
        printfn "        ExamplesCudnnMnistTest               |"
        printfn "        ExamplesCurandEstimatePiTest         |" 
        printfn "        ExamplesUnboundGemm                  |"
        printfn "        ExamplesUnboundReduceTest            |"
        printfn "        ExamplesUnboundScanTest              |"
        printfn "        ExamplesUnboundBlockRangeScanTest    |"
        printfn "        ExamplesUnboundRandomTest            |"
        printfn "        ExamplesNbodySimulation              |"
        printfn "        ExamplesRandomForestIrisExample      |"
        printfn "        ExamplesRandomForestPerformance      |"
        printfn "        ExamplesSimpleD3D9                   |"
        printfn "           ]"
    match argv with
    | [| name |] ->
        match name.ToLower() with
        | "quickstartsquarechart" -> Tutorial.Fs.quickStart.ParallelSquare.squareChart()
        | "quickstartsquaretest" -> Tutorial.Fs.quickStart.ParallelSquare.squareTest()
        | "advancedsincostest" -> Tutorial.Fs.advancedTechniques.GenericTransform.sinCosTest()
        | "advancedmatrixmulttest" -> Tutorial.Fs.advancedTechniques.GenericMatrixMult.matrixMultTest()
        | "perftuningmatrixtranspprofilef32" ->
            Tutorial.Fs.performanceTuning.GenericMatrixTransp.matrixTransposeProfileF32()
        | "perftuningmatrixtranspprofilef64" ->
            Tutorial.Fs.performanceTuning.GenericMatrixTransp.matrixTransposeProfileF64()
        | "perftuningmatrixtranspperformancejit" ->
            Tutorial.Fs.performanceTuning.GenericMatrixTransp.matrixTransposePerformanceJIT()
        | "perftuningmatrixtranspperformanceaot" ->
            Tutorial.Fs.performanceTuning.GenericMatrixTransp.matrixTransposePerformanceAOT()
        | "examplesbasicsintest" -> Tutorial.Fs.examples.basic.sinTest()
        | "examplesdevicequery" -> Tutorial.Fs.examples.deviceQuery.deviceQuery()
        | "examplessimplescan" -> Tutorial.Fs.examples.simpleScan.scanTest()
        | "examplesreducetest" -> Tutorial.Fs.examples.genericReduce.Test.reduceTest()
        | "examplesscantest" -> Tutorial.Fs.examples.genericScan.Test.scanTest()
        | "examplesmatrixmulttest" -> Tutorial.Fs.examples.matrixMultiplication.matrixMultiplyTest()
        | "examplesmatrixtranspperformance" -> Tutorial.Fs.examples.matrixTranspose.matrixTransposePerformance()
        | "examplesmovingaverage" -> Tutorial.Fs.examples.movingAverage.movingAverageTest()
        | "examplesmovingaveragedirect" -> Tutorial.Fs.examples.movingAverage.movingAverageDirectTest()
        | "examplestridiagsolvertest" -> Tutorial.Fs.examples.tridiagSolver.Solver.triDiagSolverTest()
        | "examplesheatpdetest" -> Tutorial.Fs.examples.heatPde.Solver.heatPdeTest()
        | "examplesheatpdedirect3d" -> Tutorial.Fs.examples.heatPde.Direct3d.heatPdeDirect3d()
        | "examplesunboundgemm" ->
            Tutorial.Fs.examples.unbound.MatrixMult.gpu.gemm1DArrayTest()
            Tutorial.Fs.examples.unbound.MatrixMult.gpu.gemm2DArrayTest()
        | "examplescublasaxpytest" ->
            Tutorial.Fs.examples.cublas.Axpy.daxpyTest()
            Tutorial.Fs.examples.cublas.Axpy.zaxpyTest()
        | "examplescublasgemmtest" ->
            Tutorial.Fs.examples.cublas.Gemm.dgemmTest()
            Tutorial.Fs.examples.cublas.Gemm.zgemmTest()
        | "examplescublasgemmbatchedtest" -> Tutorial.Fs.examples.cublas.GemmBatched.dgemmBatchedTest()
        | "examplescudnnmnisttest" -> Tutorial.Fs.examples.cudnn.Mnist.test()
        | "examplescurandestimatepitest" -> Tutorial.Fs.examples.curand.EstimatePi.estimatePi()
        | "examplesunboundreducetest" -> Tutorial.Fs.examples.unbound.Reduce.deviceReduceTest()
        | "examplesunboundscantest" -> Tutorial.Fs.examples.unbound.Scan.deviceScanTest()
        | "examplesunboundblockrangescantest" -> Tutorial.Fs.examples.unbound.BlockRageScan.blockRangeScanTest()
        | "examplesunboundrandomtest" -> Tutorial.Fs.examples.unbound.Random.randomTest()
        | "examplesnbodysimulation" -> Tutorial.Fs.examples.nbody.OpenGL.runSim()
        | "examplessimpled3d9" -> Tutorial.Fs.examples.simpled3d9.SimpleD3D9.main()
        | "examplesrandomforestirisexample" -> Tutorial.Fs.examples.RandomForest.IrisExample.irisExample()
        | "examplesrandomforestperformance" -> Tutorial.Fs.examples.RandomForest.Performance.``Speed of training random forests``()

        | "help" -> help()
        | _ ->
            printfn "unknown sample name %s\n" name
            help()
    | _ -> help()
    printfn "Done."
    0