[<EntryPoint>]
let main argv = 

    let help () =
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
        printfn "        ExamplesUnboundGemm                  |" 
        printfn "        ExamplesUnboundReduceTest            |"
        printfn "        ExamplesUnboundScanTest              |"
        printfn "        ExamplesUnboundBlockRangeScanTest    |"
        printfn "        ExamplesUnboundRandomTest            |"
        printfn "        ExamplesMcBasketTest                 |"
        printfn "        ExamplesNbodySimulation              |"
        printfn "        ExamplesRandomForest                 |"
        printfn "        All                                   "
        printfn "           ]"

    match argv with
    | [| name |] -> 
        match name.ToLower() with
        | "quickstartsquarechart"                -> Tutorial.Fs.quickStart.ParallelSquare.squareChart() 
        | "quickstartsquaretest"                 -> Tutorial.Fs.quickStart.ParallelSquare.squareTest() 
        | "advancedsincostest"                   -> Tutorial.Fs.advancedTechniques.GenericTransform.sinCosTest()
        | "advancedmatrixmulttest"               -> Tutorial.Fs.advancedTechniques.GenericMatrixMult.matrixMultTest()
        | "perftuningmatrixtranspprofilef32"     -> Tutorial.Fs.performanceTuning.GenericMatrixTransp.matrixTransposeProfileF32()
        | "perftuningmatrixtranspprofilef64"     -> Tutorial.Fs.performanceTuning.GenericMatrixTransp.matrixTransposeProfileF64()
        | "perftuningmatrixtranspperformancejit" -> Tutorial.Fs.performanceTuning.GenericMatrixTransp.matrixTransposePerformanceJIT()
        | "perftuningmatrixtranspperformanceaot" -> Tutorial.Fs.performanceTuning.GenericMatrixTransp.matrixTransposePerformanceAOT()
        | "examplesbasicsintest"                 -> Tutorial.Fs.examples.basic.sinTest()
        | "examplesdevicequery"                  -> Tutorial.Fs.examples.deviceQuery.deviceQuery()
        | "examplessimplescan"                   -> Tutorial.Fs.examples.simpleScan.scanTest()
        | "examplesreducetest"                   -> Tutorial.Fs.examples.genericReduce.Test.reduceTest()
        | "examplesscantest"                     -> Tutorial.Fs.examples.genericScan.Test.scanTest()
        | "examplesmatrixmulttest"               -> Tutorial.Fs.examples.matrixMultiplication.matrixMultiplyTest()
        | "examplesmatrixtranspperformance"      -> Tutorial.Fs.examples.matrixTranspose.matrixTransposePerformance()
        | "examplesmovingaverage"                -> Tutorial.Fs.examples.movingAverage.movingAverageTest()
        | "examplesmovingaveragedirect"          -> Tutorial.Fs.examples.movingAverage.movingAverageDirectTest()
        | "examplestridiagsolvertest"            -> Tutorial.Fs.examples.tridiagSolver.Solver.triDiagSolverTest()
        | "examplesheatpdetest"                  -> Tutorial.Fs.examples.heatPde.Solver.heatPdeTest()
        | "examplesheatpdedirect3d"              -> Tutorial.Fs.examples.heatPde.Direct3d.heatPdeDirect3d()
        | "examplesunboundgemm"                  -> Tutorial.Fs.examples.unbound.MatrixMult.gpu.gemm1DArrayTest()
                                                    Tutorial.Fs.examples.unbound.MatrixMult.gpu.gemm2DArrayTest()
        | "examplescublasaxpytest"               -> Tutorial.Fs.examples.cublas.Axpy.daxpyTest()
                                                    Tutorial.Fs.examples.cublas.Axpy.zaxpyTest()
        | "examplescublasgemmtest"               -> Tutorial.Fs.examples.cublas.Gemm.dgemmTest()
                                                    Tutorial.Fs.examples.cublas.Gemm.zgemmTest()
        | "examplescublasgemmbatchedtest"        -> Tutorial.Fs.examples.cublas.GemmBatched.dgemmBatchedTest()
        | "examplesunboundreducetest"            -> Tutorial.Fs.examples.unbound.Reduce.deviceReduceTest()
        | "examplesunboundscantest"              -> Tutorial.Fs.examples.unbound.Scan.deviceScanTest()
        | "examplesunboundblockrangescantest"    -> Tutorial.Fs.examples.unbound.BlockRageScan.blockRangeScanTest()
        | "examplesunboundrandomtest"            -> Tutorial.Fs.examples.unbound.Random.randomTest()
        | "examplesnbodysimulation"              -> Tutorial.Fs.examples.nbody.OpenGL.runSim()
        | "examplesrandomforest"                 -> Tutorial.Fs.examples.RandomForest.RandomForestExample.``Speed of training random forests``()
        | "all"                                  -> Tutorial.Fs.quickStart.ParallelSquare.squareTest() 
                                                    Tutorial.Fs.advancedTechniques.GenericTransform.sinCosTest()
                                                    Tutorial.Fs.advancedTechniques.GenericMatrixMult.matrixMultTest()
                                                    Tutorial.Fs.performanceTuning.GenericMatrixTransp.matrixTransposeProfileF32()
                                                    Tutorial.Fs.performanceTuning.GenericMatrixTransp.matrixTransposeProfileF64()
                                                    Tutorial.Fs.performanceTuning.GenericMatrixTransp.matrixTransposePerformanceJIT()
                                                    Tutorial.Fs.performanceTuning.GenericMatrixTransp.matrixTransposePerformanceAOT()
                                                    Tutorial.Fs.examples.basic.sinTest()
                                                    Tutorial.Fs.examples.deviceQuery.deviceQuery()
                                                    Tutorial.Fs.examples.simpleScan.scanTest()
                                                    Tutorial.Fs.examples.genericReduce.Test.reduceTest()
                                                    Tutorial.Fs.examples.genericScan.Test.scanTest()
                                                    Tutorial.Fs.examples.matrixMultiplication.matrixMultiplyTest()
                                                    Tutorial.Fs.examples.matrixTranspose.matrixTransposePerformance()
                                                    Tutorial.Fs.examples.movingAverage.movingAverageTest()
                                                    Tutorial.Fs.examples.movingAverage.movingAverageDirectTest()
                                                    Tutorial.Fs.examples.tridiagSolver.Solver.triDiagSolverTest()
                                                    Tutorial.Fs.examples.heatPde.Solver.heatPdeTest()
                                                    Tutorial.Fs.examples.heatPde.Direct3d.heatPdeDirect3d()
                                                    Tutorial.Fs.examples.unbound.MatrixMult.gpu.gemm1DArrayTest()
                                                    Tutorial.Fs.examples.unbound.MatrixMult.gpu.gemm2DArrayTest()
                                                    Tutorial.Fs.examples.cublas.Axpy.daxpyTest()
                                                    Tutorial.Fs.examples.cublas.Axpy.zaxpyTest()
                                                    Tutorial.Fs.examples.cublas.Gemm.dgemmTest()
                                                    Tutorial.Fs.examples.cublas.Gemm.zgemmTest()
                                                    Tutorial.Fs.examples.cublas.GemmBatched.dgemmBatchedTest()
                                                    Tutorial.Fs.examples.unbound.Reduce.deviceReduceTest()
                                                    Tutorial.Fs.examples.unbound.Scan.deviceScanTest()
                                                    Tutorial.Fs.examples.unbound.BlockRageScan.blockRangeScanTest()
                                                    Tutorial.Fs.examples.unbound.Random.randomTest()
                                                    Tutorial.Fs.examples.nbody.OpenGL.runSim()
                                                    Tutorial.Fs.examples.RandomForest.RandomForestExample.``Speed of training random forests``()

        | "help"                                 -> help()
        | _ -> printfn "unknown sample name %s\n" name; help()
    | _ -> help()
    printfn "Done."

    0