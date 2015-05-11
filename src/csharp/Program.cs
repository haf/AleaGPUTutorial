using System;

namespace Tutorial.Cs
{
    class Program
    {
        public static void Help()
        {
            Console.WriteLine( "Tutorial.Cs.exe [name]");
            Console.WriteLine("    name =  [");
            Console.WriteLine("        QuickStartSquareTest              |");
            Console.WriteLine("        AdvancedSinCosTest                |");
            Console.WriteLine("        ExamplesBasicSinTest              |");
            Console.WriteLine("        ExamplesMatrixMultTest            |");
            Console.WriteLine("        ExamplesMatrixTranspPerformance   |");
            Console.WriteLine("        ExamplesTriDiagSolverTest         |");
            Console.WriteLine("        ExamplescuBlasAxpyTest            |");
            Console.WriteLine("        ExamplescuBlasGemmTest            |");
            Console.WriteLine("        ExamplescuBlasGemmBatchedTest     |");
            Console.WriteLine("        ExamplesUnboundDeviceReduceTest   |");
            Console.WriteLine("        ExamplesUnboundDeviceScanTest     |");
            Console.WriteLine("        ExamplesUnboundGemm               |");
            Console.WriteLine("        ExamplesUnboundBlockRangeScanTest |");
            Console.WriteLine("        ExamplesUnboundMatrixTest         |");
            Console.WriteLine("        ExamplesNbodySimulation           |");
            Console.WriteLine("        ExamplesSimpleD3D9                |");
            Console.WriteLine("        All ]                             ");
        }

        public static void Main(string[] args)
        {
            if (args.Length > 0)
            {
                var name = args[0].ToLower();

                switch (name)
                {
                    case "quickstartsquaretest":
                        quick_start.ParallelSquare.SquareTest();
                        break;

                    case "advancedsincostest":
                        advancedTechniques.GenericTransform.Test.SinCosTest();
                        break;

                    case "examplesbasicsintest":
                        examples.basic.Test.SinTest();
                        break;

                    case "examplesmatrixmulttest":
                        examples.matrixMultiplication.Test.MatrixMultiplyTest();
                        break;

                    case "examplesmatrixtranspperformance":
                        examples.matrixTranspose.Test.MatrixTransposePerformance();
                        break;

                    case "examplestridiagsolvertest":
                        examples.tridiagSolver.Test.TriDiagSolverTest();
                        break;

                    case "examplescublasaxpytest":
                        examples.cublas.Test.DaxpyTest();
                        examples.cublas.Test.ZaxpyTest();
                        break;

                    case "examplescublasgemmtest":
                        examples.cublas.Test.DgemmTest();
                        examples.cublas.Test.ZgemmTest();
                        break;

                    case "examplescublasgemmbatchedtest":
                        examples.cublas.Test.DgemmBatchedTest();
                        break;

                    case "examplesunbounddevicereducetest":
                        examples.unbound.Test.DeviceReduceTest();
                        break;

                    case "examplesunbounddevicescantest":
                        examples.unbound.Test.DeviceScanInclusiveTest();
                        break;

                    case "examplesunboundgemm":
                        examples.unbound.Test.Gemm1DArrayTest();
                        examples.unbound.Test.Gemm2DArrayTest();
                        break;

                    case "examplesunboundblockrangescantest":
                        examples.unbound.Test.BlockRangeScanTest();
                        break;

                    case "examplesunboundmatrixtest":
                        examples.unbound.Test.BlockRangeScanTest();
                        break;

                    case "examplesnbodysimulation":
                        examples.nbody.Run.Sim();
                        break;

                    case "examplessimpled3d9":
                        examples.simpled3d9.SimpleD3D9.Main();
                        break;

                    case "all":
                        quick_start.ParallelSquare.SquareTest();
                        advancedTechniques.GenericTransform.Test.SinCosTest();
                        examples.basic.Test.SinTest();
                        examples.matrixMultiplication.Test.MatrixMultiplyTest();
                        examples.matrixTranspose.Test.MatrixTransposePerformance();
                        examples.tridiagSolver.Test.TriDiagSolverTest();
                        examples.cublas.Test.DaxpyTest();
                        examples.cublas.Test.ZaxpyTest();
                        examples.cublas.Test.DgemmTest();
                        examples.cublas.Test.ZgemmTest();
                        examples.cublas.Test.DgemmBatchedTest();
                        examples.unbound.Test.DeviceReduceTest();
                        examples.unbound.Test.DeviceScanInclusiveTest();
                        examples.unbound.Test.Gemm1DArrayTest();
                        examples.unbound.Test.Gemm2DArrayTest();
                        examples.unbound.Test.BlockRangeScanTest();
                        examples.nbody.Run.Sim();
                        break;

                    default:
                        Help();
                        break;
                }
            }
            else
            {
                Help();
            }
            Console.WriteLine("Done.");
        }
    }
}
