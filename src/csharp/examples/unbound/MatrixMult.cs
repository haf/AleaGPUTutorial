using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing.Printing;
using System.Linq;
using Alea.CUDA;
using Alea.CUDA.Unbound;
using Alea.CUDA.Unbound.LinAlg.Matrix.Multiply.GPU;
using Alea.CUDA.Utilities;
using NUnit.Framework;
using Tutorial.Cs.examples.cublas;
using Common = Alea.CUDA.Unbound.LinAlg.Matrix.Multiply.GPU.Common;

namespace Tutorial.Cs.examples.unbound
{
    //[unboundMatrixMultCPU]
    class MatrixMult
    {
        public static double[] MultiplyMatrix(double[] c, double[] a, double[] b, int wA, int wB)
        {
            var hA = a.Length/wA;
            for (var i = 0; i < hA; i++)
            {
                for (var j = 0; j < wB; j++)
                {
                    var sum = 0.0;
                    for (var k = 0; k < wA; k++)
                    {
                        var al = a[i*wA + k];
                        var bl = b[k*wB + j];
                        sum += al*bl;
                    }
                    c[i*wB + j] = sum;
                }
            }
            return c;
        }
    }
    //[/unboundMatrixMultCPU]


    public partial class Test
    {
        [Test]
        [Ignore]
        //[unboundMatrixMultGemm1DArrayTest]
        public static void Gemm1DArrayTest()
        {
            var gpuMultiplication64 = Common.DefaultMatrixMultiplyModuleF64.DefaultInstance;
            const int wA = 32, hA = 64, wB = 64, hB = 32;
            const int wC = 64, hC = 64;
            var rng = new Random(42);
            var a = (Enumerable.Repeat(rng, hA*wA).Select((random, i) => random.NextDouble())).ToArray();
            var b = (Enumerable.Repeat(rng, hB*wB).Select((random, i) => random.NextDouble())).ToArray();
            var c = Enumerable.Repeat(0.0, hC*wC).ToArray();

            var cpuOutput = MatrixMult.MultiplyMatrix(c, a, b, wA, wB);
            var gpuOutput = gpuMultiplication64.Mult(Common.Implementation.PrefetchingData, Common.Transpose.NoTranspose, Common.Transpose.NoTranspose, Common.MatrixStorageOrder.RowMajor, wA, wB, 1.0, 0.0, a, b, c);

            for (var i = 0; i < cpuOutput.Length; ++i)
            {
                Assert.AreEqual(cpuOutput[i], gpuOutput[i], 1.0e-12);
            }
        }
        //[/unboundMatrixMultGemm1DArrayTest]

        [Test]
        [Ignore]
        //[unboundMatrixMultGemm2DArrayTest]
        public static void Gemm2DArrayTest()
        {
            var gpuMultiplication64 = Common.DefaultMatrixMultiplyModuleF64.DefaultInstance;
            const int wA = 31, hA = 65, wB = 65, hB = 31;
            const int wC = 65, hC = 65;
            var rng = new Random(42);
            var a = Array2D.ofArrayRowMajor(hA, wA, (Enumerable.Repeat(rng, hA * wA).Select((random, i) => random.NextDouble())).ToArray());
            var b = Array2D.ofArrayRowMajor(hB, wB, (Enumerable.Repeat(rng, hB * wB).Select((random, i) => random.NextDouble())).ToArray());
            var c = Array2D.ofArrayRowMajor(hC, wC, Enumerable.Repeat(0.0, hC * wC).ToArray());

            var cpuOutput = MatrixMult.MultiplyMatrix(Array2D.toArrayRowMajor(c), Array2D.toArrayRowMajor(a), Array2D.toArrayRowMajor(b), wA, wB);
            var gpuOutput = gpuMultiplication64.Mult(Common.Implementation.PrefetchingData, Common.Transpose.NoTranspose, Common.Transpose.NoTranspose, Common.MatrixStorageOrder.RowMajor, 1.0, 0.0, a, b, c);

            for (var i = 0; i < cpuOutput.ToArray().Length; ++i)
            {
                Assert.AreEqual(cpuOutput[i], Array2D.toArrayRowMajor(gpuOutput)[i], 1.0e-12);
            }
        }
    }
    //[/unboundMatrixMultGemm2DArrayTest]
}
