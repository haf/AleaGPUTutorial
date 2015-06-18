using System;
using System.Collections.Generic;
using System.Linq;
using Alea.CUDA;
using Alea.CUDA.Utilities;
using Alea.CUDA.CULib;
using NUnit.Framework;
using cublasOperation_t = Alea.CUDA.CULib.CUBLASInterop.cublasOperation_t;

namespace Tutorial.Cs.examples.cublas
{
    class GemmBatched
    {
        private static readonly CUBLAS Cublas = CUBLAS.Default;
        private static readonly Worker Worker = Cublas.Worker;

        private const cublasOperation_t Transa = cublasOperation_t.CUBLAS_OP_N;
        private const cublasOperation_t Transb = cublasOperation_t.CUBLAS_OP_N;

        //[gemmBatchedCpu]
        public static class cpu
        {
            public static double[][] Dgemm(int m, int n, int k, double alpha, double[][] hAs, int lda, double[][] hBs, int ldb, double beta, double[][] hCs, int ldc)
            {
                var batchCount = hAs.Length;
                var results = new List<double[]>(batchCount);

                for (var batch = 0; batch < batchCount; ++batch)
                {
                    var A = Array2D.ofArrayColumnMajor(lda, k, hAs[batch]);
                    var B = Array2D.ofArrayColumnMajor(ldb, n, hBs[batch]);
                    var C = Array2D.ofArrayColumnMajor(ldc, n, hCs[batch]);

                    for (var j = 1; j <= n; j++)
                    {
                        for (var i = 1; i <= m; i++)
                            C[i - 1, j - 1] = beta * C[i - 1, j - 1];

                        for (var l = 1; l <= k; l++)
                        {
                            var temp = alpha * B[l - 1, j - 1];
                            for (var i = 1; i <= m; i++)
                                C[i - 1, j - 1] += temp * A[i - 1, l - 1];
                        }
                    }
                    results.Add(Array2D.toArrayColumnMajor(C));
                }

                return results.ToArray();
            }

            public static double2[][] Zgemm(int m, int n, int k, double2 alpha, double2[][] hAs, int lda, double2[][] hBs, int ldb, double2 beta, double2[][] hCs, int ldc)
            {
                var batchCount = hAs.Length;
                var results = new List<double2[]>(batchCount);

                for (var batch = 0; batch < batchCount; ++batch)
                {
                    var A = Array2D.ofArrayColumnMajor(lda, k, double2.ToComplex(hAs[batch]));
                    var B = Array2D.ofArrayColumnMajor(ldb, n, double2.ToComplex(hBs[batch]));
                    var C = Array2D.ofArrayColumnMajor(ldc, n, double2.ToComplex(hCs[batch]));

                    for (var j = 1; j <= n; j++)
                    {
                        for (var i = 1; i <= m; i++)
                            C[i - 1, j - 1] = beta.ToComplex() * C[i - 1, j - 1];

                        for (var l = 1; l <= k; l++)
                        {
                            var temp = alpha.ToComplex() * B[l - 1, j - 1];
                            for (var i = 1; i <= m; i++)
                                C[i - 1, j - 1] += temp * A[i - 1, l - 1];
                        }
                    }

                    results.Add(double2.OfComplex(Array2D.toArrayColumnMajor(C)));
                }

                return results.ToArray();
            }
        }
        //[/gemmBatchedCpu]

        //[gemmBatchedGpu]
        public static class gpu
        {
            public static double[][] Dgemm(int m, int n, int k, double alpha, double[][] hAs, int lda, double[][] hBs, int ldb, double beta, double[][] hCs, int ldc)
            {
                var batchCount = hAs.Length;
                var dAs = (from hA in hAs select Worker.Malloc(hA)).ToArray();
                var dBs = (from hB in hBs select Worker.Malloc(hB)).ToArray();
                var dCs = (from hC in hCs select Worker.Malloc(hC)).ToArray();
                try
                {
                    using (var dAPtrs = Worker.Malloc((from dA in dAs select dA.Ptr).ToArray()))
                    using (var dBPtrs = Worker.Malloc((from dB in dBs select dB.Ptr).ToArray()))
                    using (var dCPtrs = Worker.Malloc((from dC in dCs select dC.Ptr).ToArray()))
                    using (var dAlpha = Worker.Malloc(new[] { alpha }))
                    using (var dBeta = Worker.Malloc(new[] { beta }))
                    {
                        Cublas.Dgemm(Transa, Transb, m, n, k, dAlpha.Ptr, dAPtrs.Ptr, lda, dBPtrs.Ptr, ldb, dBeta.Ptr,
                                    dCPtrs.Ptr, ldc, batchCount);
                    }
                    return (from dC in dCs select dC.Gather()).ToArray();
                }
                finally
                {
                    foreach (var dA in dAs) dA.Dispose();
                    foreach (var dB in dBs) dB.Dispose();
                    foreach (var dC in dCs) dC.Dispose();
                }
            }

            public static double2[][] Zgemm(int m, int n, int k, double2 alpha, double2[][] hAs, int lda, double2[][] hBs, int ldb, double2 beta, double2[][] hCs, int ldc)
            {
                var batchCount = hAs.Length;
                var dAs = (from hA in hAs select Worker.Malloc(hA)).ToArray();
                var dBs = (from hB in hBs select Worker.Malloc(hB)).ToArray();
                var dCs = (from hC in hCs select Worker.Malloc(hC)).ToArray();
                try
                {
                    using (var dAPtrs = Worker.Malloc((from dA in dAs select dA.Ptr).ToArray()))
                    using (var dBPtrs = Worker.Malloc((from dB in dBs select dB.Ptr).ToArray()))
                    using (var dCPtrs = Worker.Malloc((from dC in dCs select dC.Ptr).ToArray()))
                    using (var dAlpha = Worker.Malloc(new[] { alpha }))
                    using (var dBeta = Worker.Malloc(new[] { beta }))
                    {
                        Cublas.Zgemm(Transa, Transb, m, n, k, dAlpha.Ptr, dAPtrs.Ptr, lda, dBPtrs.Ptr, ldb, dBeta.Ptr,
                                    dCPtrs.Ptr, ldc, batchCount);
                    }
                    return (from dC in dCs select dC.Gather()).ToArray();
                }
                finally
                {
                    foreach (var dA in dAs) dA.Dispose();
                    foreach (var dB in dBs) dB.Dispose();
                    foreach (var dC in dCs) dC.Dispose();
                }
            }
        }
        //[/gemmBatchedGpu]
    }        
    
    //[gemmBatchedTest]
    public partial class Test
    {
        //private const int M = 5;
        //private const int N = 5;
        //private const int K = 5;

        //private const int Lda = M; // lda >= max(1,m)
        //private const int Ldb = N; // ldb >= max(1,k)
        //private const int Ldc = K; // ldc >= max(1,m)

        private static double[] DGen(int n)
        {
            var R = new double[n];
            for (var i = 0; i < n; ++i)
                R[i] = TestUtil.genRandomDouble(-5.0, 5.0, 0.0);
            return R;
        }

        private static double2[] ZGen(int n)
        {
            var R = new double2[n];
            for (var i = 0; i < n; ++i)
                R[i] = TestUtil.genRandomDouble2(-5.0, 5.0, 0.0);
            return R;
        }

        [Test]
        public static void DgemmBatchedTest()
        {
            Util.FallBack(() =>
            {
                const double alpha = 2.0;
                const double beta = 2.0;
                const int batchCount = 10;

                var hAs = (Enumerable.Range(0, batchCount).Select(_ => DGen(Lda * K))).ToArray();
                var hBs = (Enumerable.Range(0, batchCount).Select(_ => DGen(Ldb * N))).ToArray();
                var hCs = (Enumerable.Range(0, batchCount).Select(_ => DGen(Ldc * N))).ToArray();
                var expecteds = GemmBatched.cpu.Dgemm(M, N, K, alpha, hAs, Lda, hBs, Ldb, beta, hCs, Ldc);
                var outputs = GemmBatched.gpu.Dgemm(M, N, K, alpha, hAs, Lda, hBs, Ldb, beta, hCs, Ldc);

                for (var batch = 0; batch < batchCount; ++batch)
                {
                    var expected = expecteds[batch];
                    var output = outputs[batch];
                    for (var i = 0; i < expected.Length; ++i)
                        Assert.AreEqual(output[i], expected[i], 1e-12);
                }
            });
        }

        [Test]
        public static void ZgemmBatchedTest()
        {
            Util.FallBack(() =>
            {
                var alpha = new double2(2.0, 2.0);
                var beta = new double2(2.0, 2.0);
                const int batchCount = 10;

                var hAs = (Enumerable.Range(0, batchCount).Select(_ => ZGen(Lda * K))).ToArray();
                var hBs = (Enumerable.Range(0, batchCount).Select(_ => ZGen(Ldb * N))).ToArray();
                var hCs = (Enumerable.Range(0, batchCount).Select(_ => ZGen(Ldc * N))).ToArray();
                var expecteds = GemmBatched.cpu.Zgemm(M, N, K, alpha, hAs, Lda, hBs, Ldb, beta, hCs, Ldc);
                var outputs = GemmBatched.gpu.Zgemm(M, N, K, alpha, hAs, Lda, hBs, Ldb, beta, hCs, Ldc);

                for (var batch = 0; batch < batchCount; ++batch)
                {
                    var expected = expecteds[batch];
                    var output = outputs[batch];
                    for (var i = 0; i < expected.Length; ++i)
                    {
                        Assert.AreEqual(output[i].x, expected[i].x, 1e-12);
                        Assert.AreEqual(output[i].y, expected[i].y, 1e-12);
                    }
                }
            });
        }
    }
    //[/gemmBatchedTest]
}
