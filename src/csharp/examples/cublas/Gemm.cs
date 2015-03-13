using System;
using System.Numerics;
using Alea.CUDA;
using Alea.CUDA.CULib;
using Alea.CUDA.Utilities;
using NUnit.Framework;
using cublasOperation_t = Alea.CUDA.CULib.CUBLASInterop.cublasOperation_t;

namespace Tutorial.Cs.examples.cublas
{
    class Gemm
    {
        private static readonly CUBLAS Cublas = CUBLAS.Default;
        private static readonly Worker Worker = Cublas.Worker;

        private const cublasOperation_t Transa = cublasOperation_t.CUBLAS_OP_N;
        private const cublasOperation_t Transb = cublasOperation_t.CUBLAS_OP_N;
        
        //[gemmCpu]
        public static class cpu
        {
            public static double[] Dgemm(int m, int n, int k, double alpha, double[] _A, int lda, double[] _B, int ldb, double beta, double[] _C, int ldc)
            {
                var A = Array2D.ofArrayColumnMajor(lda, k, _A);
                var B = Array2D.ofArrayColumnMajor(ldb, n, _B);
                var C = Array2D.ofArrayColumnMajor(ldc, n, _C);

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
                return Array2D.toArrayColumnMajor(C);
            }
            
            public static double2[] Zgemm(int m, int n, int k, double2 alpha, double2[] _A, int lda, double2[] _B, int ldb, double2 beta, double2[] _C, int ldc)
            {
                var A = Array2D.ofArrayColumnMajor(lda, k, double2.ToComplex(_A));
                var B = Array2D.ofArrayColumnMajor(ldb, n, double2.ToComplex(_B));
                var C = Array2D.ofArrayColumnMajor(ldc, n, double2.ToComplex(_C));
                var R = Array2D.ofArrayColumnMajor(ldc, n, double2.ToComplex(_C));
                for (var j = 1; j <= n; ++j)
                {
                    for (var i = 1; i <= m; ++i)
                        R[i - 1, j - 1] = beta.ToComplex() * C[i - 1, j - 1];
                    for (var l = 1; l <= k; ++l)
                    {
                        var temp = alpha.ToComplex() * B[l - 1, j - 1];
                        for (var i = 1; i <= m; ++i)
                            R[i - 1, j - 1] += temp * A[i - 1, j - 1];
                    }

                }
                return double2.OfComplex(Array2D.toArrayColumnMajor(R));
            }
        }
        //[/gemmCpu]

        //[gemmGpu]
        public static class gpu
        {
            public static double[] Dgemm(int m, int n, int k, double alpha, double[] A, int lda, double[] B, int ldb, double beta, double[] C, int ldc)
            {
                using (var dA = Worker.Malloc(A))
                using (var dB = Worker.Malloc(B))
                using (var dC = Worker.Malloc(C))
                using (var dAlpha = Worker.Malloc(new[] { alpha }))
                using (var dBeta = Worker.Malloc(new[] { beta }))
                {
                    Cublas.Dgemm(Transa, Transb, m, n, k, dAlpha.Ptr, dA.Ptr, lda, dB.Ptr, ldb, dBeta.Ptr, dC.Ptr, ldc);
                    return dC.Gather();
                }
            }

            public static double2[] Zgemm(int m, int n, int k, double2 alpha, double2[] A, int lda, double2[] B, int ldb, double2 beta, double2[] C, int ldc)
            {
                using (var dA = Worker.Malloc(A))
                using (var dB = Worker.Malloc(B))
                using (var dC = Worker.Malloc(C))
                using (var dAlpha = Worker.Malloc(new[] { alpha }))
                using (var dBeta = Worker.Malloc(new[] { beta }))
                {
                    Cublas.Zgemm(Transa, Transb, m, n, k, dAlpha.Ptr, dA.Ptr, lda, dB.Ptr, ldb, dBeta.Ptr, dC.Ptr, ldc);
                    return dC.Gather();
                }
            }
        }
        //[/gemmGpu]
    }
    
    //[gemmTest]
    public partial class Test
    {
        private const int M = 5;
        private const int N = 5;
        private const int K = 5;

        private const int Lda = M; // lda >= max(1,m)
        private const int Ldb = N; // ldb >= max(1,k)
        private const int Ldc = K; // ldc >= max(1,m)

        [Test]
        public static void DgemmTest()
        {
            var rng = new Random();

            const double alpha = 2.0;
            const double beta = 1.0;

            var A = new double[Lda*K];
            var B = new double[Ldb*N];
            var C = new double[Ldc*N];

            for (var i = 0; i < Lda*K; ++i)
                A[i] = rng.NextDouble();

            for (var i = 0; i < Lda*N; ++i)
            {
                B[i] = rng.NextDouble();
                C[i] = rng.NextDouble();
            }
            
            var outputs = Gemm.gpu.Dgemm(M, N, K, alpha, A, Lda, B, Ldb, beta, C, Ldc);
            var expected = Gemm.cpu.Dgemm(M, N, K, alpha, A, Lda, B, Ldb, beta, C, Ldc);

            for (var i = 0; i < outputs.Length; ++i)
                Assert.AreEqual(outputs[i], expected[i], 1e-12);
        }

        [Test]
        public static void ZgemmTest()
        {
            var alpha = new double2(2.0, 0.5);
            var beta = new double2(1.0, 0.5);

            var A = new double2[Lda*K];
            var B = new double2[Ldb*N];
            var C = new double2[Ldc*N];

            var outputs = Gemm.gpu.Zgemm(M, N, K, alpha, A, Lda, B, Ldb, beta, C, Ldc);
            var expected = Gemm.cpu.Zgemm(M, N, K, alpha, A, Lda, B, Ldb, beta, C, Ldc);

            for (var i = 0; i < outputs.Length; ++i)
            {
                Assert.AreEqual(outputs[i].x, expected[i].x, 1e-12);
                Assert.AreEqual(outputs[i].y, expected[i].y, 1e-12);
            }
        }
    }
    //[/gemmTest]
}
