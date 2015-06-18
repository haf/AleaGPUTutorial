using Alea.CUDA;
using Alea.CUDA.CULib;
using NUnit.Framework;

namespace Tutorial.Cs.examples.cublas
{
    class Axpy
    {
        private static readonly CUBLAS Cublas = CUBLAS.Default;
        private static readonly Worker Worker = Cublas.Worker;

        //[axpyCpu]
        public static class cpu
        {
            public static double[] Daxpy(int n, double alpha, double[] x, int incx, double[] y, int incy)
            {
                var r = new double[n];
                for (var i = 0; i < n; ++i)
                {
                    r[i] = alpha * x[i] + y[i];
                }
                return r;
            }

            public static double2[] Zaxpy(int n, double2 alpha, double2[] _x, int incx, double2[] _y, int incy)
            {
                var x = double2.ToComplex(_x);
                var y = double2.ToComplex(_y);
                var r = double2.ToComplex(new double2[n]);
                for (var i = 0; i < n; ++i)
                {
                    r[i] = alpha.ToComplex() * x[i] + y[i];
                }
                return double2.OfComplex(r);
            }
        }
        //[/axpyCpu]

        //[axpyGpu]
        public static class gpu
        {
            public static double[] Daxpy(int n, double alpha, double[] x, int incx, double[] y, int incy)
            {
                using (var dalpha = Worker.Malloc(new[] { alpha }))
                using (var dx = Worker.Malloc(x))
                using (var dy = Worker.Malloc(y))
                {
                    Cublas.Daxpy(n, dalpha.Ptr, dx.Ptr, incx, dy.Ptr, incy);
                    return dy.Gather();
                }
            }

            public static double2[] Zaxpy(int n, double2 alpha, double2[] x, int incx, double2[] y, int incy)
            {
                using (var dalpha = Worker.Malloc(new[] { alpha }))
                using (var dx = Worker.Malloc(x))
                using (var dy = Worker.Malloc(y))
                {
                    Cublas.Zaxpy(n, dalpha.Ptr, dx.Ptr, incx, dy.Ptr, incy);
                    return dy.Gather();
                }
            }
        }
        //[/axpyGpu]
    }   
    
    //[axpyTest]
    public partial class Test
    {
        private const int Incx = 1;
        private const int Incy = 1;

        [Test]
        public static void DaxpyTest()
        {
            Util.FallBack(() =>
            {
                const int N = 5;
                const double alpha = 2.0;

                var x = new double[N];
                var y = new double[N];

                for (var i = 0; i < N; ++i)
                {
                    x[i] = 2.0;
                    y[i] = 1.0;
                }

                var outputs = Axpy.gpu.Daxpy(N, alpha, x, Incx, y, Incy);
                var expected = Axpy.cpu.Daxpy(N, alpha, x, Incx, y, Incy);

                for (var i = 0; i < N; ++i)
                    Assert.AreEqual(outputs[i], expected[i], 1e-12);
            });
        }

        [Test]
        public static void ZaxpyTest()
        {
            Util.FallBack(() =>
            {
                const int N = 5;
                var alpha = new double2(2.0, 2.0);

                var x = new double2[N];
                var y = new double2[N];

                var outputs = Axpy.gpu.Zaxpy(N, alpha, x, Incx, y, Incy);
                var expected = Axpy.cpu.Zaxpy(N, alpha, x, Incx, y, Incy);

                for (var i = 0; i < N; ++i)
                {
                    Assert.AreEqual(outputs[i].x, expected[i].x, 1e-12);
                    Assert.AreEqual(outputs[i].y, expected[i].y, 1e-12);
                }
            });
        }
    }
    //[/axpyTest]
}
