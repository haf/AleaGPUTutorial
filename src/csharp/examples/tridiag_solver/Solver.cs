using System;
using System.Linq;
using Alea.CUDA;
using Alea.CUDA.IL;
using NUnit.Framework;

namespace Tutorial.Cs.examples.tridiagSolver
{
    //[triDiagSolverModule]
    [AOTCompile]
    internal class TriDiagSolverModule : ILGPUModule
    {
        public TriDiagSolverModule(GPUModuleTarget target)
            : base(target)
        {
        }

        // core solver function
        //     n  the dimension of the tridiagonal system, must fit into one block
        //     l  lower diagonal
        //     d  diagonal
        //     u  upper diagonal
        //     h  right hand side and solution at exit
        public static void Solve(int n, deviceptr<double> l, deviceptr<double> d, deviceptr<double> u, deviceptr<double> h)
        {
            var rank = threadIdx.x;

            var ltemp = 0.0;
            var utemp = 0.0;
            var htemp = 0.0;

            var span = 1;
            while (span < n)
            {
                if (rank < n)
                {
                    ltemp = (rank - span >= 0) ?
                        (d[rank - span] != 0.0) ? -l[rank] / d[rank - span] : 0.0
                        :
                        0.0;


                    utemp = (rank + span < n) ?
                        (d[rank + span] != 0.0) ? -u[rank] / d[rank + span] : 0.0
                        :
                        0.0;

                    htemp = h[rank];
                }

                Intrinsic.__syncthreads();

                if (rank < n)
                {
                    if (rank - span >= 0)
                    {
                        d[rank] = d[rank] + ltemp * u[rank - span];
                        htemp = htemp + ltemp * h[rank - span];
                        ltemp = ltemp * l[rank - span];
                    }
                    if (rank + span < n)
                    {
                        d[rank] = d[rank] + utemp * l[rank + span];
                        htemp = htemp + utemp * h[rank + span];
                        utemp = utemp * u[rank + span];
                    }
                }

                Intrinsic.__syncthreads();

                if (rank < n)
                {
                    l[rank] = ltemp;
                    u[rank] = utemp;
                    h[rank] = htemp;
                }

                Intrinsic.__syncthreads();

                span *= 2;
            }

            if (rank < n) h[rank] = h[rank] / d[rank];
        }

        // multiple systems solved in different thread blocks
        [Kernel]
        public void Kernel(int n, deviceptr<double> dl, deviceptr<double> dd, deviceptr<double> du, deviceptr<double> db, deviceptr<double> dx)
        {
            var tid = threadIdx.x;
            var gid = blockIdx.x * n + tid;

            var shared = Intrinsic.__array_to_ptr(__shared__.ExternArray<double>());
            var l = shared;
            var d = l + n;
            var u = d + n;
            var b = u + n;

            l[tid] = dl[gid];
            d[tid] = dd[gid];
            u[tid] = du[gid];
            b[tid] = db[gid];

            Intrinsic.__syncthreads();

            Solve(n, l, d, u, b);

            dx[gid] = b[tid];
        }

        public void Apply(int numSystems, int n, deviceptr<double> dl, deviceptr<double> dd, deviceptr<double> du,
            deviceptr<double> db, deviceptr<double> dx)
        {
            var sharedSize = 9*n*sizeof (double);
            var lp = new LaunchParam(numSystems, n, sharedSize);
            this.GPULaunch(this.Kernel, lp, n, dl, dd, du, db, dx);
        }

        public double[] Apply(int numSystems, double[] l, double[] d, double[] u, double[] b)
        {
            var n = d.Length/numSystems;

            // check resource availability
            var sharedSize = 9*n*sizeof (double);
            var maxBlockDimX = this.GPUWorker.Device.Attributes.MAX_BLOCK_DIM_X;
            var maxGridDimX = this.GPUWorker.Device.Attributes.MAX_GRID_DIM_X;
            var maxThreads = this.GPUWorker.Device.Attributes.MAX_THREADS_PER_BLOCK;
            var maxSharedSize = this.GPUWorker.Device.Attributes.MAX_SHARED_MEMORY_PER_BLOCK;
            if (numSystems > maxGridDimX) throw new Exception(String.Format("numSystems({0}) > maxGridDimX({1})", numSystems, maxGridDimX));
            if (n > maxBlockDimX) throw new Exception(String.Format("n({0}) > maxBlockDimX({1})", n, maxBlockDimX));
            if (n > maxThreads) throw new Exception(String.Format("n({0}) > maxThreads({1})", n, maxThreads));
            if (sharedSize > maxSharedSize) throw new Exception(String.Format("sharedSize({0}) > maxSharedSize({1})", sharedSize, maxSharedSize));  

            using (var dl = this.GPUWorker.Malloc(l))
            using (var dd = this.GPUWorker.Malloc(d))
            using (var du = this.GPUWorker.Malloc(u))
            using (var db = this.GPUWorker.Malloc(b))
            using (var dx = this.GPUWorker.Malloc<double>(d.Length))
            {
                this.Apply(numSystems, n, dl.Ptr, dd.Ptr, du.Ptr, db.Ptr, dx.Ptr);
                return dx.Gather();
            }
        }
    }
    //[/triDiagSolverModule]

    //[triDiagSolverTest]
    public class Test
    {
        public static Random Rng = new Random(42);

        private static double random(double a, double b)
        {
            return Rng.NextDouble() * (b - a) + a;
        }

        private static double[] multiply(double[] l, double[] d, double[] u, double[] x)
        {
            var n = d.Length;
            var b = new double[n];
            b[0] = d[0] * x[0] + u[0] * x[1];
            for (var i = 1; i <= n - 2; i++)
                b[i] = l[i] * x[i - 1] + d[i] * x[i] + u[i] * x[i + 1];
            b[n - 1] = l[n - 1] * x[n - 2] + d[n - 1] * x[n - 1];
            return b;
        }
        
        private static Tuple<double[], double[], double[], double[], double[]> diagonallyDominantSystem(int n)
        {
            var l = new double[n];
            var u = new double[n];
            var d = new double[n];
            var expectedSolution = new double[n];
            for (var i = 0; i < n; i++)
            {
                l[i] = random(-100.0, 100.0);
                u[i] = random(-100.0, 100.0);
                d[i] = random(0.0, 10.0) + Math.Abs(l[i]) + Math.Abs(u[i]);
                expectedSolution[i] = random(-20.0, 20.0);
            }
            var b = multiply(l, d, u, expectedSolution);
            return new Tuple<double[], double[], double[], double[], double[]>(l, d, u, b, expectedSolution);
        }
        
        [Test]
        public static void TriDiagSolverTest()
        {
            const int numSystems = 512;

            var numbers =
                Enumerable
                .Repeat(numSystems, numSystems)
                .Select(diagonallyDominantSystem)
                .ToArray();

            var l = numbers.SelectMany(s => s.Item1).ToArray();
            var d = numbers.SelectMany(s => s.Item2).ToArray();
            var u = numbers.SelectMany(s => s.Item3).ToArray();
            var b = numbers.SelectMany(s => s.Item4).ToArray();
            var expectedSolution = numbers.SelectMany(s => s.Item5).ToArray();

            var triDiagSolver = new TriDiagSolverModule(GPUModuleTarget.DefaultWorker);

            var x = triDiagSolver.Apply(numSystems, l, d, u, b);
            var errs = new double[x.Length];
            for (var i = 0; i < x.Length; ++i)
                errs[i] = Math.Abs(expectedSolution[i] - x[i]);
            var err = errs.Max();
            Console.WriteLine("error = {0}", err);
            Assert.LessOrEqual(err, 1e-8);
        }
    }
    //[/triDiagSolverTest]
}