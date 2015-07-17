using System;
using System.Diagnostics;
using System.Linq;
using Alea.CUDA;
using Alea.CUDA.CULib;
using Alea.CUDA.IL;
using NUnit.Framework;

namespace Tutorial.Cs.examples.curand
{
    
    internal class PiEstimatorModule : ILGPUModule
    {

        public PiEstimatorModule(GPUModuleTarget target)
            : base(target)
        {

        }

        //[cuRANDReduceSum]
        private int ReduceSum(int x)
        {
            var sdata = __shared__.ExternArray<int>();
            var ltid = threadIdx.x;
            sdata[ltid] = x;
            Intrinsic.__syncthreads();

            for (var s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (ltid < s) sdata[ltid] += sdata[ltid + s];
                Intrinsic.__syncthreads();
            }

            return sdata[0];
        }
        //[/cuRANDReduceSum]

        //[cuRANDComputeValue]
        [Kernel]
        public void ComputeValue(deviceptr<double> results, deviceptr<double> points, int numSims)
        {
            var bid = blockIdx.x;
            var tid = blockIdx.x * blockDim.x + threadIdx.x;
            var step = gridDim.x * blockDim.x;

            var pointx = points + tid;
            var pointy = pointx + numSims;

            var pointsInside = 0;

            for (var i = tid; i < numSims; i += step, pointx += step, pointy += step)
            {
                var x = pointx[0];
                var y = pointy[0];
                var l2norm2 = x * x + y * y;
                if (l2norm2 < 1.0) pointsInside++;
            }

            pointsInside = ReduceSum(pointsInside);

            if (threadIdx.x == 0) results[bid] = pointsInside;
        }
        //[/cuRANDComputeValue]

        //[cuRANDPiEstimator]
        public double RunEstimation(int numSims, int threadBlockSize)
        {
            const int blocksPerSm = 10;
            var numSMs = GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT;

            var block = new dim3(threadBlockSize);
            var grid = new dim3((numSims + threadBlockSize - 1) / threadBlockSize);
            while (grid.x > 2 * blocksPerSm * numSims) grid.x >>= 1;

            var n = 2 * numSims;
            using (var dPoints = GPUWorker.Malloc<double>(n))
            using (var dResults = GPUWorker.Malloc<double>(grid.x))
            {
                var curand = new CURAND(GPUWorker, CURANDInterop.curandRngType.CURAND_RNG_QUASI_SOBOL64);
                curand.SetQuasiRandomGeneratorDimensions(2);
                curand.SetGeneratorOrdering(CURANDInterop.curandOrdering.CURAND_ORDERING_QUASI_DEFAULT);
                curand.GenerateUniformDouble(dPoints.Ptr, new IntPtr(n));

                var lp = new LaunchParam(grid, block, block.x * sizeof(uint));
                GPULaunch(ComputeValue, lp, dResults.Ptr, dPoints.Ptr, numSims);
                var value = dResults.Gather().Sum();
                return (value/numSims)*4.0;
            }
        }
        //[/cuRANDPiEstimator]
    }

    [AOTCompile]
    class DefaultPiEstimatorModule : PiEstimatorModule
    {
        public DefaultPiEstimatorModule(GPUModuleTarget target)
            : base(target)
        {
        }

        private static DefaultPiEstimatorModule Instance;

        public static DefaultPiEstimatorModule DefaultInstance
        {
            get { return Instance ?? (Instance = new DefaultPiEstimatorModule(GPUModuleTarget.DefaultWorker)); }
        }
    }

    //[cuRANDEstimatePiTest]
    public class Test
    {
        // Target value
        private const double PI = 3.14159265359;

        [Test]
        public static void EstimatePi()
        {
            const int numSims = 100000;
            const int threadBlockSize = 128;
            var worker = GPUModuleTarget.DefaultWorker;
            Console.WriteLine("Estimating PI on GPU {0}\n", worker.GetWorker().Device.Name);
            var watch = Stopwatch.StartNew();
            var result = 
                DefaultPiEstimatorModule.DefaultInstance
                    .RunEstimation(numSims, threadBlockSize);
            watch.Stop();
            var elapsedTime = watch.Elapsed.TotalMilliseconds;
            const double tol = 0.01;
            var abserror = Math.Abs(result - Math.PI);
            var relerror = abserror / Math.PI;
            Console.WriteLine("Preceision:          {0}", "double");
            Console.WriteLine("Number of sims:      {0}", numSims);
            Console.WriteLine("Tolerance:           {0}", tol);
            Console.WriteLine("GPU result:          {0}", result);
            Console.WriteLine("Expected:            {0}", PI);
            Console.WriteLine("Absolute error:      {0}", abserror);
            Console.WriteLine("Relative error:      {0}", relerror);

            if (relerror > tol)
                Console.WriteLine("computed result ({0}) does not match expected result ({1}).", result, PI);

            Console.WriteLine("MonteCarloEstimatePiQ, Performance = {0} sims/s, Time = {1}(ms), NumDevsUsed = {2}, Blocksize = {3}",
                numSims / (elapsedTime/1000.0), elapsedTime, 1, threadBlockSize);

            Assert.AreEqual(PI, result, tol);
        }
        //[/cuRANDEstimatePiTest]
    }
}
