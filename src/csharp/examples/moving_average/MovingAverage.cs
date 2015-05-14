using System;
using System.Linq;
using Alea.CUDA;
using Alea.CUDA.IL;
using Alea.CUDA.Utilities;
using NUnit.Framework;
using Tutorial.Cs.examples.generic_scan;

namespace Tutorial.Cs.examples.moving_average
{
    //[MovingAvgWinDiff]
    public class WindowDifferenceModule<T> : ILGPUModule
    {
        private readonly Func<T, T, T> _sub;
        private readonly Func<T, T, T> _div;
        
        public WindowDifferenceModule(GPUModuleTarget target, Func<T,T,T> sub, Func<T,T,T> div)
            : base(target)
        {
            _sub = sub;
            _div = div;
        }

        [Kernel]
        public void Kernel(int n, int windowSize, deviceptr<T> x, deviceptr<T> y)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            var i = start + windowSize;
            T normalizer = LibDevice2.__gconv<int,T>(windowSize);
            while (i < n)
            {
                y[i - windowSize] = _div(_sub(x[i], x[i - windowSize]), normalizer);
                i += stride;
            }
        }

        public LaunchParam LaunchParams(int n)
        {
            var numSm = GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT;
            const int blockSize = 256;
            var gridSize = Math.Min(numSm, Common.divup(n, blockSize));
            return new LaunchParam(gridSize, blockSize);
        }

        public void Apply(int n, int windowSize, deviceptr<T> input, deviceptr<T> output)
        {
            var lp = LaunchParams(n);
            GPULaunch(Kernel, lp, n, windowSize, input, output);
        }
    }
    //[/MovingAvgWinDiff]

    public class MovingAverageModule<T> : ILGPUModule
    {
        private readonly Func<T, T, T> _add;
        private readonly Func<T, T, T> _mul;
        private readonly Func<T, T, T> _div; 
        private readonly T _1G;
        
        public MovingAverageModule(GPUModuleTarget target, Func<T,T,T> add, Func<T,T,T> mul, Func<T,T,T> div, T genericOne)
            : base(target)
        {
            _1G = genericOne;
            _add = add;
            _mul = mul;
            _div = div;
        }

        //[MovingAvgKernel]
        [Kernel]
        public void Kernel(int windowSize, int n, deviceptr<T> values, deviceptr<T> results)
        {
            var blockSize = blockDim.x;
            var idx = threadIdx.x;
            var iGlobal = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
            var shared = __shared__.ExternArray<T>();

            var idxShared = idx + windowSize - 1;
            var idxGlobal = iGlobal;
            while (idxShared >= 0)
            {
                var value = ((idxGlobal >= 0) && (idxGlobal < n)) ? values[idxGlobal] : default(T);
                shared[idxShared] = value;
                idxShared -= blockSize;
                idxGlobal -= blockSize;
            }

            Intrinsic.__syncthreads();

            if (iGlobal < n)
            {
                var temp = default(T);
                var k = 0;
                while (k <= Math.Min(windowSize - 1, iGlobal))
                {
                    // This looks messy because of the add,mul,div operators needed for generics.
                    // For reference, here is the original line of F# code - it's easier to read:
                    // temp <- (temp * __gconv k + shared.[idx - k + windowSize - 1]) / (__gconv k + 1G)
                    var gk = LibDevice2.__gconv<int, T>(k);
                    var s = shared[idx - k + windowSize - 1];
                    var t = _mul(temp, gk);
                    var u = _add(t, s);
                    var v = _add(gk, _1G);
                    temp = _div(u, v);
                    k++;
                }
                results[iGlobal] = temp;
            }
        }
        //[/MovingAvgKernel]

        public LaunchParam LaunchParams(int n, int windowSize)
        {
            const int maxBlockSize = 256;
            var blockSize = Math.Min(n, maxBlockSize);
            var gridSizeX = (n - 1) / blockSize + 1;
            var sharedMem = (blockSize + windowSize - 1) * Intrinsic.__sizeof<T>();
            
            if (gridSizeX <= 65535)
                return new LaunchParam(gridSizeX, blockSize, sharedMem);
            
            var gridSizeY = 1 + (n - 1) / (blockSize * 65535);
            gridSizeX = 1 + (n - 1) / (blockSize * gridSizeY);
            return new LaunchParam(new dim3(gridSizeX, gridSizeY), new dim3(blockSize), sharedMem);
        }

        public void Apply(int n, int windowSize, deviceptr<T> values, deviceptr<T> results)
        {
            var lp = LaunchParams(n, windowSize);
            GPULaunch(Kernel, lp, windowSize, n, values, results);
        }
    }

    //[MovingAvgScan]
    public class MovingAverageScan<T> : ILGPUModule
    {
        private readonly WindowDifferenceModule<T> _windowDifference;
        private readonly Func<T, T, T> _add; 

        public MovingAverageScan(GPUModuleTarget target, Func<T, T, T> add, Func<T,T,T> sub, Func<T, T, T> div, T _1G) : base(target)
        {
            _add = add;
            _windowDifference = new WindowDifferenceModule<T>(target, sub, div);
        }

        public T[] Apply(int windowSize, T[] values)
        {
            var n = values.Length;
            using (var dSums = GPUWorker.Malloc(ScanApi.Scan(_add, values, false)))
            using (var dResults = GPUWorker.Malloc<T>(n - windowSize))
            {
                _windowDifference.Apply(n, windowSize, dSums.Ptr, dResults.Ptr);
                return dResults.Gather();
            }
        }
    }
    //[/MovingAvgScan]

    public static class MovingAverage_CPU
    {
        //[MovingAvgArray]
        public static T[] MovingAverageArray<T>(int windowSize, T[] series, Func<int, T> conv, Func<T,T,T> add, Func<T,T,T> sub, Func<T,T,T> div)
        {
            var sums = ScanApi.CpuScan(add, series, false);
            var ma = new T[sums.Length - windowSize];
            for (var i = windowSize; i < sums.Length; i++)
                ma[i - windowSize] = div(sub(sums[i], sums[i - windowSize]), conv(windowSize));
            return ma;
        }
        //[/MovingAvgArray]
    }

    public static class Test
    {
        private static readonly Random _rng = new Random();

        //[MovingAvgTestFunc]
        public static void TestFunc<T>(T zero, int[] sizes, Func<int,T[]> gen, Func<int,T> conv, Func<T,T,T> add, Func<T,T,T> sub, Func<T,T,T> div, MovingAverageScan<T> movingAverageScan, Action<T[],T[]> assertArrayEqual, bool direct)
        {
            var windowSizes = new[] {2, 3, 10};
            Action<int, int> compare =
                (n, windowSize) =>
                {
                    var v = gen(n);
                    var d = movingAverageScan.Apply(windowSize, v);
                    var h = MovingAverage_CPU.MovingAverageArray(windowSize, v, conv, add, sub, div);

                    Console.WriteLine("window {0}", windowSize);
                    Console.WriteLine("gpu size: {0}", d.Length);
                    Console.WriteLine("cpu size: {0}", h.Length);
                    Console.WriteLine("gpu : {0}", d);
                    Console.WriteLine("cpu : {0}", h);

                    assertArrayEqual(h, d);
                };
            foreach (var n in sizes)
            {
                foreach (var ws in windowSizes)
                {
                    compare(n, ws);
                }   
            }
        }
        //[/MovingAvgTestFunc]

        //[MovingAvgTest]
        [Test]
        public static void MovingAverageTest()
        {
            var sizes = new[] {12};
            Func<int, double[]> gen = n =>
            {
                return Enumerable.Range(0,n).Select(_ => _rng.NextDouble()).ToArray();
            };
            var mascan = new MovingAverageScan<double>(GPUModuleTarget.DefaultWorker, (x, y) => x + y, (x, y) => x - y,
                (x, y) => x/y, 1.0);
            Action<double[], double[]> assert = (h, d) =>
            {
                for (var i = 0; i < d.Length; i++)
                    Assert.AreEqual(h[i], d[i], 1e-11);
            };

            TestFunc(0.0, sizes, gen, x => (double) x, (x,y) => x+y, (x,y) => x-y, (x,y) => x/y, mascan, assert, false);
        }
        //[/MovingAvgTest]

        //[MovingAvgDirectTest]
        [Test]
        public static void MovingAverageDirectTest()
        {
            
        }
        //[/MovingAvgDirectTest]
    }
}
