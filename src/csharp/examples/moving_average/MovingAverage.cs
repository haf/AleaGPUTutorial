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
    internal class WindowDifferenceModule<T> : ILGPUModule
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
    
    //[MovingAverage]
    internal class MovingAverageModule<T> : ILGPUModule
    {
        private readonly Func<T, T, T> _add;
        private readonly Func<T, T, T> _mul;
        private readonly Func<T, T, T> _div; 
        private readonly T _1G;
        private readonly WindowDifferenceModule<T> _windowDifference;
        private readonly Action<T[], deviceptr<T>> _scan; 
        
        
        public MovingAverageModule(GPUModuleTarget target, Func<T,T,T> add, Func<T,T,T> sub, Func<T,T,T> mul, Func<T,T,T> div, T genericOne)
            : base(target)
        {
            _1G = genericOne;
            _add = add;
            _mul = mul;
            _div = div;
            _windowDifference = new WindowDifferenceModule<T>(target, sub, div);
            _scan = ScanApi.InclusiveScan(target, add);
        }

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

        private LaunchParam LaunchParams(int windowSize, int n)
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
    
        public void MovingAverageDirect(int windowSize, int n, deviceptr<T> values, deviceptr<T> results)
        {
            GPULaunch(Kernel, LaunchParams(windowSize, n), windowSize, n, values, results);
        }

        public T[] MovingAverageDirect(int windowSize, T[] values)
        {
            var n = values.Length;
            using(var dValues = GPUWorker.Malloc(values))
            using (var dResults = GPUWorker.Malloc<T>(n))
            {
                MovingAverageDirect(windowSize, n, dValues.Ptr, dResults.Ptr);
                return dResults.Gather();
            }
        }
        
        public T[] MovingAverageScan(int windowSize, T[] values)
        {
            var n = values.Length;
            using (var dSums = GPUWorker.Malloc<T>(n))
            using (var dResults = GPUWorker.Malloc<T>(n - windowSize))
            {
                _scan(values, dSums.Ptr);
                _windowDifference.Apply(n, windowSize, dSums.Ptr, dResults.Ptr);
                return dResults.Gather();
            }
        }

        public static MovingAverageModule<double> Create(GPUModuleTarget target)
        {
            return new MovingAverageModule<double>(target, (x,y) => x+y, (x,y) => x-y, (x,y) => x*y, (x,y) => x/y, 1.0);
        } 
    }

    //[AOTCompile]
    //class MovingAverageModuleF64 : MovingAverageModule<double>
    //{
    //    public MovingAverageModuleF64(GPUModuleTarget target)
    //        : base(target, (x, y) => x + y, (x, y) => x - y, (x, y) => x * y, (x, y) => x / y, 1.0) { }

    //    private static MovingAverageModuleF64 _instance;

    //    public static MovingAverageModuleF64 DefaultInstance
    //    {
    //        get { return _instance ?? (_instance = new MovingAverageModuleF64(GPUModuleTarget.DefaultWorker)); }
    //    }
    //}
    //[/MovingAverage]

    public static class MovingAverageCPU
    {
        //[MovingAvgArray]
        public static T[] MovingAverageArray<T>(int windowSize, T[] series, Func<int, T> conv, Func<T,T,T> add, Func<T,T,T> sub, Func<T,T,T> div)
        {
            var sums = ScanApi.CpuScan(add, series, true);
            var ma = new T[sums.Length - windowSize];
            for (var i = windowSize; i < sums.Length; i++)
                ma[i - windowSize] = div(sub(sums[i], sums[i - windowSize]), conv(windowSize));
            return ma;
        }
        //[/MovingAvgArray]
    }


    public static class Test
    {
        private static readonly Random Rng = new Random();

        public static void AssertArrayEqual<T>(T[] h, T[] d, dynamic tol)
        {
            for (var i = 0; i < d.Length; i++)
                Assert.AreEqual(h[i], d[i], tol);
        }

        //[MovingAvgTestFunc]
        public static void TestFunc<T>(T zero, int[] sizes, Func<int, T[], T[]> movingAverageGpu, Func<int,T[]> gen, Func<int,T> conv, Func<T,T,T> add, Func<T,T,T> sub, Func<T,T,T> div, Action<T[],T[]> assert, bool direct)
        {
            var windowSizes = new[] {2, 3, 10};
            
            foreach (var n in sizes)
            {
                foreach (var windowSize in windowSizes)
                {
                    var v = gen(n); //.Concat(new []{default(T)}).ToArray();
                    var d = movingAverageGpu(windowSize, v.Concat(new []{default(T)}).ToArray());
                    d = direct ? d.Skip(windowSize - 1).ToArray() : d;
                    var h = MovingAverageCPU.MovingAverageArray(windowSize, v, conv, add, sub, div);
                    h = direct ? h : h.Take(h.Length - 1).ToArray();

                    Console.WriteLine("window {0}", windowSize);
                    Console.WriteLine("gpu size: {0}", d.Length);
                    Console.WriteLine("cpu size: {0}", h.Length);
                    //for (var i = 0; i < h.Length; i++)
                    //    Console.WriteLine("cpu, gpu : {0}, {1}", h[i], d[i]);
                    //Console.WriteLine("cpu : {0}", h);

                    assert(h, d);
                }   
            }
        }

        public static void TestFunc(int[] sizes, Func<int, double[], double[]> movingAverageGpu, bool direct)
        {
            TestFunc(0.0, sizes, 
                movingAverageGpu, 
                GenDoubles, x => (double) x, 
                (x,y) => x+y, (x,y) => x-y, (x,y) => x/y, 
                (h,d) => AssertArrayEqual(h,d,1e-11), 
                direct);
        }
        //[/MovingAvgTestFunc]

        public static double[] GenDoubles(int n)
        {
            return Enumerable.Range(0, n).Select(_ => TestUtil.genRandomDouble(-5.0, 5.0, 0.0)).ToArray();
        }
        
        //[MovingAvgTest]
        [Test]
        public static void MovingAverageTest()
        {
            var sizes = new[] {12, 15, 20, 32, 64, 128, 1024, 1200, 4096, 5000, 8191, 8192, 8193, 9000, 10000, 2097152, 8388608 };
            //var mamod = MovingAverageModuleF64.DefaultInstance;
            //TestFunc(sizes, mamod.MovingAverageScan, false);
            TestFunc(
                sizes,
                MovingAverageModule<double>.Create(GPUModuleTarget.DefaultWorker)
                    .MovingAverageScan,
                false);
        }
        //[/MovingAvgTest]

        //[MovingAvgDirectTest]
        [Test]
        public static void MovingAverageDirectTest()
        {
            var sizes = new[] {12, 15, 20, 32, 64, 128, 1024, 1200, 4096, 5000, 8191, 8192, 8193, 9000, 10000, 2097152, 8388608 };
            //var mamod = MovingAverageModuleF64.DefaultInstance;
            //TestFunc(sizes, mamod.MovingAverageDirect, true);
            TestFunc(
                sizes,
                MovingAverageModule<double>.Create(GPUModuleTarget.DefaultWorker)
                    .MovingAverageDirect,
                true);
        }
        //[/MovingAvgDirectTest]
    }
}
