using System;
using System.Linq;
using Alea.CUDA;
using Alea.CUDA.IL;
using Alea.CUDA.Utilities;
using Tutorial.Cs.examples.generic_scan;

namespace Tutorial.Cs.examples.moving_average
{
    public class WindowDifference : ILGPUModule
    {
        public WindowDifference(GPUModuleTarget target)
            : base(target)
        {

        }

        [Kernel]
        public static void Kernel<T>(int n, int windowSize, deviceptr<dynamic> x, deviceptr<dynamic> y)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            var i = start + windowSize;
            var normalizer = LibDevice2.__gconv<int, T>(windowSize);
            while (i < n)
            {
                y[i - windowSize] = (x[i] - x[i - windowSize]) / normalizer;
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

        public void Run<T>(int n, int windowSize, deviceptr<dynamic> input, deviceptr<dynamic> output)
        {
            var lp = LaunchParams(n);
            GPULaunch(Kernel<T>, lp, n, windowSize, input, output);
        }
    }

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

        [Kernel]
        public void MovingAvg(int windowSize, int n, deviceptr<T> values, deviceptr<T> results)
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
            GPULaunch(MovingAvg, lp, windowSize, n, values, results);
        }
    }

    public class MovingAverageScan<T> : MovingAverageModule<T>
    {
        private ScanModule<T> scanner; 

        public MovingAverageScan(GPUModuleTarget target, Func<T, T, T> add, Func<T, T, T> mul, Func<T, T, T> div, T _1G) : base(target, add, mul, div, _1G)
        {
            //scanner= new ScanModule<T>(target, () => default(T), (x,y) => x + y, x => x);
        }
    }

    public class MovingAverage_CPU
    {
        public double[] MovingAverageArray(int windowSize, double[] series)
        {
            var sums = series.Select((a, i) => series.Take(i + 1).Sum()).ToArray();
            var ma = new double[sums.Length - windowSize];
            for (var i = windowSize; i < sums.Length; i++)
                ma[i - windowSize] = (sums[i] - sums[i - windowSize]) / (LibDevice2.__gconv<int, double>(windowSize));
            return ma;
        }
    }
}
