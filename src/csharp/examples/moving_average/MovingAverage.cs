using System;
using System.Linq;
using Alea.CUDA;
using Alea.CUDA.IL;
using Alea.CUDA.Utilities;
using Microsoft.FSharp.Core;

namespace Tutorial.Cs.examples.moving_average
{
    public class WindowDifference : ILGPUModule
    {
        public WindowDifference(GPUModuleTarget target, FSharpFunc<object, FSharpFunc<CompileOptions, Template<Entry<GPUModuleEntities>>>> cudafy)
            : base(target, cudafy)
        {

        }

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
            var numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT;
            var blockSize = 256;
            var gridSize = Math.Min(numSm, Common.divup(n, blockSize));
            return new LaunchParam(gridSize, blockSize);
        }

        public void Run<T>(int n, int windowSize, deviceptr<dynamic> input, deviceptr<dynamic> output)
        {
            var lp = LaunchParams(n);
            GPULaunch(Kernel<T>, lp, n, windowSize, input, output);
        }
    }

    public class MovingAverage : ILGPUModule
    {
        public MovingAverage(GPUModuleTarget target, FSharpFunc<object, FSharpFunc<CompileOptions, Template<Entry<GPUModuleEntities>>>> cudafy)
            : base(target, cudafy)
        {
        }

        public MovingAverage(GPUModuleTarget target)
            : base(target)
        {
        }

        [Kernel]
        public void MovingAvg<T>(int windowSize, int n, deviceptr<T> values, deviceptr<T> results)
        {
            var blockSize = blockDim.x;
            var idx = threadIdx.x;
            var iGlobal = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
            var shared = __shared__.ExternArray<T>();

            var idxShared = idx + windowSize - 1;
            var idxGlobal = iGlobal;
            while (idxShared >= 0)
            {
                dynamic value;
                if (idxGlobal >= 0 && idxGlobal < n)
                    value = values[idxGlobal];
                else
                    value = 0;
                shared[idxShared] = value;
                idxShared -= blockSize;
                idxGlobal -= blockSize;
            }

            Intrinsic.__syncthreads();

            if (iGlobal < n)
            {
                dynamic temp = 0;
                dynamic dyn1 = 1;
                var k = 0;
                while (k <= Math.Min(windowSize - 1, iGlobal))
                {
                    temp = (temp * LibDevice2.__gconv<int, T>(k) + shared[idx - k + windowSize - 1]) /
                           (LibDevice2.__gconv<int, T>(k) + dyn1);
                    k++;
                }
                results[iGlobal] = temp;
            }
        }

        public void Run<T>(int n, int windowSize, deviceptr<T> values, deviceptr<T> results)
        {
            var maxBlockSize = 256;
            var blockSize = Math.Min(n, maxBlockSize);
            var gridSizeX = (n - 1) / blockSize + 1;

            var sharedMem = (blockSize + windowSize - 1) * Intrinsic.__sizeof<T>();
            var lp = new LaunchParam(gridSizeX, blockSize, sharedMem);
            if (gridSizeX > 65535)
            {
                var gridSizeY = 1 + (n - 1) / (blockSize * 65535);
                gridSizeX = 1 + (n - 1) / (blockSize * gridSizeY);
                lp = new LaunchParam(new dim3(gridSizeX, gridSizeY), new dim3(blockSize), sharedMem);
            }
            GPULaunch(MovingAvg, lp, windowSize, n, values, results);
        }
    }

    public class MovingAverage_CPU
    {
        public double[] MovingAverageArray(int windowSize, double[] series)
        {
            var count = 1;
            var sums = series.Select((a, i) => series.Take(i + 1).Sum()).ToArray();
            var ma = new double[sums.Length - windowSize];
            for (var i = windowSize; i < sums.Length; i++)
                ma[i - windowSize] = (sums[i] - sums[i - windowSize]) / (LibDevice2.__gconv<int, double>(windowSize));
            return ma;
        }
    }
}
