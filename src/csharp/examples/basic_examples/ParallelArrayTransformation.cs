using System;
using Alea.CUDA;
using Alea.CUDA.Utilities;
using Alea.CUDA.IL;
using LibDevice = Alea.CUDA.LibDevice;
using NUnit.Framework;
using System.Linq;

namespace Tutorial.Cs.examples.basic
{

    //[transformModule]
    internal class TransformModule<T> : ILGPUModule
    {
        private readonly Func<T, T> op;

        public TransformModule(GPUModuleTarget target, Func<T, T> opFunc)
            : base(target)
        {
            op = opFunc;
        }

        [Kernel]
        public void Kernel(int n, deviceptr<T> x, deviceptr<T> y)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            for (var i = start; i < n; i += stride)
                y[i] = op(x[i]);
        }

        public void Apply(int n, deviceptr<T> x, deviceptr<T> y)
        {
            const int blockSize = 256;
            var numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT;
            var gridSize = Math.Min(16 * numSm, Common.divup(n, blockSize));
            var lp = new LaunchParam(gridSize, blockSize);
            GPULaunch(Kernel, lp, n, x, y);
        }

        public T[] Apply(T[] x)
        {
            using (var dx = GPUWorker.Malloc(x))
            using (var dy = GPUWorker.Malloc<T>(x.Length))
            {
                Apply(x.Length, dx.Ptr, dy.Ptr);
                return dy.Gather();
            }
        }
    }
    //[/transformModule]


    //[transformModuleSpecialized]
    [AOTCompile]
    class SinModule : TransformModule<double>
    {
        public SinModule(GPUModuleTarget target)
            : base(target, LibDevice.__nv_sin)
        {
        }

        private static SinModule Instance = null;
        public static SinModule DefaultInstance
        {
            get { return Instance ?? (Instance = new SinModule(GPUModuleTarget.DefaultWorker)); }
        }
    }
    //[/transformModuleSpecialized]

    //[transformModuleSpecializedTest]
    public class Test
    {
        [Test]
        public static void SinTest()
        {
            using (var sinGpu = SinModule.DefaultInstance)
            {
                var rng = new Random();
                const int n = 1000;
                var x = Enumerable.Range(0, n).Select(i => rng.NextDouble()).ToArray();
                var dResult = sinGpu.Apply(x);
                var hResult = x.Select(Math.Sin);
                var err = dResult.Zip(hResult, (d, h) => Math.Abs(d - h)).Max();
                Console.WriteLine("error = {0}", err);
            }
        }
    }
    //[/transformModuleSpecializedTest]
}