//[genericTransformImport]
using System;
using Alea.CUDA;
using Alea.CUDA.Utilities;
using Alea.CUDA.IL;
using LibDevice = Alea.CUDA.LibDevice;
using NUnit.Framework;
//[/genericTransformImport]

using System.Linq;

namespace Tutorial.Cs.advancedTechniques.GenericTransform
{

    //[transformModule]
    internal class TransformModule<T> : ILGPUModule
    //[/transformModule]
    {
        //[transformConstructor]
        private readonly Func<T, T, T> op;

        public TransformModule(GPUModuleTarget target, Func<T, T, T> opFunc)
            : base(target)
        {
            op = opFunc;
        }
        //[/transformConstructor]
        

        //[transformKernel]
        [Kernel]
        public void Kernel(int n, deviceptr<T> x, deviceptr<T> y, deviceptr<T> z)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            for (var i = start; i < n; i += stride)
                z[i] = op(x[i], y[i]);
        }
        //[/transformKernel]

        //[transformGPUDevice]
        public void Apply(int n, deviceptr<T> x, deviceptr<T> y, deviceptr<T> z)
        {
            const int blockSize = 256;
            var numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT;
            var gridSize = Math.Min(16 * numSm, Common.divup(n, blockSize));
            var lp = new LaunchParam(gridSize, blockSize);
            GPULaunch(Kernel, lp, n, x, y, z);
        }
        //[/transformGPUDevice]

        //[transformGPUHost]
        public T[] Apply(T[] x, T[] y)
        {
            using (var dx = GPUWorker.Malloc(x))
            using (var dy = GPUWorker.Malloc(y))
            using (var dz = GPUWorker.Malloc<T>(x.Length))
            {
                Apply(x.Length, dx.Ptr, dy.Ptr, dz.Ptr);
                return dz.Gather();
            }
        }
        //[/transformGPUHost]
    }

    //[transformModuleSpecialized]
    [AOTCompile]
    class SinCosModule : TransformModule<double>
    {
        public SinCosModule(GPUModuleTarget target)
            : base(target, (a, b) => LibDevice.__nv_sin(a) + LibDevice.__nv_cos(b))
        {
        }

        private static SinCosModule Instance = null;
        public static SinCosModule DefaultInstance
        {
            get { return Instance ?? (Instance = new SinCosModule(GPUModuleTarget.DefaultWorker)); }
        }
    }
    //[/transformModuleSpecialized]

    //[transformModuleSpecializedTest]
    public class Test 
    {
        [Test]
        public static void SinCosTest()
        {
            using (var sinCos = SinCosModule.DefaultInstance)
            {
                var rng = new Random();
                const int n = 1000;
                var x = Enumerable.Range(0, n).Select(i => rng.NextDouble()).ToArray();
                var y = Enumerable.Range(0, n).Select(i => rng.NextDouble()).ToArray();
                var dResult = sinCos.Apply(x, y);
                var hResult = x.Zip(y, (a, b) => Math.Sin(a) + Math.Cos(b));
                var err = dResult.Zip(hResult, (d, h) => Math.Abs(d - h)).Max();
                Console.WriteLine("error = {0}", err);
            }
        }
    }
    //[/transformModuleSpecializedTest]
}