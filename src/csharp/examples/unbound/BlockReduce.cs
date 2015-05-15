using System;
using System.Collections.Generic;
using System.Linq;
using Alea.CUDA;
using Alea.CUDA.Unbound;
using NUnit.Framework;

namespace Tutorial.Cs.examples.unbound
{
    //[BlockReduceModule]
    public class BlockReduceModule : ILGPUModule
    {
        private readonly BlockReduce<int> _blockReduce;
        private readonly UnionStorage _tempStorage;
        private readonly Func<int, int, int> _op; 

        public BlockReduceModule(GPUModuleTarget target) : base(target)
        {
            _blockReduce = BlockReduce.WarpReductions<int>(new dim3(128), new DeviceArch(2, 0));
            _tempStorage = _blockReduce.TempStorage;
            _op = (a, b) => a + b;
        }

        [Kernel]
        public void Kernel(deviceptr<int> output, deviceptr<int> inputs1, deviceptr<int> inputs2)
        {
            var tempStorage = _tempStorage.AllocateShared();
            var inputs = __local__.Array<int>(2);
            var tid = threadIdx.x;
            inputs[0] = inputs1[tid];
            inputs[1] = inputs2[tid];
            var aggregate = _blockReduce.ILReduce(tempStorage, inputs, _op);
            if (tid == 0) output[0] = aggregate;
        }

        public int RunTest(int[] inputs1, int[] inputs2)
        {
            Assert.AreEqual(inputs1.Length, 128);
            Assert.AreEqual(inputs2.Length, 128);
            using (var dInputs1 = GPUWorker.Malloc(inputs1))
            using (var dInputs2 = GPUWorker.Malloc(inputs2))
            using (var dOutput = GPUWorker.Malloc<int>(1))
            {
                var lp = new LaunchParam(1, 128);
                GPULaunch(Kernel, lp, dOutput.Ptr, dInputs1.Ptr, dInputs2.Ptr);
                return dOutput.GatherScalar();
            }
        }
    }
    //[/BlockReduceModule]

    //[BlockReduceModuleTest]
    public static class Test
    {
        [Test]
        public static void BlockReduceModuleTest()
        {
            using (var gpuModule = new BlockReduceModule(GPUModuleTarget.DefaultWorker))
            {
                var inputs1 = Enumerable.Range(0, 128).ToArray();
                var inputs2 = Enumerable.Repeat(1, 128).ToArray();
                var hOutput = inputs1.Aggregate((a, b) => a + b) + inputs2.Aggregate((a, b) => a + b);
                var dOutput = gpuModule.RunTest(inputs1, inputs2);
                Assert.AreEqual(hOutput, dOutput);
            }
        }
    }
    //[/BlockReduceModuleTest]
}
