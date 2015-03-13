//[unboundReduceImport]
using System;
using System.Linq;
using Alea.CUDA.Unbound;
using NUnit.Framework;
//[/unboundReduceImport]

namespace Tutorial.Cs.examples.unbound
{
    public static class Reduce
    {

    }

    public partial class Test
    {
        [Test]
        public static void DeviceReduceTest()
        {
            const int numItems = 1000000;
            var rng = new Random(42);
            var inputs = (Enumerable.Repeat(rng, numItems).Select((random, i) => random.Next(-10, 10))).ToArray();
            var gpuReduceModule = DeviceSumModuleI32.Default;

            using (var gpuReduce = gpuReduceModule.Create(numItems))
            using (var dInputs = gpuReduceModule.GPUWorker.Malloc(inputs))
            {
                var actual = gpuReduce.Reduce(dInputs.Ptr, numItems);
                var expected = inputs.Sum(i => i);
                Assert.AreEqual(expected, actual);
            }

        }
    }
}
