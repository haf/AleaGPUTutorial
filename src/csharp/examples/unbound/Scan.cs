//[unboundScanImport]
using System;
using System.Collections.Generic;
using System.Linq;
using Alea.CUDA.Unbound;
using NUnit.Framework;
//[/unboundScanImport]

namespace Tutorial.Cs.examples.unbound
{
    public static class Scan
    {
        //[cpuInclusiveScan]
        public static IEnumerable<int> ScanInclusive(this IEnumerable<int> source, int seed, Func<int, int, int> accumulator)
        {
            foreach (var item in source)
            {
                seed = accumulator(seed, item);
                yield return seed;
            }
        }
        //[/cpuInclusiveScan]
    }

    public partial class Test
    {
        //[unboundDeviceScanTest]
        [Test]
        public static void DeviceScanInclusiveTest()
        {
            const int numItems = 1000000;
            var rng = new Random(42);
            var inputs = Enumerable.Range(0, numItems).Select(i => rng.Next(-10, 10)).ToArray();
            var gpuScanModule = DeviceSumScanModuleI32.Default;

            using (var gpuScan = gpuScanModule.Create(numItems))
            using (var dInputs = gpuScanModule.GPUWorker.Malloc(inputs))
            using (var dOutputs = gpuScanModule.GPUWorker.Malloc<int>(inputs.Length))
            {
                gpuScan.InclusiveScan(dInputs.Ptr, dOutputs.Ptr, numItems);
                var actual = dOutputs.Gather();
                var expected = inputs.ScanInclusive(0, (a, b) => a + b).ToArray();
                Assert.AreEqual(expected, actual);
            }
        }
        //[/unboundDeviceScanTest]
    }
}
