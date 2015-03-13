//[parallelSquareImport]
using System;
using Alea.CUDA;
using Alea.CUDA.Utilities;
using Alea.CUDA.IL;
using NUnit.Framework;
//[/parallelSquareImport]

using System.Linq;

namespace Tutorial.Cs.quick_start
{
    public class ParallelSquare
    {
        //[parallelSquareCPU]
        static void SquareCPU(double[] inputs)
        {
            var outputs = new double[inputs.Length];
            for (var i = 0; i < inputs.Length; i++)
            {
                outputs[i] = inputs[i] * inputs[i];
            }
        }
        //[/parallelSquareCPU]

        //[parallelSquareKernel]
        [AOTCompile]
        static void SquareKernel(deviceptr<double> outputs, deviceptr<double> inputs, int n)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            for (var i = start; i < n; i += stride)
            {
                outputs[i] = inputs[i] * inputs[i];
            }
        }
        //[/parallelSquareKernel]

        //[parallelSquareLaunch]
        static double[] SquareGPU(double[] inputs)
        {
            var worker = Worker.Default;
            using (var dInputs = worker.Malloc(inputs))
            using (var dOutputs = worker.Malloc<double>(inputs.Length))
            {
                const int blockSize = 256;
                var numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT;
                var gridSize = Math.Min(16 * numSm, Common.divup(inputs.Length, blockSize));
                var lp = new LaunchParam(gridSize, blockSize);
                worker.Launch(SquareKernel, lp, dOutputs.Ptr, dInputs.Ptr, inputs.Length);
                return dOutputs.Gather();
            }
        }
        //[/parallelSquareLaunch]

        //[parallelSquareTest]
        [Test]
        public static void SquareTest()
        {
            var inputs = Enumerable.Range(0, 101).Select(i => -5.0 + i*0.1).ToArray();
            var outputs = SquareGPU(inputs);
            Console.WriteLine("inputs = {0}", String.Join(", ", inputs));
            Console.WriteLine("outputs = {0}", String.Join(", ", outputs));
        }
        //[/parallelSquareTest]
    }
}