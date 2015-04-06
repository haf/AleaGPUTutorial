using System;
using Alea.CUDA;
using Tutorial.Cs.examples.generic_reduce;

namespace Tutorial.Cs.examples.generic_scan
{
    public static class ScanApi
    {
        public static double[] Sum(double[] input, bool inclusive)
        {
            return
                (new ScanModule<double>(
                    GPUModuleTarget.DefaultWorker,
                    () => 0.0,
                    (x,y)=> x+y,
                    x=>x,
                    Plan.Plan64)
                ).Apply(input, inclusive);
        }

        public static void Sum(float[] input)
        {

        }

        public static int[] Sum(int[] input, bool inclusive)
        {
            return
                (new ScanModule<int>(
                    GPUModuleTarget.DefaultWorker,
                    () => 0,
                    (x, y) => x + y,
                    x => x,
                    Plan.Plan32)
                ).Apply(input, inclusive);
        }
    }
}
