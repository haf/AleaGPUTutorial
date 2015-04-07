using System;
using System.Runtime.InteropServices;
using Alea.CUDA;
using NUnit.Framework;
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

        public static float[] Sum(float[] input, bool inclusive)
        {
            return
                (new ScanModule<float>(
                    GPUModuleTarget.DefaultWorker,
                    () => 0.0f,
                    (x, y) => x + y,
                    x => x,
                    Plan.Plan64)
                ).Apply(input, inclusive);
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

        public static T[] Scan<T>(T[] input, Func<T,T,T> scanOp, bool inclusive)
        {
            return
                (new ScanModule<T>(
                    GPUModuleTarget.DefaultWorker,
                    () => default(T),
                    scanOp,
                    x => x,
                    Intrinsic.__sizeof<T>() == 4 ? Plan.Plan32 : Plan.Plan64)
                ).Apply(input, inclusive);            
        }
    }
}
