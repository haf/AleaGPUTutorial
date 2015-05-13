using System;
using Alea.CUDA;
using Tutorial.Cs.examples.generic_reduce;

namespace Tutorial.Cs.examples.generic_scan
{
    //[GenericScanApi]
    public static class ScanApi
    {
        /// Sum scan specialized for integers.
        public static int[] Sum(int[] input, bool inclusive)
        {
            return Scan((x, y) => x + y, input, inclusive);
        }

        /// Sum scan specialized for singles.
        public static float[] Sum(float[] input, bool inclusive)
        {
            return Scan((x, y) => x + y, input, inclusive);
        }

        /// Sum scan specialized for doubles.
        public static double[] Sum(double[] input, bool inclusive)
        {
            return Scan((x, y) => x + y, input, inclusive);
        }

        /// Generic inclusive scan.
        public static T[] InclusiveScan<T>(Func<T, T, T> scanOp, T[] input)
        {
            return Scan(scanOp, input, true);
        }

        /// Generic exclusive scan.
        public static T[] ExclusiveScan<T>(Func<T, T, T> scanOp, T[] input)
        {
            return Scan(scanOp, input, false);
        }

        /// Generic scan using the default worker, default(T) for the init function, and
        /// x => x for the transf function.
        public static T[] Scan<T>(Func<T, T, T> scanOp, T[] input, bool inclusive)
        {
            return Scan(() => default(T), scanOp, input, inclusive);
        }

        /// Generic scan using the default worker and x => x for the transf function.
        public static T[] Scan<T>(Func<T> init, Func<T, T, T> scanOp, T[] input, bool inclusive)
        {
            return Scan(init, scanOp, x => x, input, inclusive);
        }

        /// Generic scan using the default worker.
        public static T[] Scan<T>(Func<T> init, Func<T, T, T> scanOp, Func<T, T> transf, T[] input, bool inclusive)
        {
            return Scan(GPUModuleTarget.DefaultWorker, init, scanOp, transf, input, inclusive);
        }

        /// Generic scan.
        public static T[] Scan<T>(GPUModuleTarget target, Func<T> init, Func<T,T,T> scanOp, Func<T,T> transf, T[] input, bool inclusive)
        {
            return
                (new ScanModule<T>(
                    target,
                    init,
                    scanOp,
                    transf,
                    Intrinsic.__sizeof<T>() == 4 ? Plan.Plan32 : Plan.Plan64)
                ).Apply(input, inclusive);
        }
    }
    //[/GenericScanApi]
}
