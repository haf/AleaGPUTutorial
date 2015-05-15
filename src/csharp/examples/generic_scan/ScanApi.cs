using System;
using System.Linq;
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
                    transf)
                ).Apply(input, inclusive);
        }
        
        /// Returns inclusive scan action; used when you need to reuse the scan output on the device. 
        /// Uses () => default(T) for the init function and x => x for the transf function.
        public static Action<T[], deviceptr<T>> InclusiveScan<T>(GPUModuleTarget target, Func<T, T, T> scanOp)
        {
            return Scan(target, scanOp, true);
        }

        /// Returns exclusive scan action; used when you need to reuse the scan output on the device. 
        /// Uses () => default(T) for the init function and x => x for the transf function.
        public static Action<T[], deviceptr<T>> ExclusiveScan<T>(GPUModuleTarget target, Func<T, T, T> scanOp)
        {
            return Scan(target, scanOp, false);
        }

        /// Returns scan action; used when you need to reuse the scan output on the device. 
        /// Uses () => default(T) for the init function and x => x for the transf function.
        public static Action<T[], deviceptr<T>> Scan<T>(GPUModuleTarget target, Func<T, T, T> scanOp, bool inclusive)
        {
            return Scan(target, () => default(T), scanOp, inclusive);
        }

        /// Returns scan action; used when you need to reuse the scan output on the device. 
        /// Uses x => x for the transf function.
        public static Action<T[], deviceptr<T>> Scan<T>(GPUModuleTarget target, Func<T> init, Func<T, T, T> scanOp, bool inclusive)
        {
            return Scan(target, init, scanOp, x => x, inclusive);
        }

        /// Returns scan action; used when you need to reuse the scan output on the device.
        public static Action<T[], deviceptr<T>> Scan<T>(GPUModuleTarget target, Func<T> init, Func<T, T, T> scanOp, Func<T, T> transf, bool inclusive)
        {
            return (input, output) =>
            {
                using (var dInput = target.GetWorker().Malloc(input))
                {
                    (new ScanModule<T>(
                        target,
                        init,
                        scanOp,
                        transf)
                        ).Apply(input.Length, dInput.Ptr, output, inclusive);
                }
            };
        }
        
        /// Simple generic CPU scan.
        public static T[] CpuScan<T>(Func<T, T, T> op, T[] input, bool inclusive)
        {
            var result = new T[input.Length + 1];
            result[0] = default(T);
            for (var i = 1; i < result.Length; i++)
                result[i] = op(result[i - 1], input[i - 1]);
            return inclusive ? result.Skip(1).ToArray() : result.Take(input.Length).ToArray();
        }

    }
    //[/GenericScanApi]
}
