using System;
using Alea.CUDA;

namespace Tutorial.Cs.examples.generic_reduce
{
    public static class ReduceApi
    {

        public static int Sum(int[] values)
        {
            return SumApi.Sum(values);
        }

        public static float Sum(float[] values)
        {
            return SumApi.Sum(values);
        }

        public static double Sum(double[] values)
        {
            return SumApi.Sum(values);
        }

        public static T Reduce<T>(T[] input, Func<T, T, T> reductionOp)
        {
            return
                (new ReduceModule<T>(
                    GPUModuleTarget.DefaultWorker,
                    () => default(T),
                    reductionOp,
                    x => x,
                    Intrinsic.__sizeof<T>() == 4 ? Plan.Plan32 : Plan.Plan64)
                ).Apply(input);
        }
    }
}
