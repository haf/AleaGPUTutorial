using System;
using Alea.CUDA;

namespace Tutorial.Cs.examples.generic_reduce
{
    public static class ReduceApi
    {

        public static int Sum(int[] values)
        {
            return
                (new ReduceModule<int>(
                    GPUModuleTarget.DefaultWorker,
                    () => 0,
                    (x, y) => x + y,
                    x => x,
                    Plan.Plan32)
                ).Apply(values);
        }

        public static double Sum(double[] values)
        {
            return
                (new ReduceModule<double>(
                    GPUModuleTarget.DefaultWorker, 
                    () => 0.0, 
                    (x, y) => x + y, 
                    x => x,
                    Plan.Plan64)
                ).Apply(values);
        }

        public static float Sum(float[] values)
        {
            return
                (new ReduceModule<float>(
                    GPUModuleTarget.DefaultWorker,
                    () => 0.0f,
                    (x, y) => x + y,
                    x => x,
                    Plan.Plan32)
                ).Apply(values);            
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
