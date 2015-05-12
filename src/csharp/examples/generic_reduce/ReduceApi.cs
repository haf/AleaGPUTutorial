using System;
using Alea.CUDA;

namespace Tutorial.Cs.examples.generic_reduce
{
    public static class ReduceApi
    {

        public static int ScalarProd(int[] values1, int[] values2)
        {
            return ScalarProd(values1, values2, (x, y) => x + y, (x, y) => x * y);
        }

        public static float ScalarProd(float[] values1, float[] values2)
        {
            return ScalarProd(values1, values2, (x, y) => x + y, (x, y) => x * y);
        }

        public static double ScalarProd(double[] values1, double[] values2)
        {
            return ScalarProd(values1, values2, (x, y) => x + y, (x, y) => x*y);
        }

        public static T ScalarProd<T>(T[] values1, T[] values2, Func<T,T,T> add, Func<T,T,T> mult)
        {
            return
                (new ScalarProdModule<T>(
                    GPUModuleTarget.DefaultWorker,
                    Intrinsic.__sizeof<T>() == 4 ? Plan.Plan32 : Plan.Plan64,
                    () => default(T),
                    add,
                    mult)
                    ).Apply(values1, values2);
        }
        
        public static int Sum(int[] values)
        {
            return Reduce(values, (x, y) => x + y);
        }

        public static float Sum(float[] values)
        {
            return Reduce(values, (x, y) => x + y);
        }

        public static double Sum(double[] values)
        {
            return Reduce(values, (x, y) => x + y);
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
