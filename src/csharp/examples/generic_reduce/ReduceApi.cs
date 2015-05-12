using System;
using Alea.CUDA;

namespace Tutorial.Cs.examples.generic_reduce
{
    //[GenericReduceApi]
    public static class ReduceApi
    {
        /// Scalar product reduction specialized for integers.
        public static int ScalarProd(int[] values1, int[] values2)
        {
            return ScalarProd((x, y) => x + y, (x, y) => x * y, values1, values2);
        }

        /// Scalar product reduction specialized for singles.
        public static float ScalarProd(float[] values1, float[] values2)
        {
            return ScalarProd((x, y) => x + y, (x, y) => x * y, values1, values2);
        }

        /// Scalar product reduction specialized for doubles.
        public static double ScalarProd(double[] values1, double[] values2)
        {
            return ScalarProd((x, y) => x + y, (x, y) => x*y, values1, values2);
        }

        /// Generic scalar product reduction.  Be sure to pass (x,y) => x+y for the add parameter
        /// and (x,y) => x*y for the mult parameter.
        public static T ScalarProd<T>(Func<T,T,T> add, Func<T,T,T> mult, T[] values1, T[] values2)
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
        
        /// Sum reduction specialized for integers.
        public static int Sum(int[] values)
        {
            return Reduce((x, y) => x + y, values);
        }

        /// Sum reduction specialized for singles.
        public static float Sum(float[] values)
        {
            return Reduce((x, y) => x + y, values);
        }

        /// Sum reduction specialized for doubles.
        public static double Sum(double[] values)
        {
            return Reduce((x, y) => x + y, values);
        }

        /// Generic reduction using the default worker, default(T) for the init function, and
        /// x => x for the transf function.
        public static T Reduce<T>(Func<T, T, T> reductionOp, T[] input)
        {
            return Reduce(GPUModuleTarget.DefaultWorker, () => default(T), reductionOp, x => x, input);
        }

        /// Generic reduction using the default worker and x => x for the transf function.
        public static T Reduce<T>(Func<T> init, Func<T, T, T> reductionOp, T[] input)
        {
            return Reduce(GPUModuleTarget.DefaultWorker, init, reductionOp, x => x, input);
        }

        /// Generic reduction using the default worker.
        public static T Reduce<T>(Func<T> init, Func<T, T, T> reductionOp, Func<T, T> transf, T[] input)
        {
            return Reduce(GPUModuleTarget.DefaultWorker, init, reductionOp, transf, input);
        }

        /// Generic reduction.
        public static T Reduce<T>(GPUModuleTarget target, Func<T> init, Func<T, T, T> reductionOp, Func<T, T> transf, T[] input)
        {
            return
                (new ReduceModule<T>(
                    target,
                    init,
                    reductionOp,
                    transf,
                    Intrinsic.__sizeof<T>() == 4 ? Plan.Plan32 : Plan.Plan64)
                ).Apply(input);
        }

    }
    //[/GenericReduceApi]
}
