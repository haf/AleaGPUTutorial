using System;
using Alea.CUDA;
using Alea.CUDA.IL;

namespace Tutorial.Cs.examples.generic_reduce
{
    public class ScalarProd<T> : ILGPUModule
    {
        private Plan _plan;
        private Func<T, T, T> _mult;
        private Func<T, T, T> _add;

        private Func<int, T, T> _multiReduce; 

        public ScalarProd(GPUModuleTarget target, Plan plan, Func<T> init, Func<T,T,T> add, Func<T,T,T> mult) : base(target)
        {
            _plan = plan;
            _mult = mult;
            _add = add;
            var logNumWarps = Alea.CUDA.Utilities.Common.log2(plan.NumWarps);
            _multiReduce = ReduceModule<T>.MultiReduce(init, mult, plan.NumWarps, logNumWarps);
        }

        [Kernel]
        public void Upsweep(deviceptr<T> dValues1, deviceptr<T> dValues2, deviceptr<int> dRanges, deviceptr<T> dRangeTotals)
        {
            var block = blockIdx.x;
            var tid = threadIdx.x;
            var rangeX = dRanges[block];
            var rangeY = dRanges[block + 1];

            // Loop through all elements in the interval, adding up values.
            // There is no need to synchronize until we perform the multireduce.
            var sum = default(T);
            var index = rangeX + tid;
            while (index < rangeY)
            {
                sum = _add(sum, _mult(dValues1[index], dValues2[index]));
                index += _plan.NumThreads;
            }

            // Get the total.
            var total = _multiReduce(tid, sum);

            if (tid == 0)
                dRangeTotals[block] = total;
        }

        public void Upsweep(LaunchParam lp, deviceptr<T> dValues1, deviceptr<T> dValues2, deviceptr<int> dRanges,
            deviceptr<T> dRangeTotals)
        {
            GPULaunch(Upsweep, lp, dValues1, dValues2, dRanges, dRangeTotals);
        }
    }
}
