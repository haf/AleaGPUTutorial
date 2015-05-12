using System;
using Alea.CUDA;
using Alea.CUDA.IL;

namespace Tutorial.Cs.examples.generic_reduce
{
    public class ScalarProdModule<T> : ILGPUModule
    {
        private readonly Plan _plan;
        private readonly Func<T, T, T> _mult;
        private readonly Func<T, T, T> _add;

        private readonly Func<int, T, T> _multiReduce;
        private readonly ReduceModule<T> _reduce; 

        public ScalarProdModule(GPUModuleTarget target, Plan plan, Func<T> init, Func<T,T,T> add, Func<T,T,T> mult) : base(target)
        {
            _plan = plan;
            _mult = mult;
            _add = add;
            var logNumWarps = Alea.CUDA.Utilities.Common.log2(plan.NumWarps);
            _multiReduce = ReduceModule<T>.MultiReduce(init, add, plan.NumWarps, logNumWarps);
            _reduce = new ReduceModule<T>(target, init, add, x => x, plan);
        }

        ///[genericReduceScalarProdUpsweepKernel]
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
        ///[/genericReduceScalarProdUpsweepKernel]
        
        ///[genericReduceScalarProdUse]
        public T Apply(T[] values1, T[] values2)
        {
            var n = values1.Length;
            var numSm = GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT;
            var tup = _plan.BlockRanges(numSm, n);
            var ranges = tup.Item1;
            var numRanges = tup.Item2;
            var lpUpsweep = new LaunchParam(numRanges, _plan.NumThreads);
            var lpReduce = new LaunchParam(1, _plan.NumThreadsReduction);
            
            using (var dRanges = GPUWorker.Malloc(ranges))
            using (var dRangeTotals = GPUWorker.Malloc<T>(numRanges))
            using (var dValues1 = GPUWorker.Malloc(values1))
            using (var dValues2 = GPUWorker.Malloc(values2))
            {
                // Launch range reduction kernel to calculate the totals per range.
                GPUWorker.EvalAction(
                    () =>
                    {
                        GPULaunch(Upsweep, lpUpsweep, dValues1.Ptr, dValues2.Ptr, dRanges.Ptr, dRangeTotals.Ptr);
                        if (numRanges > 1)
                        {
                            // Need to aggregate the block sums as well.
                            GPULaunch(_reduce.ReduceRangeTotals, lpReduce, numRanges, dRangeTotals.Ptr);
                        }
                    });
                return dRangeTotals.Gather()[0];
            }
        }
        ///[/genericReduceScalarProdUse]

    }
}
