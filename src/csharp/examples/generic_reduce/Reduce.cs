using System;
using Alea.CUDA;
using Alea.CUDA.IL;
using Alea.CUDA.Utilities;

namespace Tutorial.Cs.examples.generic_reduce
{
    
    public class ReduceModule<T> : ILGPUModule
    {
        public static Func<int, T, T> MultiReduce(Func<T> initFunc, Func<T, T, T> reductionOp, int numWarps,
            int logNumWarps)
        {
            var warpStride = Const.WARP_SIZE + Const.WARP_SIZE/2 + 1;
            var sharedSize = numWarps*warpStride;

            return
                (tid, x) =>
                {
                    var warp = tid/Const.WARP_SIZE;
                    var lane = tid & (Const.WARP_SIZE - 1);
                    var shared =
                        Intrinsic.__ptr_volatile(
                            Intrinsic.__array_to_ptr(
                                __shared__.Array<T>(sharedSize)));
                    var warpShared = shared + warp*warpStride;
                    var s = warpShared + (lane + Const.WARP_SIZE/2);

                    warpShared[lane] = initFunc();
                    s[0] = x;

                    // Run inclusive scan on each warp's data.
                    var warpScan = x;
                    for (var i = 0; i < Const.LOG_WARP_SIZE; i++)
                    {
                        var offset = 1 << i;
                        warpScan = reductionOp.Invoke(warpScan, s[-offset]);
                        if (i < Const.LOG_WARP_SIZE - 1)
                            s[0] = warpScan;
                    }

                    var totalsShared =
                        Intrinsic.__ptr_volatile(
                            Intrinsic.__array_to_ptr(
                                __shared__.Array<T>(2*numWarps)));

                    // Last line of warp stores the warp scan.
                    if (lane == (Const.WARP_SIZE - 1))
                        totalsShared[numWarps + warp] = warpScan;

                    // Synchronize to make all the totals available to the reduction code.
                    Intrinsic.__syncthreads();

                    // Run an exclusive scan for the warp scans.
                    if (tid < numWarps)
                    {
                        // Grab the block total for the tid'th block.  This is the last element
                        // in the block's scanned sequence.  This operation avoids bank conflicts.
                        var total = totalsShared[numWarps + tid];
                        totalsShared[tid] = initFunc();
                        var ss = totalsShared + numWarps + tid;

                        var totalsScan = total;
                        for (var i = 0; i < logNumWarps; i++)
                        {
                            var offset = 1 << i;
                            totalsScan = reductionOp.Invoke(totalsScan, ss[-offset]);
                            ss[0] = totalsScan;
                        }
                    }

                    // Synchronize to make the block scan available to all warps.
                    Intrinsic.__syncthreads();

                    // The total is the last element.
                    return totalsShared[2*numWarps - 1];

                };
        }

        private readonly Func<T> _initFunc;
        private readonly Func<T,T,T> _reductionOp;
        private readonly Func<T,T> _transform;
        public Plan Plan;
        private readonly int _numThreads;
        private readonly Func<int, T, T> _multiReduce;

        public ReduceModule(GPUModuleTarget target, Func<T> initFunc, Func<T, T, T> reductionOp, Func<T, T> transform,
            Plan plan) : base(target)
        {
            _initFunc = initFunc;
            _reductionOp = reductionOp;
            _transform = transform;
            Plan = plan;
            _numThreads = plan.NumThreads;
            var numWarps = plan.NumWarps;
            var logNumWarps = Common.log2(numWarps);
            _multiReduce = MultiReduce(initFunc, reductionOp, numWarps, logNumWarps);
        }

        [Kernel]
        public void Upsweep(deviceptr<T> dValues, deviceptr<int> dRanges, deviceptr<T> dRangeTotals)
        {
            // Each block is processing a range.
            var range = blockIdx.x;
            var tid = threadIdx.x;
            var rangeX = dRanges[range];
            var rangeY = dRanges[range + 1];

            // Loop through all elements in the interval, adding up values.
            // There is no need to synchronize until we perform the multi-reduce.
            var reduced = _initFunc();
            var index = rangeX + tid;
            while (index < rangeY)
            {
                reduced = _reductionOp.Invoke(reduced, _transform.Invoke(dValues[index]));
                index += _numThreads;
            }

            // Get the total.
            var total = _multiReduce.Invoke(tid, reduced);

            if (tid == 0)
                dRangeTotals[range] = total;
        }

        [Kernel]
        public void ReduceRangeTotals(int numRanges, deviceptr<T> dRangeTotals)
        {
            var tid = threadIdx.x;
            var x = tid < numRanges ? dRangeTotals[tid] : _initFunc();
            var total = _multiReduce.Invoke(tid, x);

            if (tid == 0)
                dRangeTotals[0] = total;
        }

        public T Apply(T[] values)
        {
            var n = values.Length;
            var numSm = GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT;
            var tup = Plan.BlockRanges(numSm, n);
            var ranges = tup.Item1;
            var numRanges = tup.Item2;
            var lpUpsweep = new LaunchParam(numRanges, Plan.NumThreads);
            var lpReduce = new LaunchParam(1, Plan.NumThreadsReduction);

            using (var dValues = GPUWorker.Malloc(values))
            using (var dRanges = GPUWorker.Malloc(ranges))
            using (var dRangeTotals = GPUWorker.Malloc<T>(numRanges))
            {
                // Launch range reduction kernel to calculate the totals per range.
                GPUWorker.EvalAction(
                    () =>
                    {
                        GPULaunch(Upsweep, lpUpsweep, dValues.Ptr, dRanges.Ptr, dRangeTotals.Ptr);
                        if (numRanges > 1)
                        {
                            // Need to aggregate the block sums as well.
                            GPULaunch(ReduceRangeTotals, lpReduce, numRanges, dRangeTotals.Ptr);
                        }
                    });
                return dRangeTotals.Gather()[0];
            }

        }
    }
}
