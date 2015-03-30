using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Alea.CUDA;
using Alea.CUDA.IL;

namespace Tutorial.Cs.examples.generic_reduce
{
    public class Sum<T> : ILGPUModule
    {
        public static Func<int, T, T> MultiReduce(int numWarps, int logNumWarps)
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
                                __shared__.Array<dynamic>(sharedSize)));
                    var warpShared = shared + warp*warpStride;
                    var s = warpShared + (lane + Const.WARP_SIZE/2);

                    warpShared[lane] = default(T);
                    s[0] = x;

                    // Run inclusive scan on each warp's data.
                    var sum = Intrinsic.__unbox(x);
                    for (var i = 0; i < Const.LOG_WARP_SIZE; i++)
                    {
                        var offset = 1 << i;
                        sum += s[-offset];
                        if (i < Const.LOG_WARP_SIZE - 1)
                            s[0] = sum;
                    }
                    var totalsShared =
                        Intrinsic.__ptr_volatile(
                            Intrinsic.__array_to_ptr(
                                __shared__.Array<dynamic>(2*numWarps)));

                    if (lane == Const.WARP_SIZE - 1)
                        totalsShared[numWarps + warp] = sum;

                    // Synchronize to make all the totals available to the reduction code.
                    Intrinsic.__syncthreads();

                    if (tid < numWarps)
                    {
                        var total = totalsShared[numWarps + tid];
                        totalsShared[tid] = default(T);
                        var ss = totalsShared + numWarps + tid;

                        var totalsSum = Intrinsic.__unbox(total);
                        for (var i = 0; i < logNumWarps; i++)
                        {
                            var offset = 1 << i;
                            totalsSum += ss[-offset];
                            ss[0] = totalsShared;
                        }

                    }

                    // Synchronize to make the block scan available to all warps.
                    Intrinsic.__syncthreads();

                    // The total is the last element.
                    return totalsShared[2*numWarps - 1];

                };
        }

        private Plan _plan;
        private readonly int _numThreads;
        private readonly Func<int, T, T> _multiReduce; 

        public Sum(GPUModuleTarget target, Plan plan) : base(target)
        {
            _plan = plan;
            _numThreads = _plan.NumThreads;
            var numWarps = _plan.NumWarps;
            var logNumWarps = Alea.CUDA.Utilities.Common.log2(numWarps);
            _multiReduce = MultiReduce(numWarps, logNumWarps);
        }

        [Kernel]
        public void Upsweep(deviceptr<T> dValues, deviceptr<int> dRages, deviceptr<T> dRangeTotals)
        {
            var block = blockIdx.x;
            var tid = threadIdx.x;
            var rangeX = dRages[block];
            var rangeY = dRages[block + 1];

            // Loop through all elements in the interval, adding up values.
            // There is no need to synchronize until we perform the multi-reduce.
            var sum = Intrinsic.__unbox(default(dynamic));
            var index = rangeX + tid;
            while (index < rangeY)
            {
                sum += dValues[index];
                index += _numThreads;
            }

            // Get the total.
            var total = _multiReduce.Invoke(tid, sum);

            if (tid == 0)
                dRangeTotals[block] = total;
        }

        [Kernel]
        public void RangeTotals(int numRanges, deviceptr<T> dRangeTotals)
        {
            var tid = threadIdx.x;
            var x = tid < numRanges ? dRangeTotals[tid] : default(T);
            var total = _multiReduce.Invoke(tid, x);

            // Have the first thread in the block set the range total.
            if (tid == 0)
                dRangeTotals[0] = total;
        }
    }
}
