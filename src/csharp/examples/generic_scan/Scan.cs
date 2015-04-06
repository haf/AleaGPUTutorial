using System;
using Alea.CUDA;
using Alea.CUDA.IL;

namespace Tutorial.Cs.examples.generic_scan
{
    public class Scan<T> : ILGPUModule
    {
        // Multi-scan function for all warps in the block.
        public static Func<int, T, T, T> MultiScan(Func<T> init, Func<T, T, T> scanOp, int numWarps, int logNumWarps)
        {
            var warpStride = Const.WARP_SIZE + Const.WARP_SIZE/2 + 1;
            return
                (tid, x, totalRef) =>
                {
                    var warp = tid/Const.WARP_SIZE;
                    var lane = tid & (Const.WARP_SIZE - 1);

                    // Allocate shared memory.
                    var shared =
                        Intrinsic.__ptr_volatile(
                            Intrinsic.__array_to_ptr(
                                __shared__.Array<T>(numWarps*warpStride)));
                    var totalsShared =
                        Intrinsic.__ptr_volatile(
                            Intrinsic.__array_to_ptr(
                                __shared__.Array<T>(2*numWarps)));

                    var warpShared = (shared + warp*warpStride).Volatile();
                    var s = (warpShared + lane + Const.WARP_SIZE/2).Volatile();
                    warpShared[lane] = init();
                    s[0] = x;

                    // Run inclusive scan on each warp's data.
                    var scan = x;
                    for (var i = 0; i < Const.LOG_WARP_SIZE; i++)
                    {
                        var offset = 1 << i;
                        scan = scanOp(scan, s[-offset]);
                        if (i < Const.LOG_WARP_SIZE - 1)
                            s[0] = scan;
                    }

                    if (lane == Const.WARP_SIZE - 1)
                        totalsShared[numWarps + warp] = scan;

                    // Synchronize to make all the totals available to the reduction code.
                    Intrinsic.__syncthreads();

                    if (tid < numWarps)
                    {
                        // Grab the block total for the tid'th block.  This is the last element
                        // in the block's scanned sequence. This operation avoids bank conflicts.
                        var total = totalsShared[numWarps + tid];
                        totalsShared[tid] = init();
                        var ss = (totalsShared + numWarps + tid).Volatile();

                        var totalsScan = total;
                        for (var i = 0; i < logNumWarps; i++)
                        {
                            var offset = 1 << i;
                            totalsScan = scanOp(totalsScan, ss[-offset]);
                            ss[0] = totalsScan;
                        }

                        // Store totalsScan shifted by one to the right for an exclusive scan.
                        totalsShared[tid + 1] = totalsScan;
                    }

                    // Synchronize to make the block scan available to all warps.
                    Intrinsic.__syncthreads();

                    // The total is the last element.
                    totalRef = totalsShared[2*numWarps - 1];

                    // Add the block scan to the inclusive scan for the block.
                    return scanOp(scan, totalsShared[warp]);
                };
        } 
        
        // Multi-scan function for all warps in the block.
        public static Func<int, T, T, T> MultiScanExcl(Func<T> init, Func<T, T, T> scanOp, int numWarps, int logNumWarps)
        {
            var warpStride = Const.WARP_SIZE + Const.WARP_SIZE/2 + 1;
            return
                (tid, x, totalRef) =>
                {
                    var warp = tid/Const.WARP_SIZE;
                    var lane = tid & (Const.WARP_SIZE - 1);

                    // Allocate shared memory.
                    var shared =
                        Intrinsic.__ptr_volatile(
                            Intrinsic.__array_to_ptr(
                                __shared__.Array<T>(numWarps*warpStride)));
                    var totalsShared =
                        Intrinsic.__ptr_volatile(
                            Intrinsic.__array_to_ptr(
                                __shared__.Array<T>(2*numWarps)));
                    var exclScan =
                        Intrinsic.__ptr_volatile(
                            Intrinsic.__array_to_ptr(
                                __shared__.Array<T>(numWarps*Const.WARP_SIZE + 1)));

                    var warpShared = (shared + warp*warpStride).Volatile();
                    var s = (warpShared + lane + Const.WARP_SIZE/2).Volatile();
                    warpShared[lane] = init();
                    s[0] = x;

                    // Run inclusive scan on each warp's data.
                    var scan = x;
                    for (var i = 0; i < Const.LOG_WARP_SIZE; i++)
                    {
                        var offset = 1 << i;
                        scan = scanOp(scan, s[-offset]);
                        if (i < Const.LOG_WARP_SIZE - 1)
                            s[0] = scan;
                    }

                    if (lane == Const.WARP_SIZE - 1)
                        totalsShared[numWarps + warp] = scan;

                    // Synchronize to make all the totals available to the reduction code.
                    Intrinsic.__syncthreads();

                    if (tid < numWarps)
                    {
                        // Grab the block total for the tid'th block.  This is the last element
                        // in the block's scanned sequence.  This operation avoids bank conflicts.
                        var total = totalsShared[numWarps + tid];
                        totalsShared[tid] = init();
                        var ss = (totalsShared + numWarps + tid).Volatile();

                        var totalsScan = total;
                        for (var i = 0; i < logNumWarps; i++)
                        {
                            var offset = 1 << i;
                            totalsScan = scanOp(totalsScan, ss[-offset]);
                            ss[0] = totalsScan;
                        }

                        // Store totalsScan shifted by one to the right for an exclusive scan.
                        totalsShared[tid + 1] = totalsScan;
                    }

                    // Synchronize to make the block scan available to all warps.
                    Intrinsic.__syncthreads();

                    // The total is the last element.
                    totalRef = totalsShared[2*numWarps - 1];

                    // Add the block scan to the inclusive scan for the block.
                    if (tid == 0)
                        exclScan[tid] = init();
                    exclScan[tid + 1] = scanOp(totalsShared[warp], scan);

                    // Synchronize to make the exclusive scan available to all threads.
                    Intrinsic.__syncthreads();

                    return exclScan[tid];

                };
        }

        private readonly Func<T> _init;
        private readonly Func<T, T, T> _scanOp;
        private readonly Func<T, T> _transform;
        public Plan Plan;
        private int numThreads;
        private int numWarps;
        private int logNumWarps;
        private Func<int, T, T, T> multiScan;

        public Scan(GPUModuleTarget target, Func<T> init, Func<T,T,T> scanOp, Func<T,T> transform, Plan plan) : base(target)
        {
            _init = init;
            _scanOp = scanOp;
            _transform = transform;
            Plan = plan;
            numThreads = plan.NumThreadsReduction;
            numWarps = plan.NumWarpsReduction;
            logNumWarps = Alea.CUDA.Utilities.Common.log2(numWarps);
            multiScan = MultiScan(init, scanOp, numWarps, logNumWarps);
        }

        [Kernel]
        public void ScanReduce(int numRanges, deviceptr<T> dRangeTotals)
        {
            var tid = threadIdx.x;
            var x = (tid < numRanges) ? dRangeTotals[tid] : _init();
            var total = _init();
            var sum = multiScan(tid, x, total);
            // Shift the value from the inclusive scan for the exclusive scan.
            if (tid < numRanges)
                dRangeTotals[tid + 1] = sum;
            // Have the first thread in the block set the scan total.
            if (tid == 0)
                dRangeTotals[0] = _init();
        }

        [Kernel]
        public void Downsweep(deviceptr<T> dValuesIn, deviceptr<T> dValuesOut, deviceptr<T> dRangeTotals,
            deviceptr<int> dRanges, int inclusive)
        {
            
        }
    }
}
