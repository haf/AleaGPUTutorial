using System;
using Alea.CUDA;
using Alea.CUDA.IL;
using Microsoft.FSharp.Core;
using Tutorial.Cs.examples.generic_reduce;

namespace Tutorial.Cs.examples.generic_scan
{
    public class ScanModule<T> : ILGPUModule
    {
        //[GenericScanMultiScan]
        public static Func<int, T, FSharpRef<T>, T> MultiScan(Func<T> init, Func<T, T, T> scanOp, int numWarps, int logNumWarps)
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
                    totalRef.Value = totalsShared[2*numWarps - 1];

                    // Add the block scan to the inclusive scan for the block.
                    return scanOp(scan, totalsShared[warp]);
                };
        } 
        //[/GenericScanMultiScan]

        //[GenericScanMultiScanExcl]
        public static Func<int, T, FSharpRef<T>, T> MultiScanExcl(Func<T> init, Func<T, T, T> scanOp, int numWarps, int logNumWarps)
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
                    totalRef.Value = totalsShared[2*numWarps - 1];

                    // Add the block scan to the inclusive scan for the block.
                    if (tid == 0)
                        exclScan[tid] = init();
                    exclScan[tid + 1] = scanOp(totalsShared[warp], scan);

                    // Synchronize to make the exclusive scan available to all threads.
                    Intrinsic.__syncthreads();

                    return exclScan[tid];

                };
        }
        //[/GenericScanMultiScanExcl]

        private readonly Func<T> _init;
        private readonly Func<T, T, T> _scanOp;
        private readonly Func<T, T> _transform;
        public Plan Plan;
        // ScanReduce values
        private int _numThreadsScanReduce;
        private readonly Func<int, T, FSharpRef<T>, T> _multiScan; 
        
        // Downsweep values
        private readonly int _numValues;
        private readonly int _valuesPerThread;
        private readonly int _valuesPerWarp;
        private readonly int _size;
        private readonly Func<int, T, FSharpRef<T>, T> _multiScanExcl;
        private readonly ReduceModule<T> _reduceModule;

        public ScanModule(Func<T, T, T> scanOp)
            : this(GPUModuleTarget.DefaultWorker, scanOp) { }

        public ScanModule(GPUModuleTarget target, Func<T, T, T> scanOp)
            : this(target, () => default(T), scanOp) { }

        public ScanModule(GPUModuleTarget target, Func<T> init, Func<T, T, T> scanOp)
            : this(target, init, scanOp, x => x) { }
        
        public ScanModule(GPUModuleTarget target, Func<T> init, Func<T, T, T> scanOp, Func<T, T> transform)
            : this(target, init, scanOp, transform, Intrinsic.__sizeof<T>() == 4 ? Plan.Plan32 : Plan.Plan64) { }        
        
        public ScanModule(GPUModuleTarget target, Func<T> init, Func<T,T,T> scanOp, Func<T,T> transform, Plan plan) : base(target)
        {
            _init = init;
            _scanOp = scanOp;
            _transform = transform;
            Plan = plan;

            // ScanReduce values
            _numThreadsScanReduce = plan.NumThreadsReduction;
            var numWarpsScanReduce = plan.NumWarpsReduction;
            var logNumWarpsScanReduce = Alea.CUDA.Utilities.Common.log2(numWarpsScanReduce);
            _multiScan = MultiScan(init, scanOp, numWarpsScanReduce, logNumWarpsScanReduce);

            // Downsweep values
            var numWarpsDownsweep = plan.NumWarps;
            _numValues = plan.NumValues;
            _valuesPerThread = plan.ValuesPerThread;
            _valuesPerWarp = plan.ValuesPerWarp;
            var logNumWarpsDownsweep = Alea.CUDA.Utilities.Common.log2(numWarpsDownsweep);
            _size = numWarpsDownsweep*_valuesPerThread*(Const.WARP_SIZE + 1);
            _multiScanExcl = MultiScanExcl(init, scanOp, numWarpsDownsweep, logNumWarpsDownsweep);

            _reduceModule = new ReduceModule<T>(target, init, scanOp, transform, plan);
        }
        

        //[GenericScanReduceKernel]
        [Kernel]
        public void ScanReduce(int numRanges, deviceptr<T> dRangeTotals)
        {
            var tid = threadIdx.x;
            var x = (tid < numRanges) ? dRangeTotals[tid] : _init();
            var total = __local__.Variable(_init());
            var sum = _multiScan(tid, x, total);
            // Shift the value from the inclusive scan for the exclusive scan.
            if (tid < numRanges)
                dRangeTotals[tid + 1] = sum;
            // Have the first thread in the block set the scan total.
            if (tid == 0)
                dRangeTotals[0] = _init();
        }

        public void ScanReduce(LaunchParam lp, int numRanges, deviceptr<T> dRangeTotals)
        {
            GPULaunch(ScanReduce, lp, numRanges, dRangeTotals);
        }
        //[/GenericScanReduceKernel]

        //[GenericScanDownsweepKernel]
        [Kernel]
        public void Downsweep(deviceptr<T> dValuesIn, deviceptr<T> dValuesOut, deviceptr<T> dRangeTotals,
            deviceptr<int> dRanges, int inclusive)
        {
            var block = blockIdx.x;
            var tid = threadIdx.x;
            var warp = tid/Const.WARP_SIZE;
            var lane = (Const.WARP_SIZE - 1) & tid;
            var index = warp*_valuesPerWarp + lane;

            var blockScan = dRangeTotals[block];
            var rangeX = dRanges[block];
            var rangeY = dRanges[block + 1];

            var shared =
                Intrinsic.__ptr_volatile(
                    Intrinsic.__array_to_ptr(
                        __shared__.Array<T>(_size)));

            // Use a stride of 33 slots per warp per value to allow conflict-free tranposes 
            // from strided to thread order.
            var warpShared = shared + warp*_valuesPerThread*(Const.WARP_SIZE + 1);
            var threadShared = warpShared + lane;

            // Transpose values into thread order.
            var offset = _valuesPerThread*lane;
            offset += offset/Const.WARP_SIZE;

            while (rangeX < rangeY)
            {

                for (var i = 0; i < _valuesPerThread; i++)
                {
                    var source = rangeX + index + i * Const.WARP_SIZE;
                    var x = (source < rangeY) ? _transform(dValuesIn[source]) : _init();
                    threadShared[i * (Const.WARP_SIZE + 1)] = x;
                }

                // Transpose into thread order by reading from transposeValues.
                // Compute the exclusive or inclusive scan of the thread values and their sum.
                var threadScan = __local__.Array<T>(_valuesPerThread);
                var scan = _init();

                for (var i = 0; i < _valuesPerThread; i++)
                {
                    var x = warpShared[offset + i];
                    threadScan[i] = scan;
                    if (inclusive != 0)
                        threadScan[i] = _scanOp(threadScan[i], x);
                    scan = _scanOp(scan, x);
                }

                // Exclusive multi-scan for each thread's scan offset within the block.
                var localTotal = __local__.Variable(_init());
                var localScan = _multiScanExcl(tid, scan, localTotal);
                var scanOffset = _scanOp(blockScan, localScan);

                // Apply the scan offset to each exclusive scan and put the values back into the shared memory
                // they came out of.
                for (var i = 0; i < _valuesPerThread; i++)
                {
                    var x = _scanOp(threadScan[i], scanOffset);
                    warpShared[offset + i] = x;
                }

                // Store the scan back to global memory.
                for (var i = 0; i < _valuesPerThread; i++)
                {
                    var x = threadShared[i * (Const.WARP_SIZE + 1)];
                    var target = rangeX + index + i * Const.WARP_SIZE;
                    if (target < rangeY)
                        dValuesOut[target] = x;
                }

                // Grab the last element of totals_shared, which was set in MultiScan.
                // This is the total for all the values encountered in this pass.
                blockScan = _scanOp(blockScan, localTotal.Value);

                rangeX += _numValues;
            }
        }

        public void Downsweep(LaunchParam lp, deviceptr<T> dValuesIn, deviceptr<T> dValuesOut, deviceptr<T> dRangeTotals,
            deviceptr<int> dRanges, int inclusive)
        {
            GPULaunch(Downsweep, lp, dValuesIn, dValuesOut, dRangeTotals, dRanges, inclusive);
        }
        //[/GenericScanDownsweepKernel]

        public void Apply(int n, deviceptr<T> input, deviceptr<T> output, bool inclusive)
        {
            var numSm = GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT;
            var tup = Plan.BlockRanges(numSm, n);
            var ranges = tup.Item1;
            var numRanges = tup.Item2;

            var lpUpsweep = new LaunchParam(numRanges, Plan.NumThreads);
            var lpReduce = new LaunchParam(1, Plan.NumThreadsReduction);
            var lpDownsweep = new LaunchParam(numRanges, Plan.NumThreads);
            var _inclusive = inclusive ? 1 : 0;

            using (var dRanges = GPUWorker.Malloc(ranges))
            using (var dRangeTotals = GPUWorker.Malloc<T>(numRanges + 1))
            {
                _reduceModule.Upsweep(lpUpsweep, input, dRanges.Ptr, dRangeTotals.Ptr);
                GPULaunch(ScanReduce, lpReduce, numRanges, dRangeTotals.Ptr);
                GPULaunch(Downsweep, lpDownsweep, input, output, dRangeTotals.Ptr, dRanges.Ptr, _inclusive);
            }
        }

        public T[] Apply(T[] input, bool inclusive)
        {
            var n = input.Length;
            using(var dInput = GPUWorker.Malloc(input))
            using (var dOutput = GPUWorker.Malloc<T>(n))
            {
                Apply(n, dInput.Ptr, dOutput.Ptr, inclusive);
                return dOutput.Gather();
            }
        }
    }
}
