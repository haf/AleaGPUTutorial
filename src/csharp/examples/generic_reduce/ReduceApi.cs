using System;
using Alea.CUDA;
using Alea.CUDA.IL;
using Microsoft.FSharp.Core;
using Tutorial.Fs.examples.genericReduce;

namespace Tutorial.Cs.examples.generic_reduce
{
    using InitFunc64 = Func<Unit, double>;
    using ReductionOp64 = Func<double, double, double>;
    using TransformFunc64 = Func<double, double>;

    using UpsweepKernel = Action<deviceptr<double>, deviceptr<int>, deviceptr<double>>;
    using ReduceKernel = Action<int, deviceptr<double>>;

    public static class ReduceApi<T>
    {
        //public static Plan Plan32 = new Plan()
        //{
        //    NumThreads = 1024,
        //    ValuesPerThread = 4,
        //    NumThreadsReduction = 256,
        //    BlockPerSm = 1
        //};

        public static Plan Plan64 = new Plan()
        {
            NumThreads = 512,
            ValuesPerThread = 4,
            NumThreadsReduction = 256,
            BlockPerSm = 1
        };

        public class ReduceDouble : ReduceModule<double>
        {
            private InitFunc64 _initFunc;
            private ReductionOp64 _reductionOp;
            private TransformFunc64 _transform;
            private readonly ReduceModule<double> _reduce; 
            //private Plan _plan;

            public ReduceDouble(GPUModuleTarget target, InitFunc64 initFunc, ReductionOp64 reductionOp, TransformFunc64 transform) 
                : base(target, initFunc, reductionOp, transform, Plan64)
            {
                _initFunc = initFunc;
                _reductionOp = reductionOp;
                _transform = transform;
                _reduce = new ReduceModule<double>(target, initFunc, reductionOp, transform, _plan);
            }

            public double Apply(double[] values)
            {
                var n = values.Length;
                var numSm = GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT;
                var tup = _plan.BlockRanges(numSm, n);
                var ranges = tup.Item1;
                var numRanges = tup.Item2;
                var lpUpsweep = new LaunchParam(numRanges, _plan.NumThreads);
                var lpReduce = new LaunchParam(1, _plan.NumThreadsReduction);


                using(var dValues = GPUWorker.Malloc(values))
                using(var dRanges = GPUWorker.Malloc(ranges))
                using (var dRangeTotals = GPUWorker.Malloc<double>(numRanges))
                {
                    // Launch range reduction kernel to calculate the totals per range.
                    GPUWorker.Launch(_reduce.Upsweep, lpUpsweep, dValues.Ptr, dRanges.Ptr, dRangeTotals.Ptr);
                    if (numRanges > 1)
                    {
                        GPUWorker.Launch(_reduce.ReduceRangeTotals, lpReduce, numRanges, dRangeTotals.Ptr);
                    }
                    return dRangeTotals.Gather()[0];
                }
            }


        }

        
    }
}
