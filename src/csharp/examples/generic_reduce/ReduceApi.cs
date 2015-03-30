using System;
using Alea.CUDA;
using Microsoft.FSharp.Core;

namespace Tutorial.Cs.examples.generic_reduce
{
    using InitFunc = Func<Unit, dynamic>;
    using ReductionOp = Func<dynamic, dynamic, dynamic>;
    using TransformFunc = Func<dynamic, dynamic>;

    using UpsweepKernel = Action<deviceptr<dynamic>, deviceptr<int>, deviceptr<dynamic>>;
    using Reducekernel = Action<int, deviceptr<dynamic>>;

    public static class ReduceApi<T>
    {
        //public static RawGeneric()
        //{
            
        //}
    }
}
