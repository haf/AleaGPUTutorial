using System;
using Microsoft.FSharp.Core;

namespace Tutorial.Cs.examples.generic_reduce
{
    using InitFunc = Func<Unit, dynamic>;
    using ReductionOp = Func<dynamic, dynamic, dynamic>;
    using TransformFunc = Func<dynamic, dynamic>;

    public static class ReduceApi<T>
    {
        //public static RawGeneric()
        //{
            
        //}
    }
}
