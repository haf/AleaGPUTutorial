using System;
using Alea.CUDA;

namespace Tutorial.Cs.examples.generic_scan
{
    using UpsweepKernel = Action<deviceptr<dynamic>, deviceptr<int>, deviceptr<dynamic>>;
    using Reducekernel = Action<int, deviceptr<dynamic>>;
    using DownsweepKernel = Action<deviceptr<dynamic>, deviceptr<dynamic>, deviceptr<int>, int>;

    public class ScanApi
    {
        private UpsweepKernel _upsweep;
        private Reducekernel _reduce;
        private DownsweepKernel _downsweep;

        public ScanApi()
        {
            
        }


    }
}
