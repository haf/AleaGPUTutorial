using Alea.CUDA;

namespace Tutorial.Cs.examples.generic_reduce
{
    public static class ReduceApi
    {

        public static double Sum(double[] values)
        {
            return
                (new ReduceModule<double>(
                    GPUModuleTarget.DefaultWorker, 
                    () => 0.0, 
                    (x, y) => x + y, 
                    x => x,
                    Plan.Plan64)
                ).Apply(values);
        }

        public static float Sum(float[] values)
        {
            return
                (new ReduceModule<float>(
                    GPUModuleTarget.DefaultWorker,
                    () => 0.0f,
                    (x, y) => x + y,
                    x => x,
                    Plan.Plan32)
                ).Apply(values);            
        }
    }
}
