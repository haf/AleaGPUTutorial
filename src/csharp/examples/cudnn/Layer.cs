using Alea.CUDA;

namespace Tutorial.Cs.examples.cudnn
{

    public class Layer
    {
        public int Inputs { get; set; }
        public int Outputs { get; set; }
        public int KernelDim { get; set; }
        public float[] DataH { get; set; }
        public DeviceMemory<float> DataD { get; set; }
        public float[] BiasH { get; set; }
        public DeviceMemory<float> BiasD { get; set; }

        public void Dispose()
        {
            DataD.Dispose();
            BiasD.Dispose();
        }
        
        public static Layer CreateLayer(Worker worker, int inputs, int outputs, int kernelDim, string fnameWeights,string fnameBias)
        {
            var pathWeights = Data.GetPath(fnameWeights);
            var pathBias = Data.GetPath(fnameBias);
            var dataH = Data.ReadBinaryFile(pathWeights);
            var dataD = worker.Malloc(dataH);
            var biasH = Data.ReadBinaryFile(pathBias);
            var biasD = worker.Malloc(biasH);
            return new Layer
            {
                Inputs = inputs,
                Outputs = outputs,
                KernelDim = kernelDim,
                DataH = dataH,
                DataD = dataD,
                BiasH = biasH,
                BiasD = biasD
            };
        }

        public static Layer Conv1(Worker worker)
        {
            return CreateLayer(worker, 1, 20, 5, Data.Conv1Bin, Data.Conv1BiasBin);
        }

        public static Layer Conv2(Worker worker)
        {
            return CreateLayer(worker, 20, 50, 5, Data.Conv2Bin, Data.Conv2BiasBin);
        }

        public static Layer Ip1(Worker worker)
        {
            return CreateLayer(worker, 800, 500, 1, Data.Ip1Bin, Data.Ip1BiasBin);
        }

        public static Layer Ip2(Worker worker)
        {
            return CreateLayer(worker, 500, 10, 1, Data.Ip2Bin, Data.Ip2BiasBin);
        }
        
    }
    
}
