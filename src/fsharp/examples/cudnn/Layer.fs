(*** hide ***)
module Tutorial.Fs.examples.cudnn.Layer
open System
open System.IO
open Alea.CUDA

(*** define:CudnnMnistLayer ***)
type Layer = 
    {
        Inputs : int
        Outputs: int

        KernelDim : int

        DataH : float32[]
        DataD : DeviceMemory<float32>

        BiasH : float32[]
        BiasD : DeviceMemory<float32>
    }
        
    static member create (worker:Worker) inputs outputs kernelDim fnameWeights fnameBias =
        let weightsPath, biasPath = getPath fnameWeights, getPath fnameBias
        
        let dataH = readBinaryFile weightsPath (inputs*outputs*kernelDim*kernelDim)
        let dataD = worker.Malloc(dataH)
            
        let biasH = readBinaryFile biasPath outputs
        let biasD = worker.Malloc(biasH)
            
        { Inputs = inputs; Outputs = outputs; KernelDim = kernelDim; DataH = dataH; DataD = dataD; BiasH = biasH; BiasD = biasD }
    
    static member conv1 worker = Layer.create worker 1 20 5 Conv1Bin Conv1BiasBin
    static member conv2 worker = Layer.create worker 20 50 5 Conv2Bin Conv2BiasBin
    static member ip1 worker = Layer.create worker 800 500 1 Ip1Bin Ip1BiasBin
    static member ip2 worker = Layer.create worker 500 10 1 Ip2Bin Ip2BiasBin