(*** hide ***)
module Tutorial.Fs.examples.cudnn.Mnist
open System
open System.IO
open Alea.CUDA
open Alea.CUDA.CULib
open Alea.CUDA.CULib.CUBLASInterop
open Alea.CUDA.CULib.CUDNNInterop
open NUnit.Framework

(**
# CUDNN

Before starting with this tutorial you will need to download the CUDNN library [here](https://developer.nvidia.com/cuDNN).
The code below is a port of the C++ MNist sample which accompanies the [library](https://developer.nvidia.com/cuDNN).

The example demonstrates how to use the CUDNN library to implement forward pass.
*)

let datadir = @"../src/csharp/examples/cudnn/data"

let getPath fname = Path.Combine(datadir, fname)

let readBinaryFile fname size = 
    let b = File.ReadAllBytes(fname)
    [for i in [0..4..b.Length-4] do yield BitConverter.ToSingle(b,i)]
    |> Seq.toArray

let loadImage fname = File.ReadAllBytes(getPath fname).[52..]

let ImageH = 28
let ImageW = 28

let FirstImage = "one_28x28.pgm"
let SecondImage = "three_28x28.pgm"
let ThirdImage = "five_28x28.pgm"

let Conv1Bin = "conv1.bin"
let Conv1BiasBin = "conv1.bias.bin"
let Conv2Bin = "conv2.bin"
let Conv2BiasBin = "conv2.bias.bin"

let Ip1Bin = "ip1.bin"
let Ip1BiasBin = "ip1.bias.bin"
let Ip2Bin = "ip2.bin"
let Ip2BiasBin = "ip2.bias.bin"


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


type nchw_t = 
    {mutable N:int; mutable C:int; mutable H:int; mutable W:int}
    member x.set n c h w = x.N <- n; x.C <- c; x.H <- h; x.W <- w
    static member create n c h w = {N = n; C = c; H = h; W = w}


type Network(worker:Worker) =
    inherit DisposableObject()

    let DataType = CUDNNInterop.cudnnDataType_t.CUDNN_DATA_FLOAT;
    let TensorFormat = CUDNNInterop.cudnnTensorFormat_t.CUDNN_TENSOR_NCHW;
    
    let cudnn = new CUDNN(worker)
    let cublas = new CUBLAS(worker)
    
    let srcTensorDesc = new CUDNNTensorDescriptor()
    let dstTensorDesc = new CUDNNTensorDescriptor()
    let biasTensorDesc = new CUDNNTensorDescriptor()
    let filterDesc = new CUDNNFilterDescriptor()
    let convDesc = new CUDNNConvolutionDescriptor()
    let poolingDesc = new CUDNNPoolingDescriptor()
    
    member net.Resize(buffer:DeviceMemory<float32> ref, length:int) =
        if   buffer.contents.Length >= length
        then ()
        else buffer.contents.Dispose(); buffer := worker.Malloc<float32>(length)

    member net.AddBias(dstTensorDesc:CUDNNTensorDescriptor, layer:Layer, c:int, data:DeviceMemory<float32> ref) =
        biasTensorDesc.Set4D(TensorFormat, DataType, 1, c, 1, 1)
        let alpha, beta = 1.f, 1.f
        cudnn.AddTensor(CUDNNInterop.cudnnAddMode_t.CUDNN_ADD_SAME_C, alpha, biasTensorDesc, layer.BiasD.Ptr, beta, dstTensorDesc, data.contents.Ptr)

    member net.FullyConnectedForward(ip:Layer, nchw:nchw_t, srcData:DeviceMemory<float32>, dstData:DeviceMemory<float32> ref) =
        if nchw.N <> 1 then failwith "Not Implemented"
        let dimX = nchw.C * nchw.H * nchw.W
        let dimY = ip.Outputs
        net.Resize(dstData, dimY)

        let alpha, beta = 1.f, 1.f
        // this cuMemcpyDtoD is a raw CUDA API call so it should be guarded with worker.eval
        worker.Eval <| fun _ -> CUDAInterop.cuMemcpyDtoD(dstData.contents.Ptr.Handle, ip.BiasD.Handle, IntPtr(dimY * sizeof<float32>)) |> ignore
        // this doesn't need worker.eval, cause cudnn is a thin wrapper, it has worke.eval 
        cublas.Sgemv(CUBLASInterop.cublasOperation_t.CUBLAS_OP_T, dimX, dimY, alpha, ip.DataD.Ptr, dimX, srcData.Ptr, 1, beta, dstData.contents.Ptr, 1)
        nchw.H <- 1; nchw.W <- 1; nchw.C <- dimY

    member net.ConvoluteForward(conv:Layer, nchw:nchw_t, srcData:DeviceMemory<float32>, dstData:DeviceMemory<float32> ref) =
        srcTensorDesc.Set4D(TensorFormat, DataType, nchw.N, nchw.C, nchw.H, nchw.W)
        filterDesc.Set4D(DataType, conv.Outputs, conv.Inputs, conv.KernelDim, conv.KernelDim)
        convDesc.Set2D(0, 0, 1, 1, 1, 1, CUDNNInterop.cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION)
        // find dimension of convoltion output
        // outputDim = 1 + (inputDim + 2*pad - filterDim) / convolutionStride
        let n,c,h,w = convDesc.Get2DForwardOutputDim(srcTensorDesc, filterDesc)
        nchw.set n c h w
        dstTensorDesc.Set4D(TensorFormat, DataType, nchw.N, nchw.C, nchw.H, nchw.W)
        let algo = cudnn.GetConvolutionForwardAlgorithm(srcTensorDesc, filterDesc, convDesc, dstTensorDesc, CUDNNInterop.cudnnConvolutionFwdPreference_t.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, IntPtr 0)

        net.Resize(dstData, nchw.N * nchw.C * nchw.H * nchw.W)
        let sizeInBytes = cudnn.GetConvolutionForwardWorkspaceSize(srcTensorDesc, filterDesc, convDesc, dstTensorDesc, algo)
        // I changed the malloc, so it can support malloc(0), where malloc(0).Ptr = 0n
        use workSpace = worker.Malloc<byte>(sizeInBytes.ToInt32())
        let alpha, beta = 1.f, 0.f
        cudnn.ConvolutionForward(alpha, srcTensorDesc, srcData.Ptr, filterDesc, conv.DataD.Ptr, convDesc, algo, workSpace.Ptr, sizeInBytes, beta, dstTensorDesc, dstData.contents.Ptr)
        net.AddBias(dstTensorDesc, conv, nchw.C, dstData)
        
    member net.PoolForward(nchw:nchw_t, srcData:DeviceMemory<float32>, dstData:DeviceMemory<float32> ref) =
        poolingDesc.Set2D(CUDNNInterop.cudnnPoolingMode_t.CUDNN_POOLING_MAX, 2, 2, 0, 0, 2, 2)
        srcTensorDesc.Set4D(TensorFormat, DataType, nchw.N, nchw.C, nchw.H, nchw.W)
        nchw.H <- nchw.H / 2
        nchw.W <- nchw.W / 2
        dstTensorDesc.Set4D(TensorFormat, DataType, nchw.N, nchw.C, nchw.H, nchw.W)
        net.Resize(dstData, nchw.N * nchw.C * nchw.H * nchw.W)
        let alpha, beta = 1.f, 0.f
        cudnn.PoolingForward(poolingDesc, alpha, srcTensorDesc, srcData.Ptr, beta, dstTensorDesc, dstData.Value.Ptr)

    member net.SoftmaxForward(nchw:nchw_t, srcData:DeviceMemory<float32>, dstData:DeviceMemory<float32> ref) =
        net.Resize(dstData, nchw.N * nchw.C * nchw.H * nchw.W)
        srcTensorDesc.Set4D(TensorFormat, DataType, nchw.N, nchw.C, nchw.H, nchw.W)
        dstTensorDesc.Set4D(TensorFormat, DataType, nchw.N, nchw.C, nchw.H, nchw.W)
        let alpha, beta = 1.f, 0.f
        cudnn.SoftmaxForward(CUDNNInterop.cudnnSoftmaxAlgorithm_t.CUDNN_SOFTMAX_ACCURATE, CUDNNInterop.cudnnSoftmaxMode_t.CUDNN_SOFTMAX_MODE_CHANNEL, alpha, srcTensorDesc, srcData.Ptr, beta, dstTensorDesc, dstData.contents.Ptr)

    member net.ActivationForward(nchw:nchw_t, srcData:DeviceMemory<float32>, dstData:DeviceMemory<float32> ref) =
        net.Resize(dstData, nchw.N * nchw.C * nchw.H * nchw.W)
        srcTensorDesc.Set4D(TensorFormat, DataType, nchw.N, nchw.C, nchw.H, nchw.W)
        dstTensorDesc.Set4D(TensorFormat, DataType, nchw.N, nchw.C, nchw.H, nchw.W)
        let alpha, beta = 1.f, 0.f
        cudnn.ActivationForward(CUDNNInterop.cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, alpha, srcTensorDesc, srcData.Ptr, beta, dstTensorDesc, dstData.contents.Ptr)

    
    override net.Dispose(disposing:bool) =
        if disposing then
            cudnn.Dispose()
            cublas.Dispose()
            srcTensorDesc.Dispose()
            dstTensorDesc.Dispose()
            biasTensorDesc.Dispose()
            filterDesc.Dispose()
            convDesc.Dispose()
            poolingDesc.Dispose()

    member net.ClassifyExample fname conv1 conv2 ip1 ip2 =
        let nchw = nchw_t.create 1 1 ImageH ImageW
        let imgDataH = Array.zeroCreate<float32> (ImageH * ImageW)
        let oHostSrc = fname |> loadImage |> Array.map (float32)
        for i = 0 to ImageH - 1 do
            for j = 0 to ImageW - 1 do
                let idx = ImageH*i+j
                imgDataH.[idx] <- oHostSrc.[idx] / 255.0f

        use srcData = worker.Malloc(imgDataH)
        use dstData = worker.Malloc<float32>(0)
        
        let src = ref srcData
        let dst = ref dstData
        
        printfn "Performing forward propigation..."
        
        net.ConvoluteForward(conv1, nchw, !src, dst)
        net.PoolForward(nchw, !dst, src)
        
        net.ConvoluteForward(conv2, nchw, !src, dst)
        net.PoolForward(nchw, !dst, src)
        
        net.FullyConnectedForward(ip1, nchw, !src, dst)
        net.ActivationForward(nchw, !dst, src)

        net.FullyConnectedForward(ip2, nchw, !src, dst)
        net.SoftmaxForward(nchw, !dst, src)
        
        printfn "Finished forward propigation."
        
        let maxDigits = 10;
        let result = src.contents.Gather().[0..maxDigits-1]
        let mutable id = 0
        for i = 1 to maxDigits - 1 do if result.[id] < result.[i] then id <- i
        printfn "Classification Complete.\n"
        id


let [<Test>] test() =
    use worker = Worker.Default
    use network = new Network(worker)
    let conv1, conv2 = Layer.conv1 worker, Layer.conv2 worker
    let ip1, ip2 = Layer.ip1 worker, Layer.ip2 worker
    
    printfn "Classifying...."
    let i1, i2, i3 =
        network.ClassifyExample FirstImage conv1 conv2 ip1 ip2,
        network.ClassifyExample SecondImage conv1 conv2 ip1 ip2,
        network.ClassifyExample ThirdImage conv1 conv2 ip1 ip2

    printfn "\n==========================================================\n"
    printfn "Result of Classification: %A, %A, %A" i1 i2 i3
    if i1 <> 1 || i2 <> 3 || i3 <> 5
    then printfn "Test Failed!!"
    else printfn "Test Passed!!"
    printfn "\n==========================================================\n"
    
    Assert.AreEqual(i1, 1)
    Assert.AreEqual(i2, 3)
    Assert.AreEqual(i3, 5)