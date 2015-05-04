using System;
using System.Linq;
using Alea.CUDA;
using Alea.CUDA.CULib;

namespace Tutorial.Cs.examples.cudnn
{
    //[mnistNetwork]
    public class Network : DisposableObject
    {
        private readonly Worker _worker;
        private readonly CUDNN _cudnn;
        private readonly CUBLAS _cublas;
        private readonly CUDNNTensorDescriptor _srcTensorDesc;
        private readonly CUDNNTensorDescriptor _dstTensorDesc;
        private readonly CUDNNTensorDescriptor _biasTensorDesc;
        private readonly CUDNNFilterDescriptor _filterDesc;
        private readonly CUDNNConvolutionDescriptor _convDesc;
        private readonly CUDNNPoolingDescriptor _poolingDesc;

        const CUDNNInterop.cudnnDataType_t DataType = CUDNNInterop.cudnnDataType_t.CUDNN_DATA_FLOAT;
        const CUDNNInterop.cudnnTensorFormat_t TensorFormat = CUDNNInterop.cudnnTensorFormat_t.CUDNN_TENSOR_NCHW;

        public Network(Worker worker)
        {
            _worker = worker;
            _cudnn = new CUDNN(_worker);
            _cublas = new CUBLAS(_worker);
            _srcTensorDesc = new CUDNNTensorDescriptor();
            _dstTensorDesc = new CUDNNTensorDescriptor();
            _biasTensorDesc = new CUDNNTensorDescriptor();
            _filterDesc = new CUDNNFilterDescriptor();
            _convDesc = new CUDNNConvolutionDescriptor();
            _poolingDesc = new CUDNNPoolingDescriptor();
        }

        // @Aaron: we need implement IDisposable, we have lot of unmanaged resources
        public override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _poolingDesc.Dispose();
                _convDesc.Dispose();
                _filterDesc.Dispose();
                _biasTensorDesc.Dispose();
                _dstTensorDesc.Dispose();
                _srcTensorDesc.Dispose();
                _cublas.Dispose();
                _cudnn.Dispose();
            }
            base.Dispose(disposing);
        }

        public class nchw_t
        {
            public int N { get; set; }
            public int C { get; set; }
            public int H { get; set; }
            public int W { get; set; }
        }

        public void Resize(ref DeviceMemory<float> buffer, int length)
        {
            if (buffer.Length >= length) return;
            buffer.Dispose();
            buffer = _worker.Malloc<float>(length);
        }

        public void AddBias(CUDNNTensorDescriptor dstTensorDesc, Layer layer, int c, DeviceMemory<float> data)
        {
            _biasTensorDesc.Set4D(TensorFormat, DataType, 1, c, 1, 1);
            const float alpha = 1.0f;
            const float beta = 1.0f;
            _cudnn.AddTensor(CUDNNInterop.cudnnAddMode_t.CUDNN_ADD_SAME_C, alpha, _biasTensorDesc, layer.BiasD.Ptr, beta, dstTensorDesc, data.Ptr);
        }

        public void FullyConnectedForward(Layer ip, nchw_t nchw, DeviceMemory<float> srcData, ref DeviceMemory<float> dstData)
        {

            if (nchw.N != 1) throw new Exception("Not Implemented");
            //Console.WriteLine("fcf n,c,h,w ===> {0}, {1}, {2}, {3}", nchw.N, nchw.C, nchw.H, nchw.W);
            var dimX = nchw.C * nchw.H * nchw.W;
            var dimY = ip.Outputs;
            Resize(ref dstData, dimY);

            const float alpha = 1.0f;
            const float beta = 1.0f;

            // this cuMemcpyDtoD is raw CUDA API, should be guard with worker.eval
            var output = dstData;
            _worker.EvalAction(() => CUDAInterop.cuMemcpyDtoD(output.Ptr.Handle, ip.BiasD.Handle, (IntPtr)(dimY * sizeof(float))));

            // this doesn't need worker.eval, cause cudnn is a thin wrapper, it has worke.eval 
            _cublas.Sgemv(CUBLASInterop.cublasOperation_t.CUBLAS_OP_T, dimX, dimY, alpha, ip.DataD.Ptr, dimX,
                srcData.Ptr, 1, beta, dstData.Ptr, 1);

            nchw.H = 1;
            nchw.W = 1;
            nchw.C = dimY;
        }

        public void ConvoluteForward(Layer conv, nchw_t nchw, DeviceMemory<float> srcData, ref DeviceMemory<float> dstData)
        {
            //Console.WriteLine("n,c,h,w ======> {0}, {1}, {2}, {3}", nchw.N, nchw.C, nchw.H, nchw.W);
            _srcTensorDesc.Set4D(TensorFormat, DataType, nchw.N, nchw.C, nchw.H, nchw.W);
            _filterDesc.Set4D(DataType, conv.Outputs, conv.Inputs, conv.KernelDim, conv.KernelDim);
            _convDesc.Set2D(0, 0, 1, 1, 1, 1, CUDNNInterop.cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION);
            // find dimension of convoltion output
            // outputDim = 1 + (inputDim + 2*pad - filterDim) / convolutionStride
            int n, c, h, w;
            _convDesc.Get2DForwardOutputDim(_srcTensorDesc, _filterDesc, out n, out c, out h, out w);
            nchw.N = n;
            nchw.C = c;
            nchw.H = h;
            nchw.W = w;
            _dstTensorDesc.Set4D(TensorFormat, DataType, nchw.N, nchw.C, nchw.H, nchw.W);
            var algo = _cudnn.GetConvolutionForwardAlgorithm(_srcTensorDesc, _filterDesc, _convDesc, _dstTensorDesc,
                CUDNNInterop.cudnnConvolutionFwdPreference_t.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, (IntPtr)0);

            //Console.WriteLine("n,c,h,w ======> {0}, {1}, {2}, {3}", nchw.N, nchw.C, nchw.H, nchw.W);
            Resize(ref dstData, nchw.N * nchw.C * nchw.H * nchw.W);
            var sizeInBytes = _cudnn.GetConvolutionForwardWorkspaceSize(_srcTensorDesc, _filterDesc, _convDesc, _dstTensorDesc, algo);

            // I changed the malloc, so it can support malloc(0), where malloc(0).Ptr = 0n
            using (var workSpace = _worker.Malloc<byte>(sizeInBytes.ToInt32()))
            {
                const float alpha = 1.0f;
                const float beta = 0.0f;
                _cudnn.ConvolutionForward(alpha, _srcTensorDesc, srcData.Ptr, _filterDesc, conv.DataD.Ptr, _convDesc, algo, workSpace.Ptr, sizeInBytes, beta, _dstTensorDesc, dstData.Ptr);
                AddBias(_dstTensorDesc, conv, c, dstData);
            }
        }

        public void PoolForward(nchw_t nchw, DeviceMemory<float> srcData, ref DeviceMemory<float> dstData)
        {
            _poolingDesc.Set2D(CUDNNInterop.cudnnPoolingMode_t.CUDNN_POOLING_MAX, 2, 2, 0, 0, 2, 2);
            _srcTensorDesc.Set4D(TensorFormat, DataType, nchw.N, nchw.C, nchw.H, nchw.W);
            nchw.H /= 2;
            nchw.W /= 2;
            _dstTensorDesc.Set4D(TensorFormat, DataType, nchw.N, nchw.C, nchw.H, nchw.W);

            Resize(ref dstData, nchw.N * nchw.C * nchw.H * nchw.W);
            const float alpha = 1.0f;
            const float beta = 0.0f;
            //Console.WriteLine("n,c,h,w ======> {0}, {1}, {2}, {3}", nchw.N, nchw.C, nchw.H, nchw.W);
            _cudnn.PoolingForward(_poolingDesc, alpha, _srcTensorDesc, srcData.Ptr, beta, _dstTensorDesc, dstData.Ptr);
        }

        public void SoftmaxForward(nchw_t nchw, DeviceMemory<float> srcData, ref DeviceMemory<float> dstData)
        {
            Resize(ref dstData, nchw.N * nchw.C * nchw.H * nchw.W);
            _srcTensorDesc.Set4D(TensorFormat, DataType, nchw.N, nchw.C, nchw.H, nchw.W);
            _dstTensorDesc.Set4D(TensorFormat, DataType, nchw.N, nchw.C, nchw.H, nchw.W);
            const float alpha = 1.0f;
            const float beta = 0.0f;
            _cudnn.SoftmaxForward(CUDNNInterop.cudnnSoftmaxAlgorithm_t.CUDNN_SOFTMAX_ACCURATE, CUDNNInterop.cudnnSoftmaxMode_t.CUDNN_SOFTMAX_MODE_CHANNEL, alpha, _srcTensorDesc, srcData.Ptr, beta, _dstTensorDesc, dstData.Ptr);
        }

        public void ActivationForward(nchw_t nchw, DeviceMemory<float> srcData, ref DeviceMemory<float> dstData)
        {
            Resize(ref dstData, nchw.N * nchw.C * nchw.H * nchw.W);
            _srcTensorDesc.Set4D(TensorFormat, DataType, nchw.N, nchw.C, nchw.H, nchw.W);
            _dstTensorDesc.Set4D(TensorFormat, DataType, nchw.N, nchw.C, nchw.H, nchw.W);
            const float alpha = 1.0f;
            const float beta = 0.0f;
            _cudnn.ActivationForward(CUDNNInterop.cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, alpha, _srcTensorDesc, srcData.Ptr, beta, _dstTensorDesc, dstData.Ptr);
        }

        public int ClassifyExample(string fname, Layer conv1, Layer conv2, Layer ip1, Layer ip2)
        {
            var nchw = new nchw_t()
            {
                N = 1,
                C = 1,
                H = Data.ImageH,
                W = Data.ImageW
            };

            var imgDataH = new float[Data.ImageH * Data.ImageW];
            var oHostSrc = Data.LoadImage(fname).Select(x => (int)x).ToArray();
            for (var i = 0; i < Data.ImageH; i++)
            {
                for (var j = 0; j < Data.ImageW; j++)
                {
                    var idx = Data.ImageH * i + j;
                    imgDataH[idx] = oHostSrc[idx] / 255.0f;
                }
            }

            using (var srcData = _worker.Malloc(imgDataH))
            using (var dstData = _worker.Malloc<float>(0))
            {
                Console.WriteLine("Performing forward propigation...");
                var src = srcData;
                var dst = dstData;

                ConvoluteForward(conv1, nchw, src, ref dst);
                PoolForward(nchw, dst, ref src);

                ConvoluteForward(conv2, nchw, src, ref dst);
                PoolForward(nchw, dst, ref src);

                FullyConnectedForward(ip1, nchw, src, ref dst);
                ActivationForward(nchw, dst, ref src);

                FullyConnectedForward(ip2, nchw, src, ref dst);
                SoftmaxForward(nchw, dst, ref src);

                Console.WriteLine("Finished forward propigation.");
                var maxDigits = 10;
                var hsrc = src.Gather();
                var result = hsrc.Take(maxDigits).ToArray();
                var id = 0;
                for (var i = 1; i < maxDigits; i++)
                    if (result[id] < result[i])
                        id = i;
                Console.WriteLine("Classification Complete.\n");
                return id;
            }

        }

    }
    //[/mnistNetwork]
}
