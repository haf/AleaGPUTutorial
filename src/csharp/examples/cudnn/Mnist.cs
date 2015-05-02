using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Alea.CUDA;
using Alea.CUDA.CULib;

namespace CUDNNTest
{
    public class Network
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

        readonly CUDNNInterop.cudnnDataType_t _dataType = CUDNNInterop.cudnnDataType_t.CUDNN_DATA_FLOAT;
        readonly CUDNNInterop.cudnnTensorFormat_t _tensorFormat = CUDNNInterop.cudnnTensorFormat_t.CUDNN_TENSOR_NCHW;

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

        public class nchw_t
        {
            public int N { get; set; }
            public int C { get; set; }
            public int H { get; set; }
            public int W { get; set; }
        }

        public DeviceMemory<float> Resize(int size)
        {
            return _worker.Malloc<float>(size);
        }

        public void AddBias(CUDNNTensorDescriptor dstTensorDesc, Common.Layer layer, int c, DeviceMemory<float> data)
        {
            _biasTensorDesc.Set4D(_tensorFormat, _dataType, 1, c, 1, 1);
            var alpha = 1.0f;
            var beta = 1.0f;
            _cudnn.AddTensor(CUDNNInterop.cudnnAddMode_t.CUDNN_ADD_SAME_C, alpha, _biasTensorDesc, layer.BiasD.Ptr, beta, dstTensorDesc, data.Ptr);
        }

        public DeviceMemory<float> FullyConnectedForward(Common.Layer ip, nchw_t nchw, DeviceMemory<float> srcData, DeviceMemory<float> dstData)
        {
            
            if(nchw.N != 1) throw new Exception("Not Implemented");
            Console.WriteLine("fcf n,c,h,w ===> {0}, {1}, {2}, {3}", nchw.N, nchw.C, nchw.H, nchw.W);
            var dimX = nchw.C*nchw.H*nchw.W;
            var dimY = ip.Outputs;
            dstData = Resize(dimY);

            var alpha = 1.0f;
            var beta = 1.0f;

            _worker.EvalAction(() => CUDAInterop.cuMemcpyDtoD(dstData.Ptr.Handle, ip.BiasD.Handle, (IntPtr) (dimY*sizeof (float))));
            _worker.Synchronize();
            _worker.EvalAction(
                () =>
                    _cublas.Sgemv(CUBLASInterop.cublasOperation_t.CUBLAS_OP_T, dimX, dimY, alpha, ip.DataD.Ptr, dimX,
                        srcData.Ptr, 1, beta, dstData.Ptr, 1));

            nchw.H = 1;
            nchw.W = 1;
            nchw.C = dimY;
            return dstData;
        }

        public DeviceMemory<float> ConvoluteForward(Common.Layer conv, nchw_t nchw, DeviceMemory<float> srcData, DeviceMemory<float> dstData)
        {
            var n = nchw.N;
            var c = nchw.C;
            var h = nchw.H;
            var w = nchw.W;
            _srcTensorDesc.Set4D(_tensorFormat, _dataType, nchw.N, nchw.C, nchw.H, nchw.W);
            _filterDesc.Set4D(_dataType, conv.Outputs, conv.Inputs, conv.KernelDim, conv.KernelDim);
            _convDesc.Set2D(0,0,1,1,1,1, CUDNNInterop.cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION);
            // find dimension of convoltion output
            _convDesc.Get2DForwardOutputDim(_srcTensorDesc,_filterDesc, out n, out c, out h, out w);
            nchw.N = n;
            nchw.C = c;
            nchw.H = h;
            nchw.W = w;
            _dstTensorDesc.Set4D(_tensorFormat, _dataType, nchw.N, nchw.C, nchw.H, nchw.W);
            var algo = _cudnn.GetConvolutionForwardAlgorithm(_srcTensorDesc, _filterDesc, _convDesc, _dstTensorDesc,
                CUDNNInterop.cudnnConvolutionFwdPreference_t.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, (IntPtr) 0);

            dstData = Resize(nchw.N*nchw.C*nchw.H*nchw.W);
            var sizeInBytes = _cudnn.GetConvolutionForwardWorkspaceSize(_srcTensorDesc, _filterDesc, _convDesc, _dstTensorDesc, algo);

            var workSpace = sizeInBytes.ToInt32() != 0
                ? _worker.Malloc<byte>(sizeInBytes.ToInt32())
                : _worker.Malloc<byte>(1);

            var alpha = 1.0f;
            var beta = 0.0f;
            _cudnn.ConvolutionForward(alpha, _srcTensorDesc, srcData.Ptr, _filterDesc, conv.DataD.Ptr, _convDesc, algo, workSpace.Ptr, sizeInBytes, beta, _dstTensorDesc, dstData.Ptr);
            AddBias(_dstTensorDesc, conv, c, dstData);

            if(sizeInBytes.ToInt32() != 0)
                workSpace.Dispose();
            return dstData;
        }

        public DeviceMemory<float> PoolForward(nchw_t nchw, DeviceMemory<float> srcData, DeviceMemory<float> dstData)
        {
            _poolingDesc.Set2D(CUDNNInterop.cudnnPoolingMode_t.CUDNN_POOLING_MAX, 2,2,0,0,2,2);
            _srcTensorDesc.Set4D(_tensorFormat, _dataType, nchw.N, nchw.C, nchw.H, nchw.W);
            nchw.H /= 2;
            nchw.W /= 2;
            _dstTensorDesc.Set4D(_tensorFormat, _dataType, nchw.N, nchw.C, nchw.H, nchw.W);

            dstData = Resize(nchw.N*nchw.C*nchw.H*nchw.W);
            var alpha = 1.0f;
            var beta = 0.0f;
            _cudnn.PoolingForward(_poolingDesc, alpha, _srcTensorDesc, srcData.Ptr, beta, _dstTensorDesc, dstData.Ptr);
            return dstData;
        }

        public DeviceMemory<float> SoftmaxForward(nchw_t nchw, DeviceMemory<float> srcData, DeviceMemory<float> dstData)
        {
            dstData = Resize(nchw.N*nchw.C*nchw.H*nchw.W);
            _srcTensorDesc.Set4D(_tensorFormat, _dataType, nchw.N, nchw.C, nchw.H, nchw.W);
            _dstTensorDesc.Set4D(_tensorFormat, _dataType, nchw.N, nchw.C, nchw.H, nchw.W);
            var alpha = 1.0f;
            var beta = 0.0f;
            _cudnn.SoftmaxForward(CUDNNInterop.cudnnSoftmaxAlgorithm_t.CUDNN_SOFTMAX_ACCURATE, CUDNNInterop.cudnnSoftmaxMode_t.CUDNN_SOFTMAX_MODE_CHANNEL, alpha, _srcTensorDesc, srcData.Ptr, beta, _dstTensorDesc, dstData.Ptr);
            return dstData;
        }

        public DeviceMemory<float> ActivationForward(nchw_t nchw, DeviceMemory<float> srcData, DeviceMemory<float> dstData)
        {
            dstData = Resize(nchw.N*nchw.C*nchw.H*nchw.W);
            _srcTensorDesc.Set4D(_tensorFormat, _dataType, nchw.N, nchw.C, nchw.H, nchw.W);
            _dstTensorDesc.Set4D(_tensorFormat, _dataType, nchw.N, nchw.C, nchw.H, nchw.W);
            var alpha = 1.0f;
            var beta = 0.0f;
            _cudnn.ActivationForward(CUDNNInterop.cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, alpha, _srcTensorDesc, srcData.Ptr, beta, _dstTensorDesc, dstData.Ptr);
            return dstData;
        }

        public int ClassifyExample(string fname, Common.Layer conv1, Common.Layer conv2, Common.Layer ip1, Common.Layer ip2)
        {
            var nchw = new nchw_t()
            {
                N = 1,
                C = 1,
                H = Common.ImageH,
                W = Common.ImageW
            };

            var imgDataH = new float[Common.ImageH*Common.ImageW];
            var oHostSrc = Common.LoadImage(fname).Select(x => (int) x).ToArray();
            for (var i = 0; i < Common.ImageH; i++)
            {
                for (var j = 0; j < Common.ImageW; j++)
                {
                    var idx = Common.ImageH*i + j;
                    imgDataH[idx] = oHostSrc[idx]/255.0f;
                }
            }

            using(var srcData = _worker.Malloc(imgDataH))
            using (var dstData = _worker.Malloc<float>(1))
            {
                Console.WriteLine("Performing forward propigation...\n");
                var src = srcData;
                var dst = dstData;

                dst = ConvoluteForward(conv1, nchw, src, dst);
                src = PoolForward(nchw, dst, src);

                dst = ConvoluteForward(conv2, nchw, src, dst);
                src = PoolForward(nchw, dst, src);

                dst = FullyConnectedForward(ip1, nchw, src, dst);
                src = ActivationForward(nchw, dst, src);

                dst = FullyConnectedForward(ip2, nchw, src, dst);
                src = SoftmaxForward(nchw, dst, src);
                
                Console.WriteLine("Finished forward propigation.");
                var maxDigits = 10;
                var hsrc = src.Gather();
                var result = hsrc.Take(maxDigits).ToArray();
                var id = 0;
                for(var i = 1; i < maxDigits; i++)
                    if (result[id] < result[i])
                        id = i;
                Console.WriteLine("done.");
                return id;
            }

        }

    }
    
}
