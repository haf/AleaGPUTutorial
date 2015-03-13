using System;
using System.Linq;
using System.Xml.Schema;
using Alea.CUDA;
using Alea.CUDA.IL;

namespace Tutorial.Cs.examples.matrixTranspose
{

    //[matrixTransposeModule]
    internal class MatrixTransposeModule<T> : ILGPUModule
    {
        public int TileDim { get; private set; }
        public int BlockRows { get; private set; }

        public MatrixTransposeModule(GPUModuleTarget target, int tileDim, int blockRows)
            : base(target)
        {
            TileDim = tileDim;
            BlockRows = blockRows;
        }

        public T[] Transpose(int sizeX, int sizeY, T[] idata)
        {
            var odata = new T[sizeX*sizeY];
            for (var y = 0; y < sizeY; y++)
            {
                for (var x = 0; x < sizeX; x++)
                    odata[x*sizeY + y] = idata[y*sizeX + x];
            }
            return odata;
        }

        [Kernel]
        public void CopyKernel(int width, int height, deviceptr<T> idata , deviceptr<T> odata)
        {
            var xIndex = blockIdx.x*TileDim + threadIdx.x;
            var yIndex = blockIdx.y*TileDim + threadIdx.y;        
            var index = xIndex + width*yIndex;
            for (var i = 0; i < TileDim; i += BlockRows)
            {
                odata[index + i*width] = idata[index + i*width];
            }        
        }

        [Kernel]
        public void TransposeNaiveKernel(int width, int height, deviceptr<T> idata, deviceptr<T> odata)
        {

            var xIndex = blockIdx.x*TileDim + threadIdx.x;
            var yIndex = blockIdx.y*TileDim + threadIdx.y;
            var index_in = xIndex + width*yIndex;
            var index_out = yIndex + height*xIndex;
            for (var i = 0; i < TileDim; i += BlockRows)
            {
                odata[index_out + i] = idata[index_in + i*width];
            }  
        }

        [Kernel]
        public void TransposeCoalescedKernel(int width, int height, deviceptr<T> idata, deviceptr<T> odata)
        {
            var tile = __shared__.Array<T>(TileDim*TileDim);
            var xIndex = blockIdx.x*TileDim + threadIdx.x;
            var yIndex = blockIdx.y*TileDim + threadIdx.y;
            var index_in = xIndex + (yIndex)*width;

            xIndex = blockIdx.y*TileDim + threadIdx.x;
            yIndex = blockIdx.x*TileDim + threadIdx.y;
            var index_out = xIndex + yIndex*height;

            for (var i = 0; i < TileDim; i += BlockRows)
            {
                tile[(threadIdx.y + i)*TileDim + threadIdx.x] = idata[index_in + i*width];
            }

            Intrinsic.__syncthreads();

            for (var i = 0; i < TileDim; i += BlockRows)
            {
                odata[index_out + i*height] = tile[threadIdx.x*TileDim + threadIdx.y + i];
            }
        }

        [Kernel]
        public void TransposeNoBankConflictsKernel(int width, int height, deviceptr<T> idata, deviceptr<T> odata)
        {
            var tile = __shared__.Array<T>(TileDim*(TileDim + 1));
            var xIndex = blockIdx.x*TileDim + threadIdx.x;
            var yIndex = blockIdx.y*TileDim + threadIdx.y;
            var index_in = xIndex + (yIndex)*width;

            xIndex = blockIdx.y*TileDim + threadIdx.x;
            yIndex = blockIdx.x*TileDim + threadIdx.y;
            var index_out = xIndex + yIndex*height;

            for (var i = 0; i < TileDim; i += BlockRows)
            {
                tile[(threadIdx.y + i)*(TileDim + 1) + threadIdx.x] = idata[index_in + i*width];
            }

            Intrinsic.__syncthreads();

            for (var i = 0; i < TileDim; i += BlockRows)
            {
                odata[index_out + i*height] = tile[threadIdx.x*(TileDim + 1) + threadIdx.y + i];
            }
        }

        public LaunchParam LaunchParams(int width , int height)
        {
            var threads = new dim3(TileDim, BlockRows);
            var grid = new dim3(width/TileDim, height/TileDim);
            return new LaunchParam(grid, threads);
        }

        public void Copy(int width, int height, deviceptr<T> idata, deviceptr<T> odata)
        {
            var lp = LaunchParams(width, height);
            GPULaunch(CopyKernel, lp, width, height, idata, odata);
        }

        public void TransposeNaive(int width, int height, deviceptr<T> idata, deviceptr<T> odata)
        {
            var lp = LaunchParams(width, height);
            GPULaunch(TransposeNaiveKernel, lp, width, height, idata, odata);
        }

        public void TransposeCoalesced(int width, int height, deviceptr<T> idata, deviceptr<T> odata)
        {
            var lp = LaunchParams(width, height);
            GPULaunch(TransposeCoalescedKernel, lp, width, height, idata, odata);
        }

        public void TransposeNoBankConflicts(int width, int height, deviceptr<T> idata, deviceptr<T> odata)
        {
            var lp = LaunchParams(width, height);
            GPULaunch(TransposeNoBankConflictsKernel, lp, width, height, idata, odata);
        }

        public double MemoryBandwidth(int memSize, double kernelTimeMs)
        {
            return 2.0 * 1000.0 * (double)memSize/(1024.0*1024.0*1024.0)/kernelTimeMs;
        } 
        
        public void Profile(int sizeX, int sizeY, Func<int, T[]> generateTestData)
        {
            if (sizeX%TileDim != 0 || sizeY%TileDim != 0 || sizeX != sizeY)
                throw new Exception("matrix sizeX and sizeY must be equal and a multiple of tile dimension");

            var size = sizeX*sizeY;
            var A = generateTestData(size); 

            using (var dA = GPUWorker.Malloc(A))
            using (var dAt = GPUWorker.Malloc<T>(size))
            {
                GPUWorker.ProfilerStart();

                Copy(sizeX, sizeY, dA.Ptr, dAt.Ptr);
                TransposeNaive(sizeX, sizeY, dA.Ptr, dAt.Ptr);
                TransposeCoalesced(sizeX, sizeY, dA.Ptr, dAt.Ptr);
                TransposeNoBankConflicts(sizeX, sizeY, dA.Ptr, dAt.Ptr);

                GPUWorker.Synchronize();
                GPUWorker.ProfilerStop();
            }
        }

        public void MeasurePerformance(int nIter, int sizeX, int sizeY, Func<int, T[]> generateTestData,
            Action<T[], T[]> validate)
        {
            if (sizeX % TileDim != 0 || sizeY % TileDim != 0 || sizeX != sizeY)
                throw new Exception("matrix sizeX and sizeY must be equal and a multiple of tile dimension");

            Console.WriteLine("Matrix Transpose Using CUDA - starting...");
            Console.WriteLine("GPU Device {0}: {1} with compute capability {2}.{3}\n",
                GPUWorker.Device.ID, GPUWorker.Device.Name,
                GPUWorker.Device.Arch.Major, GPUWorker.Device.Arch.Minor);
            Console.WriteLine("Matrix({0},{1})\n", sizeY, sizeX);

            var size = sizeX*sizeY;
            var esize = typeof(T) == typeof(double) ? 8 : 4; //sizeof<T> (T); // Bugbug Fix this
            var memSize = esize*size;
            var A = generateTestData(size);
            var At = Transpose(sizeX, sizeY, A);

            using (var dA = GPUWorker.Malloc(A))
            using (var dAt = GPUWorker.Malloc<T>(size))
            {
                // warm up and validate
                Copy(sizeX, sizeY, dA.Ptr, dAt.Ptr);
                TransposeNaive(sizeX, sizeY, dA.Ptr, dAt.Ptr);
                validate(dAt.Gather(), At);
                TransposeCoalesced(sizeX, sizeY, dA.Ptr, dAt.Ptr);
                validate(dAt.Gather(), At);
                TransposeNoBankConflicts(sizeX, sizeY, dA.Ptr, dAt.Ptr);
                validate(dAt.Gather(), At);

                using (var startEvent = GPUWorker.CreateEvent())
                using (var stopEvent = GPUWorker.CreateEvent())
                {
                    double time;
                    startEvent.Record();
                    for (var i = 0; i < nIter; i++)
                        Copy(sizeX, sizeY, dA.Ptr, dAt.Ptr);
                    stopEvent.Record();
                    stopEvent.Synchronize();
                    time = Event.ElapsedMilliseconds(startEvent, stopEvent) / nIter;
                    Console.WriteLine(
                        "copy\nthroughput = {0} Gb/s\nkernel time = {1} ms\nnum elements = {2}\nelement size = {3}\n",
                        MemoryBandwidth(memSize, time), time, (memSize/esize), esize);

                    startEvent.Record();
                    for (var i = 0; i < nIter; i++)
                        TransposeNaive(sizeX, sizeY, dA.Ptr, dAt.Ptr);
                    stopEvent.Record();
                    stopEvent.Synchronize();
                    time = Event.ElapsedMilliseconds(startEvent, stopEvent) / nIter;
                    Console.WriteLine(
                        "naive transpose\nthroughput = {0} Gb/s\nkernel time = {1} ms\nnum elements = {2}\nelement size = {3}\n",
                        MemoryBandwidth(memSize, time), time, (memSize/esize), esize);

                    startEvent.Record();
                    for (var i = 0; i < nIter; i++)
                        TransposeCoalesced(sizeX, sizeY, dA.Ptr, dAt.Ptr);
                    stopEvent.Record();
                    stopEvent.Synchronize();
                    time = Event.ElapsedMilliseconds(startEvent, stopEvent) / nIter;
                    Console.WriteLine(
                        "coalesced transpose\nthroughput = {0} Gb/s\nkernel time = {1} ms\nnum elements = {2}\nelement size = {3}\n",
                        MemoryBandwidth(memSize, time), time, (memSize/esize), esize);

                    startEvent.Record();
                    for (var i = 0; i < nIter; i++)
                        TransposeNoBankConflicts(sizeX, sizeY, dA.Ptr, dAt.Ptr);
                    stopEvent.Record();
                    stopEvent.Synchronize();
                    time = Event.ElapsedMilliseconds(startEvent, stopEvent) / nIter;
                    Console.WriteLine(
                        "coalesced no bank conflict transpose\nthroughput = {0} Gb/s\nkernel time = {1} ms\nnum elements = {2}\nelement size = {3}\n",
                        MemoryBandwidth(memSize, time), time, (memSize/esize), esize);
                }
            }
        }
    }
    //[/matrixTransposeModule]

    //[matrixTransposeAOT]
    [AOTCompile]
    class MatrixTransposeF32 : MatrixTransposeModule<float>
    {
        public MatrixTransposeF32(GPUModuleTarget target)
            : base(target, 32, 8)
        {
        }

        private static MatrixTransposeF32 Instance = null;
        public static MatrixTransposeF32 DefaultInstance
        {
            get { return Instance ?? (Instance = new MatrixTransposeF32(GPUModuleTarget.DefaultWorker)); }
        }
    }

    [AOTCompile]
    class MatrixTransposeF64 : MatrixTransposeModule<double>
    {
        public MatrixTransposeF64(GPUModuleTarget target)
            : base(target, 32, 8)
        {
        }

        private static MatrixTransposeF64 Instance = null;
        public static MatrixTransposeF64 DefaultInstance
        {
            get { return Instance ?? (Instance = new MatrixTransposeF64(GPUModuleTarget.DefaultWorker)); }
        }
    }
    //[/matrixTransposeAOT]

    //[matrixTransposePerformance]
    public class Test
    {

        public static float[] CreateF32(int n)
        {
            return Enumerable.Range(0, n).Select(i => (float) i).ToArray();
        }

        public static void ValidateF32(float[] a, float[] b)
        {
            var err = a.Zip(b, (ai, bi) => Math.Abs(ai - bi)).Max();
            if (err > 1e-8) throw new Exception(String.Format("failed with error {0}", err));
        }

        public static double[] CreateF64(int n)
        {
            return Enumerable.Range(0, n).Select(i => (double) i).ToArray();
        }

        public static void ValidateF64(double[] a, double[] b)
        {
            var err = a.Zip(b, (ai, bi) => Math.Abs(ai - bi)).Max();
            if (err > 1e-14) throw new Exception(String.Format("failed with error {0}", err));
        }

        public static void MatrixTransposePerformance()
        {
            var sizeX = 2560;
            var sizeY = 2560;   
            var nIter = 100;

            Console.WriteLine("Performance single precision");
            Console.WriteLine("============================");

            var matrixTransposeF32 = new MatrixTransposeF32(GPUModuleTarget.DefaultWorker);
            matrixTransposeF32.MeasurePerformance(nIter, sizeX, sizeY, CreateF32, ValidateF32);

            Console.WriteLine("");
            Console.WriteLine("Performance double precision");
            Console.WriteLine("============================");

            var matrixTransposeF64 = new MatrixTransposeF64(GPUModuleTarget.DefaultWorker);
            matrixTransposeF64.MeasurePerformance(nIter, sizeX, sizeY, CreateF64, ValidateF64);
        }
    }
    //[/matrixTransposePerformance]
}