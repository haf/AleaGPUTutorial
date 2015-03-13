using System;
using System.Linq;
using Alea.CUDA;
using Alea.CUDA.IL;
using NUnit.Framework;

namespace Tutorial.Cs.examples.matrixMultiplication
{

    //[matrixMultiplyModule]
    internal class MatrixMultiplyModule : ILGPUModule
    {
        public int BlockSize { get; private set; }

        public MatrixMultiplyModule(GPUModuleTarget target, int blockSize)
            : base(target)
        {
            BlockSize = blockSize;
        }

        [Kernel]
        public void Kernel(int wA, int wB, deviceptr<double> A, deviceptr<double> B, deviceptr<double> C)
        {
            var blx = blockIdx.x;
            var bly = blockIdx.y;
            var tx = threadIdx.x;
            var ty = threadIdx.y;

            // offset to first element of the first sub-matrix of A processed by the block
            var aBegin = wA * BlockSize * bly;

            // index of the last sub-matrix of A processed by the block
            var aEnd = aBegin + wA - 1;

            // step size used to iterate through the sub-matrices of A
            var aStep = BlockSize;

            // offset to first element of the first sub-matrix of B processed by the block
            var bBegin = BlockSize * blx;

            // step size used to iterate through the sub-matrices of B
            var bStep = BlockSize * wB;

            // Csub is used to store the element of the block sub-matrix that is computed by the thread
            var Csub = 0.0;

            // loop over all the sub-matrices of A and B required to compute the block sub-matrix
            var a = aBegin;
            var b = bBegin;
            for (; a <= aEnd; a += aStep, b += bStep)
            {
                var As = __shared__.Array2D<double>(BlockSize, BlockSize);
                var Bs = __shared__.Array2D<double>(BlockSize, BlockSize);

                // load the matrices from device memory to shared memory; each thread loads one element of each matrix 
                As[ty, tx] = A[a + wA * ty + tx];
                Bs[ty, tx] = B[b + wB * ty + tx];

                Intrinsic.__syncthreads();

                // multiply the two matrices together; each thread computes one element of the block sub-matrix
                for (var k = 0; k < BlockSize; ++k)
                    Csub += As[ty, k] * Bs[k, tx];

                Intrinsic.__syncthreads();
            }

            // write the block sub-matrix to device memory; each thread writes one element
            var c = wB * BlockSize * bly + BlockSize * blx;
            C[c + wB * ty + tx] = Csub;
        }
 
        public void Mult(int wA, int wB, int hC, deviceptr<double> A, deviceptr<double> B, deviceptr<double> C)
        {
            var block = new dim3(BlockSize, BlockSize);
            var grid = new dim3(wB/block.x, hC/block.y);
            var lp = new LaunchParam(grid, block);
            GPULaunch(Kernel, lp, wA, wB, A, B, C);
        }

        public double[] Mult(int wA, int wB, double[] A, double[] B)
        {
            var wC = wB;
            var hC = A.Length / wA;
            using (var dA = GPUWorker.Malloc(A))
            using (var dB = GPUWorker.Malloc(B))
            using (var dC = GPUWorker.Malloc<double>(wC * hC))
            {
                Mult(wA, wB, hC, dA.Ptr, dB.Ptr, dC.Ptr);
                return dC.Gather();
            }
        }
    }
    //[/matrixMultiplyModule]

    //[defaultMatrixMultiplyModule]
    [AOTCompile]
    class DefaultMatrixMultiplyModule : MatrixMultiplyModule
    {
        public DefaultMatrixMultiplyModule(GPUModuleTarget target)
            : base(target, 32)
        {
        }

        private static DefaultMatrixMultiplyModule Instance;
        public static DefaultMatrixMultiplyModule DefaultInstance
        {
            get { return Instance ?? (Instance = new DefaultMatrixMultiplyModule(GPUModuleTarget.DefaultWorker)); }
        }
    }    
    //[/defaultMatrixMultiplyModule]

    //[matrixMultiplyTest]
    public class Test 
    {
        public static double[] MatrixMultiplyCPU(int wA, int wB, double[] A, double[] B)
        {
            var hA = A.Length / wA;
            var C = new double[hA * wB];

            for (var i = 0; i < hA; ++i)
            {
                for (var j = 0; j < wB; ++j)
                {
                    var sum = 0.0;
                    for (var k = 0; k < wA; ++k)
                    {
                        sum += A[i * wA + k] * B[k * wB + j];
                    }
                    C[i * wB + j] = sum;
                }
            }

            return C;
        }

        public static void Validate(int wA, int wB)
        {
            var sizeA = wA*wA;
            var sizeB = wB*wB;
            var rng = new Random();
            var A = Enumerable.Range(0, sizeA).Select(i => rng.NextDouble()).ToArray();
            var B = Enumerable.Range(0, sizeB).Select(i => rng.NextDouble()).ToArray();
            var dAB = DefaultMatrixMultiplyModule.DefaultInstance.Mult(wA, wB, A, B);
            var hAB = MatrixMultiplyCPU(wA, wB, A, B);

            for (var i = 0; i < hAB.Length; ++i)
                Assert.AreEqual(hAB[i], dAB[i], 1e-12);
            
            var err = dAB.Zip(hAB, (x, y) => Math.Abs(x - y)).Max();
            Console.WriteLine("dimA {0}, dimB {1}, error = {2}", wA, wB, err);
        }

        [Test]
        public static void MatrixMultiplyTest()
        {
            int[] dimensions = {128, 512, 1024, 2048};
            dimensions.ToList().ForEach(dimension => Validate(dimension, dimension));
        }
    }
    //[/matrixMultiplyTest]
}