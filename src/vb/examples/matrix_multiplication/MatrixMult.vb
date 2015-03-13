Imports System
Imports Alea.CUDA
Imports Alea.CUDA.IL

Namespace Tutorial.Vb.examples.matrixMultiplication

    '[matrixMultiplyModule]
    Public Class MatrixMultiplyModule
        Inherits ILGPUModule

        Private blockSize As Integer

        Sub New(target As GPUModuleTarget, bs As Integer)
            MyBase.New(target)
            blockSize = bs
        End Sub

        Property PropertyBlockSize() As Integer
            Get
                Return blockSize
            End Get
            Set(ByVal value As Integer)
                blockSize = value
            End Set
        End Property

        <Kernel>
        Public Sub Kernel(wA As Integer, wB As Integer, A As deviceptr(Of Double), B As deviceptr(Of Double), C As deviceptr(Of Double))
            Dim bx = blockIdx.x
            Dim by = blockIdx.y
            Dim tx = threadIdx.x
            Dim ty = threadIdx.y

            ' offset to first element of the first sub-matrix of A processed by the block
            Dim aBegin = wA * blockSize * by

            ' index of the last sub-matrix of A processed by the block
            Dim aEnd = aBegin + wA - 1

            ' step size used to iterate through the sub-matrices of A
            Dim aStep = blockSize

            ' offset to first element of the first sub-matrix of B processed by the block
            Dim bBegin = blockSize * bx

            ' step size used to iterate through the sub-matrices of B
            Dim bStep = blockSize * wB

            ' Csub is used to store the element of the block sub-matrix that is computed by the thread
            Dim Csub = 0.0

            ' loop over all the sub-matrices of A and B required to compute the block sub-matrix
            Dim ai = aBegin
            Dim bi = bBegin
            While ai <= aEnd
                Dim Ashared = __shared__.Array2D(Of Double)(blockSize, blockSize)
                Dim Bshared = __shared__.Array2D(Of Double)(blockSize, blockSize)

                ' load the matrices from device memory to shared memory; each thread loads one element of each matrix 
                Ashared(ty, tx) = A(ai + wA * ty + tx)
                Bshared(ty, tx) = B(bi + wA * ty + tx)

                Intrinsic.__syncthreads()

                ' multiply the two matrices together; each thread computes one element of the block sub-matrix
                For k As Integer = 0 To blockSize - 1
                    Csub += Ashared(ty, k) * Bshared(k, tx)
                Next

                Intrinsic.__syncthreads()

                ai = ai + aStep
                bi = bi + bStep
            End While

            ' write the block sub-matrix to device memory; each thread writes one element
            Dim ci = wB * blockSize * by + blockSize * bx
            C(ci + wB * ty + tx) = Csub
        End Sub

        Public Sub Mult(wA As Integer, wB As Integer, hC As Integer, A As deviceptr(Of Double), B As deviceptr(Of Double), C As deviceptr(Of Double))
            Dim block = New dim3(blockSize, blockSize)
            Dim grid = New dim3(wB / block.x, hC / block.y)
            Dim lp = New LaunchParam(grid, block)
            GPULaunch(AddressOf Kernel, lp, wA, wB, A, B, C)
        End Sub

        Public Function Mult(wA As Integer, wB As Integer, A As Double(), B As Double()) As Double()
            Dim wC = wB
            Dim hC = A.Length / wA
            Dim C(wC * hC - 1) As Double
            Using dA = GPUWorker.Malloc(A)
                Using dB = GPUWorker.Malloc(B)
                    Using dC = GPUWorker.Malloc(C)
                        Mult(wA, wB, hC, dA.Ptr, dB.Ptr, dC.Ptr)
                        Return dC.Gather()
                    End Using
                End Using
            End Using
        End Function

    End Class
    '[/matrixMultiplyModule]

    '[defaultMatrixMultiplyModule]
    <AOTCompile>
    Public Class DefaultMatrixMultiplyModule
        Inherits MatrixMultiplyModule

        Sub New(target As GPUModuleTarget)
            MyBase.New(target, 32)
        End Sub

        Private Shared Instance As DefaultMatrixMultiplyModule

        Public Shared ReadOnly Property DefaultInstance As DefaultMatrixMultiplyModule
            Get
                If IsNothing(Instance) Then
                    Instance = New DefaultMatrixMultiplyModule(GPUModuleTarget.DefaultWorker)
                End If
                Return Instance
            End Get
        End Property

    End Class
    '[/defaultMatrixMultiplyModule]

    '[matrixMultiplyTest]
    Public Class Test

        Public Shared Function MatrixMultiplyCPU(wA As Integer, wB As Integer, A As Double(), B As Double()) As Double()
            Dim hA = A.Length / wA
            Dim C(hA * wB - 1) As Double

            For i As Integer = 0 To hA - 1
                For j As Integer = 0 To wB - 1
                    Dim sum = 0.0F
                    For k As Integer = 0 To wA - 1
                        sum += A(i * wA + k) * B(k * wB + j)
                    Next
                    C(i * wB + j) = sum
                Next
            Next

            Return C
        End Function

        Public Shared Sub Validate(wA As Integer, wB As Integer)
            Dim sizeA = wA * wA
            Dim sizeB = wB * wB
            Dim rng = New Random()
            Dim A = Enumerable.Range(0, sizeA).Select(Function(i) rng.NextDouble()).ToArray()
            Dim B = Enumerable.Range(0, sizeB).Select(Function(i) rng.NextDouble()).ToArray()
            Dim dAB = DefaultMatrixMultiplyModule.DefaultInstance.Mult(wA, wB, A, B)
            Dim hAB = MatrixMultiplyCPU(wA, wB, A, B)

            Dim err = dAB.Zip(hAB, Function(x, y) Math.Abs(x - y)).Max()
            Console.WriteLine("dimA {0}, dimB {1}, error = {2}", wA, wB, err)
        End Sub

        Public Shared Sub Test()
            Dim dimensions = {128, 512, 1024, 2048}
            dimensions.ToList().ForEach(Sub(dimension) Validate(dimension, dimension))
        End Sub

    End Class
    '[/matrixMultiplyTest]

End Namespace
