Imports System
Imports Alea.CUDA
Imports Alea.CUDA.IL
Imports Alea.CUDA.Utilities

Namespace Tutorial.Vb.examples.basic

    '[transformModule]
    Public Class TransformModule
        Inherits ILGPUModule

        Private op As Func(Of Double, Double)

        Public Sub New(worker As GPUModuleTarget, opFunc As Func(Of Double, Double))
            MyBase.New(worker)
            op = opFunc
        End Sub

        <Kernel>
        Public Sub Kernel(n As Integer, x As deviceptr(Of Double), y As deviceptr(Of Double))
            Dim start = blockIdx.x * blockDim.x + threadIdx.x
            Dim stride = gridDim.x * blockDim.x
            For i As Integer = start To n - 1 Step stride
                y(i) = op(x(i))
            Next
        End Sub

        Public Sub Apply(n As Integer, x As deviceptr(Of Double), y As deviceptr(Of Double))
            Dim blockSize = 256
            Dim numSm = GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
            Dim gridSize = Math.Min(16 * numSm, Common.divup(n, blockSize))
            Dim lp = New LaunchParam(gridSize, blockSize)
            GPULaunch(AddressOf Kernel, lp, n, x, y)
        End Sub

        Public Function Apply(x As Double()) As Double()
            Dim n = x.Length
            Using dx = GPUWorker.Malloc(x)
                Using dy = GPUWorker.Malloc(Of Double)(x.Length)
                    Apply(n, dx.Ptr, dy.Ptr)
                    Return dy.Gather()
                End Using
            End Using
        End Function

    End Class
    '[/transformModule]

    '[transformModuleSpecialized]
    <AOTCompile>
    Public Class SinModule
        Inherits TransformModule

        Sub New(target As GPUModuleTarget)
            MyBase.New(target, Function(x) Alea.CUDA.LibDevice.__nv_sin(x))
        End Sub

        Private Shared Instance As SinModule

        Public Shared ReadOnly Property DefaultInstance As SinModule
            Get
                If IsNothing(Instance) Then
                    Instance = New SinModule(GPUModuleTarget.DefaultWorker)
                End If
                Return Instance
            End Get
        End Property

    End Class
    '[/transformModuleSpecialized]

    '[transformModuleSpecializedTest]
    Public Class Test

        Public Shared Sub SinTest()
            Using sinGpu = SinModule.DefaultInstance
                Dim rng = New Random()
                Dim n = 1000
                Dim x = Enumerable.Range(0, n).Select(Function(i) rng.NextDouble()).ToArray()
                Dim dResult = sinGpu.Apply(x)
                Dim hResult = x.Select(Function(a) Math.Sin(a))
                Dim err = dResult.Zip(hResult, Function(d, h) Math.Abs(d - h)).Max()
                Console.WriteLine("error = {0}", err)
            End Using
        End Sub

    End Class
    '[/transformModuleSpecializedTest]

End Namespace
