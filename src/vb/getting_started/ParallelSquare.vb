'[parallelSquareImport]
Imports System
Imports Alea.CUDA
Imports Alea.CUDA.Utilities
Imports Alea.CUDA.IL
Imports Alea.CUDA.IL.WorkerExtensions
'[/parallelSquareImport]

Imports System.Linq

Namespace Tutorial.Vb.quickStart
    Public Class ParallelSquare

        '[parallelSquareCPU]
        Public Shared Function SquareCPU(inputs As Double())
            Dim outputs(inputs.Length - 1) As Double
            For i As Integer = 0 To inputs.Length - 1
                outputs(i) = inputs(i) * inputs(i)
            Next
            Return outputs
        End Function
        '[/parallelSquareCPU]

        '[parallelSquareKernel]
        <AOTCompile>
        Public Shared Sub SquareKernel(outputs As deviceptr(Of Double), inputs As deviceptr(Of Double), n As Integer)
            Dim start = blockIdx.x * blockDim.x + threadIdx.x
            Dim stride = gridDim.x * blockDim.x
            For i As Integer = start To n - 1 Step stride
                outputs(i) = inputs(i) * inputs(i)
            Next
        End Sub
        '[/parallelSquareKernel]

        '[parallelSquareLaunch]
        Public Shared Function SquareGPU(inputs As Double()) As Double()
            Dim worker = Alea.CUDA.Worker.Default
            Dim n = inputs.Length
            Using dInputs = worker.Malloc(inputs)
                Using dOutput = worker.Malloc(Of Double)(inputs.Length)
                    Const blockSize As Integer = 256
                    Dim numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT
                    Dim gridSize = Math.Min(16 * numSm, Alea.CUDA.Utilities.Common.divup(inputs.Length, blockSize))
                    Dim lp = New LaunchParam(gridSize, blockSize)
                    'worker.Launch(AddressOf SquareKernel, lp, dOutput.Ptr, dInputs.Ptr, n)
                    Return dOutput.Gather()
                End Using
            End Using
        End Function
        '[/parallelSquareLaunch]

        '[parallelSquareTest]
        Public Shared Sub SquareTest()
            Dim inputs = Enumerable.Range(0, 101).Select(Function(i) -5.0 + i * 0.1).ToArray()
            Dim outputs = SquareGPU(inputs)
            Console.WriteLine("inputs = {0}", String.Join(", ", inputs))
            Console.WriteLine("outputs = {0}", String.Join(", ", outputs))
        End Sub
        '[/parallelSquareTest]

    End Class
End Namespace




