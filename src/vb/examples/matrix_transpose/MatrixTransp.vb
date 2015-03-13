Imports System
Imports Alea.CUDA
Imports Alea.CUDA.IL
Imports Tutorial.Vb.Tutorial.Vb.examples.matrixMultiplication

Namespace Tutorial.Vb.examples.matrixTranspose

    Public Delegate Function GenerateTestData(n As Integer) As Double()
    Public Delegate Sub Validate(x As Double(), y As Double())

    '[matrixTransposeModule]
    Public Class MatrixTransposeModule
        Inherits ILGPUModule

        Private tileDim As Integer
        Private blockRows As Integer

        Sub New(target As GPUModuleTarget, td As Integer, br As Integer)
            MyBase.New(target)
            tileDim = td
            blockRows = br
        End Sub

        Public Sub Transpose(A As Double(), B As Double(), sizeX As Integer, sizeY As Integer)
            For y As Integer = 0 To sizeY - 1
                For x As Integer = 0 To sizeX - 1
                    B(x * sizeY + y) = A(y * sizeX + x)
                Next
            Next
        End Sub

        <Kernel>
        Public Sub CopyKernel(width As Integer, height As Integer, idata As deviceptr(Of Double), odata As deviceptr(Of Double))
            Dim xIndex = blockIdx.x * tileDim + threadIdx.x
            Dim yIndex = blockIdx.y * tileDim + threadIdx.y

            Dim index = xIndex + width * yIndex
            For i As Integer = 0 To tileDim - 1 Step blockRows
                odata(index + i * width) = idata(index + i * width)
            Next
        End Sub

        <Kernel>
        Public Sub TransposeNaiveKernel(width As Integer, height As Integer, idata As deviceptr(Of Double), odata As deviceptr(Of Double))
            Dim xIndex = blockIdx.x * tileDim + threadIdx.x
            Dim yIndex = blockIdx.y * tileDim + threadIdx.y
            Dim index_in = xIndex + width * yIndex
            Dim index_out = yIndex + height * xIndex

            For i As Integer = 0 To tileDim - 1 Step blockRows
                odata(index_out + i) = idata(index_in + i * width)
            Next
        End Sub

        <Kernel>
        Public Sub TransposeCoalescedKernel(width As Integer, height As Integer, idata As deviceptr(Of Double), odata As deviceptr(Of Double))
            Dim tile = __shared__.Array(Of Double)(tileDim * tileDim)
            Dim xIndex = blockIdx.x * tileDim + threadIdx.x
            Dim yIndex = blockIdx.y * tileDim + threadIdx.y
            Dim index_in = xIndex + (yIndex) * width
            xIndex = blockIdx.y * tileDim + threadIdx.x
            yIndex = blockIdx.x * tileDim + threadIdx.y
            Dim index_out = xIndex + yIndex * height

            For i As Integer = 0 To tileDim - 1 Step blockRows
                tile((threadIdx.y + i) * tileDim + threadIdx.x) = idata(index_in + i * width)
            Next

            Intrinsic.__syncthreads()

            For i As Integer = 0 To tileDim - 1 Step blockRows
                odata(index_out + i * height) = tile(threadIdx.x * tileDim + threadIdx.y + i)
            Next
        End Sub

        <Kernel>
        Public Sub TransposeNoBankConflictsKernel(width As Integer, height As Integer, idata As deviceptr(Of Double), odata As deviceptr(Of Double))
            Dim tile = __shared__.Array(Of Double)(tileDim * (tileDim + 1))
            Dim xIndex = blockIdx.x * tileDim + threadIdx.x
            Dim yIndex = blockIdx.y * tileDim + threadIdx.y
            Dim index_in = xIndex + yIndex * width
            xIndex = blockIdx.y * tileDim + threadIdx.x
            yIndex = blockIdx.x * tileDim + threadIdx.y
            Dim index_out = xIndex + yIndex * height

            For i As Integer = 0 To tileDim - 1 Step blockRows
                tile((threadIdx.y + i) * (tileDim + 1) + threadIdx.x) = idata(index_in + i * width)
            Next

            Intrinsic.__syncthreads()

            For i As Integer = 0 To tileDim - 1 Step blockRows
                odata(index_out + i * height) = tile(threadIdx.x * (tileDim + 1) + threadIdx.y + i)
            Next
        End Sub

        Public Function LaunchParams(width As Integer, height As Integer) As LaunchParam
            Dim threads = New dim3(tileDim, blockRows)
            Dim grid = New dim3(width / tileDim, height / tileDim)
            Return New LaunchParam(grid, threads)
        End Function

        Public Sub Copy(width As Integer, height As Integer, idata As deviceptr(Of Double), odata As deviceptr(Of Double))
            Dim lp = LaunchParams(width, height)
            GPULaunch(AddressOf CopyKernel, lp, width, height, idata, odata)
        End Sub

        Public Sub TransposeNaive(width As Integer, height As Integer, idata As deviceptr(Of Double), odata As deviceptr(Of Double))
            Dim lp = LaunchParams(width, height)
            GPULaunch(AddressOf TransposeNaiveKernel, lp, width, height, idata, odata)
        End Sub

        Public Sub TransposeCoalesced(width As Integer, height As Integer, idata As deviceptr(Of Double), odata As deviceptr(Of Double))
            Dim lp = LaunchParams(width, height)
            GPULaunch(AddressOf TransposeCoalescedKernel, lp, width, height, idata, odata)
        End Sub

        Public Sub TransposeNoBankConflicts(width As Integer, height As Integer, idata As deviceptr(Of Double), odata As deviceptr(Of Double))
            Dim lp = LaunchParams(width, height)
            GPULaunch(AddressOf TransposeNoBankConflictsKernel, lp, width, height, idata, odata)
        End Sub

        Public Function MemoryBandwidth(memSize As Double, kernelTimeMs As Double)
            Return (2.0 * 1000.0 * memSize / (1024.0 * 1024.0 * 1024.0) / kernelTimeMs)
        End Function

        Public Sub Profile(sizeX As Integer, sizeY As Integer, generateTestData As GenerateTestData, validate As Validate)
            If sizeX Mod tileDim <> 0 Or sizeY Mod tileDim <> 0 Or sizeX <> sizeY Then
                Throw New Exception("matrix sizeX and sizeY must be equal and a multiple of tile dimension")
            End If

            Dim size = sizeX * sizeY
            Const esize As Integer = 8
            Dim memSize = esize * size
            Dim A = generateTestData(size)

            Using dA = GPUWorker.Malloc(A)
                Using dAt = GPUWorker.Malloc(Of Double)(size)
                    GPUWorker.ProfilerStart()

                    Copy(sizeX, sizeY, dA.Ptr, dAt.Ptr)
                    TransposeNaive(sizeX, sizeY, dA.Ptr, dAt.Ptr)
                    TransposeCoalesced(sizeX, sizeY, dA.Ptr, dAt.Ptr)
                    TransposeNoBankConflicts(sizeX, sizeY, dA.Ptr, dAt.Ptr)

                    GPUWorker.Synchronize()
                    GPUWorker.ProfilerStop()
                End Using
            End Using
        End Sub

        Public Sub MeasurePerformance(nIter As Integer, sizeX As Integer, sizeY As Integer, generateTestData As GenerateTestData, validate As Validate)
            If sizeX Mod tileDim <> 0 Or sizeY Mod tileDim <> 0 Or sizeX <> sizeY Then
                Throw New Exception("matrix sizeX and sizeY must be equal and a multiple of tile dimension")
            End If

            Console.WriteLine("Matrix Transpose Using CUDA - starting...")
            Console.WriteLine("GPU Device {0}: {1} with compute capability {2}.{3}" & vbCrLf,
                GPUWorker.Device.ID, GPUWorker.Device.Name,
                GPUWorker.Device.Arch.Major, GPUWorker.Device.Arch.Minor)
            Console.WriteLine("Matrix({0},{1})" & vbCrLf, sizeY, sizeX)

            Dim size = sizeX * sizeY
            Const esize As Integer = 8
            Dim memSize = esize * size
            Dim A = generateTestData(size)

            Using dA = GPUWorker.Malloc(A)
                Using dAt = GPUWorker.Malloc(Of Double)(size)

                    ' warm up and validate
                    Copy(sizeX, sizeY, dA.Ptr, dAt.Ptr)
                    TransposeNaive(sizeX, sizeY, dA.Ptr, dAt.Ptr)
                    TransposeCoalesced(sizeX, sizeY, dA.Ptr, dAt.Ptr)
                    TransposeNoBankConflicts(sizeX, sizeY, dA.Ptr, dAt.Ptr)

                    Using startEvent = GPUWorker.CreateEvent()
                        Using stopEvent = GPUWorker.CreateEvent()
                            Dim time = 0.0
                            startEvent.Record()
                            For i As Integer = 0 To nIter - 1
                                Copy(sizeX, sizeY, dA.Ptr, dAt.Ptr)
                            Next
                            stopEvent.Record()
                            stopEvent.Synchronize()
                            time = [Event].ElapsedMilliseconds(startEvent, stopEvent) / nIter
                            Console.WriteLine("copy" & vbCrLf & "throughput = {0} Gb/s" & vbCrLf & "kernel time = {1} ms" & vbCrLf & "num elements = {2}" & vbCrLf & "element size = {3}" & vbCrLf,
                                MemoryBandwidth(memSize, time), time, (memSize / esize), esize)

                            startEvent.Record()
                            For i As Integer = 0 To nIter - 1
                                TransposeNaive(sizeX, sizeY, dA.Ptr, dAt.Ptr)
                            Next
                            stopEvent.Record()
                            stopEvent.Synchronize()
                            time = [Event].ElapsedMilliseconds(startEvent, stopEvent) / nIter
                            Console.WriteLine("naive transpose" & vbCrLf & "throughput = {0} Gb/s" & vbCrLf & "kernel time = {1} ms" & vbCrLf & "num elements = {2}" & vbCrLf & "element size = {3}" & vbCrLf,
                                MemoryBandwidth(memSize, time), time, (memSize / esize), esize)

                            startEvent.Record()
                            For i As Integer = 0 To nIter - 1
                                TransposeCoalesced(sizeX, sizeY, dA.Ptr, dAt.Ptr)
                            Next
                            stopEvent.Record()
                            stopEvent.Synchronize()
                            time = [Event].ElapsedMilliseconds(startEvent, stopEvent) / nIter
                            Console.WriteLine("coalesced transpose" & vbCrLf & "throughput = {0} Gb/s" & vbCrLf & "kernel time = {1} ms" & vbCrLf & "num elements = {2}" & vbCrLf & "element size = {3}" & vbCrLf,
                                MemoryBandwidth(memSize, time), time, (memSize / esize), esize)

                            startEvent.Record()
                            For i As Integer = 0 To nIter - 1
                                TransposeNoBankConflicts(sizeX, sizeY, dA.Ptr, dAt.Ptr)
                            Next
                            stopEvent.Record()
                            stopEvent.Synchronize()
                            time = [Event].ElapsedMilliseconds(startEvent, stopEvent) / nIter
                            Console.WriteLine("coalesced no bank conflict transpose" & vbCrLf & "throughput = {0} Gb/s" & vbCrLf & "kernel time = {1} ms" & vbCrLf & "num elements = {2}" & vbCrLf & "element size = {3}" & vbCrLf,
                                MemoryBandwidth(memSize, time), time, (memSize / esize), esize)
                        End Using
                    End Using
                End Using
            End Using
        End Sub
    End Class
    '[/matrixTransposeModule]

    '[matrixTransposeAOT]
    <AOTCompile>
    Public Class MatrixTransposeF64
        Inherits MatrixTransposeModule

        Sub New(target As GPUModuleTarget)
            MyBase.New(target, 32, 8)
        End Sub

        Private Shared Instance As MatrixTransposeF64

        Public Shared ReadOnly Property DefaultInstance As MatrixTransposeF64
            Get
                If IsNothing(Instance) Then
                    Instance = New MatrixTransposeF64(GPUModuleTarget.DefaultWorker)
                End If
                Return Instance
            End Get
        End Property
    End Class
    '[/matrixTransposeAOT]

    '[matrixTransposePerformance]
    Public Class Test
        Public Shared Function CreateF32(n As Integer) As Single()
            Return Enumerable.Range(0, n).Select(Function(i) Convert.ToSingle(i)).ToArray()
        End Function

        Public Shared Sub ValidateF32(a As Single(), b As Single())
            Dim err = a.Zip(b, Function(ai, bi) Math.Abs(ai - bi)).Max()
            If (err > 0.00000001) Then Throw New Exception(String.Format("failed with error {0}", err))
        End Sub

        Public Shared Function CreateF64(n As Integer) As Double()
            Return Enumerable.Range(0, n).Select(Function(i) Convert.ToDouble(i)).ToArray()
        End Function

        Public Shared Sub ValidateF64(a As Double(), b As Double())
            Dim err = a.Zip(b, Function(ai, bi) Math.Abs(ai - bi)).Max()
            If (err > 0.00000000000001) Then Throw New Exception(String.Format("failed with error {0}", err))
        End Sub

        Public Shared Sub MatrixTransposePerformance()
            Dim sizeX = 2560
            Dim sizeY = 2560
            Dim nIter = 100

            'Console.WriteLine("Performance single precision")
            'Console.WriteLine("============================")

            'Dim matrixTransposeF32 = New MatrixTransposeF32(GPUModuleTarget.DefaultWorker)
            'matrixTransposeF32.MeasurePerformance(nIter, sizeX, sizeY, CreateF32, ValidateF32)

            Console.WriteLine("")
            Console.WriteLine("Performance double precision")
            Console.WriteLine("============================")

            Dim matrixTransposeF64 = New MatrixTransposeF64(GPUModuleTarget.DefaultWorker)
            matrixTransposeF64.MeasurePerformance(nIter, sizeX, sizeY, AddressOf CreateF64, AddressOf ValidateF64)
        End Sub
    End Class
    '[/matrixTransposePerformance]

End Namespace
