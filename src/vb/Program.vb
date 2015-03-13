Module Program

    Sub Help()
        Console.WriteLine("Tutorial.Fs.exe [name]")
        Console.WriteLine("    name = [")
        Console.WriteLine("        QuickStartSquareTest              |")
        Console.WriteLine("        ExampleBasicSinTest               |")
        Console.WriteLine("        ExamplesMatrixMultTest            |")
        Console.WriteLine("        ExamplesMatrixTranspPerformance   |")
        Console.WriteLine("        All                                ")
    End Sub


    Sub Main(args As String())
        If args.Length > 0 Then
            Dim name = args(0).ToLower()

            Select Case name
                Case "quickstartsquaretest"
                    Tutorial.Vb.quickStart.ParallelSquare.SquareTest()
                Case "examplesmatrixmultest"
                    Tutorial.Vb.examples.matrixMultiplication.Test.Test()
                Case "examplesmatrixtranspperformance"
                    Tutorial.Vb.examples.matrixTranspose.Test.MatrixTransposePerformance()
                Case "all"
                    Tutorial.Vb.quickStart.ParallelSquare.SquareTest()
                    Tutorial.Vb.examples.basic.Test.SinTest()
                    Tutorial.Vb.examples.matrixMultiplication.Test.Test()
                    Tutorial.Vb.examples.matrixTranspose.Test.MatrixTransposePerformance()
                Case Else
                    Help()
            End Select
        Else
            Help()
        End If
    End Sub

End Module
