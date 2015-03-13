module Tutorial.Fs.examples.RandomForest.Cublas

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.CULib

type internal MatrixLayout<'T> =
    | Empty of rows : int * cols : int
    | OneD of 'T[] * cols : int
    | TwoD of 'T[,]
    | Jagged of 'T[][]

type Matrix<'T> private (worker:Worker, matrixType) =
    inherit DisposableObject()

    let nRows, nCols, deviceData =
        match matrixType with
        | Empty (rows, cols) -> rows, cols, worker.Malloc(rows * cols)
        | OneD (arr, cols) -> arr.Length / cols, cols, worker.Malloc(arr)
        | TwoD arr -> arr.GetLength(0), arr.GetLength(1), worker.Malloc(arr |> Array2D.toArrayRowMajor)
        | Jagged arr ->
            let rows = arr.Length
            if rows = 0 then
                0, 0, worker.Malloc(0)
            else 
                let cols = arr.[0].Length
                if cols = 0 then
                    rows, 0, worker.Malloc(0)
                else
                    let host = arr |> Array.collect id
                    let deviceData = worker.Malloc (host)
                    rows, cols, deviceData
        
    new (worker : Worker, hostData : 'T[,]) =
        new Matrix<'T>(worker, TwoD hostData)

    new (hostData : 'T[,]) =
        new Matrix<'T>(Worker.Default, TwoD hostData)

    new (worker : Worker, nRows : int, nCols : int, hostData : 'T[]) =
        if hostData.Length <> nRows * nCols then
            failwith "host data length does not agree with nCols and nRows"
        new Matrix<'T>(worker, OneD (hostData, nCols))

    new (worker : Worker, nRows : int, nCols : int) =
        new Matrix<'T>(worker, Empty (nRows, nCols))

    new (worker : Worker, hostData : 'T[][]) =
        new Matrix<'T>(worker, Jagged hostData )

    new (hostData : 'T[][]) =
        new Matrix<'T>(Worker.Default, hostData )

    member this.DeviceData = deviceData

    member this.NumRows = nRows

    member this.NumCols = nCols

    member this.IsEmpty = nRows = 0 || nCols = 0

    member this.ToArray () = this.DeviceData.Gather ()

    member this.ToArray2D () = this.ToArray () |> Array2D.ofArrayRowMajor nRows nCols

    member this.ToJaggedArray () =
        let host = this.ToArray()
        Array.init nRows (fun rowIdx ->
            Array.sub host (rowIdx * nCols) nCols
        ) 

    member this.Gather (rowIdx, colIdx) =
        if rowIdx >= nRows then failwith "rowIdx out of bounds"
        if colIdx >= nCols then failwith "colIdx out of bounds"
        let offset = rowIdx * nCols + colIdx
        let host = Array.zeroCreate 1
        MemoryUtil.gather worker.Thread (deviceData.Ptr + offset) host 0 1
        host.[0]

    member this.Scatter(host : 'T[]) =
        if host.Length <> nRows * nCols then invalidArg "host" "length does not match nRows * nCols"
        deviceData.Scatter (host)

    override this.ToString () =
        this.ToArray2D() |> sprintf "%A"
    
    override this.Dispose(disposing) =
        if disposing then
            deviceData.Dispose()
        base.Dispose(disposing)

type CublasHelper (Cublas : CUBLAS) =

    member this.Transpose (src : Matrix<_>) = 
        let alpha = 1.0f
        let beta = 0.0f
        let nRows = src.NumRows
        let nCols = src.NumCols

        let dst = new Matrix<_>(Cublas.Worker, nCols, nRows)
        
        Cublas.Sgeam(
            cublasOperation_t.CUBLAS_OP_T,
            cublasOperation_t.CUBLAS_OP_N,
            nRows, nCols, alpha, src.DeviceData.Ptr, nCols, beta, dst.DeviceData.Ptr, nRows, dst.DeviceData.Ptr, nRows)
        dst

    member this.Transpose (src : Matrix<_>) = 
        let alpha = 1.0
        let beta = 0.0
        let nRows = src.NumRows
        let nCols = src.NumCols

        let dst = new Matrix<_>(Cublas.Worker, nCols, nRows)

        Cublas.Dgeam(
            cublasOperation_t.CUBLAS_OP_T,
            cublasOperation_t.CUBLAS_OP_N,
            nRows, nCols, alpha, src.DeviceData.Ptr, nCols, beta, dst.DeviceData.Ptr, nRows, dst.DeviceData.Ptr, nRows)
        dst

    member this.Multiply(a : Matrix<_>, aTrans, b : Matrix<_>, bTrans) =
        let stdRows (m : Matrix<_>) trans = if trans then m.NumCols else m.NumRows
        let stdCols (m : Matrix<_>) trans = if trans then m.NumRows else m.NumCols

        if (stdCols a aTrans) <> (stdRows b bTrans) then
            failwith "Matrix dimensions mismatch."

        if a.IsEmpty || b.IsEmpty then
            new Matrix<_>(Cublas.Worker, 0, 0)
        else
            let out = new Matrix<_>(Cublas.Worker, (stdRows a aTrans), (stdCols b bTrans))
            let alpha = 1.0f;
            let beta = 0.0f;

            let transOp flag = if flag then cublasOperation_t.CUBLAS_OP_T else cublasOperation_t.CUBLAS_OP_N

            Cublas.Sgemm(
                transOp(bTrans), transOp(aTrans), 
                (stdCols b bTrans), (stdRows a aTrans), (stdCols a aTrans), alpha, b.DeviceData.Ptr, b.NumCols,
                a.DeviceData.Ptr, a.NumCols, beta, out.DeviceData.Ptr, (stdCols b bTrans));
            out

    member this.Multiply(a : Matrix<_>, aTrans, b : Matrix<_>, bTrans) =
        let stdRows (m : Matrix<_>) trans = if trans then m.NumCols else m.NumRows
        let stdCols (m : Matrix<_>) trans = if trans then m.NumRows else m.NumCols

        if (stdCols a aTrans) <> (stdRows b bTrans) then
            failwith "Matrix dimensions mismatch."

        if a.IsEmpty || b.IsEmpty then
            new Matrix<_>(Cublas.Worker, 0, 0)
        else
            let out = new Matrix<_>(Cublas.Worker, (stdRows a aTrans), (stdCols b bTrans))
            let alpha = 1.0;
            let beta = 0.0;

            let transOp flag = if flag then cublasOperation_t.CUBLAS_OP_T else cublasOperation_t.CUBLAS_OP_N

            Cublas.Dgemm(
                transOp(bTrans), transOp(aTrans), 
                (stdCols b bTrans), (stdRows a aTrans), (stdCols a aTrans), alpha, b.DeviceData.Ptr, b.NumCols,
                a.DeviceData.Ptr, a.NumCols, beta, out.DeviceData.Ptr, (stdCols b bTrans));
            out

    member this.Multiply(a : Matrix<float>, b : Matrix<float>) =
        this.Multiply(a, false, b, false)
        
    member this.Multiply(a : Matrix<float32>, b : Matrix<float32>) =
        this.Multiply(a, false, b, false)
        

    static member Default = CublasHelper(CUBLAS.Default)