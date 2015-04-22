(**
The Class `Matrix` helps alllocation, scattering and gathering data on, to and from the GPU memory.
*)
module Tutorial.Fs.examples.RandomForest.Cublas

open Alea.CUDA
open Alea.CUDA.Utilities

type internal MatrixLayout<'T> =
    | Empty of rows : int * cols : int
    | Jagged of 'T[][]

type Matrix<'T> private (worker : Worker, matrixType) =
    inherit DisposableObject()

    let nRows, nCols, deviceData =
        match matrixType with
        | Empty(rows, cols) -> rows, cols, worker.Malloc(rows * cols)
        | Jagged arr ->
            let rows = arr.Length
            if rows = 0 then 0, 0, worker.Malloc(0)
            else
                let cols = arr.[0].Length
                if cols = 0 then rows, 0, worker.Malloc(0)
                else
                    let host = arr |> Array.collect id
                    let deviceData = worker.Malloc(host)
                    rows, cols, deviceData

    new(worker : Worker, nRows : int, nCols : int) = new Matrix<'T>(worker, Empty(nRows, nCols))
    new(worker : Worker, hostData : 'T[][]) = new Matrix<'T>(worker, Jagged hostData)
    new(hostData : 'T[][]) = new Matrix<'T>(Worker.Default, hostData)
    member this.DeviceData = deviceData
    member this.NumRows = nRows
    member this.NumCols = nCols
    member this.ToArray() = this.DeviceData.Gather()
    member this.ToArray2D() = this.ToArray() |> Array2D.ofArrayRowMajor nRows nCols

    member this.Gather(rowIdx, colIdx) =
        if rowIdx >= nRows then failwith "rowIdx out of bounds"
        if colIdx >= nCols then failwith "colIdx out of bounds"
        let offset = rowIdx * nCols + colIdx
        let host = Array.zeroCreate 1
        MemoryUtil.gather worker.Thread (deviceData.Ptr + offset) host 0 1
        host.[0]

    member this.Scatter(host : 'T[]) =
        if host.Length <> nRows * nCols then invalidArg "host" "length does not match nRows * nCols"
        worker.Eval <| fun _ -> deviceData.Scatter(host)

    override this.ToString() = this.ToArray2D() |> sprintf "%A"
    override this.Dispose(disposing) =
        if disposing then deviceData.Dispose()
        base.Dispose(disposing)