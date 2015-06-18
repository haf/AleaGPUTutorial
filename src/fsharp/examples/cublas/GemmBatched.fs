(*** hide ***)
module Tutorial.Fs.examples.cublas.GemmBatched

open System
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.CULib
open NUnit.Framework
open NUnit.Framework.Constraints
open FsUnit

(**
Batched matrix multiplication with `<t>gemmBatched`. 
*)

let cuBLAS = CUBLAS.Default

(*** define:gemmBatchedCpu ***)
module cpu =
    let private dgemm transa transb m n k alpha (A:float[]) lda (B:float[]) ldb beta (C:float[]) ldc =
        let A = Array2D.ofArrayColumnMajor lda k A
        let B = Array2D.ofArrayColumnMajor ldb n B
        let C = Array2D.ofArrayColumnMajor ldc n C
        for j in [1..n] do
            for i in [1..m] do
                C.[i - 1,j - 1] <- beta * C.[i - 1,j - 1]
            for l in [1..k] do
                let temp = alpha * B.[l - 1,j - 1]
                for i in [1..m] do
                    C.[i - 1,j - 1] <- C.[i - 1,j - 1] + temp * A.[i - 1,l - 1]
        C |> Array2D.toArrayColumnMajor

    let dgemmBatched transa transb m n k alpha (As:float[][]) lda (Bs:float[][]) ldb beta (Cs:float[][]) ldc =
        let worker = Worker.Default
        let batchCount = As.Length
        [for i in 0..batchCount-1 ->
            dgemm transa transb m n k alpha As.[i] lda Bs.[i] ldb beta Cs.[i] ldc]
        |> Array.ofList

(*** define:gemmBatchedGpu ***)
module gpu = 
    let dgemmBatched (transa:cublasOperation_t) (transb:cublasOperation_t) m n k alpha (As:float[][]) lda (Bs:float[][]) ldb beta (Cs:float[][]) ldc =
        let worker = Worker.Default
        let batchCount = As.Length
        use dalpha = worker.Malloc([|alpha|])
        use dbeta = worker.Malloc([|beta|])
        [for i in 0..batchCount-1 ->
            use dA = worker.Malloc(As.[i])
            use dB = worker.Malloc(Bs.[i])
            use dC = worker.Malloc(Cs.[i])
            cuBLAS.Dgemm(transa, transb, m, n, k, dalpha.Ptr, dA.Ptr, lda, dB.Ptr, ldb, dbeta.Ptr, dC.Ptr, ldc)
            dC.Gather()] |> Array.ofList

(*** define:gemmBatchedTest ***)
[<Test>]
let dgemmBatchedTest() =
    Util.fallback <| fun _ ->
        let gen rows cols = 
            Array.init (rows*cols) (TestUtil.genRandomDouble -5.0 5.0)
    
        let batchCount = 10

        let transa = cublasOperation_t.CUBLAS_OP_N
        let transb = cublasOperation_t.CUBLAS_OP_N
        let m, n, k = 5, 5, 5
        let lda = m // lda >= max(1,m)
        let ldb = n // ldb >= max(1,k)
        let ldc = k // ldc >= max(1,m)
    
        let alpha, beta = 1.5, 0.5

        let As = [for i in [1..batchCount] -> gen lda k] |> Array.ofList
        let Bs = [for i in [1..batchCount] -> gen ldb n] |> Array.ofList
        let Cs = [for i in [1..batchCount] -> gen ldc n] |> Array.ofList

        let outputs = gpu.dgemmBatched transa transb m n k alpha As lda Bs ldb beta Cs ldc
        let expected = cpu.dgemmBatched transa transb m n k alpha As lda Bs ldb beta Cs ldc

        printfn "cpu result: %A" expected
        printfn "gpu result: %A" outputs

        outputs |> should (equalWithin 1e-12) expected