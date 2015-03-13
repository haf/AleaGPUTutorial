(*** hide ***)
module Tutorial.Fs.examples.cublas.Gemm

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.CULib
open NUnit.Framework
open NUnit.Framework.Constraints
open FsUnit

(**
Matrix multiplication with `<t>gemm`. 
*)

let cublas = CUBLAS.Default

(*** define:gemmCpu ***)
module cpu =     
    let dgemm transa transb m n k alpha (A:float[]) lda (B:float[]) ldb beta (C:float[]) ldc =
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

    let zgemm transa transb m n k (alpha:double2) (A:double2[]) lda (B:double2[]) ldb (beta:double2) (C:double2[]) ldc =
        let A = double2.ToComplex(A) |> Array2D.ofArrayColumnMajor lda k 
        let B = double2.ToComplex(B) |> Array2D.ofArrayColumnMajor ldb n 
        let C = double2.ToComplex(C) |> Array2D.ofArrayColumnMajor ldc n
        for j in [1..n] do
            for i in [1..m] do
                C.[i - 1,j - 1] <- beta.ToComplex() * C.[i - 1,j - 1]
            for l in [1..k] do
                let temp = alpha.ToComplex() * B.[l - 1,j - 1]
                for i in [1..m] do
                    C.[i - 1,j - 1] <- C.[i - 1,j - 1] + temp * A.[i - 1,l - 1]
        C |> Array2D.toArrayColumnMajor |> double2.OfComplex
        
(*** define:gemmGpu ***)
module gpu =    
    let dgemm transa transb m n k alpha (A:float[]) lda (B:float[]) ldb beta (C:float[]) ldc =
        let worker = Worker.Default
        use dalpha = worker.Malloc([|alpha|])
        use dA = worker.Malloc(A)
        use dB = worker.Malloc(B)
        use dbeta = worker.Malloc([|beta|])
        use dC = worker.Malloc(C)
        cublas.Dgemm(transa, transb, m, n, k, dalpha.Ptr, dA.Ptr, lda, dB.Ptr, ldb, dbeta.Ptr, dC.Ptr, ldc)
        dC.Gather()

    let zgemm transa transb m n k alpha (A:double2[]) lda (B:double2[]) ldb beta (C:double2[]) ldc =
        let worker = Worker.Default
        use dalpha = worker.Malloc([|alpha|])
        use dA = worker.Malloc(A)
        use dB = worker.Malloc(B)
        use dbeta = worker.Malloc([|beta|])
        use dC = worker.Malloc(C)
        cublas.Zgemm(transa, transb, m, n, k, dalpha.Ptr, dA.Ptr, lda, dB.Ptr, ldb, dbeta.Ptr, dC.Ptr, ldc)
        dC.Gather()

(*** define:gemmTest ***)
[<Test>]
let dgemmTest() =
    let gen rows cols = 
        Array.init (rows*cols) (TestUtil.genRandomDouble -5.0 5.0)
        
    let transa = cublasOperation_t.CUBLAS_OP_N
    let transb = cublasOperation_t.CUBLAS_OP_N
    let m,n,k = 5,5,5
    let lda = m // lda >= max(1,m)
    let ldb = n // ldb >= max(1,k)
    let ldc = k // ldc >= max(1,m)
    
    let alpha, beta = 1.5, 0.5
    
    let A = gen lda k
    let B = gen ldb n
    let C = gen ldc n

    let outputs = gpu.dgemm transa transb m n k alpha A lda B ldb beta C ldc
    let expected = cpu.dgemm transa transb m n k alpha A lda B ldb beta C ldc

    printfn "cpu result: %A" expected
    printfn "gpu result: %A" outputs

    outputs |> should (equalWithin 1e-12) expected 

[<Test>]
let zgemmTest() =
    let gen rows cols = 
        Array.init (rows*cols) (TestUtil.genRandomDouble2 -5.0 5.0)
        
    let transa = cublasOperation_t.CUBLAS_OP_N
    let transb = cublasOperation_t.CUBLAS_OP_N
    let m,n,k = 5,5,5
    let lda = m // lda >= max(1,m)
    let ldb = n // ldb >= max(1,k)
    let ldc = k // ldc >= max(1,m)
    
    let alpha = double2(2.0, 0.5)
    let beta = double2(1.5, 0.5)
    
    let A = gen lda k
    let B = gen ldb n
    let C = gen ldc n

    let outputs = gpu.zgemm transa transb m n k alpha A lda B ldb beta C ldc
    let expected = cpu.zgemm transa transb m n k alpha A lda B ldb beta C ldc

    (outputs, expected)
    ||> Array.iter2 (fun o e -> 
        o.x |> should (equalWithin 1e-9) e.x
        o.y |> should (equalWithin 1e-9) e.y)

    printfn "cpu result: %A" expected
    printfn "gpu result: %A" outputs
