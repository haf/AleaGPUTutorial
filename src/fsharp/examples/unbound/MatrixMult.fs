(*** hide ***)
module Tutorial.Fs.examples.unbound.MatrixMult

open System
open Alea.CUDA.Utilities
open FsUnit

open Alea.CUDA.Unbound.LinAlg.Matrix.Multiply.GPU.Common


(**
Matrix multiplication with Alea Unbound. 

we start with a simple CPU implementation of a matrix-matrix multiplication:
$C = AB$.
*)
(*** define:unboundMatrixMultCPU ***)
module cpu =
    let inline multiplyMatrix (C:'T[]) (A:'T[]) (B:'T[]) (wA:int) (wB:int) =
        let hA = A.Length / wA
        for i = 0 to hA-1 do
            for j = 0 to wB-1 do
                let mutable (sum:'T) = 0G
                for k = 0 to wA-1 do
                    let a = A.[i*wA + k]
                    let b = B.[k*wB + j]
                    sum <- sum + a*b
                C.[i * wB + j] <- sum
        C

module gpu =
(**
Alea Unbound offers you several implementations of matrix multipications for single as well as double precision, for 
    `GEMM` (GEneral Matrix Multiplication):

$$$ 
\begin{equation}
    C = \alpha T_A(A) T_B(B) + \beta C,
\end{equation}

where $T_A$, resp. $T_B$ is the transposition operator if `atransp` resp. `btransp` set to `Transpose`, else it is the idendity.

You can access them by first creating an instance of the class `DefaultMatrixMultiplyModuleF32` resp. `DefaultMatrixMultiplyModuleF64` for double precision.

Two methods, two different implementation and different matrix storages order are available:
    
1. The first method takes the matrices as one dimensional arrays. It requires the matrix height and length beeing a multiple of the blocksize (32 for the here called defaultInstance) in the `Simple` case.
2. The second method uses 2D-arrays and makes a padding if needed (above length requirements are not needed).
    
Lets call the first method using the following **input parameters**:
    
- `implementation` might be: `Simple`, `PrefetchingData`, `AutoSelection`. Where `PrefetchingData` is an enhanced implementation hiding data fetching behind the computation. `AutoSelection` selects between the two methods according to the size of the input matrices.
- `transpA` : should Matrix $A^T$ enter into computation instead of $A$?
- `transpB` : should Matrix $B^T$ enter into computation instead of $B$?
- `majorType` : what major are matrices stored in? Options: `RowMajor`, `ColumnMajor`
- `wA` : width of matrix $A$.
- `wB` : width of matrix $B$.
- `alpha` : prefactor $\alpha$.
- `beta` : prefactor $\beta$.
- `A` : Matrix $A$, **Note:** If `Simple` or `AutoSelection` with small system sizes is used, this its size must be a multiple of Blocksize set in `MatrixMultiplyModule`, 32 in the here used default case.
- `B` : Matrix $B$, **Note:** If `Simple` or `AutoSelection` with small system sizes is used, this its size must be a multiple of Blocksize set in `MatrixMultiplyModule`, 32 in the here used default case.
- `C` : Matrix $C$, **Note:** If `Simple` or `AutoSelection` with small system sizes is used, this its size must be a multiple of Blocksize set in `MatrixMultiplyModule`, 32 in the here used default case.
*)
(*** define:unboundMatrixMultGemm1DArrayTest ***)
    let gemm1DArrayTest() =
        let GPUMultiplication64 = DefaultMatrixMultiplyModuleF64.DefaultInstance
        let wA, hA, wB, hB = 32, 64, 64, 32
        let wC, hC = 64, 64
        let A = Array.init (hA*wA) (TestUtil.genRandomDouble -5.0 5.0)
        let B = Array.init (hB*wB) (fun _ -> TestUtil.genRandomDouble -4.0 4.0 1.0)
        let C = Array.zeroCreate (hC*wC)
    
        let cpuOutput = cpu.multiplyMatrix C A B wA wB
        let gpuOutput = GPUMultiplication64.Mult(PrefetchingData, NoTranspose, NoTranspose, RowMajor, wA, wB, 1.0, 0.0, A, B, C)
        gpuOutput |> should (equalWithin 1e-12) cpuOutput
    
(**
Lets call the second method using the following **input parameters**:
    
- `implementation` might be: `Simple`, `PrefetchingData`, `AutoSelection`. Where `PrefetchingData` is an enhanced implementation hiding data fetching below computation.
- `transpA` : should Matrix $A^T$ enter into computation instead of $A$?
- `transpB` : should Matrix $B^T$ enter into computation instead of $B$?
- `majorType` : what major are matrices stored in? Options: `RowMajor`, `ColumnMajor`
- `wA` : width of matrix $A$.
- `wB` : width of matrix $B$.
- `alpha` : prefactor $\alpha$.
- `beta` : prefactor $\beta$.
- `A` : Matrix $A$
- `B` : Matrix $B$
- `C` : Matrix $C$
*)
(*** define:unboundMatrixMultGemm2DArrayTest ***)
    let gemm2DArrayTest() =
        let GPUMultiplication64 = DefaultMatrixMultiplyModuleF64.DefaultInstance
        let wA, hA, wB, hB = 31, 65, 65, 31
        let wC, hC = 65, 65
        let A = Array2D.init hA wA (fun _ _ -> TestUtil.genRandomDouble -5.0 5.0 1.0)
        let B = Array2D.init hB wB (fun _ _ -> TestUtil.genRandomDouble -4.0 4.0 1.0)
        let C = Array2D.zeroCreate hC wC
    
        let cpuOutput = cpu.multiplyMatrix (Array2D.toArrayRowMajor C) (Array2D.toArrayRowMajor A) (Array2D.toArrayRowMajor B) wA wB
        let cpuOutput = cpuOutput |> Array2D.ofArrayRowMajor hC wC
        let gpuOutput = GPUMultiplication64.Mult(PrefetchingData, NoTranspose, NoTranspose, RowMajor, 1.0, 0.0, A, B, C)
        gpuOutput |> should (equalWithin 1e-12) cpuOutput
