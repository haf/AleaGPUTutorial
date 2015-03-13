(*** hide ***)
module Tutorial.Fs.examples.tridiagSolver.Solver

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.IL
open Alea.CUDA.Utilities
open NUnit.Framework
open FsUnit

(**
# Parallel Tridiagonal Linear System Solver 

Optimized implementation of the parallel cyclic reduction solver for 
multiple systems with dimensions smaller than the maximal number of threads per block.
A single system is solved with a thread block and each thread handles a single row 
of the system.
*)

(*** define:triDiagSolverModule ***)
[<AOTCompile>]
type TriDiagSolverModule(target) =
    inherit ILGPUModule(target)

    // core solver function
    //     n  the dimension of the tridiagonal system, must fit into one block
    //     l  lower diagonal
    //     d  diagonal
    //     u  upper diagonal
    //     h  right hand side and solution at exit
    [<ReflectedDefinition>]
    let solve n (l:deviceptr<float>) (d:deviceptr<float>) (u:deviceptr<float>) (h:deviceptr<float>) =
        let rank = threadIdx.x

        let mutable ltemp = 0.0
        let mutable utemp = 0.0
        let mutable htemp = 0.0
        
        let mutable span = 1
        while span < n do
              
            if rank < n then
                if rank - span >= 0 then
                    ltemp <- if d.[rank - span] <> 0.0 then -l.[rank] / d.[rank - span] else 0.0
                else
                    ltemp <- 0.0
                if rank + span < n then
                    utemp <- if d.[rank + span] <> 0.0 then -u.[rank] / d.[rank + span] else 0.0
                else
                    utemp <- 0.0
                htemp <- h.[rank]
            
            __syncthreads()

            if rank < n then    
                if rank - span >= 0 then              
                    d.[rank] <- d.[rank] + ltemp * u.[rank - span]
                    htemp <- htemp + ltemp * h.[rank - span]
                    ltemp <-ltemp * l.[rank - span]
                
                if rank + span < n then               
                    d.[rank] <- d.[rank] + utemp * l.[rank + span]
                    htemp <- htemp + utemp * h.[rank + span]
                    utemp <- utemp * u.[rank + span]
                           
            __syncthreads()
            
            if rank < n then
                l.[rank] <- ltemp
                u.[rank] <- utemp
                h.[rank] <- htemp

            __syncthreads()

            span <- 2*span
               
        if rank < n then
            h.[rank] <- h.[rank] / d.[rank]

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (n:int) (dl:deviceptr<float>) (dd:deviceptr<float>) (du:deviceptr<float>) (db:deviceptr<float>) (dx:deviceptr<float>) =
        let tid = threadIdx.x
        let gid = blockIdx.x * n + tid

        let shared = __shared__.ExternArray<float>() |> __array_to_ptr
        let l = shared
        let d = l + n
        let u = d + n
        let b = u + n
        
        l.[tid] <- dl.[gid]
        d.[tid] <- dd.[gid]
        u.[tid] <- du.[gid]
        b.[tid] <- db.[gid]
        
        __syncthreads()

        solve n l d u b

        dx.[gid] <- b.[tid]

    member this.Apply (numSystems:int, n:int, dl:deviceptr<float>, dd:deviceptr<float>, du:deviceptr<float>, db:deviceptr<float>, dx:deviceptr<float>) =
        let sharedSize = 9 * n * __sizeof<float>()
        let lp = LaunchParam(numSystems, n, sharedSize)
        this.GPULaunch <@ this.Kernel @> lp n dl dd du db dx

    member this.Solve(numSystems, l:float[], d:float[], u:float[], b:float[]) =
        let n = d.Length / numSystems

        // check resource availability
        let sharedSize = 9 * n * __sizeof<float>()
        let maxBlockDimX = this.GPUWorker.Device.Attributes.MAX_BLOCK_DIM_X
        let maxGridDimX = this.GPUWorker.Device.Attributes.MAX_GRID_DIM_X
        let maxThreads = this.GPUWorker.Device.Attributes.MAX_THREADS_PER_BLOCK
        let maxSharedSize = this.GPUWorker.Device.Attributes.MAX_SHARED_MEMORY_PER_BLOCK
        if numSystems > maxGridDimX then failwithf "numSystems(%d) > maxGridDimX(%d)" numSystems maxGridDimX  
        if n > maxBlockDimX then failwithf "n(%d) > maxBlockDimX(%d)" n maxBlockDimX  
        if n > maxThreads then failwithf "n(%d) > maxThreads(%d)" n maxThreads 
        if sharedSize > maxSharedSize then failwithf "sharedSize(%d) > maxSharedSize(%d)" sharedSize maxSharedSize  

        use dl = this.GPUWorker.Malloc(l)
        use dd = this.GPUWorker.Malloc(d)
        use du = this.GPUWorker.Malloc(u)
        use db = this.GPUWorker.Malloc(b)
        use dx = this.GPUWorker.Malloc(d.Length)

        this.Apply(numSystems, n, dl.Ptr, dd.Ptr, du.Ptr, db.Ptr, dx.Ptr)
        dx.Gather()

(**
We test the tridiagonal solver by setting up multiple diagonally dominant sytems.
*)

(*** define:triDiagSolverTest ***)
[<Test>]
let triDiagSolverTest () =
    let rng = System.Random(42)
    
    let random a b _ =
        rng.NextDouble() * (b - a) + a
    
    let multiply (l:float[]) (d:float[]) (u:float[]) (x:float[]) =
        let n = d.Length
        let b = Array.zeroCreate n
        b.[0] <- d.[0] * x.[0] + u.[0] * x.[1]
        for i = 1 to n - 2 do
            b.[i] <- l.[i] * x.[i - 1] + d.[i] * x.[i] + u.[i] * x.[i + 1]
        b.[n - 1] <- l.[n - 1] * x.[n - 2] + d.[n - 1] * x.[n - 1]
        b
    
    let generateDiagonallyDominantSystem n =
        let l = Array.init n (random -100.0 100.0)
        let u = Array.init n (random -100.0 100.0)
        let d = Array.map2 (fun l u -> random  0.0 10.0 () + (abs l) + (abs u)) l u
        let expectedSolution = Array.init n (random -20.0 20.0)
        let b = multiply l d u expectedSolution
        expectedSolution, l, d, u, b

    let unzip5 = Array.fold (fun (acc1, acc2, acc3, acc4, acc5) (e1, e2, e3, e4, e5) -> (Array.append acc1 e1, Array.append acc2 e2, Array.append acc3 e3, Array.append acc4 e4, Array.append acc5 e5)) ([||],[||],[||],[||],[||])
    
    let n = 100
    let expectedSolutions, l, d, u, b = Array.init n (fun _ -> generateDiagonallyDominantSystem 512) |> unzip5
    use tridiagSolver = new TriDiagSolverModule(GPUModuleTarget.DefaultWorker)
    let solutions = tridiagSolver.Solve(n, l, d, u, b)
    solutions |> should (equalWithin 1e-12) expectedSolutions
