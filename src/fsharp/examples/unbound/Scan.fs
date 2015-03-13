(*** hide ***)
module Tutorial.Fs.examples.unbound.Scan

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities 
open Alea.CUDA.Unbound
open NUnit.Framework
open FsUnit

(**
Scan with Alea Unbound. 
*)
[<Test>]
let deviceScanTest() =
    let worker = Worker.Default
    
    let cpuInclusiveScan op (x:'T[]) = Array.scan op x.[0] x.[1..]

    let gpuInclusiveScan (op:Expr<'T -> 'T -> 'T>) (x:'T[]) =
        use scanModule = new DeviceScanModule<'T>(GPUModuleTarget.Worker(worker), op)
        use scan = scanModule.Create(x.Length)
        use dx = worker.Malloc(x)
        use dr = worker.Malloc<'T>(x.Length)
        scan.InclusiveScan(dx.Ptr, dr.Ptr, x.Length)
        dr.Gather()

    let n = 1000
    let rng = Random()
    let x = Array.init n (fun _ -> rng.Next(-5,5))

    let cpuResult = cpuInclusiveScan (+) x
    let gpuResult = gpuInclusiveScan <@(+)@> x
    
    printfn "cpuResult:\n%A" cpuResult
    printfn "gpuResult:\n%A" gpuResult
    gpuResult |> should equal cpuResult

