(*** hide ***)
module Tutorial.Fs.examples.unbound.Reduce

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound
open NUnit.Framework
open FsUnit

(**
Reduction with Alea Unbound. 
*)
[<Test>]
let deviceReduceTest() =
    
    let cpuReduce = Array.reduce

    let gpuReduce (op:Expr<'T -> 'T -> 'T>) (x:'T[]) =
        let worker = Worker.Default
        use blob = new Blob(worker)
        worker.LoadProgram(
            DeviceReduceImpl.DeviceReduce(op, worker.Device.Arch, PlatformUtil.Instance.ProcessBitness).Template
            ).Entry.Create(blob, x.Length)
            .Reduce(None, blob.CreateArray(x).Ptr, x.Length)
    
    let n = 1000
    let rng = Random()
    let x = Array.init n (fun _ -> rng.Next())
    
    let cpuResult = cpuReduce (+) x
    let gpuResult = gpuReduce <@ (+) @> x
    
    printfn "cpuResult, gpuResult = %A, %A" cpuResult gpuResult
    cpuResult |> should equal gpuResult