(**
# Scripting with F#

This script can be executed in a command 

<img src="../content/images/fsharpScripting1.png" width="600" alt="FSharp Interactive">

First we have to set the include paths and reference the required assemblies.
*)
#load @"..\..\..\packages\Alea.CUDA\Alea.CUDA.fsx"
#I @"..\..\..\packages\NUnit\lib"
#I @"..\..\..\packages\FsUnit\Lib\Net40"
#r "nunit.framework.dll"
#r "FsUnit.NUnit.dll"
#r "System.Configuration.dll"

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework
open FsUnit

(**
We code a binary transform on the GPU. 
*)
type TransformModule<'T>(target, op:Expr<'T -> 'T -> 'T>) =
    inherit GPUModule(target)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (n:int) (x:deviceptr<'T>) (y:deviceptr<'T>) (z:deviceptr<'T>) =
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start 
        while i < n do
            z.[i] <- __eval(op) x.[i] y.[i]
            i <- i + stride

    member this.Apply(n:int, x:deviceptr<'T>, y:deviceptr<'T>, z:deviceptr<'T>) =
        let blockSize = 256
        let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
        let gridSize = min (16 * numSm) (divup n blockSize)
        let lp = LaunchParam(gridSize, blockSize)
        this.GPULaunch <@ this.Kernel @> lp n x y z

    member this.Apply(x:'T[], y:'T[]) =
        let n = x.Length
        use x = this.GPUWorker.Malloc(x)
        use y = this.GPUWorker.Malloc(y)
        use z = this.GPUWorker.Malloc(n)
        this.Apply(n, x.Ptr, y.Ptr, z.Ptr)
        z.Gather()

let generate n =
    let rng = Random()
    let n = 1000
    let x = Array.init n (fun _ -> rng.NextDouble())
    let y = Array.init n (fun _ -> rng.NextDouble())
    x, y

(**
We can instantiate a `TransformModule` for a specific binary operator and JIT compile it. 
*)
let sinCos = new TransformModule<float>(GPUModuleTarget.DefaultWorker, <@ fun x y -> (__nv_sin x) + (__nv_cos y) @>)

let x, y = generate 1000
let dz = sinCos.Apply(x, y)
let hz = Array.map2 (fun a b -> (sin a) + (cos b)) x y

dz |> should (equalWithin 1e-10) hz

printfn "result = %A" dz

let err = Array.map2 (fun d h -> abs (d - h)) dz hz |> Array.max 
printfn "error = %A" err
