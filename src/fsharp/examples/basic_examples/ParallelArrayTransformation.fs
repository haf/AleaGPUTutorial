(*** hide ***)
module Tutorial.Fs.examples.basic

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.IL
open Alea.CUDA.Utilities
open NUnit.Framework

(**
# Parallel Array Transformation

For very large arrays we cannot assume that the thread grid is large enough to cover the entire data array.
Therefore the kernel loops over the data array one grid-size at a time until all of the array elements are processed.
The stride of the loop is blockDim.x * gridDim.x = total number of threads in the grid.
For example, if there are 1024 threads in the grid, thread 0 will compute elements 0, 1024, 2048, ... . 
A loop with stride equal to the grid size also causes all addressing within warps to be unit-strided
which implies a maximally coalescing memory access pattern.
*)

(*** define:transformModule ***)
type TransformModule<'T>(target, op:Expr<'T -> 'T>) =
    inherit ILGPUModule(target)

    new (target, op:Func<'T, 'T>) =
        new TransformModule<'T>(target, <@ fun x -> op.Invoke(x) @>)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (n:int) (x:deviceptr<'T>) (y:deviceptr<'T>) =
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start 
        while i < n do
            y.[i] <- __eval(op) x.[i] 
            i <- i + stride

    member this.Apply(n:int, x:deviceptr<'T>, y:deviceptr<'T>) =
        let blockSize = 256
        let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
        let gridSize = min (16 * numSm) (divup n blockSize)
        let lp = LaunchParam(gridSize, blockSize)
        this.GPULaunch <@ this.Kernel @> lp n x y 

    member this.Apply (x:'T[]) =
        use x = this.GPUWorker.Malloc(x)
        use y = this.GPUWorker.Malloc(x.Length)
        this.Apply(x.Length, x.Ptr, y.Ptr)
        y.Gather()

(**
The class `TransformModule` is still generic. We now provide concrete specializations and attribute the class with `AOTCompile`. 
To instantiate the GPU module `SinModule` we provide a convenience method `SinModule.DefaultInstance` which uses the default worker.
*)

(*** define:transformModuleSpecialized ***)
[<AOTCompile>]
type SinModule(target) = 
    inherit TransformModule<float>(target, fun x -> __nv_sin x)
    static let instance = lazy new SinModule(GPUModuleTarget.DefaultWorker)
    static member DefaultInstance = instance.Value

(**
We verify the correctness with the following test. 
*)

(*** define:transformModuleSpecializedTest ***)
[<Test>]
let sinTest () =
    let sinGpu = SinModule.DefaultInstance
    let rng = Random()
    let n = 1000
    let x = Array.init n (fun _ -> rng.NextDouble())
    let dResult = sinGpu.Apply(x)
    let hResult = Array.map (sin) x
    let err = Array.map2 (fun d h -> abs (d - h)) dResult hResult |> Array.max 
    printfn "error = %A" err
