(*** hide ***)
module Tutorial.Fs.advancedTechniques.GenericTransform

#if SCRIPT_REFS
#r "..\\..\\..\\packages\\Alea.IL\\lib\\net40\\Alea.IL.dll"
#r "..\\..\\..\\packages\\Alea.CUDA\\lib\\net40\\Alea.CUDA.dll"
#r "..\\..\\..\\packages\\Alea.CUDA.IL\\lib\\net40\\Alea.CUDA.IL.dll"
#r "..\\..\\..\\packages\\NUnit\\lib\\nunit.framework.dll"
#endif

(**
# Instance Based Technique

The instance based technique allows more flexibility than the method based approach 
with only minimal additional coding effort.  
*)

(*** define:genericTransformImport ***) 
open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.IL
open NUnit.Framework

(**
We create a generic class and inherit it from `ILGPUModule`, which provides functionality
to define and manage GPU resources such as kernels, constant memory or textures. 
*)

(*** define:transformModule ***)
type TransformModule<'T>(target, op:Expr<'T -> 'T -> 'T>) =
    inherit ILGPUModule(target)
(**
The constructor take a binary function. The F# version has two constructors, one from a binary 
function expression and the second from a delegate `Func<'T, 'T, 'T>`.   
*)

(*** define:transformConstructor ***)
    new (target, op:Func<'T, 'T, 'T>) =
        new TransformModule<'T>(target, <@ fun x y -> op.Invoke(x, y) @>)

(**
The thread organization in GPU kernel is as in the [quick start example.](../quick_start/quick_start_example.html).
In F# the operation is stored as an expression of a binary function. We extract the function with `__eval(op)`. 
In C# we can directly apply the delegate. 
*)

(*** define:transformKernel ***)
    [<Kernel;ReflectedDefinition>]
    member this.Kernel (n:int) (x:deviceptr<'T>) (y:deviceptr<'T>) (z:deviceptr<'T>) =
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start 
        while i < n do
            z.[i] <- __eval(op) x.[i] y.[i]
            i <- i + stride

(**
For convenience we create an overload which assumes that all the data is already on the GPU. 
*)

(*** define:transformGPUDevice ***)
    member this.Apply(n:int, x:deviceptr<'T>, y:deviceptr<'T>, z:deviceptr<'T>) =
        let blockSize = 256
        let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
        let gridSize = min (16 * numSm) (divup n blockSize)
        let lp = LaunchParam(gridSize, blockSize)
        this.GPULaunch <@ this.Kernel @> lp n x y z

(**
The standard calling function also handles memory copy, memory allocation for the result and 
copies the result back from the GPU to the host memory. 
*)

(*** define:transformGPUHost ***)
    member this.Apply(x:'T[], y:'T[]) =
        let n = x.Length
        use x = this.GPUWorker.Malloc(x)
        use y = this.GPUWorker.Malloc(y)
        use z = this.GPUWorker.Malloc(n)
        this.Apply(n, x.Ptr, y.Ptr, z.Ptr)
        z.Gather()

(**
The class `TransformModule` is still generic. We now provide concrete specializations and attribute the class with `AOTCompile`. 
To instantiate the GPU module `SinCosModule` we provide a convenience method `SinCosModule.DefaultInstance` which uses the default worker.
*)

(*** define:transformModuleSpecialized ***)
[<AOTCompile>]
type SinCosModule(target) = 
    inherit TransformModule<float>(target, fun x y -> (__nv_sin x) + (__nv_cos y))
    static let instance = lazy new SinCosModule(GPUModuleTarget.DefaultWorker)
    static member DefaultInstance = instance.Value

(**
After instantiation `SinCosModule` the transform is called by `sinCos.Apply(x, y)` as shown in the following test. 
*)

(*** define:transformModuleSpecializedTest ***)
[<Test>]
let sinCosTest () =
    let sinCos = SinCosModule.DefaultInstance
    let rng = Random()
    let n = 1000
    let x = Array.init n (fun _ -> rng.NextDouble())
    let y = Array.init n (fun _ -> rng.NextDouble())
    let dResult = sinCos.Apply(x, y)
    let hResult = Array.map2 (fun a b -> (sin a) + (cos b)) x y
    let err = Array.map2 (fun d h -> abs (d - h)) dResult hResult |> Array.max 
    printfn "error = %A" err