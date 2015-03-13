#I @"..\..\packages\Alea.CUDA\lib\net40"
#I @"..\..\packages\FSharp.Charting\lib\net40"
#I @"..\..\packages\NUnit\lib"
#I @"..\..\packages\FsUnit\Lib\Net40"
#r "Alea.CUDA.dll"
#r "nunit.framework.dll"
#r "FsUnit.NUnit.dll"
#r "FSharp.Charting.dll"
#r "System.Configuration.dll"
#r "System.Windows.Forms"
#r "System.Windows.Forms.DataVisualization"

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open FSharp.Charting
open NUnit.Framework
open FsUnit

(**
We have to set the assembly path so that the JIT compiler finds the dlls. 
*)
Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- __SOURCE_DIRECTORY__ + @"\..\..\packages\Alea.CUDA
3057\private"
Alea.CUDA.Settings.Instance.Resource.Path <- __SOURCE_DIRECTORY__ + @"\..\..\release"

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

(**
We create a `TransformModule` instance for a specific binary operator, JIT compile it,
run the transform operation, and visualize the result.
*)
let sinCos () =
    let sinCos = new TransformModule<float>(GPUModuleTarget.DefaultWorker, <@ fun x y -> (__nv_sin x) + (__nv_cos y) @>)

    let x = [|-2.0*Math.PI..0.01..2.0*Math.PI|] 
    let y = [|-2.0*Math.PI..0.01..2.0*Math.PI|] 
    let dz = sinCos.Apply(x, y)
    let hz = Array.map2 (fun a b -> (sin a) + (cos b)) x y

    dz |> should (equalWithin 1e-10) hz

    printfn "result = %A" dz

    let chart = 
        Chart.Line (Array.zip x dz) 
        |> Chart.WithTitle("Sin*Cos")
        |> Chart.WithXAxis(Min = -2.0*Math.PI, Max = 2.0*Math.PI)

    chart.ShowChart() |> ignore

    chart.SaveChartAs("sinCos.png", ChartTypes.ChartImageFormat.Png)

sinCos()
