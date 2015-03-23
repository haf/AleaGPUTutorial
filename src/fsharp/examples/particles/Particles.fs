//#I @"..\..\packages\Alea.CUDA\lib\net40"
//#I @"..\..\packages\Alea.CUDA.IL\lib\net40"
//#I @"..\..\packages\Alea.CUDA.Unbound\lib\net40"
//#I @"..\..\packages\FSharp.Charting\lib\net40"
//#I @"..\..\packages\NUnit\lib"
//#I @"..\..\packages\FsUnit\Lib\Net40"
//#r "Alea.CUDA.dll"
//#r "Alea.CUDA.IL.dll"
//#r "Alea.CUDA.Unbound.dll"
//#r "nunit.framework.dll"
//#r "FsUnit.NUnit.dll"
//#r "FSharp.Charting.dll"
//#r "System.Configuration.dll"
//#r "System.Windows.Forms"
//#r "System.Windows.Forms.DataVisualization"
module Tutorial.Fs.examples.particles.Particles

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Unbound
open Alea.CUDA.Utilities
open FSharp.Charting
open NUnit.Framework
open FsUnit

(**
We have to set the assembly path so that the JIT compiler finds the dlls. 
*)
//Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- __SOURCE_DIRECTORY__ + @"\..\..\packages\Alea.CUDA\private"
//Alea.CUDA.Settings.Instance.Resource.Path <- __SOURCE_DIRECTORY__ + @"\..\..\release"

type ParticleType =
| Single = 0
| Cluster = 1
| PartOfCluster = 2

[<Struct>]
type Point3 =
    val mutable x : float
    val mutable y : float
    val mutable z : float
    val mutable ptype : ParticleType

    [<ReflectedDefinition>]
    new (x, y, z, pt) = { x = x; y = y; z = z; ptype = pt }

    [<ReflectedDefinition>]
    static member Dist (lhs:Point3) (rhs:Point3) =
        let x = lhs.x - rhs.x
        let y = lhs.y - rhs.y
        let z = lhs.z - rhs.z
        sqrt (x*x + y*y + z*z)

    override this.ToString() = sprintf "(%f,%f,%f)" this.x this.y this.z

(**
Example code. 
*)
type ParticlesModule(target, maxPoints:int) =
    inherit GPUModule(target)

    [<ReflectedDefinition>]
    let collect (a1, a2) (b1, b2) =
         a1 + b1, b2

    let scanModule = new DeviceScanModule<_>(target, <@ (+) @>)
    let scan = scanModule.Create maxPoints

    override this.Dispose disposing =
        if disposing then
            scan.Dispose()
            scanModule.Dispose()
        base.Dispose disposing

    member this.LaunchParam n =
        let blockSize = 256
        let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
        let gridSize = min (16 * numSm) (divup n blockSize)
        LaunchParam(gridSize, blockSize)

    [<Kernel;ReflectedDefinition>]
    member this.DistKernel (minDist:float) (refPoint:Point3) (n:int) (points:deviceptr<Point3>) (offsets:deviceptr<int>) (inCluster:deviceptr<bool>) =
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start 
        while i < n do
            let mutable point = points.[i] 
            if point.ptype = ParticleType.Single then
                let dist = Point3.Dist refPoint point 
                offsets.[i] <- if dist < minDist then 1 else 0
                inCluster.[i] <- dist < minDist 
            i <- i + stride

    [<Kernel;ReflectedDefinition>]
    member this.CompactCopyKernel (n:int) (points:deviceptr<Point3>) (offsets:deviceptr<int>) (inCluster:deviceptr<bool>) (clusterPoints:deviceptr<Point3>) (numClusterPoints:deviceptr<int>) =
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start 
        while i < n do
            let j = offsets.[i] 
            let copy = inCluster.[i] 
            if copy then 
                clusterPoints.[j] <- points.[i]            
            
            i <- i + stride

        if i = n - 1 + stride then 
            numClusterPoints.[0] <- offsets.[n-1] 

    member this.Apply(minDist:float, n:int, refPoint:Point3, points:deviceptr<Point3>, offsets:deviceptr<int>, inCluster:deviceptr<bool>, clusterPoints:deviceptr<Point3>, numClusterPoints:deviceptr<int>) =
        let lp = this.LaunchParam n
        
        // calculate the distances and indentify points which change from `Single` to `PartOfCluster` 
        this.GPULaunch <@ this.DistKernel @> lp minDist refPoint n points offsets inCluster

        // copy all the points which are new in cluster to `clusterPoints`
        scan.ExclusiveScan(offsets, offsets, 0, n)
        this.GPULaunch <@ this.CompactCopyKernel @> lp n points offsets inCluster clusterPoints numClusterPoints

    // example how to call the kernels with data on the CPU
    member this.Apply(minDist:float, refPoint:Point3, points:Point3[]) =
        let n = points.Length
        use points = this.GPUWorker.Malloc(points)
        use offsets = this.GPUWorker.Malloc<int>(n)
        use inCluster = this.GPUWorker.Malloc<bool>(n)
        use clusterPoints = this.GPUWorker.Malloc<Point3>(n)
        use numClusterPoints = this.GPUWorker.Malloc<int>(1)
        this.Apply(minDist, n, refPoint, points.Ptr, offsets.Ptr, inCluster.Ptr, clusterPoints.Ptr, numClusterPoints.Ptr)
        let numPoints = numClusterPoints.Gather().[0]
        let points = clusterPoints.Gather().[0..numPoints - 1]
        points, numPoints, (points |> Array.map (Point3.Dist refPoint))

let testParticles () =
    if Device.Devices.[0].Arch.Major < 3 then 
        printfn "The example: testParticles needs Compute-Capability 3.0 or higher, \nbut your default GPU has: %d.%d" Device.Devices.[0].Arch.Major Device.Devices.[0].Arch.Minor
    else 
        let particles = 
           let rng = Random(0)
           Array.init 10000000 (fun i -> Point3(rng.NextDouble(), rng.NextDouble(), rng.NextDouble(), ParticleType.Single))

        use particlesModule = new ParticlesModule(GPUModuleTarget.DefaultWorker, particles.Length)
        let refPoint = Point3(0.5, 0.5, 0.5, ParticleType.Cluster)
        particlesModule.Apply(0.01, refPoint, particles)  |> ignore