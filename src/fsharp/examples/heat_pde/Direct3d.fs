(*** hide ***)
module Tutorial.Fs.examples.heatPde.Direct3d

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Direct3D9
open NUnit.Framework

open Tutorial.Fs.examples.heatPde.Solver

let inline kernels (real:RealTraits<'T>)
                   (initCondExpr:Expr<'T -> 'T -> 'T -> 'T>) 
                   (boundaryExpr:Expr<'T -> 'T -> 'T -> 'T>) 
                   (sourceExpr:Expr<'T -> 'T -> 'T -> 'T>) = cuda {

    let! initCondKernel =     
        <@ fun nx ny t (x:deviceptr<'T>) (y:deviceptr<'T>) (u:deviceptr<'T>) ->
            let initCond = %initCondExpr
            let i = blockIdx.x*blockDim.x + threadIdx.x
            let j = blockIdx.y*blockDim.y + threadIdx.y
            let mstride = ny
            if i < nx && j < ny then u.[i*mstride+j] <- initCond t x.[i] y.[j] @> |> Compiler.DefineKernel

    let! xSweepKernel =     
        <@ fun nx ny (x:deviceptr<'T>) (y:deviceptr<'T>) Cx Cy dt t0 t1 (u0:deviceptr<'T>) (u1:deviceptr<'T>) ->     
            let boundary = %boundaryExpr
            let source = %sourceExpr     
            xSweep boundary source nx ny x y Cx Cy dt t0 t1 u0 u1 @> |> Compiler.DefineKernel

    let! ySweepKernel =     
        <@ fun nx ny (x:deviceptr<'T>) (y:deviceptr<'T>) Cx Cy dt t0 t1 (u0:deviceptr<'T>) (u1:deviceptr<'T>) ->          
            let boundary = %boundaryExpr
            let source = % sourceExpr     
            ySweep boundary source nx ny x y Cx Cy dt t0 t1 u0 u1 @> |> Compiler.DefineKernel

    return initCondKernel, xSweepKernel, ySweepKernel }

type Example =
    {
        Name      : string
        InitCond  : Expr<float -> float -> float -> float>
        Boundary  : Expr<float -> float -> float -> float>
        Source    : Expr<float -> float -> float -> float>
        LoopTime  : float
        TimeRatio : float
        ValSpace  : AxisSpace<float>
    }

    member this.Real = RealTraits.Real64
    member this.Kernels = kernels this.Real this.InitCond this.Boundary this.Source

let example1 =
    {
        Name      = "exp(-t) * sin(pi*x) * cos(pi*y)"
        InitCond  = <@ fun t x y -> exp(-t) * sin(__pi()*x) * cos(__pi()*y) @>
        Boundary  = <@ fun t x y -> exp(-t) * sin(__pi()*x) * cos(__pi()*y) @>
        Source    = <@ fun t x y -> exp(-t) * sin(__pi()*x) * cos(__pi()*y) * (2.0*__pi()*__pi() - 1.0) @>
        LoopTime  = 8000.0
        TimeRatio = 8.0
        ValSpace  = { MinValue = -1.0; MaxValue = 1.0; Ratio = 2.0 }  
    }

let example2 =
    {
        Name      = "heat box (instable solution)"
        InitCond  = <@ fun t x y -> if x >= 0.4 && x <= 0.6 && y >= 0.4 && y <= 0.6 then 1.0 else 0.0 @>
        Boundary  = <@ fun t x y -> 0.0 @>
        Source    = <@ fun t x y -> 0.0 @>
        LoopTime  = 5000.0
        TimeRatio = 0.03
        ValSpace  = { MinValue = -0.13; MaxValue = 1.0; Ratio = 1.3 }  
    }

let example3 =
    let sigma1 = 0.04
    let sigma2 = 0.04
    let sigma3 = 0.04
    {
        Name      = "heat gauss"
        InitCond  =  <@ fun t x y -> 1.0/3.0*exp (-((x-0.2)*(x-0.2) + (y-0.2)*(y-0.2))/(2.0*sigma1*sigma1)) / (sigma1*sigma1*2.0*__pi()) +
                                     1.0/3.0*exp (-((x-0.8)*(x-0.8) + (y-0.8)*(y-0.8))/(2.0*sigma2*sigma2)) / (sigma2*sigma2*2.0*__pi()) +
                                     1.0/3.0*exp (-((x-0.8)*(x-0.8) + (y-0.2)*(y-0.2))/(2.0*sigma3*sigma3)) / (sigma3*sigma3*2.0*__pi()) @>
        Boundary  = <@ fun t x y -> 0.0 @>
        Source    = <@ fun t x y -> 0.0 @>
        LoopTime  = 8000.0
        TimeRatio = 0.005
        ValSpace = { MinValue = 0.0; MaxValue = 35.0; Ratio = 1.0 }  
    }

let examples = [| example1; example2; example3 |]

let heatPdeDirect3d() =
    let example =
        printfn "Please choose the equation:"
        examples |> Array.iteri (fun i example -> printfn "(%d) %s" i example.Name)
        printf "Please choose: "
        let selection = int32(Console.Read()) - 48
        examples.[selection]

    let nx = 256
    let ny = 256
    let tstop = 1.0
    let diffusionCoeff = 1.0
    let tstart = 0.0
    let xMin = 0.0
    let xMax = 1.0
    let yMin = 0.0
    let yMax = 1.0
    let dt = 0.01

    let plotter = cuda {
        let! initCondKernel, xSweepKernel, ySweepKernel = example.Kernels

        return Entry(fun program (ctx:ApplicationContext) ->
            let worker = program.Worker
            let initCondKernel = program.Apply initCondKernel
            let xSweepKernel = program.Apply xSweepKernel
            let ySweepKernel = program.Apply ySweepKernel
            let real = example.Real

            let timeGrid = exponentiallyCondensedGrid real 5 tstart tstop dt
            let xgrid, dx = homogeneousGrid real nx xMin xMax
            let ygrid, dy = homogeneousGrid real ny yMin yMax
            let nu = nx * ny
            let lp0 = LaunchParam(dim3(divup nx 16, divup ny 16), dim3(16, 16))
            let lpx = LaunchParam(ny, nx, 4*nx*sizeof<float>)
            let lpy = LaunchParam(nx, ny, 4*ny*sizeof<float>)

            use x = worker.Malloc(xgrid)
            use y = worker.Malloc(ygrid)
            use u0 = worker.Malloc<float>(nu)
            use u1 = worker.Malloc<float>(nu)

            let initCondKernelFunc = initCondKernel.Launch lp0 
            let xSweepKernelFunc = xSweepKernel.Launch lpx
            let ySweepKernelFunc = ySweepKernel.Launch lpy

            let step t0 t1 =
                let dt = t1 - t0
                if dt > 0.0 then
                    //printfn "t1(%f) - t0(%f) = dt(%f)" t1 t0 dt
                    let Cx = diffusionCoeff * dt / (dx * dx)
                    let Cy = diffusionCoeff * dt / (dy * dy)
                    xSweepKernelFunc nx ny x.Ptr y.Ptr Cx Cy dt t0 (t0 + __half() * dt) u0.Ptr u1.Ptr
                    ySweepKernelFunc nx ny x.Ptr y.Ptr Cx Cy dt (t0 + __half() * dt) t1 u1.Ptr u0.Ptr

            let t0 = ref -10.0
            let maxu = ref Double.NegativeInfinity
            let minu = ref Double.PositiveInfinity

            let frame (time:float) =
                let result =
                    if !t0 < 0.0 then
                        // first frame
                        let t1 = time / example.LoopTime * example.TimeRatio
                        t0 := 0.0
                        initCondKernelFunc nx ny tstart x.Ptr y.Ptr u0.Ptr
                        step !t0 t1
                        t0 := t1
                        u0.Ptr, Some example.ValSpace
                    else
                        let t1 = time / example.LoopTime * example.TimeRatio
                        if !t0 > t1 then
                            // a new loop
                            t0 := 0.0
                            initCondKernelFunc nx ny tstart x.Ptr y.Ptr u0.Ptr
                            step !t0 t1
                            t0 := t1
                            u0.Ptr, None
                        else 
                            // a step
                            step !t0 t1
                            t0 := t1
                            u0.Ptr, None

                // just to check the max value
                if false then
                    let u = u0.Gather()
                    let maxu' = u |> Array.max
                    let minu' = u |> Array.min
                    if maxu' > !maxu then
                        maxu := maxu'
                        printfn "maxu: %f" !maxu
                    if minu' < !minu then
                        minu := minu'
                        printfn "minu: %f" !minu

                result

            let param : SurfacePlotter.AnimationParam<float> =
                {
                    Order = MatrixStorageOrder.RowMajor
                    RowLength = nx
                    ColLength = ny
                    RowSpace = { MinValue = xMin; MaxValue = xMax; Ratio = 1.0 }
                    ColSpace = { MinValue = yMin; MaxValue = yMax; Ratio = 1.0 }
                    ValSpace = example.ValSpace |> Some
                    RowPtr = x.Ptr
                    ColPtr = y.Ptr
                    Frame = frame
                    LoopTime = Some example.LoopTime
                }

            SurfacePlotter.animationLoop ctx param ) }

    let param : ApplicationParam =
        {
            CUDADevice = Device.Devices.[0]
            FormTitle = "Heat PDE"
            DrawingSize = Drawing.Size(800, 800)
        }

    let loop ctx =
        printf "Compiling ... "
        use plotter = plotter |> Compiler.load ctx.Worker
        printfn "[OK]"
        plotter.Run ctx

    let app = Application(param, loop)
    app.Start()


