(*** hide ***)
module Tutorial.Fs.examples.heatPde.Direct3d

open System
open System.Threading
open System.Diagnostics
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open SharpDX
open SharpDX.Multimedia
open SharpDX.RawInput
open SharpDX.Windows
open SharpDX.Direct3D
open SharpDX.Direct3D9
open NUnit.Framework
open Tutorial.Fs.examples.heatPde.Solver

#nowarn "9"
#nowarn "51"

[<AutoOpen>]
module Common =
    let registerGraphicsResource (worker:Worker) (d3d9Res:CppObject) =
        worker.Eval <| fun _ ->
            let mutable cudaRes = 0n
            cuSafeCall(cuGraphicsD3D9RegisterResource(&&cudaRes, d3d9Res.NativePointer, 0u))
            cudaRes

    let unregisterGraphicsResource (worker:Worker) (cudaRes:nativeint) =
        worker.Eval <| fun _ ->
            cuSafeCall(cuGraphicsUnregisterResource(cudaRes))

    type CUDADevice = Alea.CUDA.Device
    type D3D9Device = SharpDX.Direct3D9.Device
    type RawInputDevice = SharpDX.RawInput.Device

    let transform2d (transform:Expr<int -> int -> 'T -> 'U>) = cuda {

        let kernel (transform:Expr<int -> int -> 'T -> 'U>) =
            <@ fun (majors:int) (minors:int) (inputs:deviceptr<'T>) (outputs:deviceptr<'U>) ->
                let minorStart = blockIdx.x * blockDim.x + threadIdx.x
                let majorStart = blockIdx.y * blockDim.y + threadIdx.y
            
                let minorStride = gridDim.x * blockDim.x
                let majorStride = gridDim.y * blockDim.y

                let mutable major = majorStart
                while major < majors do
                    let mutable minor = minorStart
                    while minor < minors do
                        let i = major * minors + minor
                        outputs.[i] <- (%transform) major minor inputs.[i]
                        minor <- minor + minorStride
                    major <- major + majorStride @>

        let! kernelRowMajor = transform |> kernel |> Compiler.DefineKernel
        let! kernelColMajor = <@ fun c r v -> (%transform) r c v @> |> kernel |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker
            let kernelRowMajor = program.Apply kernelRowMajor
            let kernelColMajor = program.Apply kernelColMajor

            let lp =
                let blockSize = dim3(32, 8)
                let gridSize = dim3(16, 16)
                LaunchParam(gridSize, blockSize)

            let run (order:MatrixStorageOrder) (rows:int) (cols:int) (inputs:deviceptr<'T>) (outputs:deviceptr<'U>) =
                order |> function
                | MatrixStorageOrder.RowMajor -> kernelRowMajor.Launch lp rows cols inputs outputs
                | MatrixStorageOrder.ColMajor -> kernelColMajor.Launch lp cols rows inputs outputs

            run ) }

    type RenderType =
        | Mesh
        | Point

    [<Record>]
    type AxisSpace<'T> =
        {
            MinValue : 'T
            MaxValue : 'T
            Ratio : 'T
        }

    type Data1D<'T> =
        | ByHostArray of 'T[]
        | ByDeviceMemory of DeviceMemory<'T>
        | ByDevicePtr of deviceptr<'T> * int

    type Data2D<'T> =
        | ByHostArray of 'T[] * MatrixStorageOrder * int * int
        | ByDeviceMemory of DeviceMemory<'T> * MatrixStorageOrder
        | ByDevicePtr of deviceptr<'T> * MatrixStorageOrder * int * int

[<AutoOpen>]
module Application =
    type ApplicationParam =
        {
            CUDADevice : CUDADevice
            FormTitle : string
            DrawingSize : Drawing.Size
        }

        static member Create(cudaDevice) =
            {
                CUDADevice = cudaDevice
                FormTitle = "NoName"
                DrawingSize = Drawing.Size(800, 600)
            }

    type ApplicationContext =
        {
            Form : RenderForm
            D3D9Device : D3D9Device
            CUDADevice : CUDADevice
            Worker : Worker
        }

        member this.RegisterGraphicsResource(d3d9Res:CppObject) = registerGraphicsResource this.Worker d3d9Res
        member this.UnregisterGraphicsResource(cudaRes:nativeint) = unregisterGraphicsResource this.Worker cudaRes

    type Application(param:ApplicationParam, loop:ApplicationContext -> unit) =
        let proc() =
            use form = new RenderForm(Text = param.FormTitle, ClientSize = param.DrawingSize)

            let cudaDevice = param.CUDADevice

            use d3d9Device = new D3D9Device(new Direct3D(),
                                            cudaDevice.ID,
                                            DeviceType.Hardware,
                                            form.Handle,
                                            CreateFlags.HardwareVertexProcessing,
                                            PresentParameters(form.ClientSize.Width, form.ClientSize.Height))

            use worker =
                let generate() =
                    let mutable ctx = 0n
                    let mutable dev = -1
                    cuSafeCall(cuD3D9CtxCreate(&&ctx, &&dev, 0u, d3d9Device.NativePointer))
                    if dev <> cudaDevice.ID then printfn "warning: returned dev is %d, but you require %d" dev cudaDevice.ID
                    let dev = Device.DeviceDict.[dev]
                    dev, ctx
                Worker.Create(generate)

            let context =
                {
                    Form = form
                    D3D9Device = d3d9Device
                    CUDADevice = cudaDevice
                    Worker = worker
                }

            loop context

        member this.Start(?forceNewThread:bool, ?waitForStop:bool) =
            let forceNewThread = defaultArg forceNewThread false
            let waitForStop = defaultArg waitForStop true
            match forceNewThread with
            | true ->
                let thread = Thread(proc)
                thread.SetApartmentState(ApartmentState.STA)
                thread.Start()
                if waitForStop then thread.Join()
            | false ->
                match Thread.CurrentThread.GetApartmentState() with
                | ApartmentState.STA -> proc()
                | _ -> 
                    let thread = Thread(proc)
                    thread.SetApartmentState(ApartmentState.STA)
                    thread.Start()
                    if waitForStop then thread.Join()

module SurfacePlotter =
    [<Struct;Align(16)>]
    type Vector4 =
        val x : float32
        val y : float32
        val z : float32
        val w : float32

        [<ReflectedDefinition>]
        new (x, y, z, w) = { x = x; y = y; z = z; w = w }

        override this.ToString() = sprintf "(%f,%f,%f,%f)" this.x this.y this.z this.w
    
    [<Struct;Align(16)>]
    type Vertex =
        val position : Vector4
        val color : Vector4

        [<ReflectedDefinition>]
        new (position, color) = { position = position; color = color }

        override this.ToString() = sprintf "[Position%O,Color%O]" this.position this.color

    [<ReflectedDefinition>]
    let inline mapColor (value:'T) (minv:'T) (maxv:'T) =
        let mapB (level:'T) = max 0G (cos (level * __pi()))
        let mapG (level:'T) = sin (level * __pi())
        let mapR (level:'T) = max 0.0 (-(cos (level * __pi())))
        let level = (value - minv) / (maxv - minv)
        Vector4(float32(mapR level), float32(mapG level), float32(mapB level), 1.0f)

    let inline fill (vbRes:CUgraphicsResource) (rowLength:int) (colLength:int) = cuda {
        let! cRows = Compiler.DefineConstantArray<'T>(rowLength)
        let! cCols = Compiler.DefineConstantArray<'T>(colLength)
        let! cRowSpace = Compiler.DefineConstantVariable<AxisSpace<'T>>()
        let! cColSpace = Compiler.DefineConstantVariable<AxisSpace<'T>>()
        let! cValSpace = Compiler.DefineConstantVariable<AxisSpace<'T>>()

        let transform =
            <@ fun (r:int) (c:int) (v:'T) ->
                let colSpace = cColSpace.Value |> __unbox
                let rowSpace = cRowSpace.Value |> __unbox
                let valSpace = cValSpace.Value |> __unbox

                let x = (cCols.[c] - colSpace.MinValue) / (colSpace.MaxValue - colSpace.MinValue) * colSpace.Ratio - colSpace.Ratio / 2G |> float32
                let z = (cRows.[r] - rowSpace.MinValue) / (rowSpace.MaxValue - rowSpace.MinValue) * rowSpace.Ratio - rowSpace.Ratio / 2G |> float32
                let y = (v - valSpace.MinValue) / (valSpace.MaxValue - valSpace.MinValue) * valSpace.Ratio - valSpace.Ratio / 2G |> float32

                let position = Vector4(x, y, z, 1.0f)
                let color = mapColor v valSpace.MinValue valSpace.MaxValue

                Vertex(position, color) @>

        let! transform = transform |> transform2d

        return Entry(fun program order (rowPtr:deviceptr<'T>) (colPtr:deviceptr<'T>) rowSpace colSpace (valSpace:AxisSpace<'T> option) ->
            let worker = program.Worker
            let transform = transform.Apply program
            let transform = transform order rowLength colLength
            let cRows = program.Apply cRows
            let cCols = program.Apply cCols
            let cRowSpace = program.Apply cRowSpace
            let cColSpace = program.Apply cColSpace
            let cValSpace = program.Apply cValSpace

            cRows.Copy(rowPtr, rowLength)
            cCols.Copy(colPtr, colLength)
            cRowSpace.Scatter(rowSpace)
            cColSpace.Scatter(colSpace)
            valSpace |> Option.iter (fun valSpace -> cValSpace.Scatter(valSpace))

            let run (valSpace:AxisSpace<'T> option) (inputs:deviceptr<'T>) =
                worker.Eval <| fun _ ->
                    valSpace |> Option.iter (fun valSpace -> cValSpace.Scatter(valSpace))

                    let mutable vbRes = vbRes
                    cuSafeCall(cuGraphicsMapResources(1u, &&vbRes, 0n))

                    let mutable vbPtr = 0n
                    let mutable vbSize = 0n
                    cuSafeCall(cuGraphicsResourceGetMappedPointer(&&vbPtr, &&vbSize, vbRes))

                    let vb = deviceptr<Vertex>(vbPtr)
                    transform inputs vb
                    
                    cuSafeCall(cuGraphicsUnmapResources(1u, &&vbRes, 0n))

            run ) }

    let createVertexBuffer (ctx:ApplicationContext) (elements:int) =
        new VertexBuffer(ctx.D3D9Device, __sizeof<Vertex>() * elements, Usage.WriteOnly, VertexFormat.None, Pool.Default)

    let createVertexDeclaration (ctx:ApplicationContext) =
        let ves = [| VertexElement(0s,  0s, DeclarationType.Float4, DeclarationMethod.Default, DeclarationUsage.Position, 0uy)
                     VertexElement(0s, 16s, DeclarationType.Float4, DeclarationMethod.Default, DeclarationUsage.Color,    0uy)
                     VertexElement.VertexDeclarationEnd |]
        new VertexDeclaration(ctx.D3D9Device, ves)

    let createPointIndexBuffer (ctx:ApplicationContext) =
        new IndexBuffer(ctx.D3D9Device, sizeof<int>, Usage.WriteOnly, Pool.Managed, false)

    let renderingLoop (ctx:ApplicationContext) (renderType:RenderType) (vd:VertexDeclaration) (vb:VertexBuffer) (order:MatrixStorageOrder) (rows:int) (cols:int) (hook:Stopwatch -> unit) =
        let elements = rows * cols

        use ib = renderType |> function
            | RenderType.Point -> createPointIndexBuffer ctx
            | RenderType.Mesh -> failwith "Mesh rendering not implemented yet."

        let eye = Vector3(0.0f, 2.0f, -2.0f)
        let lookat = Vector3(0.0f, 0.0f, 0.0f)
        let up = Vector3(0.0f, 1.0f, 0.0f)

        let view = Matrix.LookAtLH(eye, lookat, up)
        let proj = Matrix.PerspectiveFovLH(Math.PI * 0.25 |> float32, 1.0f, 1.0f, 100.0f)
        let world = ref (Matrix.RotationY(Math.PI * 0.25 |> float32))

        ctx.D3D9Device.SetTransform(TransformState.View, view)
        ctx.D3D9Device.SetTransform(TransformState.Projection, proj)
        ctx.D3D9Device.SetRenderState(RenderState.Lighting, false)

        ctx.D3D9Device.Indices <- ib
        ctx.D3D9Device.VertexDeclaration <- vd
        ctx.D3D9Device.SetStreamSource(0, vb, 0, sizeof<Vertex>)

        let isMouseLeftButtonDown = ref false
        RawInputDevice.RegisterDevice(UsagePage.Generic, UsageId.GenericMouse, DeviceFlags.None)
        RawInputDevice.MouseInput.Add(fun args ->
            //printfn "(x,y):(%d,%d) Buttons: %A State: %A Wheel: %A" args.X args.Y args.ButtonFlags args.Mode args.WheelDelta
            if uint32(args.ButtonFlags &&& MouseButtonFlags.LeftButtonDown) <> 0u then isMouseLeftButtonDown := true
            if uint32(args.ButtonFlags &&& MouseButtonFlags.LeftButtonUp) <> 0u then isMouseLeftButtonDown := false

            if !isMouseLeftButtonDown && args.X <> 0 then
                let r = float(-args.X) / 150.0 * Math.PI * 0.25 |> float32
                world := Matrix.Multiply(!world, Matrix.RotationY(r))

            if !isMouseLeftButtonDown && args.Y <> 0 then
                let r = float(-args.Y) / 150.0 * Math.PI * 0.25 |> float32
                world := Matrix.Multiply(!world, Matrix.RotationX(r))

            match args.WheelDelta with
            | delta when delta > 0 -> world := Matrix.Multiply(!world, Matrix.Scaling(1.01f))
            | delta when delta < 0 -> world := Matrix.Multiply(!world, Matrix.Scaling(0.99f))
            | _ -> ())

        let clock = System.Diagnostics.Stopwatch.StartNew()

        let render () = 
            hook clock

            ctx.D3D9Device.Clear(ClearFlags.Target ||| ClearFlags.ZBuffer, ColorBGRA(0uy, 40uy, 100uy, 0uy), 1.0f, 0)
            ctx.D3D9Device.BeginScene()

            ctx.D3D9Device.SetTransform(TransformState.World, world)

            match renderType with
            | RenderType.Point -> ctx.D3D9Device.DrawPrimitives(PrimitiveType.PointList, 0, elements)
            | RenderType.Mesh -> failwith "Mesh rendering not implemented yet."

            ctx.D3D9Device.EndScene()
            ctx.D3D9Device.Present()

        RenderLoop.Run(ctx.Form, RenderLoop.RenderCallback(render))

    type AnimationParam<'T> =
        {
            Order : MatrixStorageOrder
            RowLength : int
            ColLength : int
            RowSpace : AxisSpace<'T>
            ColSpace : AxisSpace<'T>
            ValSpace : AxisSpace<'T> option
            RowPtr : deviceptr<'T>
            ColPtr : deviceptr<'T>
            Frame : float -> deviceptr<'T> * (AxisSpace<'T> option)
            LoopTime : float option
        }

        member this.Elements = this.RowLength * this.ColLength

    let inline animationLoop (ctx:ApplicationContext) (param:AnimationParam<'T>) =
        use vb = createVertexBuffer ctx param.Elements
        use vd = createVertexDeclaration ctx
        let vbRes = ctx.RegisterGraphicsResource(vb)

        try
            use program = fill vbRes param.RowLength param.ColLength |> Compiler.load ctx.Worker
            let fill = program.Run param.Order param.RowPtr param.ColPtr param.RowSpace param.ColSpace param.ValSpace

            let hook (clock:Stopwatch) =
                let time = param.LoopTime |> function
                    | None -> clock.Elapsed.TotalMilliseconds
                    | Some loopTime ->
                        if loopTime <= 0.0 then clock.Elapsed.TotalMilliseconds
                        else
                            let time = clock.Elapsed.TotalMilliseconds
                            if time > loopTime then clock.Restart()
                            clock.Elapsed.TotalMilliseconds
                let valuePtr, valSpace = param.Frame time
                fill valSpace valuePtr

            renderingLoop ctx RenderType.Point vd vb param.Order param.RowLength param.ColLength hook

        finally ctx.UnregisterGraphicsResource(vbRes)


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


