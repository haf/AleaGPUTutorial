(*** define:simpled3d9 ***)
module Tutorial.Fs.examples.simpled3d9.SimpleD3D9

open System
open System.Runtime.InteropServices
open Microsoft.FSharp.NativeInterop
open SharpDX
open SharpDX.Direct3D9
open SharpDX.Windows
open Alea.CUDA

// these two warning switch will turn off those warning that we are operating on unsafe 
// native stuff
#nowarn "9"
#nowarn "51"

type D3D9Device = SharpDX.Direct3D9.Device

// We define the point numbers in width and height. They should be multiple of 16 according
// to the implementation.
let width = 1024
let height = 1024
let total = width * height

// a switcher for gpu or cpu calculation
let usegpu = true

let windowSize = Drawing.Size(1024, 768)

[<AOTCompile>]
type VerticesUpdater(target) =
    inherit GPUModule(target)

    let blockSize = dim3(16, 16)
    let gridSize = dim3(width / blockSize.x, height / blockSize.y)
    let lp = LaunchParam(gridSize, blockSize)

    // a host impl
    member this.Update(vertices:VertexBuffer, time) = 
        let genwave (time:float32) =
            Array.init total (fun i ->
                let x = i % width
                let y = i / width

                let u = float32(x) / float32(width)
                let v = float32(y) / float32(height)
                let u = u * 2.0f - 1.0f
                let v = v * 2.0f - 1.0f

                let freq = 4.0f
                let w = sin(u * freq + time) * cos(v * freq + time) * 0.5f

                Vector4(u, w, v, __nv_int_as_float(0xff00ff00u)))

        vertices.Lock(0, 0, LockFlags.None).WriteRange(genwave time)
        vertices.Unlock()

    // The kernel to generate wave-like points.
    [<Kernel;ReflectedDefinition>]
    member this.Kernel (pos:deviceptr<float4>) (width:int) (height:int) (time:float32) =
        let x = blockIdx.x * blockDim.x + threadIdx.x
        let y = blockIdx.y * blockDim.y + threadIdx.y

        let u = float32(x) / float32(width)
        let v = float32(y) / float32(height)
        let u = u * 2.0f - 1.0f
        let v = v * 2.0f - 1.0f

        let freq = 4.0f
        let w = sin(u * freq + time) * cos(v * freq + time) * 0.5f

        pos.[y * width + x] <- float4(u, w, v, __nv_int_as_float(0xff00ff00u))

    // Launch the kernel, by mapping the vertex buffer to device pointer.
    member this.Update(vbRes:CUgraphicsResource, time:float32) =
        // 1. map resource to cuda space, means lock to cuda space
        let mutable vbRes = vbRes
        cuSafeCall(cuGraphicsMapResources(1u, &&vbRes, 0n))

        // 2. get memory pointer from mapped resource
        let mutable vbPtr = 0n
        let mutable vbSize = 0n
        cuSafeCall(cuGraphicsResourceGetMappedPointer(&&vbPtr, &&vbSize, vbRes))

        // 3. create device pointer, and run the kernel
        let pos = deviceptr<float4>(vbPtr)
        this.GPULaunch <@ this.Kernel @> lp pos width height time
                
        // 4. unmap resource, means unlock, so that DirectX can then use it again
        cuSafeCall(cuGraphicsUnmapResources(1u, &&vbRes, 0n))

// Create D3D9 device, then create cuda context, then loop. In each iteration, we calculate
// the time, then map (lock) the vertex buffer into CUDA device pointer, and update it
// directly with kernel.
let main () =
    // create a form
    use form = new RenderForm("SimpleD3D9 by F#")
    form.ClientSize <- windowSize

    // create a D3D9 device
    use device = new D3D9Device(new Direct3D(), // Direct3D interface (COM)
                                Device.Default.ID, // display adapter (device id)
                                DeviceType.Hardware, // device type
                                form.Handle, // focus window
                                CreateFlags.HardwareVertexProcessing, // behavior flags
                                PresentParameters(form.ClientSize.Width, form.ClientSize.Height))

    // create vertex buffer, NOTICE, the pool type MUST be Pool.Default, which let it possible for CUDA to process.
    use vertices = new VertexBuffer(device, Utilities.SizeOf<Vector4>() * total, Usage.WriteOnly, VertexFormat.None, Pool.Default)

    // define the FVF of the vertex, first 3 float is for position, last float will be reinterpreted to 4 bytes for the color
    let vertexElems = [| VertexElement(0s, 0s, DeclarationType.Float3, DeclarationMethod.Default, DeclarationUsage.Position, 0uy)
                         VertexElement(0s, 12s, DeclarationType.Ubyte4, DeclarationMethod.Default, DeclarationUsage.Color, 0uy)
                         VertexElement.VertexDeclarationEnd |]
    use vertexDecl = new VertexDeclaration(device, vertexElems)

    let view = Matrix.LookAtLH(Vector3(0.0f, 3.0f, -2.0f),          // the camera position
                               Vector3(0.0f, 0.0f, 0.0f),           // the look-at position
                               Vector3(0.0f, 1.0f, 0.0f))           // the up direction
    let proj = Matrix.PerspectiveFovLH(float32(Math.PI / 4.0),      // the horizontal field of view
                                       1.0f,
                                       1.0f,
                                       100.0f)

    // IMPORTANT: to interop with CUDA, CUDA context must be created using special API, so we use a customized device worker
    // constructor, which takes a context generation function: unit -> nativeint * Engine.Device.
    use worker =
        let generate() =
            let mutable ctx = 0n
            let mutable dev = -1
            cuSafeCall(cuD3D9CtxCreate(&&ctx, &&dev, 0u, device.NativePointer))
            let dev = Device.DeviceDict.[dev]
            dev, ctx
        Worker.CreateOnCurrentThread(generate)

    use updater = new VerticesUpdater(GPUModuleTarget.Worker(worker))

    let registerVerticesResource() =
        let mutable res = 0n
        cuSafeCall(cuGraphicsD3D9RegisterResource(&&res, vertices.NativePointer, 0u))
        res

    let unregisterVerticesResource res =
        cuSafeCall(cuGraphicsUnregisterResource(res))

    let vbres = if usegpu then registerVerticesResource() else 0n

    device.SetTransform(TransformState.View, view)
    device.SetTransform(TransformState.Projection, proj)
    device.SetRenderState(RenderState.Lighting, false)

    let clock = System.Diagnostics.Stopwatch.StartNew()

    let render () = 
        // on each render, first calculate the vertex buffer.
        let time = float32(clock.Elapsed.TotalMilliseconds / 300.0 )
        match usegpu with
        | true -> updater.Update(vbres, time)
        | false -> updater.Update(vertices, time)
        
        // Now normal D3D9 render procedure.
        device.Clear(ClearFlags.Target ||| ClearFlags.ZBuffer, ColorBGRA(0uy, 40uy, 100uy, 0uy), 1.0f, 0)
        device.BeginScene()

        device.VertexDeclaration <- vertexDecl
        device.SetStreamSource(0, vertices, 0, Utilities.SizeOf<Vector4>())
        // we use PointList as the graphics primitives
        device.DrawPrimitives(PrimitiveType.PointList, 0, total)

        device.EndScene()
        device.Present()

    RenderLoop.Run(form, RenderLoop.RenderCallback(render))

    // unregister the vertex buffer
    if usegpu then unregisterVerticesResource vbres
