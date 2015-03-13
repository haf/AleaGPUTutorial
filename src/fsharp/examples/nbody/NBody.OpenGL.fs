(**
Visualizing simulation using [OpenGL](https://www.opengl.org/) respectively [OpenTK](http://www.opentk.com/).
*)
(*** define:StartOpenGL ***)
module Tutorial.Fs.examples.NBody.OpenGL
open System
open System.Collections.Generic
open System.Diagnostics
open OpenTK
open OpenTK.Graphics
open OpenTK.Graphics.OpenGL
open Alea.CUDA

#nowarn "9"
#nowarn "51"

(**
We inhere from the [OpenTK](http://www.opentk.com/) class `GameWindow`.
*)
(*** define:startSimWindow ***)
type SimWindow() as this =
    inherit GameWindow(800, 600, GraphicsMode.Default, "Gravitational n-body simulation")

    let numBodies = 256*64
    let clusterScale = 1.0f
    let velocityScale = 1.0f
    //let initializeBodies = initializeBodies1 clusterScale velocityScale
    //let initializeBodies = initializeBodies2 clusterScale velocityScale
    let initializeBodies = initializeBodies3 clusterScale velocityScale
    let deltaTime = 0.001f
    let softeningSquared = 0.00125f
    let damping = 0.9995f
    let stopWatch = Stopwatch.StartNew()
    let fpsCalcLag = 128
    let mutable frameCounter = 0

(**
function to create cuda GL context, needed in order to initialize worker.
*)
(*** define:createGLContextGenerator ***)
    let generate() =
        let mutable cuContext = 0n
        let cuDevice = Device.Default
        cuSafeCall(cuGLCtxCreate(&&cuContext, 0u, Device.Default.ID))
        cuDevice, cuContext
(**
create worker using Cuda GL context generation function.
*)
(*** define:CreateWorker ***)
    // Note, we don't need worker.Eval cause we will run it in single thread.
    let worker =
        Worker.CreateOnCurrentThread(generate)

(**
Create several simulators: dynamic & static with different `blockSize`s for comparison.
*)
(***  define:CreateSimulatros ***)
    let simulators, disposeSimulators =
        let simulators = Queue<ISimulator>()
        let target = GPUModuleTarget.Worker(worker)

        let simulatorCPUSimpleModule = Impl.CPU.Simple.SimulatorModule()
        let simulatorGPUDynamicBlockSizeModule = new Impl.GPU.DynamicBlockSize.SimulatorModule(target)
        let simulatorGPUStaticBlockSizeModule64 = new Impl.GPU.StaticBlockSize.SimulatorModule64(target)
        let simulatorGPUStaticBlockSizeModule128 = new Impl.GPU.StaticBlockSize.SimulatorModule128(target)
        let simulatorGPUStaticBlockSizeModule256 = new Impl.GPU.StaticBlockSize.SimulatorModule256(target)
        let simulatorGPUStaticBlockSizeModule512 = new Impl.GPU.StaticBlockSize.SimulatorModule512(target)

        // first, enquene one simulator which is 256 blocksize so we can compare with C code for performance
        simulators.Enqueue(simulatorGPUStaticBlockSizeModule256.CreateSimulator())

        // now, enqueue several dynamic block size simulators
        simulators.Enqueue(simulatorGPUDynamicBlockSizeModule.CreateSimulator(64))
        simulators.Enqueue(simulatorGPUDynamicBlockSizeModule.CreateSimulator(128))
        simulators.Enqueue(simulatorGPUDynamicBlockSizeModule.CreateSimulator(256))
        simulators.Enqueue(simulatorGPUDynamicBlockSizeModule.CreateSimulator(512))

        // now, enqueue serveral static block size simulators
        simulators.Enqueue(simulatorGPUStaticBlockSizeModule64.CreateSimulator())
        simulators.Enqueue(simulatorGPUStaticBlockSizeModule128.CreateSimulator())
        simulators.Enqueue(simulatorGPUStaticBlockSizeModule256.CreateSimulator())
        simulators.Enqueue(simulatorGPUStaticBlockSizeModule512.CreateSimulator())

        // last, enqueue cpu simulator, this is quite slow, FPS = 0 :)
        //simulators.Enqueue(simulatorCPUSimpleModule.CreateSimulator(worker, numBodies))

        let dispose() =
            simulatorGPUDynamicBlockSizeModule.Dispose()
            simulatorGPUStaticBlockSizeModule64.Dispose()
            simulatorGPUStaticBlockSizeModule128.Dispose()
            simulatorGPUStaticBlockSizeModule256.Dispose()
            simulatorGPUStaticBlockSizeModule512.Dispose()

        simulators, dispose

    let mutable simulator = simulators.Dequeue()

(**
Method to describe simulation in window-title.
*)
(*** define:GLdescription***)
    let description() =
        let time = stopWatch.ElapsedMilliseconds
        let fps = (float frameCounter) * 1000.0 / (float time)
        this.Title <- sprintf "bodies %d, %s %A %d cores, %s, fps %f" numBodies worker.Device.Name worker.Device.Arch worker.Device.Cores simulator.Description fps
        stopWatch.Restart()

(**
Switch between different simulators in order to compare the performence for dynamic & static GPU implementation and different `blockSize`s.
*)
(*** define:GLswitchSimulators***)
    let switchSimulator() =
        simulators.Enqueue(simulator)
        simulator <- simulators.Dequeue()
        description()
        frameCounter <- 0
        stopWatch.Restart()

(**
Display options: exit simulation or switch simulator. 
*)
(*** define:GLhelp***)
    let help() =
        printfn "Press these keys:"
        printfn "[ESC]    : Exit"
        printfn "S        : Switch to next simulator"

(**
Create `buffer`s and `vel`, where the particles position and velocities will be stored in.
Positions are read from one `buffer` and stored to the other, hence the `buffer`s have to change between every integration step. This is what `swapPos` is for.
*)
(*** define:CreateBuffers***)
    let buffers = 
        let buffers = Array.zeroCreate<GLuint> 2
        GL.GenBuffers(buffers.Length, buffers)

        for buffer in buffers do
            GL.BindBuffer(BufferTarget.ArrayBuffer, buffer)            
            GL.BufferData(BufferTarget.ArrayBuffer, nativeint (sizeof<float4>*numBodies), (null : _ []), BufferUsageHint.DynamicDraw )
            let size = ref 0
            GL.GetBufferParameter(BufferTarget.ArrayBuffer, BufferParameterName.BufferSize, size)
            if !size <> sizeof<float4>*numBodies then
                failwith "Pixel Buffer Object allocation failed!"
            GL.BindBuffer(BufferTarget.ArrayBuffer, 0)
            cuSafeCall(cuGLRegisterBufferObject(buffer))

        buffers

    let resources = buffers |> Array.map (fun buffer ->
        let mutable res = 0n
        cuSafeCall(cuGraphicsGLRegisterBuffer(&&res, buffer, 0u))
        res)

    let mutable vel = worker.Malloc<float4>(numBodies)

    let swapPos() =
        let buffer' = buffers.[0]
        buffers.[0] <- buffers.[1]
        buffers.[1] <- buffer'

        let resource' = resources.[0]
        resources.[0] <- resources.[1]
        resources.[1] <- resource'

(**
Locks pointers from acces by [OpenTK](http://www.opentk.com/), s.t. function `f` can work on these pointers.
*)
(*** define:LockPositions ***)
    let lockPos(f:deviceptr<float4> -> deviceptr<float4> -> 'T) =
        cuSafeCall(cuGraphicsResourceSetMapFlags(resources.[0], uint32 CUgraphicsMapResourceFlags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY))
        cuSafeCall(cuGraphicsResourceSetMapFlags(resources.[1], uint32 CUgraphicsMapResourceFlags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD))
        cuSafeCall(cuGraphicsMapResourcesEx(2u, resources, 0n))
        let mutable bytes = 0n
        let mutable handle0 = 0n
        let mutable handle1 = 0n
        cuSafeCall(cuGraphicsResourceGetMappedPointer(&&handle0, &&bytes, resources.[0]))
        cuSafeCall(cuGraphicsResourceGetMappedPointer(&&handle1, &&bytes, resources.[1]))
        let pos0 = deviceptr<float4>(handle0)
        let pos1 = deviceptr<float4>(handle1)
        try f pos0 pos1
        finally cuSafeCall(cuGraphicsUnmapResourcesEx(2u, resources, 0n))

(**
Final steps in order to initialize the simulaiton and the visualization.
*)
(*** define:FinalizeGL ***)
    do
        let hpos, hvel = initializeBodies numBodies
        worker.Scatter(hvel, vel.Ptr)
        lockPos <| fun pos0 pos1 -> worker.Scatter(hpos, pos1)
        help()
        description()

        this.KeyDown.Add(fun e ->
            match e.Key.ToString() with
            | "Escape" -> this.Exit()
            | "S" -> switchSimulator()
            | _ -> () )

(**
Override needed functionality from `GameWindow`.
*)
(*** define:overrideFuctions ***)
    override this.Dispose(disposing) =
        for resource in resources do cuSafeCall(cuGraphicsUnregisterResource(resource))
        for buffer in buffers do cuSafeCall(cuGLUnregisterBufferObject(buffer))
        if buffers.Length > 0 then GL.DeleteBuffers(buffers.Length, buffers)
        if disposing then vel.Dispose()
        if disposing then disposeSimulators()
        if disposing then worker.Dispose()
        base.Dispose(disposing)

    override this.OnLoad e =
        base.OnLoad(e)
        this.VSync <- VSyncMode.Off
        GL.ClearColor(0.0f, 0.0f, 0.0f, 0.0f) // black as the universe
        GL.Enable(EnableCap.DepthTest)

    override this.OnResize e =
        base.OnResize(e)
        GL.Viewport(this.ClientRectangle.X, this.ClientRectangle.Y, this.ClientRectangle.Width, this.ClientRectangle.Height)
        let projection = Matrix4.CreatePerspectiveFieldOfView((float32)Math.PI / 4.0f, float32 this.Width / (float32)this.Height, 1.0f, 64.0f)
        GL.MatrixMode(MatrixMode.Projection)
        GL.LoadMatrix(ref projection)

(**
Render function:

- Updates frame-per-second calculation and if needed description in title.
- Calls `Integrate` method from choosen `simulator`.
- Displays all the particles using [OpenTK](http://www.opentk.com/) functionality.
*)
(*** define:render ***)
    override this.OnRenderFrame e =
        base.OnRenderFrame(e)

        frameCounter <- frameCounter + 1
        if frameCounter >= fpsCalcLag then
            description()
            frameCounter <- 0

        swapPos()
        lockPos <| fun pos0 pos1 -> simulator.Integrate pos1 pos0 vel.Ptr numBodies deltaTime softeningSquared damping

        GL.Clear(ClearBufferMask.ColorBufferBit ||| ClearBufferMask.DepthBufferBit)
        let modelview = Matrix4.LookAt(Vector3.Zero, Vector3.UnitZ, Vector3.UnitY)
        GL.MatrixMode(MatrixMode.Modelview)
        GL.LoadMatrix(ref modelview)

        GL.Color3(1.0f, 215.0f/255.0f, 0.0f) // golden as the stars
        GL.EnableClientState(ArrayCap.VertexArray)
        GL.BindBuffer(BufferTarget.ArrayBuffer, buffers.[1])
        GL.VertexPointer(4, VertexPointerType.Float, 0, 0)
        GL.DrawArrays(PrimitiveType.Points, 0, numBodies)
        GL.DisableClientState(ArrayCap.VertexArray)

        GL.Finish()
        this.SwapBuffers()

(**
Defaults to use `SimWindow`.
*)
(*** define:runSimWindow ***)
let runSim() =
    use window = new SimWindow()
    window.Run()

let runPerformance() =
    Impl.GPU.DynamicBlockSize.Performance()
    Impl.GPU.StaticBlockSize.Performance()