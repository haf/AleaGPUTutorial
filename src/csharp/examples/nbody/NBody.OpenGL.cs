using System;
using System.Collections.Generic;
using System.Diagnostics;
using Alea.CUDA;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;

namespace Tutorial.Cs.examples.nbody
{
    //[startSimWindow]
    public delegate void Disposer();
    public delegate void Del(deviceptr<float4> a, deviceptr<float4> b);

    public class SimWindow : GameWindow
    {
        private readonly int _numBodies;
        private readonly float _deltaTime;
        private readonly float _softeningSquared;
        private readonly float _damping;
        private readonly Worker _worker;
        private readonly Queue<ISimulator> _simulators;
        private readonly Disposer _disposeSimulators;
        private ISimulator _simulator;
        private readonly uint[] _buffers;
        private readonly DeviceMemory<float4> _vel;
        private readonly IntPtr[] _resources;
        private readonly Stopwatch _stopwatch;
        private readonly int _fpsCalcLag;
        private int _frameCounter;
        //[/startSimWindow]

        //[createGLContextGenerator]
        unsafe Tuple<Device, IntPtr> Generate()
        {
            var cuContext = IntPtr.Zero;
            var cuDevice = Device.Default;
            CUDAInterop.cuSafeCall(CUDAInterop.cuGLCtxCreate(&cuContext, 0u, Device.Default.ID));
            return (new Tuple<Device, IntPtr>(cuDevice, cuContext));
        }
        //[/createGLContextGenerator]

        //[GLhelp]
        static void Help()
        {
            Console.WriteLine("Press these keys:");
            Console.WriteLine("[ESC]    : Exit");
            Console.WriteLine("S        : Switch to next simulator");
        }
        //[/GLhelp]

        //[GLdescription]
        void Description()
        {
            var time = _stopwatch.ElapsedMilliseconds;
            var fps = ((float) _frameCounter)*1000.0/((float) time);
            Title = string.Format("bodies {0}, {1} {2} {3} cores, {4}, fps {5}", _numBodies, _worker.Device.Name,
                _worker.Device.Arch, _worker.Device.Cores, _simulator.Description, fps);
            _stopwatch.Restart();
        }
        //[/GLdescription]

        //[LockPositions]
        void LockPos(Del f)
        {
            CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsResourceSetMapFlags(_resources[0],
                (uint) CUDAInterop.CUgraphicsMapResourceFlags_enum.CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY));
            CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsResourceSetMapFlags(_resources[1],
                (uint) CUDAInterop.CUgraphicsMapResourceFlags_enum.CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD));
            CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsMapResourcesEx(2u, _resources, IntPtr.Zero));

            var bytes = IntPtr.Zero;
            var handle0 = IntPtr.Zero;
            var handle1 = IntPtr.Zero;
            unsafe
            {
                CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsResourceGetMappedPointer(&handle0, &bytes, 
                                                                                      _resources[0]));
                CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsResourceGetMappedPointer(&handle1, &bytes, 
                                                                                      _resources[1]));
            }
            var pos0 = new deviceptr<float4>(handle0);
            var pos1 = new deviceptr<float4>(handle1);
            try
            {
                f(pos0, pos1);
            }
            finally
            {
                CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsUnmapResourcesEx(2u, _resources, IntPtr.Zero));
            }
        }
        //[/LockPositions]

        public SimWindow() : base(800, 600, GraphicsMode.Default, "Gravitational n-body simulation")
        {
            _numBodies = 256*64;
            const float clusterScale = 1.0f;
            const float velocityScale = 1.0f;
            _deltaTime = 0.001f;
            _softeningSquared = 0.00125f;
            _damping = 0.9995f;
            //[CreateWorker]
            _worker = Worker.CreateOnCurrentThreadByFunc(Generate);
            //[/CreateWorker]

            _stopwatch = Stopwatch.StartNew();
            _fpsCalcLag = 128;
            _frameCounter = 0;

            //[CreateSimulatros]
            _simulators = new Queue<ISimulator>();
            var target = GPUModuleTarget.Worker(_worker);

            var simulatorGpuDynamicBlockSizeModule = new GpuDynamicSimulatorModule(target);         // need dispose
            var simulatorGpuDynamicBlockSize64 = simulatorGpuDynamicBlockSizeModule.Create(64);
            var simulatorGpuDynamicBlockSize128 = simulatorGpuDynamicBlockSizeModule.Create(128);
            var simulatorGpuDynamicBlockSize256 = simulatorGpuDynamicBlockSizeModule.Create(256);
            var simulatorGpuDynamicBlockSize512 = simulatorGpuDynamicBlockSizeModule.Create(512);

            var simulatorGpuStaticBlockSizeModule64 = new GpuStaticSimulatorModule64(target);       // need dispose
            var simulatorGpuStaticBlockSizeModule128 = new GpuStaticSimulatorModule128(target);     // need dispose
            var simulatorGpuStaticBlockSizeModule256 = new GpuStaticSimulatorModule256(target);     // need dispose
            var simulatorGpuStaticBlockSizeModule512 = new GpuStaticSimulatorModule512(target);     // need dispose

            // First, enquene one simulator which is 256 blocksize so we can compare with C code for performance.
            _simulators.Enqueue(simulatorGpuStaticBlockSizeModule256);

            // Enqueue several dynamic block size simulators.
            _simulators.Enqueue(simulatorGpuDynamicBlockSize64);
            _simulators.Enqueue(simulatorGpuDynamicBlockSize128);
            _simulators.Enqueue(simulatorGpuDynamicBlockSize256);
            _simulators.Enqueue(simulatorGpuDynamicBlockSize512);

            // Enqueue several static block size simulators.
            _simulators.Enqueue(simulatorGpuStaticBlockSizeModule64);
            _simulators.Enqueue(simulatorGpuStaticBlockSizeModule128);
            _simulators.Enqueue(simulatorGpuStaticBlockSizeModule256);
            _simulators.Enqueue(simulatorGpuStaticBlockSizeModule512);

            // We do not enqueue any cpu simulator as it is much too slow.
            //_simulators.Enqueue(new CpuSimulator(_worker, _numBodies));

            _disposeSimulators = () =>
            {
                simulatorGpuDynamicBlockSizeModule.Dispose();
                simulatorGpuStaticBlockSizeModule64.Dispose();
                simulatorGpuStaticBlockSizeModule128.Dispose();
                simulatorGpuStaticBlockSizeModule256.Dispose();
                simulatorGpuStaticBlockSizeModule512.Dispose();
            };

            _simulator = _simulators.Dequeue();
            //[/CreateSimulatros]

            //[CreateBuffers]
            _buffers = new uint[2];
            for (var i = 0; i < _buffers.Length; i++)
            {
                _buffers[i] = 0;
            }
            GL.GenBuffers(_buffers.Length, _buffers);
            foreach (var buffer in _buffers)
            {
                GL.BindBuffer(BufferTarget.ArrayBuffer, buffer);
                GL.BufferData(BufferTarget.ArrayBuffer,
                              (IntPtr) (Microsoft.FSharp.Core.Operators.SizeOf<float4>()*_numBodies), 
                              IntPtr.Zero, BufferUsageHint.DynamicDraw);
                var size = 0;
                unsafe
                {
                    GL.GetBufferParameter(BufferTarget.ArrayBuffer, BufferParameterName.BufferSize, &size);
                }
                if (size != Microsoft.FSharp.Core.Operators.SizeOf<float4>()*_numBodies)
                {
                    throw new Exception("Pixel Buffer Object allocation failed!");
                }
                GL.BindBuffer(BufferTarget.ArrayBuffer, 0);
                CUDAInterop.cuSafeCall(CUDAInterop.cuGLRegisterBufferObject(buffer));
            }

            _resources = new IntPtr[_buffers.Length];
            for (var i = 0; i < _buffers.Length; i++)
            {
                var res = IntPtr.Zero;
                unsafe
                {
                    CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsGLRegisterBuffer(&res, _buffers[i], 0u));
                }
                _resources[i] = res;
            }
            //[/CreateBuffers]

            //[FinalizeGL]
            _vel = _worker.Malloc<float4>(_numBodies);

            float4[] hpos, hvel;
            BodyInitializer.Initialize(new BodyInitializer3(), clusterScale, velocityScale, _numBodies, 
                                       out hpos, out hvel);
            _worker.Scatter(hvel, _vel.Ptr, Microsoft.FSharp.Core.FSharpOption<int>.None,
                Microsoft.FSharp.Core.FSharpOption<int>.None);
            LockPos(
                (pos0, pos1) =>
                    _worker.Scatter(hpos, pos1, Microsoft.FSharp.Core.FSharpOption<int>.None,
                        Microsoft.FSharp.Core.FSharpOption<int>.None));

            Help();
            Description();
            //[/FinalizeGL]
        }


        //[GLswitchSimulators]
        public void SwitchSimulator()
        {
            _simulators.Enqueue(_simulator);
            _simulator = _simulators.Dequeue();
            Description();
            _frameCounter = 0;
            _stopwatch.Restart();
        }
        //[/GLswitchSimulators]

        public void SwapPos()
        {
            var buffer = _buffers[0];
            _buffers[0] = _buffers[1];
            _buffers[1] = buffer;

            var resource = _resources[0];
            _resources[0] = _resources[1];
            _resources[1] = resource;
        }

        void HandleKeyDown(object sender, KeyboardKeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.Escape:
                    Exit();
                    break;
                case Key.S:
                    SwitchSimulator();
                    break;
            }
        }

        //[overrideFuctions]
        protected override void Dispose(bool disposing)
        {
            foreach (var resource in _resources)
            {
                CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsUnregisterResource(resource));
            }
            foreach (var buffer in _buffers)
            {
                CUDAInterop.cuSafeCall(CUDAInterop.cuGLUnregisterBufferObject(buffer));
            }
            if (_buffers.Length > 0)
            {
                GL.DeleteBuffers(_buffers.Length, _buffers);
            }
            if (disposing)
            {
                _vel.Dispose();
                _disposeSimulators();
                _worker.Dispose();
            }
            base.Dispose(disposing);
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
            Keyboard.KeyDown += HandleKeyDown;
            VSync = VSyncMode.Off;
            GL.ClearColor(0.0f, 0.0f, 0.0f, 0.0f); // black as the universe
            GL.Enable(EnableCap.DepthTest);
        }
        
        protected override void OnResize(EventArgs e)
        {
            base.OnResize(e);
            GL.Viewport(ClientRectangle.X, ClientRectangle.Y, ClientRectangle.Width, ClientRectangle.Height);
            var projection = Matrix4.CreatePerspectiveFieldOfView((float) Math.PI/4.0f, 
                                                                  (float) Width/Height, 1.0f, 64.0f);
            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadMatrix(ref projection);
        }
        //[/overrideFuctions]

        //[render]
        protected override void OnRenderFrame(FrameEventArgs e)
        {
            base.OnRenderFrame(e);

            _frameCounter++;
            if (_frameCounter >= _fpsCalcLag)
            {
                Description();
                _frameCounter = 0;
            }

            SwapPos();
            LockPos( (pos0,pos1) => _simulator.Integrate(pos1, pos0, _vel.Ptr, _numBodies, _deltaTime, 
                                                         _softeningSquared, _damping));

            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            var modelview = Matrix4.LookAt(Vector3.Zero, Vector3.UnitZ, Vector3.UnitY);
            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadMatrix(ref modelview);

            GL.Color3(1.0f, 215.0f/255.0f, 0.0f); // golden as the stars
            GL.EnableClientState(ArrayCap.VertexArray);
            GL.BindBuffer(BufferTarget.ArrayBuffer, _buffers[1]);
            GL.VertexPointer(4, VertexPointerType.Float, 0, 0);
            GL.DrawArrays(PrimitiveType.Points, 0, _numBodies);
            GL.DisableClientState(ArrayCap.VertexArray);

            GL.Finish();
            SwapBuffers();
        }
        //[/render]
    }

    //[runSimWindow]
    public static class Run
    {
        public static void Sim()
        {
            using (var window = new SimWindow())
            {
                window.Run();
            }
        }

        public static void Performance()
        {
            GpuDynamicSimulatorTests.Performance();
            GpuStaticSimulatorTests.Performence();
        }
    }
    //[/runSimWindow]
}