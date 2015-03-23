//[simpled3d9]
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenTK.Graphics.OpenGL;
using SharpDX;
using SharpDX.Direct3D9;
using SharpDX.Windows;
using Alea.CUDA;
using Alea.CUDA.IL;
using D3D9Device = SharpDX.Direct3D9.Device;
using CUDADevice = Alea.CUDA.Device;

namespace Tutorial.Cs.examples.simpled3d9
{
    [AOTCompile]
    class SimpleD3D9 : ILGPUModule
    {
        public const int Width = 1024;
        public const int Height = 1024;
        public const int Total = Width*Height;

        public static readonly dim3 BlockSize = new dim3(16, 16);
        public static readonly dim3 GridSize = new dim3(Width/BlockSize.x, Height/BlockSize.y);
        public static readonly LaunchParam LaunchParam = new LaunchParam(GridSize, BlockSize);

        public SimpleD3D9(GPUModuleTarget target) : base(target)
        {
        }

        [Kernel]
        public void Kernel(deviceptr<float4> pos, float time)
        {
            var x = blockIdx.x*blockDim.x + threadIdx.x;
            var y = blockIdx.y*blockDim.y + threadIdx.y;

            var u = ((float) x)/((float) Width);
            var v = ((float) y)/((float) Height);
            u = u*2.0f - 1.0f;
            v = v*2.0f - 1.0f;

            const float freq = 4.0f;
            var w = LibDevice.__nv_sinf(u*freq + time)*LibDevice.__nv_cosf(v*freq + time)*0.5f;

            unchecked
            {
                pos[y * Width + x] = new float4(u, w, v, LibDevice.__nv_int_as_float((int)0xff00ff00));
            }
        }

        unsafe public void Update(IntPtr vbRes, float time)
        {
            // 1. map resource to cuda space, means lock to cuda space
            var vbRes1 = vbRes;
            CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsMapResources(1, &vbRes1, IntPtr.Zero));

            // 2. get memory pointer from mapped resource
            var vbPtr = IntPtr.Zero;
            var vbSize = IntPtr.Zero;
            CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsResourceGetMappedPointer(&vbPtr, &vbSize, vbRes1));

            // 3. create device pointer, and run the kernel
            var pos = new deviceptr<float4>(vbPtr);
            GPULaunch(Kernel, LaunchParam, pos, time);

            // 4. unmap resource, means unlock, so that DirectX can then use it again
            CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsUnmapResources(1u, &vbRes1, IntPtr.Zero));
        }

        unsafe static Tuple<CUDADevice, IntPtr> Generate(D3D9Device d3d9Device)
        {
            var cuContext = IntPtr.Zero;
            var cuDevice = -1;
            CUDAInterop.cuSafeCall(CUDAInterop.cuD3D9CtxCreate(&cuContext, &cuDevice, 0u, d3d9Device.NativePointer));
            return (new Tuple<CUDADevice, IntPtr>(CUDADevice.DeviceDict[cuDevice], cuContext));
        }

        unsafe static IntPtr RegisterVerticesResource(VertexBuffer vertices)
        {
            var res = IntPtr.Zero;
            CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsD3D9RegisterResource(&res, vertices.NativePointer, 0));
            return res;
        }

        static void UnregisterVerticesResource(IntPtr res)
        {
            CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsUnregisterResource(res));
        }

        public static void Main()
        {
            var form = new RenderForm("SimpleD3D9 by C#") { ClientSize = new Size(1024, 768) };

            var device = new D3D9Device(
                new Direct3D(),
                CUDADevice.Default.ID,
                DeviceType.Hardware,
                form.Handle,
                CreateFlags.HardwareVertexProcessing,
                new PresentParameters(form.ClientSize.Width, form.ClientSize.Height));

            var vertices = new VertexBuffer(device, Utilities.SizeOf<Vector4>()*Total, Usage.WriteOnly,
                VertexFormat.None, Pool.Default);

            var vertexElems = new []
            {
                new VertexElement(0, 0, DeclarationType.Float3, DeclarationMethod.Default, DeclarationUsage.Position, 0),
                new VertexElement(0, 12, DeclarationType.Ubyte4, DeclarationMethod.Default, DeclarationUsage.Color, 0),
                VertexElement.VertexDeclarationEnd
            };

            var vertexDecl = new VertexDeclaration(device, vertexElems);

            var worker = Worker.CreateOnCurrentThreadByFunc(() => Generate(device));
            var updater = new SimpleD3D9(GPUModuleTarget.Worker(worker));

            var view = Matrix.LookAtLH(
                new Vector3(0.0f, 3.0f, -2.0f), // the camera position
                new Vector3(0.0f, 0.0f, 0.0f),  // the look-at position
                new Vector3(0.0f, 1.0f, 0.0f)); // the up direction

            var proj = Matrix.PerspectiveFovLH(
                (float) (Math.PI/4.0), // the horizontal field of view
                1.0f,
                1.0f,
                100.0f);

            device.SetTransform(TransformState.View, view);
            device.SetTransform(TransformState.Projection, proj);
            device.SetRenderState(RenderState.Lighting, false);

            var vbres = RegisterVerticesResource(vertices);
            var clock = System.Diagnostics.Stopwatch.StartNew();

            RenderLoop.Run(form, () =>
            {
                var time = (float) (clock.Elapsed.TotalMilliseconds)/300.0f;
                updater.Update(vbres, time);

                // Now normal D3D9 rendering procedure.
                device.Clear(ClearFlags.Target | ClearFlags.ZBuffer, new ColorBGRA(0, 40, 100, 0), 1.0f, 0);
                device.BeginScene();

                device.VertexDeclaration = vertexDecl;
                device.SetStreamSource(0, vertices, 0, Utilities.SizeOf<Vector4>());
                // we use PointList as the graphics primitives
                device.DrawPrimitives(SharpDX.Direct3D9.PrimitiveType.PointList, 0, Total);

                device.EndScene();
                device.Present();
            });

            UnregisterVerticesResource(vbres);

            updater.Dispose();
            worker.Dispose();
            vertexDecl.Dispose();
            vertices.Dispose();
            device.Dispose();
            form.Dispose();
        }
    }
}
//[/simpled3d9]
