//[startStatic]
using Alea.CUDA;
using Alea.CUDA.IL;
using Microsoft.FSharp.Core;
using NUnit.Framework;

namespace Tutorial.Cs.examples.nbody
{
    public class GpuStaticSimulatorModule : ILGPUModule, ISimulator, ISimulatorTester
    {
        private readonly int _blockSize;
        private readonly string _description;

        public GpuStaticSimulatorModule(GPUModuleTarget target, int blockSize) : base(target)
        {
            _blockSize = blockSize;
            _description = string.Format("GPU.StaticBlockSize({0})", _blockSize);
        }
        //[/startStatic]

        //[StaticComputeBodyAccel]
        public float3 ComputeBodyAccel(float softeningSquared, float4 bodyPos, deviceptr<float4> positions,
                                       int numTiles)
        {
            var sharedPos = __shared__.Array<float4>(_blockSize);
            var acc = new float3(0.0f, 0.0f, 0.0f);

            for (var tile = 0; tile < numTiles; tile++)
            {
                sharedPos[threadIdx.x] = positions[tile*blockDim.x + threadIdx.x];

                Intrinsic.__syncthreads();

                // This is the "tile_calculation" function from the GPUG3 article.
                for (var counter = 0; counter < _blockSize; counter++)
                {
                    acc = Common.BodyBodyInteraction(softeningSquared, acc, bodyPos, sharedPos[counter]);
                }

                Intrinsic.__syncthreads();
            }
            return (acc);
        }
        //[/StaticComputeBodyAccel]

        //[StaticStartKernel]
        [Kernel]
        public void IntegrateBodies(deviceptr<float4> newPos, deviceptr<float4> oldPos, deviceptr<float4> vel,
            int numBodies, float deltaTime, float softeningSquared, float damping, int numTiles)
        {
            var index = threadIdx.x + blockIdx.x*_blockSize;

            if (index >= numBodies) return;
            var position = oldPos[index];
            var accel = ComputeBodyAccel(softeningSquared, position, oldPos, numTiles);

            // acceleration = force \ mass
            // new velocity = old velocity + acceleration*deltaTime
            // note we factor out the body's mass from the equation, here and in bodyBodyInteraction 
            // (because they cancel out).  Thus here force = acceleration
            var velocity = vel[index];

            velocity.x = velocity.x + accel.x*deltaTime;
            velocity.y = velocity.y + accel.y*deltaTime;
            velocity.z = velocity.z + accel.z*deltaTime;

            velocity.x = velocity.x*damping;
            velocity.y = velocity.y*damping;
            velocity.z = velocity.z*damping;

            // new position = old position + velocity*deltaTime
            position.x = position.x + velocity.x*deltaTime;
            position.y = position.y + velocity.y*deltaTime;
            position.z = position.z + velocity.z*deltaTime;

            // store new position and velocity
            newPos[index] = position;
            vel[index] = velocity;
        }
        //[/StaticStartKernel]

        //[StaticPrepareAndLaunchKernel]
        public void IntegrateNbodySystem(deviceptr<float4> newPos, deviceptr<float4> oldPos, 
                                         deviceptr<float4> vel, int numBodies, float deltaTime, 
                                         float softeningSquared, float damping)
        {
            var numBlocks = Alea.CUDA.Utilities.Common.divup(numBodies, _blockSize);
            var numTiles = Alea.CUDA.Utilities.Common.divup(numBodies, _blockSize);
            var lp = new LaunchParam(numBlocks, _blockSize);
            GPULaunch(IntegrateBodies, lp, newPos, oldPos, vel, numBodies, deltaTime, softeningSquared, damping,
                numTiles);
        }
        //[/StaticPrepareAndLaunchKernel]

        //[StaticCreateInfrastructure]
        string ISimulator.Description
        {
            get { return _description; }
        }

        void ISimulator.Integrate(deviceptr<float4> newPos, deviceptr<float4> oldPos, deviceptr<float4> vel,
                                  int numBodies, float deltaTime, float softeningSquared, float damping)
        {
            IntegrateNbodySystem(newPos, oldPos, vel, numBodies, deltaTime, softeningSquared, damping);
        }

        string ISimulatorTester.Description
        {
            get { return _description; }
        }

        void ISimulatorTester.Integrate(float4[] pos, float4[] vel, int numBodies, float deltaTime,
                                        float softeningSquared, float damping, int steps)
        {
            using (var dpos0 = GPUWorker.Malloc<float4>(numBodies))
            using (var dpos1 = GPUWorker.Malloc(pos))
            using (var dvel = GPUWorker.Malloc(vel))
            {
                var pos0 = dpos0.Ptr;
                var pos1 = dpos1.Ptr;
                for (var i = 0; i < steps; i++)
                {
                    var tempPos = pos0;
                    pos0 = pos1;
                    pos1 = tempPos;
                    IntegrateNbodySystem(pos1, pos0, dvel.Ptr, numBodies, deltaTime, softeningSquared, damping);
                }
                GPUWorker.Gather(pos1, pos, FSharpOption<int>.None, FSharpOption<int>.None);
                GPUWorker.Gather(dvel.Ptr, vel, FSharpOption<int>.None, FSharpOption<int>.None);
            }
        }
        //[/StaticCreateInfrastructure]
    }

    //[CompileArchitectures]
    [AOTCompile(AOTOnly = true, SpecificArchs = "sm20;sm30;sm35")]
    public class GpuStaticSimulatorModule64 : GpuStaticSimulatorModule
    {
        public GpuStaticSimulatorModule64(GPUModuleTarget target)
            : base(target, 64)
        {
        }
    }

    [AOTCompile(AOTOnly = true, SpecificArchs = "sm20;sm30;sm35")]
    public class GpuStaticSimulatorModule128 : GpuStaticSimulatorModule
    {
        public GpuStaticSimulatorModule128(GPUModuleTarget target)
            : base(target, 128)
        {
        }
    }

    [AOTCompile(AOTOnly = true, SpecificArchs = "sm20;sm30;sm35")]
    public class GpuStaticSimulatorModule256 : GpuStaticSimulatorModule
    {
        public GpuStaticSimulatorModule256(GPUModuleTarget target)
            : base(target, 256)
        {
        }
    }

    [AOTCompile(AOTOnly = true, SpecificArchs = "sm20;sm30;sm35")]
    public class GpuStaticSimulatorModule512 : GpuStaticSimulatorModule
    {
        public GpuStaticSimulatorModule512(GPUModuleTarget target)
            : base(target, 512)
        {
        }
    }
    //[/CompileArchitectures]

    //[StaticTest]
    public static class GpuStaticSimulatorTests
    {
        [Test]
        public static void Correctness256()
        {
            var target = GPUModuleTarget.DefaultWorker;
            const int numBodies = 256*56;
            var expectedSimulator = new CpuSimulator(target.GetWorker(), numBodies);
            using (var actualSimulator = new GpuStaticSimulatorModule(target, 256))
            {
                Common.Test(expectedSimulator, actualSimulator, numBodies);
            }
        }

        public static void Performence()
        {
            var target = GPUModuleTarget.DefaultWorker;
            const int numBodies = 256*56;
            using (var simulatorModule64 = new GpuStaticSimulatorModule(target, 64))
            using (var simulatorModule128 = new GpuStaticSimulatorModule(target, 128))
            using (var simulatorModule256 = new GpuStaticSimulatorModule(target, 256))
            using (var simulatorModule512 = new GpuStaticSimulatorModule(target, 512))
            using (var simulatorModule640 = new GpuStaticSimulatorModule(target, 640))
            {
                Common.Performance(simulatorModule64, numBodies);
                Common.Performance(simulatorModule128, numBodies);
                Common.Performance(simulatorModule256, numBodies);
                Common.Performance(simulatorModule512, numBodies);
                Common.Performance(simulatorModule640, numBodies);
            }
        }
    }
    //[/StaticTest]
}