using Alea.CUDA;
using Alea.CUDA.IL;
using Microsoft.FSharp.Core;
using NUnit.Framework;

namespace Tutorial.Cs.examples.nbody
{
    [AOTCompile(AOTOnly = true, SpecificArchs = "sm20;sm30;sm35")]
    public class GpuDynamicSimulatorModule : ILGPUModule
    {
        public GpuDynamicSimulatorModule(GPUModuleTarget target) : base(target)
        {
        }

        public float3 ComputeBodyAccel(float softeningSquared, float4 bodyPos, deviceptr<float4> positions, int numTiles)
        {
            var sharedPos = __shared__.ExternArray<float4>();
            var acc = new float3(0.0f, 0.0f, 0.0f);

            for (var tile = 0; tile < numTiles; tile++)
            {
                sharedPos[threadIdx.x] = positions[tile*blockDim.x + threadIdx.x];

                Intrinsic.__syncthreads();

                // This is the "tile_calculation" function from the GPUG3 article.
                for (var counter = 0; counter < blockDim.x; counter++)
                {
                    acc = Common.BodyBodyInteraction(softeningSquared, acc, bodyPos, sharedPos[counter]);
                }
                Intrinsic.__syncthreads();
            }
            return (acc);
        }

        [Kernel]
        public void IntegrateBodies(deviceptr<float4> newPos, deviceptr<float4> oldPos, deviceptr<float4> vel,
            int numBodies, float deltaTime, float softeningSquared, float damping, int numTiles)
        {
            var index = threadIdx.x + blockIdx.x*blockDim.x;

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

        public void IntegrateNbodySystem(deviceptr<float4> newPos, deviceptr<float4> oldPos, deviceptr<float4> vel,
            int numBodies, float deltaTime, float softeningSquared, float damping, int blockSize)
        {
            var numBlocks = Alea.CUDA.Utilities.Common.divup(numBodies, blockSize);
            var numTiles = Alea.CUDA.Utilities.Common.divup(numBodies, blockSize);
            var sharedMemSize = blockSize * Operators.SizeOf<float4>();
            var lp = new LaunchParam(numBlocks, blockSize, sharedMemSize);
            GPULaunch(IntegrateBodies, lp, newPos, oldPos, vel, numBodies, deltaTime, softeningSquared, damping, numTiles);
        }

        public GpuDynamicSimulator Create(int blockSize)
        {
            return new GpuDynamicSimulator(this, blockSize);
        }
    }
    
    public class GpuDynamicSimulator : ISimulator, ISimulatorTester
    {
        private readonly GpuDynamicSimulatorModule _simMod;
        private readonly int _blockSize;
        private readonly string _description;

        public GpuDynamicSimulator(GpuDynamicSimulatorModule simMod, int blockSize)
        {
            _simMod = simMod;
            _blockSize = blockSize;
            _description = string.Format("GPU.DynamicBlockSize({0})", _blockSize);
        }

        string ISimulator.Description()
        {
            return _description;
        }

        void ISimulator.Integrate(deviceptr<float4> newPos, deviceptr<float4> oldPos, deviceptr<float4> vel, int numBodies, float deltaTime, float softeningSquared, float damping)
        {
            _simMod.IntegrateNbodySystem(newPos, oldPos, vel, numBodies, deltaTime, softeningSquared, damping, _blockSize);
        }

        string ISimulatorTester.Description()
        {
            return _description;
        }

        void ISimulatorTester.Integrate(float4[] pos, float4[] vel, int numBodies, float deltaTime, float softeningSquared, float damping,
            int steps)
        {
            using (var dpos0 = _simMod.GPUWorker.Malloc<float4>(numBodies))
            using (var dpos1 = _simMod.GPUWorker.Malloc(pos))
            using (var dvel = _simMod.GPUWorker.Malloc(vel))
            {
                var pos0 = dpos0.Ptr;
                var pos1 = dpos1.Ptr;
                for (var i = 0; i < steps; i++)
                {
                    var tempPos = pos0;
                    pos0 = pos1;
                    pos1 = tempPos;
                    _simMod.IntegrateNbodySystem(pos1, pos0, dvel.Ptr, numBodies, deltaTime, softeningSquared, damping, _blockSize);
                }
                _simMod.GPUWorker.Gather(pos1, pos, FSharpOption<int>.None, FSharpOption<int>.None);
                _simMod.GPUWorker.Gather(dvel.Ptr, vel, FSharpOption<int>.None, FSharpOption<int>.None);
            }
        }
    }

    [AOTCompile]
    public static class GpuDynamicSimulatorTests
    {
        [Test]
        public static void Correctness()
        {
            var target = GPUModuleTarget.DefaultWorker;
            const int numBodies = 256*56;
            var expectedSimulator = new CpuSimulator(target.GetWorker(), numBodies);
            using (var actualSimulatorModule = new GpuDynamicSimulatorModule(target))
            {
                Common.Test(expectedSimulator, actualSimulatorModule.Create(128), numBodies);
                Common.Test(expectedSimulator, actualSimulatorModule.Create(512), numBodies);
            }
        }

        public static void Performance()
        {
            var target = GPUModuleTarget.DefaultWorker;
            const int numBodies = 256*56;
            using (var simulatorModule = new GpuDynamicSimulatorModule(target))
            {
                Common.Performance(simulatorModule.Create(64), numBodies);
                Common.Performance(simulatorModule.Create(128), numBodies);
                Common.Performance(simulatorModule.Create(256), numBodies);
                Common.Performance(simulatorModule.Create(512), numBodies);
            }
        }
    }
}