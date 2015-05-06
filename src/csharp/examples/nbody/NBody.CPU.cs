using Alea.CUDA;
using Microsoft.FSharp.Core;

namespace Tutorial.Cs.examples.nbody
{
    public static class CpuIntegrator
    {
        //[IntegrateCommonNbodySystem]
        public static void IntegrateNbodySystem(float3[] accel, float4[] pos, float4[] vel, int numBodies,
            float deltaTime, float softeningSquared, float damping)
        {
            // Force of particle i on itselfe is 0 because of the regularisatino of the force.
            // As fij = -fji we could save half of the time, but implement it here as on GPU.            
            for (var i = 0; i < numBodies; i++)
            {
                var acc = new float3(0.0f, 0.0f, 0.0f);
                for (var j = 0; j < numBodies; j++)
                {
                    acc = Common.BodyBodyInteraction(softeningSquared, acc, pos[i], pos[j]);
                }
                accel[i] = acc;
            }
            for (var i = 0; i < numBodies; i++)
            {

                var position = pos[i];
                var localAccel = accel[i];

                // acceleration = force \ mass
                // new velocity = old velocity + acceleration*deltaTime
                // note we factor out the body's mass from the equation, here and in bodyBodyInteraction 
                // (because they cancel out).  Thus here force = acceleration
                var velocity = vel[i];

                velocity.x = velocity.x + localAccel.x*deltaTime;
                velocity.y = velocity.y + localAccel.y*deltaTime;
                velocity.z = velocity.z + localAccel.z*deltaTime;

                velocity.x = velocity.x*damping;
                velocity.y = velocity.y*damping;
                velocity.z = velocity.z*damping;

                // new position = old position + velocity*deltaTime
                position.x = position.x + velocity.x*deltaTime;
                position.y = position.y + velocity.y*deltaTime;
                position.z = position.z + velocity.z*deltaTime;

                // store new position and velocity
                pos[i] = position;
                vel[i] = velocity;
            }
        }
        //[/IntegrateCommonNbodySystem]
    }

    //[CPUTestFunctionality]
    public class CpuSimulator : ISimulator, ISimulatorTester
    {
        private readonly Worker _worker;
        private readonly int _numBodies;
        private readonly float3[] _haccel;
        private readonly float4[] _hpos;
        private readonly float4[] _hvel;
        private readonly string _description;

        public CpuSimulator(Worker worker, int numBodies)
        {
            _worker = worker;
            _numBodies = numBodies;
            _haccel = new float3[numBodies];
            _hpos = new float4[numBodies];
            _hvel = new float4[numBodies];

            for (var i = 0; i < numBodies; i++)
            {
                _haccel[i] = new float3(0.0f, 0.0f, 0.0f);
                _hpos[i] = new float4(0.0f, 0.0f, 0.0f, 0.0f);
                _hvel[i] = new float4(0.0f, 0.0f, 0.0f, 0.0f);
            }

            _description = "CPU.Simple";
        }

        string ISimulator.Description
        {
            get { return _description; }
        }

        void ISimulator.Integrate(deviceptr<float4> newPos, deviceptr<float4> oldPos, deviceptr<float4> vel, int numBodies, float deltaTime, float softeningSquared, float damping)
        {
            _worker.Gather(oldPos, _hpos, FSharpOption<int>.None, FSharpOption<int>.None);
            _worker.Gather(vel, _hvel, FSharpOption<int>.None, FSharpOption<int>.None);
            CpuIntegrator.IntegrateNbodySystem(_haccel, _hpos, _hvel, _numBodies, deltaTime, softeningSquared, damping);
        }

        string ISimulatorTester.Description
        {
            get { return _description; }
        }

        void ISimulatorTester.Integrate(float4[] pos, float4[] vel, int numBodies, float deltaTime, float softeningSquared, float damping, int steps)
        {
            for (var i = 0; i < steps; i++)
            {
                CpuIntegrator.IntegrateNbodySystem(_haccel, pos, vel, numBodies, deltaTime, softeningSquared, damping);
            }
        }
    }
    //[/CPUTestFunctionality]
}