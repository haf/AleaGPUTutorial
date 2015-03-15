//[startCommon]
using System;
using Alea.CUDA;
using NUnit.Framework;

namespace Tutorial.Cs.examples.nbody
{
    //[/startCommon]

    //[ISimulator]
    public interface ISimulator
    {
        string Description();
        void Integrate(deviceptr<float4> newPos, deviceptr<float4> oldPos, deviceptr<float4> vel, int numBodies, float deltaTime, float softeningSquared, float damping);
    }
    //[/ISimulator]

    //[ISimulatorTester]
    public interface ISimulatorTester
    {
        string Description();
        void Integrate(float4[] pos, float4[] vel, int numBodies, float deltaTime, float softeningSquared, float damping, int steps);
    }
    //[/ISimulatorTester]

    public static class Common
    {
        delegate float Del(float a, float b);
        delegate float Deleg();

        //[initializeBodies1]
        public static Tuple<float4[], float4[]> InitializeBodies1(float clusterScale, float velocityScale, int numBodies)
        {
            var rng = new Random(42);
            Del random = (a, b) => (float) (rng.NextDouble()*a - b);
            var scale = clusterScale * Math.Max( 1.0f, numBodies/1024.0f);
            var vscale = velocityScale*scale;

            Deleg randomPos = () => scale * random(1.0f, 0.5f);
            Deleg randomVel = () => vscale * random(1.0f, 0.5f);

            var pos = new float4[numBodies];
            var vel = new float4[numBodies];
            for (var i = 0; i < numBodies; i++)
            {
                pos[i] = new float4(randomPos(), randomPos(), randomPos() + 50.0f, 1.0f);
                vel[i] = new float4(randomVel(), randomVel(), randomVel(), 1.0f);

            }
            var totalMomentum = new float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (var i = 0; i < numBodies; i++)
            {
                totalMomentum = new float4(totalMomentum.x + vel[i].x*vel[i].w, totalMomentum.y + vel[i].y*vel[i].w, totalMomentum.z + vel[i].z*vel[i].w, totalMomentum.w + vel[i].w);
            }
            for (var i = 0; i < numBodies; i++)
            {
                vel[i] = new float4(vel[i].x - totalMomentum.x / totalMomentum.w / vel[i].w, vel[i].y - totalMomentum.y / totalMomentum.w / vel[i].w, vel[i].z - totalMomentum.z / totalMomentum.w / vel[i].w, vel[i].w);
            }
            return (new Tuple<float4[], float4[]>(pos, vel));
        }
        //[/initializeBodies1]

        //[initializeBodies2]
        public static Tuple<float4[], float4[]> InitializeBodies2(float clusterScale, float velocityScale, int numBodies)
        {
            var rng = new Random(42);
            Del random = (a, b) => (float)(rng.NextDouble()*a - b);
            var scale = clusterScale * Math.Max(1.0f, numBodies/1024.0f);
            var vscale = velocityScale*scale;

            Deleg randomPos = () => scale * random(1.0f, 0.5f);
            Deleg randomVel = () => vscale * random(1.0f, 0.5f);
            var pos = new float4[numBodies];
            var vel = new float4[numBodies];
            for (var i = 0; i < numBodies; i++)
            {
                if (i < numBodies/2)
                {
                    pos[i] = new float4(randomPos() + 0.5f*scale, randomPos(), randomPos() + 50.0f, 1.0f);
                    vel[i] = new float4(randomVel(), randomVel() + 0.01f * vscale * pos[i].x * pos[i].x, randomVel(), 1.0f);
                }
                else
                {
                    pos[i] = new float4(randomPos() - 0.5f*scale, randomPos(), randomPos() + 50.0f, 1.0f);
                    vel[i] = new float4(randomVel(), randomVel() - 0.01f*vscale*pos[i].x*pos[i].x, randomVel(), 1.0f);
                }
            }
            var totalMomentum = new float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (var i = 0; i < numBodies; i++)
            {
                totalMomentum = new float4(totalMomentum.x + vel[i].x*vel[i].w, totalMomentum.y + vel[i].y*vel[i].w, totalMomentum.z + vel[i].z*vel[i].w, totalMomentum.w + vel[i].w);
            }
            for (var i = 0; i < numBodies; i++)
            {
                vel[i] = new float4(vel[i].x - totalMomentum.x / totalMomentum.w / vel[i].w, vel[i].y - totalMomentum.y / totalMomentum.w / vel[i].w, vel[i].z - totalMomentum.z / totalMomentum.w / vel[i].w, vel[i].w);
            }
            return (new Tuple<float4[], float4[]>(pos, vel));
        }
        //[/initializeBodies2]

        //[initializeBodies3]
        public static Tuple<float4[], float4[]> InitializeBodies3(float clusterScale, float velocityScale, int numBodies)
        {
            var rng = new Random(42);
            Del random = (a, b) => (float) (rng.NextDouble()*a - b);
            var scale = clusterScale * Math.Max( 1.0f, numBodies/1024.0f);
            var vscale = velocityScale*scale;

            Deleg randomPos = () => scale * random(1.0f, 0.5f);
            Deleg randomVel = () => vscale * random(1.0f, 0.5f);
            Deleg randomMass = () => (float) (rng.NextDouble()*(1.3 - 0.7) + 0.7);

            var pos = new float4[numBodies];
            var vel = new float4[numBodies];
            for (var i = 0; i < numBodies; i++)
            {
                if (i < numBodies/2)
                {
                    pos[i] = new float4(randomPos() + 0.5f*scale, randomPos(), randomPos() + 50.0f, randomMass());
                    vel[i] = new float4(randomVel(), randomVel() + 0.01f*vscale*pos[i].x*pos[i].x, randomVel(), 1.0f / pos[i].w);
                }
                else
                {
                    pos[i] = new float4(randomPos() - 0.5f*scale, randomPos(), randomPos() + 50.0f, randomMass());
                    vel[i] = new float4(randomVel(), randomVel() - 0.01f*vscale*pos[i].x*pos[i].x, randomVel(), 1.0f/pos[i].w);
                }
            }
            var totalMomentum = new float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (var i = 0; i < numBodies; i++)
            {
                totalMomentum = new float4(totalMomentum.x + vel[i].x*vel[i].w, totalMomentum.y + vel[i].y*vel[i].w, totalMomentum.z + vel[i].z*vel[i].w, totalMomentum.w + vel[i].w);
            }
            for (var i = 0; i < numBodies; i++)
            {
                vel[i] = new float4(vel[i].x - totalMomentum.x / totalMomentum.w / vel[i].w, vel[i].y - totalMomentum.y / totalMomentum.w / vel[i].w, vel[i].z - totalMomentum.z / totalMomentum.w / vel[i].w, vel[i].w);
            }
            return (new Tuple<float4[], float4[]>(pos, vel));
        }
        //[/initializeBodies3]

        //[commonTester]
        public static void Test(ISimulatorTester expectedSimulator, ISimulatorTester actualSimulator, int numBodies)
        {
            const float clusterScale = 1.0f;
            const float velocityScale = 1.0f;
            const float deltaTime = 0.001f;
            const float softeningSquared = 0.00125f;
            const float damping = 0.9995f;
            const int steps = 5;

            Console.WriteLine("Testing {0} against {1} with {2} bodies...", actualSimulator.Description(),
                expectedSimulator.Description(), numBodies);

            var res = InitializeBodies1(clusterScale, velocityScale, numBodies);
            var expectedPos = res.Item1;
            var expectedVel = res.Item2;

            for (var i = 0; i < steps; i++)
            {
                const double tol = 1e-5;
                var actualPos = new float4[numBodies];
                var actualVel = new float4[numBodies];
                Array.Copy(expectedPos, actualPos, numBodies);
                Array.Copy(expectedVel, actualVel, numBodies);
                expectedSimulator.Integrate(expectedPos, expectedVel, numBodies, deltaTime, softeningSquared, damping, 1);
                actualSimulator.Integrate(actualPos, actualVel, numBodies, deltaTime, softeningSquared, damping, 1);
                for (var j = 0; j < expectedPos.Length; j++)
                {
                    Assert.AreEqual(actualPos[j].x, expectedPos[j].x, tol);
                    Assert.AreEqual(actualPos[j].y, expectedPos[j].y, tol);
                    Assert.AreEqual(actualPos[j].z, expectedPos[j].z, tol);
                    Assert.AreEqual(actualPos[j].w, expectedPos[j].w, tol);
                }
            }

            res = InitializeBodies3(clusterScale, velocityScale, numBodies);
            expectedPos = res.Item1;
            expectedVel = res.Item2;

            for (var i = 0; i < steps; i++)
            {
                const double tol = 1e-4;
                var actualPos = new float4[numBodies];
                var actualVel = new float4[numBodies];
                Array.Copy(expectedPos, actualPos, numBodies);
                Array.Copy(expectedVel, actualVel, numBodies);
                expectedSimulator.Integrate(expectedPos, expectedVel, numBodies, deltaTime, softeningSquared, damping, 1);
                actualSimulator.Integrate(actualPos, actualVel, numBodies, deltaTime, softeningSquared, damping, 1);
                for (var j = 0; j < expectedPos.Length; j++)
                {
                    Assert.AreEqual(actualPos[j].x, expectedPos[j].x, tol);
                    Assert.AreEqual(actualPos[j].y, expectedPos[j].y, tol);
                    Assert.AreEqual(actualPos[j].z, expectedPos[j].z, tol);
                    Assert.AreEqual(actualPos[j].w, expectedPos[j].w, tol);
                }
            }
        }
        //[/commonTester]

        //[commonPerfTester]
        public static void Performance(ISimulatorTester simulator, int numBodies)
        {
            const float clusterScale = 1.0f;
            const float velocityScale = 1.0f;
            const float deltaTime = 0.001f;
            const float softeningSquared = 0.00125f;
            const float damping = 0.9995f;
            const int steps = 10;

            Console.WriteLine("Perfomancing {0} with {1} bodies...", simulator.Description(), numBodies);

            var result = InitializeBodies1(clusterScale, velocityScale, numBodies);
            var pos = result.Item1;
            var vel = result.Item2;
            simulator.Integrate(pos, vel, numBodies, deltaTime, softeningSquared, damping, steps);
        }
        //[/commonPerfTester]


        //[bodyBodyInteraction]
        public static float3 BodyBodyInteraction(float softeningSquared, float3 ai, float4 bi, float4 bj) 
        {
            // r_ij  [3 FLOPS]
            var r = new float3(bj.x - bi.x, bj.y - bi.y, bj.z - bi.z);

            // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
            var distSqr = r.x * r.x + r.y * r.y + r.z * r.z + softeningSquared;

            // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
            var invDist = LibDevice.__nv_rsqrtf(distSqr);
            var invDistCube = invDist * invDist * invDist;
    
            // s = m_j * invDistCube [1 FLOP]
            var s = bj.w * invDistCube;

            // a_i =  a_i + s * r_ij [6 FLOPS]
            return(new float3(ai.x + r.x*s, ai.y + r.y*s, ai.z + r.z*s));
        }
        //[/bodyBodyInteraction]
    }
}