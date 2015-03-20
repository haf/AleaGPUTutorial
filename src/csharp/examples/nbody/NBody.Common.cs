//[startCommon]
using System;
using System.Linq;
using Alea.CUDA;
using NUnit.Framework;

namespace Tutorial.Cs.examples.nbody
{
    //[/startCommon]

    //[ISimulator]
    public interface ISimulator
    {
        string Description { get; }
        void Integrate(deviceptr<float4> newPos, deviceptr<float4> oldPos, deviceptr<float4> vel, int numBodies, float deltaTime, float softeningSquared, float damping);
    }
    //[/ISimulator]

    //[ISimulatorTester]
    public interface ISimulatorTester
    {
        string Description { get; }
        void Integrate(float4[] pos, float4[] vel, int numBodies, float deltaTime, float softeningSquared, float damping, int steps);
    }
    //[/ISimulatorTester]

    public abstract class BodyInitializer
    {
        abstract public int NumBodies { get; set; }
        abstract public float PScale { get; set; }
        abstract public float VScale { get; set; }
        public abstract float4 Position(int i);
        public abstract float4 Velocity(float4 position, int i);

        static float4 Momentum(float4 velocity)
        {
            // we store mass in velocity.w
            var mass = velocity.w;
            return new float4(velocity.x * mass,
                              velocity.y * mass,
                              velocity.z * mass,
                              mass);
        }

        //[adjustMomentum]
        static public void Initialize(BodyInitializer initializer, float clusterScale, float velocityScale, int numBodies,
            out float4[] positions, out float4[] velocities)
        {
            var pscale = clusterScale*Math.Max(1.0f, numBodies/1024.0f);
            var vscale = velocityScale*pscale;
            initializer.NumBodies = numBodies;
            initializer.PScale = pscale;
            initializer.VScale = vscale;
            positions = Enumerable.Range(0, numBodies).Select(initializer.Position).ToArray();
            velocities = positions.Select(initializer.Velocity).ToArray();

            // now we try to adjust velocity to make total momentum = zero.
            var momentums = velocities.Select(Momentum).ToArray();
            var totalMomentum = momentums.Aggregate(new float4(0.0f, 0.0f, 0.0f, 0.0f),
                (accum, momentum) =>
                    new float4(accum.x + momentum.x,
                               accum.y + momentum.y,
                               accum.z + momentum.z, 
                               accum.w + momentum.w));
            Console.WriteLine("total momentum and mass 0 = {0}", totalMomentum);

            // adjust velocities
            velocities = velocities.Select((vel, i) => new float4(
                vel.x - totalMomentum.x / totalMomentum.w,
                vel.y - totalMomentum.y / totalMomentum.w,
                vel.z - totalMomentum.z / totalMomentum.w,
                vel.w)).ToArray();

            // see total momentum after adjustment
            momentums = velocities.Select(Momentum).ToArray();
            totalMomentum = momentums.Aggregate(new float4(0.0f, 0.0f, 0.0f, 0.0f),
                (accum, momentum) =>
                    new float4(accum.x + momentum.x,
                               accum.y + momentum.y,
                               accum.z + momentum.z,
                               accum.w + momentum.w));
            Console.WriteLine("total momentum and mass 1 = {0}", totalMomentum);
        }
        //[/adjustMomentum]
    }

    //[initializeBodies1]
    public class BodyInitializer1 : BodyInitializer
    {
        private readonly Random _random = new Random(42);

        public override int NumBodies { get; set; }
        public override float PScale { get; set; }
        public override float VScale { get; set; }

        private float Rand(float scale, float location)
        {
            return (float) (_random.NextDouble()*scale + location);
        }

        private float RandP()
        {
            return PScale*Rand(1.0f, -0.5f);
        }

        private float RandV()
        {
            return VScale*Rand(1.0f, -0.5f);
        }

        public override float4 Position(int i)
        {
            return new float4(RandP(), RandP(), RandP() + 50.0f, 1.0f);
        }

        public override float4 Velocity(float4 position, int i)
        {
            return new float4(RandV(), RandV(), RandV(), 1.0f);
        }
    }
    //[/initializeBodies1]

    //[initializeBodies2]
    public class BodyInitializer2 : BodyInitializer
    {
        private readonly Random _random = new Random(42);

        public override int NumBodies { get; set; }
        public override float PScale { get; set; }
        public override float VScale { get; set; }

        private float Rand(float scale, float location)
        {
            return (float)(_random.NextDouble() * scale + location);
        }

        private float RandP()
        {
            return PScale * Rand(1.0f, -0.5f);
        }

        private float RandV()
        {
            return VScale * Rand(1.0f, -0.5f);
        }

        public override float4 Position(int i)
        {
            if (i < NumBodies / 2)
            {
                return new float4(RandP() + 0.5f * PScale, RandP(), RandP() + 50.0f, 1.0f);
            }
            else
            {
                return new float4(RandP() - 0.5f * PScale, RandP(), RandP() + 50.0f, 1.0f);
            }
        }

        public override float4 Velocity(float4 position, int i)
        {
            if (i < NumBodies / 2)
            {
                return new float4(RandV(), RandV() + 0.01f * VScale * position.x * position.x, RandV(), 1.0f);
            }
            else
            {
                return new float4(RandV(), RandV() - 0.01f * VScale * position.x * position.x, RandV(), 1.0f);
            }
        }
    }
    //[/initializeBodies2]

    //[initializeBodies3]
    public class BodyInitializer3 : BodyInitializer
    {
        private readonly Random _random = new Random();

        public override int NumBodies { get; set; }
        public override float PScale { get; set; }
        public override float VScale { get; set; }

        private float Rand(float scale, float location)
        {
            return (float)(_random.NextDouble() * scale + location);
        }

        private float RandP()
        {
            return PScale * Rand(1.0f, -0.5f);
        }

        private float RandV()
        {
            return VScale * Rand(1.0f, -0.5f);
        }

        private float RandM()
        {
            return Rand(0.6f, 0.7f);
        }

        public override float4 Position(int i)
        {
            if (i < NumBodies / 2)
            {
                return new float4(RandP() + 0.5f * PScale, RandP(), RandP() + 50.0f, RandM());
            }
            else
            {
                return new float4(RandP() - 0.5f * PScale, RandP(), RandP() + 50.0f, RandM());
            }
        }

        public override float4 Velocity(float4 position, int i)
        {
            if (i < NumBodies / 2)
            {
                return new float4(RandV(), RandV() + 0.01f * VScale * position.x * position.x, RandV(), position.w);
            }
            else
            {
                return new float4(RandV(), RandV() - 0.01f * VScale * position.x * position.x, RandV(), position.w);
            }
        }
    }
    //[/initializeBodies3]

    public static class Common
    {
        //[commonTester]
        public static void Test(ISimulatorTester expectedSimulator, ISimulatorTester actualSimulator, int numBodies)
        {
            const float clusterScale = 1.0f;
            const float velocityScale = 1.0f;
            const float deltaTime = 0.001f;
            const float softeningSquared = 0.00125f;
            const float damping = 0.9995f;
            const int steps = 5;

            Console.WriteLine("Testing {0} against {1} with {2} bodies...", actualSimulator.Description,
                expectedSimulator.Description, numBodies);

            {
                float4[] expectedPos, expectedVel;
                BodyInitializer.Initialize(new BodyInitializer1(), clusterScale, velocityScale, numBodies, out expectedPos, out expectedVel);

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
            }

            {
                float4[] expectedPos, expectedVel;
                BodyInitializer.Initialize(new BodyInitializer3(), clusterScale, velocityScale, numBodies,
                    out expectedPos, out expectedVel);

                for (var i = 0; i < steps; i++)
                {
                    const double tol = 1e-4;
                    var actualPos = new float4[numBodies];
                    var actualVel = new float4[numBodies];
                    Array.Copy(expectedPos, actualPos, numBodies);
                    Array.Copy(expectedVel, actualVel, numBodies);
                    expectedSimulator.Integrate(expectedPos, expectedVel, numBodies, deltaTime, softeningSquared,
                        damping, 1);
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

            Console.WriteLine("Perfomancing {0} with {1} bodies...", simulator.Description, numBodies);

            float4[] pos, vel;
            BodyInitializer.Initialize(new BodyInitializer1(), clusterScale, velocityScale, numBodies, out pos, out vel);
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