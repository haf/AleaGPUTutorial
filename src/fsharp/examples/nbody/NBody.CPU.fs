(**
CPU implementation of the NBody simulation. It is mainly given to test the GPU implementations and show the difference in performance.
*)
(*** define:StartCPU ***)
module Tutorial.Fs.examples.nbody.Impl.CPU.Simple

open Alea.CUDA
open Tutorial.Fs.examples.nbody.Common

type SimulatorModule() =
(**
CPU integration method. First calculates all forces, then integrates including damping the volicity using a `damping`-factor.
*)
(*** define:IntegrateCommonNbodySystem ***)
    member this.IntegrateNbodySystem (accel:float3[])
                                     (pos:float4[])
                                     (vel:float4[])
                                     (numBodies:int)
                                     (deltaTime:float32)
                                     (softeningSquared:float32)
                                     (damping:float32) =

        // Force of i particle on itselfe is 0 because of the regularisatino of the force.
        // As fij = -fji we could save half of time, but implement it here as on GPU.
        for i = 0 to numBodies - 1 do
            let mutable acc = float3(0.0f, 0.0f, 0.0f)
            for j = 0 to numBodies - 1 do
                acc <- bodyBodyInteraction softeningSquared acc pos.[i] pos.[j]
            accel.[i] <- acc

        for i = 0 to numBodies - 1 do

            let mutable position = pos.[i]
            let accel = accel.[i]

            // acceleration = force \ mass
            // new velocity = old velocity + acceleration*deltaTime
            // note we factor out the body's mass from the equation, here and in bodyBodyInteraction 
            // (because they cancel out).  Thus here force = acceleration
            let mutable velocity = vel.[i]

            velocity.x <- velocity.x + accel.x * deltaTime
            velocity.y <- velocity.y + accel.y * deltaTime
            velocity.z <- velocity.z + accel.z * deltaTime

            velocity.x <- velocity.x * damping
            velocity.y <- velocity.y * damping
            velocity.z <- velocity.z * damping

            // new position = old position + velocity*deltaTime
            position.x <- position.x + velocity.x * deltaTime
            position.y <- position.y + velocity.y * deltaTime
            position.z <- position.z + velocity.z * deltaTime

            // store new position and velocity
            pos.[i] <- position
            vel.[i] <- velocity

(**
Creator functionality for Simulator and SimulatorTester
*)
(*** define:CPUTestFunctionality ***)
    member this.CreateSimulator(worker:Worker, numBodies:int) =
        // create arrays for work, store in closure, to save time for allocation.
        let haccel = Array.zeroCreate<float3> numBodies
        let hpos = Array.zeroCreate<float4> numBodies
        let hvel = Array.zeroCreate<float4> numBodies
        let description = "CPU.Simple"
        
        { new ISimulator with

            member x.Description = description

            member x.Integrate newPos oldPos vel numBodies deltaTime softeningSquared damping =
                worker.Gather(oldPos, hpos)
                worker.Gather(vel, hvel)
                this.IntegrateNbodySystem haccel hpos hvel numBodies deltaTime softeningSquared damping
        }

    member this.CreateSimulatorTester(numBodies:int) =
        let description = "CPU.Simple"
        let accel = Array.zeroCreate<float3> numBodies

        { new ISimulatorTester with

            member x.Description = description

            member x.Integrate pos vel numBodies deltaTime softeningSquared damping steps =
                for i = 1 to steps do
                    this.IntegrateNbodySystem accel pos vel numBodies deltaTime softeningSquared damping
       }