(*** hide ***)
module Tutorial.Fs.examples.nbody.Impl.CPU.Simple

open Alea.CUDA
open Tutorial.Fs.examples.nbody.Common

(**
# CPU implementation of the NBody simulation. 

It consists of:

- An integration method (or function in the case of F#)
- Methods creating instances of the classes `ISimulator` and `ISimulatorTester` defined in `Common`.

CPU integration method. First calculates all forces, then integrates including damping the velocity using the `damping`-factor.
*)
(*** define:IntegrateCommonNbodySystem ***)
let integrateNbodySystem (accel : float3[])
                         (pos : float4[])
                         (vel : float4[])
                         (numBodies : int)
                         (deltaTime : float32)
                         (softeningSquared : float32)
                         (damping : float32) =

    // Force of particle i on itselfe is 0 because of the regularisatino of the force.
    // As fij = -fji we could save half of the time, but implement it here as on GPU.
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

        position.x <- position.x + velocity.x * deltaTime
        position.y <- position.y + velocity.y * deltaTime
        position.z <- position.z + velocity.z * deltaTime

        pos.[i] <- position
        vel.[i] <- velocity

(**
Creator functionality for `Simulator` and `SimulatorTester`.
*)
(*** define:CPUTestFunctionality ***)
let createSimulator(worker : Worker, numBodies : int) =
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
            integrateNbodySystem haccel hpos hvel numBodies deltaTime softeningSquared damping
    }

let createSimulatorTester(numBodies : int) =
    let description = "CPU.Simple"
    let accel = Array.zeroCreate<float3> numBodies

    { new ISimulatorTester with

        member x.Description = description

        member x.Integrate pos vel numBodies deltaTime softeningSquared damping steps =
            for i = 1 to steps do
                integrateNbodySystem accel pos vel numBodies deltaTime softeningSquared damping
    }