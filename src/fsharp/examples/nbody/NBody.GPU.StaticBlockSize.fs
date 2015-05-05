(*** hide ***)
module Tutorial.Fs.examples.nbody.Impl.GPU.StaticBlockSize

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework
open Tutorial.Fs.examples.nbody


(**
# N-Body GPU implementation with static-block-size

GPU implementation of the N-Body problem where the block-size is known at compile time. Infrastructure around kernel might be slightly more coplex, but the compiler can optimize the code better.

Define a class `SimulatorModule` which takes the `blockSize` as an argument. We will use additional classes in order to inhere from `SimulatorModule` and which specialize for specific `blockSize`s.
*)
(*** define:startStatic ***)
type SimulatorModule(target, blockSize:int) =
    inherit GPUModule(target)

(**
Computation of the accelerations between the particles. The parallelization strategy is nicely described in: 
[GPU Gems 3](http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html),
essentailly it is parallelized over the particles. Particle positions for `blockDim.x` are loaded to shared memory in order to have faster access.
In this version the `blockSize` is known at compile time, in contrast to the `DynamicBlockSize` implementation. Hence the size of shared memory can be set here and does 
not need to be handed over via launch parameters.
*)
(*** define:StaticComputeBodyAccel ***)
    [<ReflectedDefinition>]
    member this.ComputeBodyAccel softeningSquared 
                                 (bodyPos : float4) 
                                 (positions : deviceptr<float4>) 
                                 (numTiles : int) =
        let sharedPos = __shared__.Array<float4>(blockSize)

        let mutable acc = float3(0.0f, 0.0f, 0.0f)

        for tile = 0 to numTiles - 1 do
            sharedPos.[threadIdx.x] <- positions.[tile * blockDim.x + threadIdx.x]

            __syncthreads()

            // This is the "tile_calculation" function from the GPUG3 article.
            __unroll()
            for counter = 0 to blockSize - 1 do
                acc <- bodyBodyInteraction softeningSquared acc bodyPos sharedPos.[counter]

            __syncthreads()

        acc

(**
Integration method on GPU, calls `ComputeBodyAccel` and integrates the equation of motion, including a `damping`-term.
*)
(*** define:StaticStartKernel ***)
    [<Kernel;ReflectedDefinition>]
    member this.IntegrateBodies (newPos : deviceptr<float4>)
                                (oldPos : deviceptr<float4>)
                                (vel : deviceptr<float4>)
                                (numBodies : int)
                                (deltaTime : float32)
                                (softeningSquared : float32)
                                (damping : float32)
                                (numTiles : int) =

        let index = threadIdx.x + blockIdx.x*blockSize

        if index < numBodies then

            let mutable position = oldPos.[index]
            let accel = this.ComputeBodyAccel softeningSquared position oldPos numTiles

            // acceleration = force \ mass
            // new velocity = old velocity + acceleration*deltaTime
            // note we factor out the body's mass from the equation, here and in bodyBodyInteraction 
            // (because they cancel out). Thus here force = acceleration
            let mutable velocity = vel.[index]

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
            newPos.[index] <- position
            vel.[index] <- velocity

(**
Prepare and launch kernel.
Note: `blockSize` will be used as a compile-time constant.
*)
(*** define:StaticPrepareAndLaunchKernel ***)
    member this.IntegrateNbodySystem (newPos : deviceptr<float4>)
                                     (oldPos : deviceptr<float4>)
                                     (vel : deviceptr<float4>)
                                     (numBodies : int)
                                     (deltaTime : float32)
                                     (softeningSquared : float32)
                                     (damping : float32) =

        let numBlocks = divup numBodies blockSize
        let numTiles = divup numBodies blockSize
        let lp = LaunchParam(numBlocks, blockSize)
        this.GPULaunch <@this.IntegrateBodies @> lp newPos oldPos vel numBodies deltaTime 
                                                    softeningSquared damping numTiles

(**
Creating infrastructure for launching and testing.
*)
(*** define:StaticCreateInfrastructure ***)
    member this.CreateSimulator() =
        let description = sprintf "GPU.StaticBlockSize(%d)" blockSize
        { new ISimulator with

            member x.Description = description

            member x.Integrate newPos oldPos vel numBodies deltaTime softeningSquared damping =
                this.IntegrateNbodySystem newPos oldPos vel numBodies deltaTime 
                                          softeningSquared damping
        }

    member this.CreateSimulatorTester() =
        let description = sprintf "GPU.StaticBlockSize(%d)" blockSize
        { new ISimulatorTester with

            member x.Description = description

            member x.Integrate pos vel numBodies deltaTime softeningSquared damping steps =
                use dpos0 = this.GPUWorker.Malloc<float4>(numBodies)
                use dpos1 = this.GPUWorker.Malloc(pos)
                use dvel = this.GPUWorker.Malloc(vel)
                let mutable pos0 = dpos0.Ptr
                let mutable pos1 = dpos1.Ptr

                for i = 1 to steps do
                    let pos' = pos0
                    pos0 <- pos1
                    pos1 <- pos'
                    this.IntegrateNbodySystem pos1 pos0 dvel.Ptr numBodies deltaTime 
                                              softeningSquared damping

                this.GPUWorker.Gather(pos1, pos)
                this.GPUWorker.Gather(dvel.Ptr, vel)
       }

(** 
Fixing `blockSize` and compile.
Compile for all architectures: `sm20`, `sm30`, `sm35` separately.
*)
(*** define:CompileArchitectures ***)
type [<AOTCompile(AOTOnly = true, SpecificArchs = "sm20;sm30;sm35")>] SimulatorModule64(target) = 
    inherit SimulatorModule(target, 64)
type [<AOTCompile(AOTOnly = true, SpecificArchs = "sm20;sm30;sm35")>] SimulatorModule128(target) = 
    inherit SimulatorModule(target, 128)
type [<AOTCompile(AOTOnly = true, SpecificArchs = "sm20;sm30;sm35")>] SimulatorModule256(target) = 
    inherit SimulatorModule(target, 256)
type [<AOTCompile(AOTOnly = true, SpecificArchs = "sm20;sm30;sm35")>] SimulatorModule512(target) = 
    inherit SimulatorModule(target, 512)

(**
Infrastructure for (performance) testing.
*)
(*** define:StaticTest ***)
[<Test>]
let Correctness256() =
    let target = GPUModuleTarget.DefaultWorker
    let numBodies = 256*56
    use actualSimulatorModule = new SimulatorModule256(target)
    let expectedSimulator = Impl.CPU.Simple.createSimulatorTester(numBodies)
    let actualSimulator = actualSimulatorModule.CreateSimulatorTester()
    test expectedSimulator actualSimulator numBodies

let Performance() =
    let target = GPUModuleTarget.DefaultWorker
    let numBodies = 256*56
    use simulatorModule64 = new SimulatorModule64(target)
    use simulatorModule128 = new SimulatorModule128(target)
    use simulatorModule256 = new SimulatorModule256(target)
    use simulatorModule512 = new SimulatorModule512(target)
    performance (simulatorModule64.CreateSimulatorTester()) numBodies
    performance (simulatorModule128.CreateSimulatorTester()) numBodies
    performance (simulatorModule256.CreateSimulatorTester()) numBodies
    performance (simulatorModule512.CreateSimulatorTester()) numBodies