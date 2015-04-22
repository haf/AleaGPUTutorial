(*** hide ***)
module Tutorial.Fs.examples.nbody.Impl.GPU.DynamicBlockSize

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework
open Tutorial.Fs.examples.nbody
(**
# GPU N-Body Implementation with Dynamic Block Size

GPU implementation of the n-body problem where the block-size is not known at compile time. It makes the infrastructure around the kernel a bit
simpler, but might give away possible performance gains due to better optization possibilities by the compiler.


Start a Class `SimulatorModule` and make sure it is (GPU)-compiled ahead of time (AOT).
We specify to compile and optimize for the three specific architectures: `sm20`, `sm30` and `sm35`.
*)
(*** define:DynamicAOTCompile***)
[<AOTCompile(SpecificArchs = "sm20;sm30;sm35")>]
type SimulatorModule(target) =
    inherit GPUModule(target)

(**
Computing the accelerations between the particles. The parallelization strategy is nicely described in: 
[GPU Gems 3](http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html),
essentailly it is parallelized over the particles. Particle positions for `blockDim.x` are loaded to shared memory in order to have faster access.
In this version the `blockDim.x` is not known at compile time and hence loop-unrolling of the inner loop is not possible, see the `StaticBlockSize` 
implementation for comparison.
*)
(*** define:DynamicComputeBodyAccel ***)
    [<ReflectedDefinition>]
    let computeBodyAccel softeningSquared
                         (bodyPos : float4)
                         (positions : deviceptr<float4>)
                         (numTiles : int) =
        let sharedPos = __shared__.ExternArray<float4>()

        let mutable acc = float3(0.0f, 0.0f, 0.0f)

        for tile = 0 to numTiles - 1 do
            sharedPos.[threadIdx.x] <- positions.[tile * blockDim.x + threadIdx.x]

            __syncthreads()

            // This is the "tile_calculation" function from the GPUG3 article.
            for counter = 0 to blockDim.x - 1 do
                acc <- bodyBodyInteraction softeningSquared acc bodyPos sharedPos.[counter]

            __syncthreads()

        acc

(**
Integration method on GPU, calls `ComputeBodyAccel` and integrates the equation of motion, including a `damping`-term.
*)
(*** define:DynamicStartKernel ***)
    [<Kernel;ReflectedDefinition>]
    member this.IntegrateBodies (newPos : deviceptr<float4>)
                                (oldPos : deviceptr<float4>)
                                (vel : deviceptr<float4>)
                                (numBodies : int)
                                (deltaTime : float32)
                                (softeningSquared : float32)
                                (damping : float32)
                                (numTiles : int) =

        let index = threadIdx.x + blockIdx.x*blockDim.x

        if index < numBodies then

            let mutable position = oldPos.[index]
            let accel = computeBodyAccel softeningSquared position oldPos numTiles

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
*)
(*** define:DynamicPrepareAndLaunchKernel ***)
    member this.IntegrateNbodySystem (newPos : deviceptr<float4>)
                                     (oldPos : deviceptr<float4>)
                                     (vel : deviceptr<float4>)
                                     (numBodies : int)
                                     (deltaTime : float32)
                                     (softeningSquared : float32)
                                     (damping : float32)
                                     (blockSize : int) =

        let numBlocks = divup numBodies blockSize
        let numTiles = divup numBodies blockSize
        let sharedMemSize = blockSize * sizeof<float4>
        let lp = LaunchParam(numBlocks, blockSize, sharedMemSize)
        this.GPULaunch <@ this.IntegrateBodies @> lp newPos oldPos vel numBodies deltaTime 
                                                     softeningSquared damping numTiles

(**
Creating infrastructure for launching.
*)
(*** define:DynamicCreateInfrastructure ***)
    member this.CreateSimulator(blockSize : int) =
        let description = sprintf "GPU.DynamicBlockSize(%d)" blockSize
        { new ISimulator with

            member x.Description = description

            member x.Integrate newPos oldPos vel numBodies deltaTime softeningSquared damping =
                this.IntegrateNbodySystem newPos oldPos vel numBodies deltaTime 
                                          softeningSquared damping blockSize
        }

    member this.CreateSimulatorTester(blockSize : int) =
        let description = sprintf "GPU.DynamicBlockSize(%d)" blockSize
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
                    this.IntegrateNbodySystem pos1 pos0 dvel.Ptr numBodies deltaTime softeningSquared damping blockSize

                this.GPUWorker.Gather(pos1, pos)
                this.GPUWorker.Gather(dvel.Ptr, vel)
       }

(**
Infrastructure for (performance) testing.
*)
(*** define:DynamicTest ***)
[<Test>]
let Correctness256() =
    let target = GPUModuleTarget.DefaultWorker
    let numBodies = 256*56
    use actualSimulatorModule = new SimulatorModule(target)
    let expectedSimulator = Impl.CPU.Simple.createSimulatorTester(numBodies)
    let actualSimulator = actualSimulatorModule.CreateSimulatorTester(256)
    test expectedSimulator actualSimulator numBodies

let Performance() =
    let target = GPUModuleTarget.DefaultWorker
    let numBodies = 256*56
    use simulatorModule = new SimulatorModule(target)
    performance (simulatorModule.CreateSimulatorTester(64)) numBodies
    performance (simulatorModule.CreateSimulatorTester(128)) numBodies
    performance (simulatorModule.CreateSimulatorTester(256)) numBodies
    performance (simulatorModule.CreateSimulatorTester(512)) numBodies