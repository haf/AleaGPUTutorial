(*** hide ***)
module Tutorial.Fs.examples.unbound.BlockReduce

(*** hide ***)
open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound
open NUnit.Framework
open FsUnit

module ByTemplate =
     let template = cuda {
        // specialize BlockReduce for a 1D block of 128 threads on type int
        // uhmm one problem, in our code, we miss a auto dispatch method, 
        // now you have to specify the implementation
        let BlockReduce = BlockReduce.WarpReductions<int>(dim3(128), DeviceArch(2, 0))
        let TempStorage = BlockReduce.TempStorage
        
        let! kernel = 
            <@ fun (output:deviceptr<int>) (inputs1:deviceptr<int>) (inputs2:deviceptr<int>) ->
                let tempStorage = TempStorage.AllocateShared()
                let inputs = __local__.Array<int>(2)
                let tid = threadIdx.x
                inputs.[0] <- inputs1.[tid]
                inputs.[1] <- inputs2.[tid]
                let aggregate = BlockReduce.Reduce(tempStorage, inputs, (+))
                if tid = 0 then output.[0] <- aggregate @>
            |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            let run (inputs1:int[]) (inputs2:int[]) =
                Assert.AreEqual(inputs1.Length, 128)
                Assert.AreEqual(inputs2.Length, 128)
                use inputs1 = worker.Malloc(inputs1)
                use inputs2 = worker.Malloc(inputs2)
                use output = worker.Malloc<int>(1)
                let lp = LaunchParam(1, 128)
                kernel.Launch lp output.Ptr inputs1.Ptr inputs2.Ptr
                output.GatherScalar()

            run ) }

    let test() =
        use program = Worker.Default.LoadProgram(template)
        let n = 128
        let inputs1 = Array.init n id
        let inputs2 = Array.create n 1
        let dOutput = program.Run inputs1 inputs2
        let hOutput = (Array.sum inputs1) + (Array.sum inputs2)
        Assert.AreEqual(hOutput, dOutput)

module ByModule =
(*** define:BlockReduceModule ***)
    type BlockReduceModule(target) =
        inherit GPUModule(target)

        let BlockReduce = BlockReduce.WarpReductions<int>(dim3(128), DeviceArch(2, 0))
        let TempStorage = BlockReduce.TempStorage

        [<ReflectedDefinition;Kernel>]
        member this.Kernel (output:deviceptr<int>) (inputs1:deviceptr<int>) (inputs2:deviceptr<int>) =
            let tempStorage = TempStorage.AllocateShared()
            let inputs = __local__.Array<int>(2)
            let tid = threadIdx.x
            inputs.[0] <- inputs1.[tid]
            inputs.[1] <- inputs2.[tid]
            let aggregate = BlockReduce.Reduce(tempStorage, inputs, (+))
            if tid = 0 then output.[0] <- aggregate

        member this.RunTest (inputs1:int[]) (inputs2:int[]) =
            let worker = this.GPUWorker
            Assert.AreEqual(inputs1.Length, 128)
            Assert.AreEqual(inputs2.Length, 128)
            use inputs1 = worker.Malloc(inputs1)
            use inputs2 = worker.Malloc(inputs2)
            use output = worker.Malloc<int>(1)
            let lp = LaunchParam(1, 128)
            this.GPULaunch <@ this.Kernel @> lp output.Ptr inputs1.Ptr inputs2.Ptr
            output.GatherScalar()

(*** define:BlockReduceModuleTest ***)
    let test() =
        use gpuModule = new BlockReduceModule(GPUModuleTarget.DefaultWorker)
        let n = 128
        let inputs1 = Array.init n id
        let inputs2 = Array.create n 1
        let dOutput = gpuModule.RunTest inputs1 inputs2
        let hOutput = (Array.sum inputs1) + (Array.sum inputs2)
        Assert.AreEqual(hOutput, dOutput)

[<Test>]
let BlockReduceTests() =
    ByTemplate.test()
    ByModule.test()