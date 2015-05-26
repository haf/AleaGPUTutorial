(*** hide ***)
module Tutorial.Fs.examples.cudnn.Mnist
open System
open System.IO
open Alea.CUDA
open Alea.CUDA.CULib
open Alea.CUDA.CULib.CUBLASInterop
open Alea.CUDA.CULib.CUDNNInterop
open NUnit.Framework
open Layer
open Network

(*** define:CudnnMnistTest ***)
let [<Test>] test() =
    if Worker.Default.Device.Arch.Number < 300 then
        Assert.Inconclusive("running cudnn need at least cuda device of arch 3.0")
    else
        try
            let worker = Worker.Default
            use network = new Network(worker)
            let conv1, conv2 = Layer.conv1 worker, Layer.conv2 worker
            let ip1, ip2 = Layer.ip1 worker, Layer.ip2 worker
    
            printfn "Classifying...."
            let i1, i2, i3 =
                network.ClassifyExample FirstImage conv1 conv2 ip1 ip2,
                network.ClassifyExample SecondImage conv1 conv2 ip1 ip2,
                network.ClassifyExample ThirdImage conv1 conv2 ip1 ip2

            printfn "\n==========================================================\n"
            printfn "Result of Classification: %A, %A, %A" i1 i2 i3
            if i1 <> 1 || i2 <> 3 || i3 <> 5
            then printfn "Test Failed!!"
            else printfn "Test Passed!!"
            printfn "\n==========================================================\n"
    
            Assert.AreEqual(i1, 1)
            Assert.AreEqual(i2, 3)
            Assert.AreEqual(i3, 5)

        with :? System.DllNotFoundException -> Assert.Inconclusive("You need set the environment for cudnn native library to do this test.")