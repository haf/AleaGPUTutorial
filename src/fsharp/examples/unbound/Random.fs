(*** hide ***)
module Tutorial.Fs.examples.unbound.Random

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound
open NUnit.Framework
open FsUnit

(**
Random number generation with Alea Unbound. 
*)

open Alea.CUDA.Unbound.Rng

[<Test>]
let randomTest() =
    let numStreams = 1
    let numDimensions = 1
    let seed = 42u
    
    let numPoints = 100
    
    use hostRandom = XorShift7.Host.RandomI32.Create(numStreams, numDimensions, 42u) :> IRandom<uint32>
    use cudaRandom = XorShift7.CUDA.RandomModuleI32.Default.Create(numStreams, numDimensions, 42u) :> IRandom<uint32>
    let hostBuffer = hostRandom.AllocHostStreamBuffer numPoints
    use cudaBuffer = cudaRandom.AllocCUDAStreamBuffer numPoints

    let check streamId =
        hostRandom.Fill(streamId, numPoints, hostBuffer)
        cudaRandom.Fill(streamId, numPoints, cudaBuffer)
        hostBuffer |> should equal (cudaBuffer.Gather())

    [0..numStreams-1] |> List.iter check
