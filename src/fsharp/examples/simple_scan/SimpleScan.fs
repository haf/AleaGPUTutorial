(*** hide ***)
module Tutorial.Fs.examples.simpleScan

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework
open FsUnit

(**
# Simple Scan

We implement a simplified scan kernel using shared memory. 
*)
[<ReflectedDefinition; AOTCompile>]
let scanKernel (g_odata:deviceptr<int>) (g_idata:deviceptr<int>) (n:int) =
(**
The shared memory is allocated when invoking the kernel. See the setup of the launch params below.
*)
    let temp = __shared__.ExternArray<int>()  
    let thid = threadIdx.x
    let mutable pout = 0
    let mutable pin = 1

(**
Load input into shared memory. 
*)
    temp.[pout * n + thid] <- if thid > 0 then g_idata.[thid - 1] else 0
    __syncthreads()

(**
Then we do an exclusive scan so we shift one right and set first element to 0.
*)
    let mutable offset = 1
    while offset < n do
        // swap in and out buffer
        pout <- 1 - pout
        pin <- 1 - pin

        // TODO check if we can safely drop the if condition because we initialized the shared array with leading zeros
        if thid >= offset then
            temp.[pout * n + thid] <- temp.[pin * n + thid] + temp.[pin * n + thid - offset]
        else
            temp.[pout * n + thid] <- temp.[pin * n + thid]

        __syncthreads()
        offset <- offset * 2

    g_odata.[thid] <- temp.[pout * n + thid]

(**
Scan host function with memory copy to and from GPU.  
*)
let scan (inputs:int[]) =
    let worker = Worker.Default
    let n = inputs.Length
    use dInputs = worker.Malloc(inputs)
    use dOutputs = worker.Malloc(n)

(**
Setup the launch parameters with sufficient amount of shared memory.
*)
    let lp = LaunchParam(1, n, sizeof<int> * n * 2)
    worker.Launch <@ scanKernel @> lp dOutputs.Ptr dInputs.Ptr inputs.Length
    dOutputs.Gather()

(**
Here is a CPU reference implementation of scan. Note that we have to drop the last element. 
*)
let cpuScan a = 
    let scan = Array.scan (+) 0 a      
    Array.sub scan 0 a.Length

(**
Testing against CPU reference implementation. 
*)
[<Test>]
let scanTest () =
    let inputs = [| 3; 1; 7; 0; 4; 1; 6; 3 |]
    let outputs = scan inputs
    let expected = cpuScan inputs

    outputs |> should equal expected
    
    printfn "cpu result: %A" expected
    printfn "gpu result: %A" outputs  

