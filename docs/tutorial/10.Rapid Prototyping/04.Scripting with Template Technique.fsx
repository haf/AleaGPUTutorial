(**
# Scripting with Template Technique  

We look at how the template technique can be used for scripting and to explore
the generated IR and PTX code. 

First we have to set the include paths and reference the required assemblies.
*)

#I @"..\..\..\packages\Alea.CUDA\lib\net40"
#I @"..\..\..\packages\NUnit\lib"
#I @"..\..\..\packages\FsUnit\Lib\Net40"
#r "Alea.CUDA.dll"
#r "nunit.framework.dll"
#r "FsUnit.NUnit.dll"

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework
open FsUnit

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- __SOURCE_DIRECTORY__ + @"\..\..\..\packages\Alea.CUDA\private"
Alea.CUDA.Settings.Instance.Resource.Path <- __SOURCE_DIRECTORY__ + @"\..\..\..\Release"

(**
We code the GPU kernel code in with the template technique.
*)
[<ReflectedDefinition>]
let scan (g_odata:deviceptr<int>) (g_idata:deviceptr<int>) (n:int) =
    let temp = __shared__.ExternArray<int>()  
    let thid = threadIdx.x
    let mutable pout = 0
    let mutable pin = 1

    temp.[pout * n + thid] <- if thid > 0 then g_idata.[thid - 1] else 0
    __syncthreads()

    let mutable offset = 1
    while offset < n do
        // swap in and out buffer
        pout <- 1 - pout
        pin <- 1 - pin

        if thid >= offset then
            temp.[pout * n + thid] <- temp.[pin * n + thid] + temp.[pin * n + thid - offset]
        else
            temp.[pout * n + thid] <- temp.[pin * n + thid]

        __syncthreads()
        offset <- offset * 2

    g_odata.[thid] <- temp.[pout * n + thid]

let template = cuda {
    let! kernel = <@ scan @> |> Compiler.DefineKernel

    return Entry(fun program ->
        let worker = program.Worker
        let kernel = program.Apply kernel

        let run (input:int[]) =
            let n = input.Length
            use input = worker.Malloc(input)
            use output = worker.Malloc<int>(n)
            let lp = LaunchParam(1, n, sizeof<int> * n * 2)
            kernel.Launch lp output.Ptr input.Ptr n
            output.Gather()

        run ) }

(**
We compile and link the template in separate stages, so that we can access th LLVM IR and PTX code.
*)
let worker = Worker.Default
let irm = Compiler.Compile(template, CompileOptions.OptimizedConfig).IRModule
let ptxm = Compiler.Link(irm, worker.Device.Arch).PTXModule

(**
Now we can dump the generated code to the console.
*)
irm.Dump()
ptxm.Dump()

let input = [| 3; 1; 7; 0; 4; 1; 6; 3 |]

let program = worker.LoadProgram(ptxm)
let output = program.Run input

(**
CPU reference implementation of scan. Note that we have to drop the last element. 
*)
let cpuScan a = 
    let scan = Array.scan (+) 0 a      
    Array.sub scan 0 input.Length
      
let expected = cpuScan input

output |> should equal expected
    
printfn "cpu result: %A" expected
printfn "gpu result: %A" output

(**
We finish by disposing the program resources.
*)
program.Dispose
