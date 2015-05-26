(*** hide ***)
module Tutorial.Fs.quickStart.ParallelSquare

#if SCRIPT_REFS
#r "..\\..\\..\\packages\\Alea.IL\\lib\\net40\\Alea.IL.dll"
#r "..\\..\\..\\packages\\Alea.CUDA\\lib\\net40\\Alea.CUDA.dll"
#r "..\\..\\..\\packages\\Alea.CUDA.IL\\lib\\net40\\Alea.CUDA.IL.dll"
#r "..\\..\\..\\packages\\NUnit\\lib\\nunit.framework.dll"
#endif

open FSharp.Charting
open System

(**
# Quick Start Example 

This example explains how to square the elements of an array in parallel on the GPU. 

To use Alea GPU we need to open three modules respectively name spaces. For convenience we also
open the `System` namespace.  

*)

(*** define:parallelSquareImport ***) 
open System
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.IL
open NUnit.Framework

let worker = Worker.Default

(**
## Method Based Approach  
 
For this simple example we apply the __method based__ approach which is the most basic GPU programming
technique of Alea GPU.  

With the method based approach everything is implemented in terms of static member functions of a class. 
This technique has some limitations but covers already several common use cases. 

More advanced methods are covered in the section on [advanced programming techniques](../advanced_techniques/index.html). 
A guidance when to use which method can be found [here](../advanced_techniques/comparing_different_techniques.html).
 
A straightforward CPU implementation uses a `for` loop to iterate serially over the whole array. 
For each iteration the computations are all independent and can be executed in parallel. 

*)

(*** define:parallelSquareCPU ***)  
let squareCPU (inputs:float[]) =
    let outputs = Array.zeroCreate inputs.Length
    for i = 0 to inputs.Length - 1 do
        outputs.[i] <- inputs.[i]*inputs.[i]  
    outputs

(**
     
## Basic CUDA Concepts

Before we develop the parallel CUDA version we discuss the main concepts behind the CUDA programming model. 

The first important concept is a __kernel__. A kernel is a program which is executed multiple times on the GPU in each time in a different thread.

The threads which execute a kernel are hierarchically grouped into __blocks of threads__. Every thread block has the same number of threads. 
The collection of all thread blocks form a __grid of blocks__. 

A single thread is identified with a __block index__ representing the block within the grid, and a __thread index__, which identifies 
the thread in the block. For convenience the thread and block index are three dimensional and are accessible within the scope of a kernel 
through the built-in variables `threadIdx.x`, `threadIdx.y`, `threadIdx.z` and `blockIdx.x`, `blockIdx.y`, `blockIdx.z`.
The shape of a block and the extend of the grid are also available through the built-in variables `blockDim.x`, `blockDim.y`, `blockDim.z`
respectively `gridDim.x`, `gridDim.y`, `gridDim.z`.   

There are some hardware limitations: a thread block may contain up to 1024 threads. The limitation of the grid 
in each dimension is 65535 for GPUs of at least compute capability 2.0.

## Parallel CUDA Implementation
 
A common CUDA practice is to launch one thread per data element, which assumes we have enough threads to cover the entire array. 
The downside of this approach is that the grid and block layout becomes data dependent. A slightly more advanced approach is 
to use a fixed grid and block shape and write a GPU kernel which processes more than one element per thread and loops over the 
data array one grid-size at a time until all of the array elements are processed. 
The stride of the loop is equal to the total number of threads in the grid. Assuming a one-dimensional grid and block shape 
the stride becomes `blockDim.x * gridDim.x` 

For the more experienced GPU programmers it is worth mentioning that loops with a stride equal to the grid size also causes 
all addressing within warps to be unit-strided, which implies a maximally coalescing memory access pattern.

Here is a schematic illustration how the threads of a grid of 3 blocks with 4 threads each are mapped to the different array elements:

<img src="../content/images/gpuTransformThreadMapping.png" width="1000" alt="Thread Mapping">

*)
  

(**
## Kernel

The kernel is a static member function with the attribute `AOTCompile`. This attribute selects the Ahead of Time compilation mode. 
For more details about different compilation modes we refer to [Licensing and Deployment](quick_start/licensing_and_deployment.html) section. 
The kernel function obtains a pointer to the output and input data on the GPU of type `deviceptr<float>` and the length of the array to be processed. 

Each thread first calculates where to start in the array and then loops over the array with stride `gridDim.x * blockDim.x`.
*)

(*** define:parallelSquareKernel ***)  
[<ReflectedDefinition; AOTCompile>]
let squareKernel (outputs:deviceptr<float>) (inputs:deviceptr<float>) (n:int) =
    let start = blockIdx.x * blockDim.x + threadIdx.x
    let stride = gridDim.x * blockDim.x
    let mutable i = start 
    while i < n do
        outputs.[i] <- inputs.[i] * inputs.[i]
        i <- i + stride

(**
## Launch Function

We first create a `Worker` object bound to the GPU on which we want to launch our computations. 
The property `Worker.Default` returns a `Worker` bound to the GPU with device id zero. With a worker we can
copy the input array `inputs` to the GPU and allocate memory for the result on the GPU. 

`DeviceMemory` objects returned by `Worker.Malloc` are disposable objects and destroyed once they go out of scope.  

The next step is to determine the grid and block shape. In our example we use a fixed thread block size of 256 threads.
and a grid shape which is partly data dependent: for small arrays we choose enough blocks to cover the entire array
but limit the grid size at 16 times the number of streaming multiprocessors. This would give us enough thread blocks
to efficiently shadow memory access. We then use the `worker` object to launch the kernel on the GPU. 
The last step is to copy the result back to the CPU by using the `Gather` method. 

*)

(*** define:parallelSquareLaunch ***)          
let squareGPU (inputs:float[]) =
    use dInputs = worker.Malloc(inputs)
    use dOutputs = worker.Malloc(inputs.Length)
    let blockSize = 256
    let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT
    let gridSize = Math.Min(16 * numSm, divup inputs.Length blockSize)
    let lp = new LaunchParam(gridSize, blockSize)
    worker.Launch <@ squareKernel @> lp dOutputs.Ptr dInputs.Ptr inputs.Length
    dOutputs.Gather()

(**
## Test 

A simple test program calculates the squares from -5 to 5.

*)

(*** define:parallelSquareTest ***)
[<Test>]  
let squareTest() =
    let inputs = [|-5.0..0.1..5.0|]
    let outputs = squareGPU inputs
    printfn "inputs = %A" inputs
    printfn "outputs = %A" outputs

(*** hide ***)
let squareChart() =
    let inputs = [|-5.0..0.1..5.0|]
    let outputs = squareGPU inputs
    let chart = 
        Chart.Line (Array.zip inputs outputs) 
        |> Chart.WithTitle("Squares")
        |> Chart.WithXAxis(Min = -5.0, Max = 5.0)

    chart.ShowChart() |> ignore

    chart.SaveChartAs("parallelSquareResult.png", ChartTypes.ChartImageFormat.Png)

    Windows.Forms.Application.Run()
    Console.ReadKey(true) |> ignore

(**

A plot shows the correctness of the result.

<img src="../content/images/parallelSquareResult.png" width="500" alt="parallel square result"> 

*)