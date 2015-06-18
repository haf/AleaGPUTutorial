(*** hide ***)
module Tutorial.Fs.examples.cublas.Axpy

open System
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.CULib
open NUnit.Framework
open FsUnit

(**
# Axpy

Here we see a basic example of how to use the cuBLAS library.
The function [<t>axpy](http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-axpy) multiplies the vector 
x by the scalar alpha and adds it to the vector y.
*)

(**
To use the cuBLAS library from Alea GPU you simply need to open `Alea.CUDA.cuBLAS`.
The default implementation is available in `Alea.CUDA.CULib.CUBLAS.Default`.
*)
let cublas = CULib.CUBLAS.Default

(**
First, we implement a simple CPU version of the function `<t>axpy`.
*)
(*** define:axpyCpu ***)
module cpu = 
    let daxpy n alpha (x:float[]) incx (y:float[]) incy =
        [for i in [0..n-1] -> alpha * x.[i] + y.[i]]
        |> Array.ofList

    let zaxpy n (alpha:double2) (x:double2[]) incx (y:double2[]) incy =
        let alpha = alpha.ToComplex()
        let x = double2.ToComplex(x)
        let y = double2.ToComplex(y)
        [for i in [0..n-1] -> alpha * x.[i] + y.[i]]
        |> Array.ofList

(**
The GPU implementation of `<t>axpy` is quite simple. Because cuBlas functions are executed from the host side, 
all you need to do is to allocate the device memory with a worker.
*)
(*** define:axpyGpu ***)
module gpu =
    let daxpy n (alpha:float) (x:float[]) incx (y:float[]) incy =
        let worker = Worker.Default
        use dalpha = worker.Malloc([|alpha|])
        use dx = worker.Malloc(x)
        use dy = worker.Malloc(y)
        cublas.Daxpy(n, dalpha.Ptr, dx.Ptr, incx, dy.Ptr, incy)
        dy.Gather()

    let zaxpy n (alpha:double2) (x:double2[]) incx (y:double2[]) incy =
        let worker = Worker.Default
        use dalpha = worker.Malloc([|alpha|])
        use dx = worker.Malloc(x)
        use dy = worker.Malloc(y)
        cublas.Zaxpy(n, dalpha.Ptr, dx.Ptr, incx, dy.Ptr, incy)
        dy.Gather()

(**
When using cuBLAS in your own implementations, review the official cuBLAS 
documentation and make sure you are gathering the appropriate data from the GPU.

Testing against CPU reference implementation.
*)
(*** define:axpyTest ***)
[<Test>]
let daxpyTest() =
    Util.fallback <| fun _ ->
        let n = 5
        let incx,incy = 1,1
        let alpha = 2.0
        let x = Array.create n 2.0
        let y = Array.create n 1.0
    
        let outputs = gpu.daxpy n alpha x incx y incy
        let expected = cpu.daxpy n alpha x incx y incy

        printfn "cpu result: %A" expected
        printfn "gpu result: %A" outputs

        outputs |> should equal expected

[<Test>]
let zaxpyTest() =
    Util.fallback <| fun _ ->
        let n = 5
        let incx,incy = 1,1
        let alpha = double2(2.0, 2.0)
        let x = Array.init n (fun _ -> double2(2.0, 0.5))
        let y = Array.init n (fun _ -> double2(1.0, 0.5))
    
        let outputs = gpu.zaxpy n alpha x incx y incy |> Array.map (fun x -> Numerics.Complex(x.x, x.y))
        let expected = cpu.zaxpy n alpha x incx y incy

        printfn "cpu result: %A" expected
        printfn "gpu result: %A" outputs
    
        outputs |> should equal expected
