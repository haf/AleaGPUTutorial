(*** hide ***)
module Tutorial.Fs.examples.cublas.Util

open System
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.CULib
open NUnit.Framework
open FsUnit

(**
# Util

Alea GPU doesn't ship native cublas library. You need make sure the cublas native library can be found
on your system. To ensure this, you can try:

- If you installed CUDA Toolkit, then the libraries are in your system PATH. If not, please
  add it. Alternatively, you can add `<cuBLAS path="/path/to/nvidia/bin/">` in your `app.config`
  file. Usually this path is `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\bin`.
- By default, Alea GPU uses CUDA 6.5 version to locate the native libraries. You can change 
  this by adding `<cuBLAS version="7.0"/>` in your `app.config` file.

Here we create a fallback function to our tests in case the native library cannot be found.
*)
let fallback (f:unit -> unit) =
    match Alea.CUDA.PlatformUtil.Instance.OperatingSystem with
    | OperatingSystem.MacOSX ->
        Assert.Inconclusive("CUBLAS destroy has some issues in macosx when deinit it in finalizer.")
    | _ ->
        try
            f()
        with :? TypeInitializationException as ex ->
            match ex.InnerException with
            | :? DllNotFoundException ->
                Assert.Inconclusive("Native libraries cannot be found, please setup your environment, or use app.config.")
            | _ -> raise ex

