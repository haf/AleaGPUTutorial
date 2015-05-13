(*** hide ***)
module Tutorial.Fs.examples.genericScan.Test

open System
open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework
open FsUnit

open Tutorial.Fs.examples.genericScan.Plan
open Tutorial.Fs.examples.genericScan.ScanApi

(**
Creating a sum reduction work flow with data in CPU memory
*)
let inline sum () = cuda {      
    let! scanner = rawSum Planner.Default  
    return Entry(fun program (values:'T[]) inclusive ->
        let worker = program.Worker
        let scanner = scanner program values.Length
        use ranges = worker.Malloc(scanner.Ranges)
        use rangeTotals = worker.Malloc<'T>(scanner.NumRanges)
        use values = worker.Malloc(values)
        use results = worker.Malloc(values.Length)
        scanner.Scan ranges.Ptr rangeTotals.Ptr values.Ptr results.Ptr inclusive
        results.Gather()
    ) }

(**
Create a generic reduction work flow with data in CPU memory
*)
let inline scan init op transf = cuda {      
    let! scanner = rawGeneric Planner.Default init op transf 
    return Entry(fun program (values:'T[]) inclusive ->
        let worker = program.Worker
        let scanner = scanner program values.Length
        use ranges = worker.Malloc(scanner.Ranges)
        use rangeTotals = worker.Malloc<'T>(scanner.NumRanges)
        use values = worker.Malloc(values)
        use results = worker.Malloc(values.Length)
        scanner.Scan ranges.Ptr rangeTotals.Ptr values.Ptr results.Ptr inclusive
        results.Gather()
    ) }

(**
CPU scan implementation used for testing.
*)
(*** define:GenericScanCPUScan ***)
let inline sumScan (v:'T[]) incl =
    let vs = Array.scan (+) 0G v
    if incl then Array.sub vs 1 v.Length else Array.sub vs 0 v.Length

(**
Finally we test the different scan interfaces with a few different types.
*)

(**
Sum scan of integers.
*)
(*** define:GenericScanSumInts ***)
let ``sum<int>`` () =

    let sum = sum() |> Compiler.load Worker.Default

    let values1 n = Array.init n (fun _ -> 1)
    let values2 n = Array.init n (fun _ -> -1)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.Next(-100, 100))

    let test incl n = 
        let v1 = values1 n
        let v2 = values2 n
        let v3 = values3 n
        let d1, h1 = sum.Run v1 incl, sumScan v1 incl
        let d2, h2 = sum.Run v2 incl, sumScan v2 incl
        let d3, h3 = sum.Run v3 incl, sumScan v3 incl
        
        d1 |> should equal h1   
        d2 |> should equal h2   
        d3 |> should equal h3  
    
    // Inclusive scan
    [1; 2; 8; 10; 128; 100; 1024; 1<<<22] |> Seq.iter (test true) 
    // Exclusive scan
    [1; 2; 8; 10; 128; 100; 1024; 1<<<22] |> Seq.iter (test false) 

(**
Generic sum scan of integers.
*)
(*** define:GenericScanGenericSumInts ***)
let ``sum<int> generic`` () =

    let sum = scan <@ fun () -> 0 @> <@ ( + ) @> <@ fun x -> x @> |> Compiler.load Worker.Default

    let values1 n = Array.init n (fun _ -> 1)
    let values2 n = Array.init n (fun _ -> -1)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.Next(-100, 100))

    let test incl n = 
        let v1 = values1 n
        let v2 = values2 n
        let v3 = values3 n
        let d1, h1 = sum.Run v1 incl, sumScan v1 incl
        let d2, h2 = sum.Run v2 incl, sumScan v2 incl
        let d3, h3 = sum.Run v3 incl, sumScan v3 incl
        
        d1 |> should equal h1   
        d2 |> should equal h2   
        d3 |> should equal h3   

    // Inclusive scan
    [1; 2; 8; 10; 128; 100; 1024; 1<<<22] |> Seq.iter (test true) 
    // Exclusive scan
    [1; 2; 8; 10; 128; 100; 1024; 1<<<22] |> Seq.iter (test false) 

(**
Sum scan of doubles.
*)
(*** define:GenericScanSumDoubles ***)
let ``sum<float>`` () =

    let sum = sum() |> Compiler.load Worker.Default

    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = Array.init n (fun _ -> -1.0)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> -100.0 + 200.0 * rng.NextDouble())

    let test incl (n, tol) = 
        let v1 = values1 n
        let v2 = values2 n
        let v3 = values3 n
        let d1, h1 = sum.Run v1 incl, sumScan v1 incl
        let d2, h2 = sum.Run v2 incl, sumScan v2 incl
        let d3, h3 = sum.Run v3 incl, sumScan v3 incl
        
        d1 |> should (equalWithin tol) h1   
        d2 |> should (equalWithin tol) h2   
        d3 |> should (equalWithin tol) h3   

    // Inclusive scan
    [1, 1e-12; 2, 1e-12; 8, 1e-12; 10, 1e-12; 128, 1e-12; 100, 1e-12; 1024, 1e-10; 1<<<22, 1e-8] |> Seq.iter (test true) 
    // Exclusive scan
    [1, 1e-12; 2, 1e-12; 8, 1e-12; 10, 1e-12; 128, 1e-12; 100, 1e-12; 1024, 1e-10; 1<<<22, 1e-8] |> Seq.iter (test false) 

(**
Generic sum scan of doubles.
*)
(*** define:GenericScanGenericSumDoubles ***)
let ``sum<float> generic`` () =

    let sum = scan <@ fun () -> 0.0 @> <@ ( + ) @> <@ fun x -> x @> |> Compiler.load Worker.Default

    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = Array.init n (fun _ -> -1.0)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> -100.0 + 200.0 * rng.NextDouble())

    let test incl (n, tol) = 
        let v1 = values1 n
        let v2 = values2 n
        let v3 = values3 n
        let d1, h1 = sum.Run v1 incl, sumScan v1 incl
        let d2, h2 = sum.Run v2 incl, sumScan v2 incl
        let d3, h3 = sum.Run v3 incl, sumScan v3 incl
        
        d1 |> should (equalWithin tol) h1   
        d2 |> should (equalWithin tol) h2   
        d3 |> should (equalWithin tol) h3   

    // Inclusive scan
    [1, 1e-12; 2, 1e-12; 8, 1e-12; 10, 1e-12; 128, 1e-12; 100, 1e-12; 1024, 1e-10; 1<<<22, 1e-8] |> Seq.iter (test true) 
    // Exclusive scan
    [1, 1e-12; 2, 1e-12; 8, 1e-12; 10, 1e-12; 128, 1e-12; 100, 1e-12; 1024, 1e-10; 1<<<22, 1e-8] |> Seq.iter (test false) 

(**
Scan tests.
*)
(*** define:GenericScanTests ***)
[<Test>]
let scanTest () =
    ``sum<int>`` ()
    ``sum<int> generic`` ()
    ``sum<float>`` ()
    ``sum<float> generic`` ()