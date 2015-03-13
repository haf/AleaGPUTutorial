(*** hide ***)
module Tutorial.Fs.examples.genericReduce.Test

open System
open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework
open FsUnit

open Tutorial.Fs.examples.genericReduce.Plan
open Tutorial.Fs.examples.genericReduce.ReduceApi

let worker = Worker.Default

/// Creating a sum reduction work flow with data in CPU memory
let inline sum () = cuda {      
    let! reducer = rawSum Planner.Default  
    return Entry(fun program (values:'T[]) ->
        let worker = program.Worker
        let reducer = reducer program values.Length
        use ranges = worker.Malloc(reducer.Ranges)
        use rangeTotals = worker.Malloc<'T>(reducer.NumRanges)
        use values = worker.Malloc(values)
        reducer.Reduce ranges.Ptr rangeTotals.Ptr values.Ptr
        let rangeTotals = rangeTotals.Gather()
        rangeTotals.[0]
    ) }

/// Creating a scalar product reduction work flow with data in CPU memory
let inline scalarProd () = cuda {      
    let! reducer = rawScalarProd Planner.Default  
    return Entry(fun program (values1:'T[]) (values2:'T[]) ->
        if values1.Length <> values2.Length then failwith "lenght of vectors must be equal"
        let worker = program.Worker
        let reducer = reducer program values1.Length
        use ranges = worker.Malloc(reducer.Ranges)
        use rangeTotals = worker.Malloc<'T>(reducer.NumRanges)
        use values1 = worker.Malloc(values1)
        use values2 = worker.Malloc(values2)
        reducer.Reduce ranges.Ptr rangeTotals.Ptr values1.Ptr values2.Ptr
        let rangeTotals = rangeTotals.Gather()
        rangeTotals.[0]
    ) }

/// Create a generic reduction work flow with data in CPU memory
let inline reduce init op transf = cuda {      
    let! reducer = rawGeneric Planner.Default init op transf 
    return Entry(fun program (values:'T[]) ->
        let worker = program.Worker
        let reducer = reducer program values.Length
        use ranges = worker.Malloc(reducer.Ranges)
        use rangeTotals = worker.Malloc<'T>(reducer.NumRanges)
        use values = worker.Malloc(values)
        reducer.Reduce ranges.Ptr rangeTotals.Ptr values.Ptr 
        let rangeTotals = rangeTotals.Gather()
        rangeTotals.[0]
    ) }

let ``sum<int>`` () =

    let sum = sum() |> Compiler.load Worker.Default

    let values1 n = Array.init n (fun _ -> 1)
    let values2 n = Array.init n (fun _ -> -1)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> rng.Next(-100, 100))

    let test n = 
        let v1 = values1 n
        let v2 = values2 n
        let v3 = values3 n
        let d1, h1 = sum.Run v1, Array.sum v1
        let d2, h2 = sum.Run v2, Array.sum v2
        let d3, h3 = sum.Run v3, Array.sum v3       
        
        h1 |> should equal d1
        h2 |> should equal d2
        h3 |> should equal d3

    [1; 2; 8; 10; 128; 100; 1024; 1<<<22] |> Seq.iter (test) 

let ``sum<float>`` () =

    let sum = sum() |> Compiler.load Worker.Default

    let values1 n = Array.init n (fun _ -> 1.0)
    let values2 n = Array.init n (fun _ -> -1.0)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> -100.0 + 200.0 * rng.NextDouble())

    let test (n, tol) = 
        let v1 = values1 n
        let v2 = values2 n
        let v3 = values3 n
        let d1, h1 = sum.Run v1, Array.sum v1
        let d2, h2 = sum.Run v2, Array.sum v2
        let d3, h3 = sum.Run v3, Array.sum v3
        
        d1 |> should (equalWithin tol) h1   
        d2 |> should (equalWithin tol) h2   
        d3 |> should (equalWithin tol) h3   

    [1, 1e-11; 2, 1e-11; 8, 1e-11; 10, 1e-11; 128, 1e-11; 100, 1e-11; 1024, 1e-11; 1<<<22, 1e-8] |> Seq.iter (test) 

let ``scalar product<float>`` () =

    let scalarProd = scalarProd() |> Compiler.load Worker.Default

    let values1 n = Array.init n (fun _ -> 1.0), Array.init n (fun _ -> 1.5)
    let values2 n = Array.init n (fun _ -> -1.0), Array.init n (fun _ -> -1.5)
    let values3 n = let rng = Random(2) in Array.init n (fun _ -> -100.0 + 200.0 * rng.NextDouble()), Array.init n (fun _ -> -100.0 + 200.0 * rng.NextDouble())

    let test (n, tol) = 
        let v1, w1 = values1 n
        let v2, w2 = values2 n
        let v3, w3 = values3 n
        let d1, h1 = scalarProd.Run v1 w1, Array.sum (Array.map2 (*) v1 w1)
        let d2, h2 = scalarProd.Run v2 w2, Array.sum (Array.map2 (*) v2 w2)
        let d3, h3 = scalarProd.Run v3 w3, Array.sum (Array.map2 (*) v3 w3)
        
        d1 |> should (equalWithin tol) h1   
        d2 |> should (equalWithin tol) h2   
        d3 |> should (equalWithin tol) h3  

    [1, 1e-11; 2, 1e-11; 8, 1e-10; 10, 1e-10; 128, 1e-10; 100, 1e-10; 1024, 1e-10; 1<<<22, 1e-6] |> Seq.iter (test) 

let ``max<float>`` () =    

    let reduce = reduce <@ fun () -> __neginf() @> <@ max @> <@ id @> |> Compiler.load Worker.Default

    let values1 n = let rng = Random(2) in Array.init n (fun _ -> -100.0 + 200.0 * rng.NextDouble())

    let test (n, tol) = 
        let v1 = values1 n
        let d1, h1 = reduce.Run v1, Array.max v1
            
        d1 |> should (equalWithin tol) h1   

    [1, 1e-11; 2, 1e-11; 8, 1e-10; 10, 1e-10; 128, 1e-10; 100, 1e-10; 1024, 1e-10; 1<<<22, 1e-6] |> Seq.iter (test) 

let ``min<float>`` () =    

    let reduce = reduce <@ fun () -> __posinf() @> <@ min @> <@ id @> |> Compiler.load Worker.Default

    let values1 n = let rng = Random(2) in Array.init n (fun _ -> -100.0 + 200.0 * rng.NextDouble())

    let test (n, tol) = 
        let v1 = values1 n
        let d1, h1 = reduce.Run v1, Array.min v1
        
        d1 |> should (equalWithin tol) h1   

    [1, 1e-11; 2, 1e-11; 8, 1e-10; 10, 1e-10; 128, 1e-10; 100, 1e-10; 1024, 1e-10; 1<<<22, 1e-6] |> Seq.iter (test) 

[<Test>]
let reduceTest () =
    ``sum<int>`` ()
    ``sum<float>`` ()
    ``scalar product<float>`` ()
    ``max<float>`` ()
    ``min<float>`` ()
