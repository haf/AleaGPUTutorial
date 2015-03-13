(*** hide ***)
module Tutorial.Fs.examples.genericScan.ScanApi

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Plan
open Tutorial.Fs.examples.genericReduce

// UpsweepKernel values ranges rangeTotals
type UpsweepKernel<'T> = ReduceApi.UpsweepKernel<'T>

// ReduceKernel numRanges rangeTotals
type ReduceKernel<'T> = ReduceApi.ReduceKernel<'T>

// DownsweepKernel values -> results -> rangeTotals -> ranges -> inclusive
type DownsweepKernel<'T> = deviceptr<'T> -> deviceptr<'T> -> deviceptr<'T> -> deviceptr<int> -> int -> unit

let plan32 : Plan = { NumThreads = 1024; ValuesPerThread = 4; NumThreadsReduction = 256; BlockPerSm = 1 }
let plan64 : Plan = { NumThreads = 512; ValuesPerThread = 4; NumThreadsReduction = 256; BlockPerSm = 1 }

// Raw scanner interface. It is created by given the number of scanning values.
// Then it calcuate the ranges, which is an integer array of size numRanges + 1.
// The member NumRangeTotals gives you a hint on the minumal size requirement on rangeTotals.
// The member Scan takes this signature:
// Scan : program -> num range totals -> ranges -> rangeTotals -> values -> results -> inclusive -> unit
type Raw<'T when 'T : unmanaged> =
    /// num ranges
    abstract NumRanges : int
    /// ranges
    abstract Ranges : int[]    
    /// ranges rangeTotals input output inclusive
    abstract Scan : deviceptr<int> -> deviceptr<'T> -> deviceptr<'T> -> deviceptr<'T> -> bool -> unit

/// Scan builder to unify scan cuda monad with a function taking the kernel1, kernel2, kernel3 as args.
let raw (planner:Planner) (upsweep:Plan -> Expr<UpsweepKernel<'T>>) (reduce:Plan -> Expr<ReduceKernel<'T>>) (downsweep:Plan -> Expr<DownsweepKernel<'T>>) = cuda {
    let plan =
        match planner with
        | Planner.Default -> if sizeof<'T> > 4 then plan64 else plan32
        | Planner.Specific(plan) -> plan

    let! upsweep = upsweep plan |> Compiler.DefineKernel  
    let! reduce = reduce plan |> Compiler.DefineKernel  
    let! downsweep = downsweep plan |> Compiler.DefineKernel  

    return (fun (program:Program) ->
        let worker = program.Worker
        let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT 
        let upsweep = program.Apply(upsweep)
        let reduce = program.Apply(reduce)
        let downsweep = program.Apply(downsweep)

        fun (n:int) ->
            let ranges, numRanges = plan.BlockRanges numSm n
            let lpUpsweep = LaunchParam(numRanges, plan.NumThreads)
            let lpReduce = LaunchParam(1, plan.NumThreadsReduction)
            let lpDownsweep = LaunchParam(numRanges, plan.NumThreads)

            { new Raw<'T> with
                member this.NumRanges = numRanges + 1
                member this.Ranges = ranges             
                member this.Scan ranges rangeTotals input output inclusive = 
                    let inclusive = if inclusive then 1 else 0

                    fun () ->
                        upsweep.Launch lpUpsweep input ranges rangeTotals
                        reduce.Launch lpReduce numRanges rangeTotals
                        downsweep.Launch lpDownsweep input output rangeTotals ranges inclusive
                    |> worker.Eval // the three kernels should be launched together without interrupt.
            } ) }

/// <summary>
/// Global scan algorithm template. 
/// </summary>
let rawGeneric planner init op transf =
    let upsweep = Reduce.reduceUpSweepKernel init op transf
    let reduce = Scan.scanReduceKernel init op transf
    let downsweep = Scan.scanDownSweepKernel init op transf
    raw planner upsweep reduce downsweep

/// <summary>
/// Global scan algorithm template. 
/// </summary>
let inline rawSum planner = 
    let upsweep = Tutorial.Fs.examples.genericReduce.Sum.reduceUpSweepKernel
    let reduce = Tutorial.Fs.examples.genericScan.Sum.scanReduceKernel
    let downsweep = Tutorial.Fs.examples.genericScan.Sum.scanDownSweepKernel
    raw planner upsweep reduce downsweep


