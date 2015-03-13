(*** hide ***)
module Tutorial.Fs.examples.genericReduce.ReduceApi

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Plan

// UpsweepKernel values ranges rangeTotals
type UpsweepKernel<'T> = deviceptr<'T> -> deviceptr<int> -> deviceptr<'T> -> unit

// ReduceKernel numRanges rangeTotals
type ReduceKernel<'T> = int -> deviceptr<'T> -> unit

let plan32 : Plan = { NumThreads = 1024; ValuesPerThread = 4; NumThreadsReduction = 256; BlockPerSm = 1 }
let plan64 : Plan = { NumThreads = 512; ValuesPerThread = 4; NumThreadsReduction = 256; BlockPerSm = 1 }

/// Raw reducer interface. It is created by given the number of reducing values.
/// Then it calcuate the ranges, which is an integer array of size numRanges + 1.
/// The member Ranges returns for a program an size the ranges and the number of range totals.
/// The member Reduce takes a program, the num range totals, the ranges on the device, 
/// the range totals on the device and the data array on the device and executes the reduction. 
/// After calling this function, the result is stored in rangeTotals[0], you should decide
/// a way to extract that value, because it will dispear once you run Reduce for another input.
type Raw<'T when 'T:unmanaged> =
    /// num ranges
    abstract NumRanges : int
    /// ranges
    abstract Ranges : int[]
    /// ranges -> rangeTotals -> values -> unit
    abstract Reduce : deviceptr<int> -> deviceptr<'T> -> deviceptr<'T> -> unit

let raw (planner:Planner) (upsweep:Plan -> Expr<UpsweepKernel<'T>>) (reduce:Plan -> Expr<ReduceKernel<'T>>) = cuda {
    let plan =
        match planner with
        | Planner.Default -> if sizeof<'T> > 4 then plan64 else plan32
        | Planner.Specific(plan) -> plan

    let! upsweep = upsweep plan |> Compiler.DefineKernelWithName "reduceUpSweep"  
    let! reduce = reduce plan |> Compiler.DefineKernelWithName "reduce"

    return (fun (program:Program) ->
        let worker = program.Worker
        let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT 
        let upsweep = program.Apply(upsweep)
        let reduce = program.Apply(reduce)

        fun (n:int) ->
            let ranges, numRanges = plan.BlockRanges numSm n
            let lpUpsweep = LaunchParam(numRanges, plan.NumThreads)
            let lpReduce = LaunchParam(1, plan.NumThreadsReduction)

            { new Raw<'T> with
                member this.Ranges = ranges
                member this.NumRanges = numRanges
                member this.Reduce ranges rangeTotals values = 
                    fun () ->
                        // Launch range reduction kernel to calculate the totals per range.
                        upsweep.Launch lpUpsweep values ranges rangeTotals

                        // Need to aggregate the block sums as well.
                        if numRanges > 1 then reduce.Launch lpReduce numRanges rangeTotals
                    |> worker.Eval // the two kernels should be launched together without interrupt.
            } 
        )        
    }

let rawGeneric planner init op transf =
    let upsweep = Reduce.reduceUpSweepKernel init op transf
    let reduce = Reduce.reduceRangeTotalsKernel init op
    raw planner upsweep reduce

let inline rawSum planner =
    let upsweep = Sum.reduceUpSweepKernel
    let reduce = Sum.reduceRangeTotalsKernel
    raw planner upsweep reduce

// UpsweepKernel2 values ranges rangeTotals
type UpsweepKernel2<'T> = deviceptr<'T> -> deviceptr<'T> -> deviceptr<int> -> deviceptr<'T> -> unit

type Raw2<'T when 'T:unmanaged> =
    /// num ranges
    abstract NumRanges : int
    /// ranges
    abstract Ranges : int[]
    /// ranges -> rangeTotals -> values1 -> values2 -> unit
    abstract Reduce : deviceptr<int> -> deviceptr<'T> -> deviceptr<'T> -> deviceptr<'T> -> unit

let raw2 (planner:Planner) (upsweep:Plan -> Expr<UpsweepKernel2<'T>>) (reduce:Plan -> Expr<ReduceKernel<'T>>) = cuda {
    let plan =
        match planner with
        | Planner.Default -> if sizeof<'T> > 4 then plan64 else plan32
        | Planner.Specific(plan) -> plan

    let! upsweep = upsweep plan |> Compiler.DefineKernelWithName "reduceUpSweep"
    let! reduce = reduce plan |> Compiler.DefineKernelWithName "reduce"

    return (fun (program:Program) ->
        let worker = program.Worker
        let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT 
        let upsweep = program.Apply(upsweep)
        let reduce = program.Apply(reduce)

        fun (n:int) ->
            let ranges, numRanges = plan.BlockRanges numSm n
            let lpUpsweep = LaunchParam(numRanges, plan.NumThreads)
            let lpReduce = LaunchParam(1, plan.NumThreadsReduction)

            { new Raw2<'T> with
                member this.Ranges = ranges
                member this.NumRanges = numRanges
                member this.Reduce ranges rangeTotals values1 values2 = 
                    fun () ->
                        // Launch range reduction kernel to calculate the totals per range.
                        upsweep.Launch lpUpsweep values1 values2 ranges rangeTotals

                        // Need to aggregate the block sums as well.
                        if numRanges > 1 then reduce.Launch lpReduce numRanges rangeTotals
                    |> worker.Eval // the two kernels should be launched together without interrupt.
            }
        ) 
    }

let inline rawScalarProd planner =
    let upsweep = ScalarProd.reduceUpSweepKernel
    let reduce = Sum.reduceRangeTotalsKernel
    raw2 planner upsweep reduce


