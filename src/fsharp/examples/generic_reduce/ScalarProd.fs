(*** hide ***)
module Tutorial.Fs.examples.genericReduce.ScalarProd

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Plan

(**
Scalar product reduces ranges and store reduced values in array of the range totals. 
*)
//(*** define:genericReduceScalarProdUpsweepKernel ***)
let inline reduceUpSweepKernel (plan:Plan) =
    let numThreads = plan.NumThreads
    let numWarps = plan.NumWarps
    let logNumWarps = log2 numWarps
    let multiReduce = Sum.multiReduce numWarps logNumWarps

    <@ fun (dValues1:deviceptr<'T>) (dValues2:deviceptr<'T>) (dRanges:deviceptr<int>) (dRangeTotals:deviceptr<'T>) ->
        let block = blockIdx.x
        let tid = threadIdx.x
        let rangeX = dRanges.[block]
        let rangeY = dRanges.[block + 1]

        // Loop through all elements in the interval, adding up values.
        // There is no need to synchronize until we perform the multireduce.
        let mutable sum = 0G
        let mutable index = rangeX + tid
        while index < rangeY do              
            sum <- sum + dValues1.[index]*dValues2.[index]
            index <- index + numThreads

        // Get the total.
        let total = (%multiReduce) tid sum 

        if tid = 0 then dRangeTotals.[block] <- total @>

