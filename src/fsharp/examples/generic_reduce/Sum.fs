(*** hide ***)
module Tutorial.Fs.examples.genericReduce.Sum

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Plan

(**
Sum multi-reduce function for all warps in the block.  This sum-specialized multi reduce function
is unique to the F# implementation of generic reduce.
*)
let inline multiReduce numWarps logNumWarps =
    let warpStride = WARP_SIZE + WARP_SIZE / 2 + 1
    let sharedSize = numWarps * warpStride

    <@ fun tid (x:'T) ->
        let warp = tid / WARP_SIZE
        let lane = tid &&& (WARP_SIZE - 1)
        let shared = __shared__.Array<'T>(sharedSize) |> __array_to_ptr |> __ptr_volatile 
        let warpShared = shared + warp * warpStride     
        let s = warpShared + (lane + WARP_SIZE / 2)

        warpShared.[lane] <- 0G
        s.[0] <- x

        // Run inclusive scan on each warp's data.
        let mutable sum = x |> __unbox
        for i = 0 to LOG_WARP_SIZE - 1 do
            let offset = 1 <<< i
            sum <- sum + s.[-offset]   
            if i < LOG_WARP_SIZE - 1 then s.[0] <- sum
        
        let totalsShared = __shared__.Array<'T>(2*numWarps) |> __array_to_ptr |> __ptr_volatile 

        if lane = WARP_SIZE - 1 then
            totalsShared.[numWarps + warp] <- sum  

        // Synchronize to make all the totals available to the reduction code.
        __syncthreads()

        if tid < numWarps then
            // Grab the block total for the tid'th block. This is the last element
            // in the block's scanned sequence. This operation avoids bank conflicts.
            let total = totalsShared.[numWarps + tid]
            totalsShared.[tid] <- 0G
            let s = totalsShared + numWarps + tid

            let mutable totalsSum = total |> __unbox
            for i = 0 to logNumWarps - 1 do
                let offset = 1 <<< i
                totalsSum <- totalsSum + s.[-offset]
                s.[0] <- totalsSum

        // Synchronize to make the block scan available to all warps.
        __syncthreads()

        // The total is the last element.
        totalsShared.[2 * numWarps - 1] @>

(**
Sum reduces ranges and store reduced values in array of the range totals.  This sum-specialized 
upsweep kernel is unique to the F# implementation of generic reduce.
*)
let inline reduceUpSweepKernel (plan:Plan) =
    let numThreads = plan.NumThreads
    let numWarps = plan.NumWarps
    let logNumWarps = log2 numWarps
    let multiReduce = multiReduce numWarps logNumWarps

    <@ fun (dValues:deviceptr<'T>) (dRanges:deviceptr<int>) (dRangeTotals:deviceptr<'T>) ->
        let block = blockIdx.x
        let tid = threadIdx.x
        let rangeX = dRanges.[block]
        let rangeY = dRanges.[block + 1]

        // Loop through all elements in the interval, adding up values.
        // There is no need to synchronize until we perform the multireduce.
        let mutable sum = 0G |> __unbox
        let mutable index = rangeX + tid
        while index < rangeY do              
            sum <- sum + dValues.[index] 
            index <- index + numThreads

        // Get the total.
        let total = (%multiReduce) tid sum 

        if tid = 0 then dRangeTotals.[block] <- total @>

(**
Sum reduces range totals to a single total, which is written back to the first element in the 
range totals input array.   This sum-specialized reduce range totals kernel is unique to the F# 
implementation of generic reduce.
*)
let inline reduceRangeTotalsKernel (plan:Plan) =
    let numThreads = plan.NumThreadsReduction
    let numWarps = plan.NumWarpsReduction
    let logNumWarps = log2 numWarps
    let multiReduce = multiReduce numWarps logNumWarps

    <@ fun numRanges (dRangeTotals:deviceptr<'T>) ->
        let tid = threadIdx.x

        let x = if tid < numRanges then dRangeTotals.[tid] else 0G
        let total = (%multiReduce) tid x

        // Have the first thread in the block set the range total.
        if tid = 0 then dRangeTotals.[0] <- total @>

