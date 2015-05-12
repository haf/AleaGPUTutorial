(*** hide ***)
module Tutorial.Fs.examples.genericReduce.Reduce

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Plan

(**
Multi-reduce function for all warps in the block.
*)
(*** define:GenericReduceMultiReduce ***)
let multiReduce (initExpr:Expr<unit -> 'T>) (opExpr:Expr<'T -> 'T -> 'T>) numWarps logNumWarps =
    let warpStride = WARP_SIZE + WARP_SIZE / 2 + 1
    let sharedSize = numWarps * warpStride

    <@ fun tid (x:'T) ->
        let init = %initExpr
        let op = %opExpr
        let warp = tid / WARP_SIZE
        let lane = tid &&& (WARP_SIZE - 1)
        let shared = __shared__.Array<'T>(sharedSize) |> __array_to_ptr |> __ptr_volatile
        let warpShared = shared + warp * warpStride     
        let s = warpShared + (lane + WARP_SIZE / 2)

        warpShared.[lane] <- init()  
        s.[0] <- x

        // Run inclusive scan on each warp's data.
        let mutable warpScan = x
        for i = 0 to LOG_WARP_SIZE - 1 do
            let offset = 1 <<< i
            warpScan <- op warpScan s.[-offset]   
            if i < LOG_WARP_SIZE - 1 then s.[0] <- warpScan
        
        let totalsShared = __shared__.Array<'T>(2*numWarps) |> __array_to_ptr |> __ptr_volatile 

        // Last line of warp stores the warp scan.
        if lane = WARP_SIZE - 1 then totalsShared.[numWarps + warp] <- warpScan  

        // Synchronize to make all the totals available to the reduction code.
        __syncthreads()

        // Run an exclusive scan for the warp scans. 
        if tid < numWarps then
            // Grab the block total for the tid'th block. This is the last element
            // in the block's scanned sequence. This operation avoids bank conflicts.
            let total = totalsShared.[numWarps + tid]
            totalsShared.[tid] <- init()
            let s = totalsShared + numWarps + tid

            let mutable totalsScan = total
            for i = 0 to logNumWarps - 1 do
                let offset = 1 <<< i
                totalsScan <- op totalsScan s.[-offset]
                s.[0] <- totalsScan

        // Synchronize to make the block scan available to all warps.
        __syncthreads()

        // The total is the last element.
        totalsShared.[2 * numWarps - 1] @>

(**
Reduces ranges and store reduced values in array of the range totals.  
*)
(*** define:GenericReduceUpsweepKernel ***)   
let reduceUpSweepKernel (initExpr:Expr<unit -> 'T>) (opExpr:Expr<'T -> 'T -> 'T>) (transfExpr:Expr<'T -> 'T>) (plan:Plan) =
    let numThreads = plan.NumThreads
    let numWarps = plan.NumWarps
    let logNumWarps = log2 numWarps
    let multiReduce = multiReduce initExpr opExpr numWarps logNumWarps

    <@ fun (dValues:deviceptr<'T>) (dRanges:deviceptr<int>) (dRangeTotals:deviceptr<'T>) ->
        let init = %initExpr
        let op = %opExpr
        let transf = %transfExpr

        // Each block is processing a range.
        let range = blockIdx.x
        let tid = threadIdx.x
        let rangeX = dRanges.[range]
        let rangeY = dRanges.[range + 1]

        // Loop through all elements in the interval, adding up values.
        // There is no need to synchronize until we perform the multireduce.
        let mutable reduced = init()
        let mutable index = rangeX + tid
        while index < rangeY do              
            reduced <- op reduced (transf dValues.[index]) 
            index <- index + numThreads

        // Get the total.
        let total = (%multiReduce) tid reduced 

        if tid = 0 then dRangeTotals.[range] <- total @>

(**
Reduces range totals to a single total, which is written back to the first element in the range totals input array.
*)
(*** define:GenericReduceRangeTotalsKernel ***)
let reduceRangeTotalsKernel (initExpr:Expr<unit -> 'T>) (opExpr:Expr<'T -> 'T -> 'T>) (plan:Plan) =
    let numThreads = plan.NumThreadsReduction
    let numWarps = plan.NumWarpsReduction
    let logNumWarps = log2 numWarps
    let multiReduce = multiReduce initExpr opExpr numWarps logNumWarps

    <@ fun numRanges (dRangeTotals:deviceptr<'T>) ->
        let init = %initExpr
        let op = %opExpr

        let tid = threadIdx.x
        let x = if tid < numRanges then dRangeTotals.[tid] else (init())
        let total = (%multiReduce) tid x

        // Have the first thread in the block set the total and store it in the first element of the input array.
        if tid = 0 then dRangeTotals.[0] <- total @>

