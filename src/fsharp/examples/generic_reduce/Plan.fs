(*** hide ***)
module Tutorial.Fs.examples.genericReduce.Plan

open Alea.CUDA.Utilities

let WARP_SIZE = 32
let LOG_WARP_SIZE = 5

(*** define:genericReducePlan ***)
type Plan =
    {
        NumThreads : int
        ValuesPerThread : int
        NumThreadsReduction : int
        BlockPerSm:int
    }
     
    member this.ValuesPerWarp = this.ValuesPerThread * WARP_SIZE
    member this.NumWarps = this.NumThreads / WARP_SIZE
    member this.NumWarpsReduction = this.NumThreadsReduction / WARP_SIZE
    member this.NumValues = this.NumThreads * this.ValuesPerThread


    /// Finds the ranges for each block to process. 
    /// Note that each range must begin a multiple of the block size.
    /// It returns a sequence of length 1 + effective num blocks (which is equal to min numRanges numBricks)
    /// and the number off effective blocks, i.e. number of ranges
    /// The range pairs can be obtained by blockRanges numRanges count |> Seq.pairwise  
    member this.BlockRanges numSm count =
        let numBlocks = min (this.BlockPerSm * numSm) this.NumThreadsReduction
        let blockSize = this.NumThreads * this.ValuesPerThread     
        let numBricks = divup count blockSize
        let numBlocks = min numBlocks numBricks 

        let brickDivQuot = numBricks / numBlocks 
        let brickDivRem = numBricks % numBlocks

        let ranges = [| 1..numBlocks |] |> Array.scan (fun s i -> 
            let bricks = if (i-1) < brickDivRem then brickDivQuot + 1 else brickDivQuot
            min (s + bricks * blockSize) count) 0
           
        ranges, ranges.Length - 1

type Planner =
    | Default
    | Specific of Plan



