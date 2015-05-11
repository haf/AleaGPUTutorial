

using System;
using System.Linq;

namespace Tutorial.Cs.examples.generic_reduce
{
    public static class Const
    {
        public static int WARP_SIZE = 32;
        public static int LOG_WARP_SIZE = 5;
    }

    //[genericReducePlan]
    public class Plan
    {
        public int NumThreads { get; set; }
        public int ValuesPerThread { get; set; }
        public int NumThreadsReduction { get; set; }
        public int BlockPerSm { get; set; }

        public int ValuesPerWarp { get { return ValuesPerThread*Const.WARP_SIZE; } }
        public int NumWarps { get { return NumThreads/Const.WARP_SIZE; } }
        public int NumWarpsReduction { get { return NumThreadsReduction/Const.WARP_SIZE; } }
        public int NumValues { get { return NumThreads*ValuesPerThread; } }

        public Plan()
        {
            
        }

        public Tuple<int[],int> BlockRanges(int numSm, int count)
        {
            var numBlocks = Math.Min(BlockPerSm*numSm, NumThreadsReduction);
            var blockSize = NumThreads*ValuesPerThread;
            var numBricks = Alea.CUDA.Utilities.Common.divup(count, blockSize);
            numBlocks = Math.Min(numBlocks, numBricks);

            var brickDivQuot = numBricks/numBlocks;
            var brickDivRem = numBricks%numBlocks;

            var r = Enumerable.Range(0, numBlocks + 1).ToArray();
            int idx = 1;
            var ranges = 
                r.Select(i =>
                {
                    var s = r.Take(idx++).Sum();
                    var bricks = (i - 1) < brickDivRem ? brickDivQuot + 1 : brickDivQuot;
                    return Math.Min(s + bricks * blockSize, count);
                }).ToArray();
            ranges[0] = 0;
            return new Tuple<int[], int>(ranges, ranges.Length - 1);
        }
        
        public static Plan Plan32 = new Plan()
        {
            NumThreads = 1024,
            ValuesPerThread = 4,
            NumThreadsReduction = 256,
            BlockPerSm = 1
        };

        public static Plan Plan64 = new Plan()
        {
            NumThreads = 512,
            ValuesPerThread = 4,
            NumThreadsReduction = 256,
            BlockPerSm = 1
        };
    }
    //[/genericReducePlan]

    public enum Planner
    {
        Default,
        Specific
    }
}
