

using System;

namespace Tutorial.Cs.examples.generic_reduce
{
    public static class Const
    {
        public static int WARP_SIZE = 32;
        public static int LOG_WARP_SIZE = 5;
    }

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
            var numBricks = 0; // divup
            numBlocks = Math.Min(numBlocks, numBricks);

            var brickDivQuot = numBricks/numBlocks;
            var brickDivRem = numBricks%numBlocks;

            var ranges = new int[numBlocks];
            for (var i = 0; i < numBlocks; i++) ranges[i] = i + 1;
            for (var i = 0; i < numBlocks; i++)
            {
                int bricks = 0;
                if ((i - 1) < brickDivRem)
                    bricks = brickDivQuot + 1;
                else
                    bricks = brickDivQuot;
                ranges[i] = Math.Min(ranges[i] + bricks*blockSize, count);
            }
            return new Tuple<int[], int>(ranges, ranges.Length - 1);
        }

        public static Plan Plan32()
        {
            return new Plan {NumThreads = 1024, ValuesPerThread = 4, NumThreadsReduction = 256, BlockPerSm = 1};
        }
    }

    public enum Planner
    {
        Default,
        Specific
    }
}
