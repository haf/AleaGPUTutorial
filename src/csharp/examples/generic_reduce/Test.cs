using System;
using Alea.CUDA;
using NUnit.Framework;
using OpenTK.Graphics.ES20;

namespace Tutorial.Cs.examples.generic_reduce
{
    class Test
    {


        public int[] GenInts(int n)
        {
            var vals = new int[n];
            var rng = new Random();
            for (var i = 0; i < n; i++)
                vals[i] = rng.Next();
            return vals;
        }

        [Test]
        public void Sum()
        {
            var nums = new[]{1,2,8,128,100,1024};
            var reduce = new Reduce<int>(GPUModuleTarget.DefaultWorker, x => x, (x, y) => x + y, x => x, Plan.Plan32());
            foreach (int n in nums)
            {
                reduce.   
            }
        }
    }
}
