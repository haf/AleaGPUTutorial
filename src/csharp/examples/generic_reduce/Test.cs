using System;
using System.Linq;
using NUnit.Framework;

namespace Tutorial.Cs.examples.generic_reduce
{
    class Test
    {


        public double[] Gen(int n)
        {
            var vals = new double[n];
            var rng = new Random();
            for (var i = 0; i < n; i++)
                vals[i] = rng.NextDouble();
            return vals;
        }

        [Test]
        public void Sum()
        {
            var nums = new[]{1,2,8,128,100,1024};
            foreach (int n in nums)
            {
                var values = Gen(n);
                var dr = ReduceApi.Sum(values);
                var hr = values.Sum();
                Console.WriteLine("hr, dr ===> {0}, {1}", hr, dr);
                Assert.AreEqual(dr, hr, 1e-12);
            }
        }
    }
}
