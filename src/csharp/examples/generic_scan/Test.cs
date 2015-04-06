using System;
using System.Linq;
using NUnit.Framework;

namespace Tutorial.Cs.examples.generic_scan
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

        // TODO: Add functionality for operations other than sum
        public double[] cpuScan(double[] input, bool inclusive)
        {
            int idx = 1;
            var result = input.Select(_ => input.Take(idx++).Sum()).ToArray();
            return inclusive ? result.Skip(1).ToArray() : result;
        }

        [Test]
        public void InclusiveSum()
        {
            var nums = new[] { 1, 2, 8, 128, 100, 1024 };
            foreach (int n in nums)
            {
                var values = Gen(n);
                var dr = ScanApi.Sum(values, true);
                var hr = cpuScan(values, true);
                
                //Console.WriteLine("hr, dr ===> {0}, {1}", hr, dr);
                
                for(var i = 0; i < hr.Length; i++)
                    Assert.AreEqual(hr[i], dr[i], 1e-12);
            }
        }
    }
}
