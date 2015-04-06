using System;
using System.Linq;
using NUnit.Framework;

namespace Tutorial.Cs.examples.generic_scan
{
    class Test
    {
        public double[] GenDoubles(int n)
        {
            var vals = new double[n];
            var rng = new Random();
            for (var i = 0; i < n; i++)
                vals[i] = rng.NextDouble();
            return vals;
        }

        public int[] GenInts(int n)
        {
            var vals = new int[n];
            var rng = new Random();
            for (var i = 0; i < n; i++)
                vals[i] = rng.Next(-100, 100);
            return vals;
        }

        // TODO: Add functionality for operations other than sum
        public int[] cpuScan(int[] input, bool inclusive)
        {
            int idx = inclusive ? 1 : 0;
            return input.Select(_ => input.Take(idx++).Sum()).ToArray();
        }

        public double[] cpuScan(double[] input, bool inclusive)
        {
            int idx = inclusive ? 1 : 0;
            return input.Select(_ => input.Take(idx++).Sum()).ToArray();
        }

        [Test]
        public void InclusiveSumInts()
        {
            var nums = new[] { 1, 2, 8, 128, 100, 1024 };
            foreach (int n in nums)
            {
                var values = GenInts(n);
                var dr = ScanApi.Sum(values, true);
                var hr = cpuScan(values, true);
                
                //for (var i = 0; i < hr.Length; i++)
                //    Console.WriteLine("hr{0}, dr{0} ===> {1}, {2}", i, hr[i], dr[i]);
                
                for (var i = 0; i < hr.Length; i++)    
                    Assert.AreEqual(hr[i], dr[i]);
            }
        }

        [Test]
        public void InclusiveSumDoubles()
        {
            var nums = new[] { 1, 2, 8, 128, 100, 1024 };
            foreach (int n in nums)
            {
                var values = GenDoubles(n);
                var dr = ScanApi.Sum(values, true);
                var hr = cpuScan(values, true);

                for (var i = 0; i < hr.Length; i++)
                    Assert.AreEqual(hr[i], dr[i], 1e-12);
            }
        }
    }
}
