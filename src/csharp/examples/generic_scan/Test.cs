using System;
using System.Linq;
using NUnit.Framework;

namespace Tutorial.Cs.examples.generic_scan
{
    public static class Test
    {
        public static Random Rng = new Random();
        public static int[] Nums = { 1, 2, 8, 128, 100, 1024 };
        public static bool Verbose = false;
        public static int ShowLimit = 8;

        //[GenericScanSumInts]
        public static void SumInts()
        {
            foreach (var n in Nums)
            {
                var values = Gen(Rng.Next, n);
                // Inclusive scan
                var dri = ScanApi.Sum(values, true);
                var hri = cpuScan((x,y)=> x+y, values, true);
                Compare(hri,dri);
                // Exclusive scan
                var dre = ScanApi.Sum(values, false);
                var hre = cpuScan((x, y) => x + y, values, false);
                Compare(hre, dre);
            }
        }
        //[/GenericScanSumInts]

        //[GenericScanGenericSumInts]
        public static void SumIntsGeneric()
        {
            foreach (var n in Nums)
            {
                var values = Gen(Rng.Next, n);
                // Inclusive scan
                var dri = ScanApi.Scan((x, y) => x + y, values, true);
                var hri = cpuScan((x, y) => x + y, values, true);
                Compare(hri,dri);
                // Exclusive scan
                var dre = ScanApi.Scan((x, y) => x + y, values, false);
                var hre = cpuScan((x, y) => x + y, values, false);
                Compare(hre, dre);
            }
        }
        //[/GenericScanGenericSumInts]

        //[GenericScanSumDoubles]
        public static void SumDoubles()
        {
            foreach (var n in Nums)
            {
                var values = Gen(Rng.NextDouble, n);
                // Inclusive scan
                var dri = ScanApi.Sum(values, true);
                var hri = cpuScan((x,y) => x + y, values, true);
                Compare(hri,dri,1e-11);
                // Exclusive scan
                var dre = ScanApi.Sum(values, false);
                var hre = cpuScan((x, y) => x + y, values, false);
                Compare(hre, dre, 1e-11);
            }
        }
        //[/GenericScanSumDoubles]

        //[GenericScanGenericSumDoubles]
        public static void SumDoublesGeneric()
        {
            foreach (var n in Nums)
            {
                var values = Gen(Rng.NextDouble, n);
                // Inclusive scan
                var dri = ScanApi.Scan((x, y) => x + y, values, true);
                var hri = cpuScan((x, y) => x + y, values, true);
                Compare(hri, dri, 1e-11);
                // Exclusive scan
                var dre = ScanApi.Scan((x, y) => x + y, values, false);
                var hre = cpuScan((x, y) => x + y, values, false);
                Compare(hre, dre, 1e-11);
            }
        }
        //[/GenericScanGenericSumDoubles]

        //[GenericScanCPUScan]
        public static T[] cpuScan<T>(Func<T, T, T> op, T[] input, bool inclusive)
        {
            var result = new T[input.Length + 1];
            result[0] = default(T);
            for (var i = 1; i < result.Length; i++)
                result[i] = op(result[i - 1], input[i - 1]);
            return inclusive ? result.Skip(1).ToArray() : result.Take(input.Length).ToArray();
        }
        //[/GenericScanCPUScan]

        public static T[] Gen<T>(Func<T> g, int n)
        {
            return Enumerable.Range(0, n).Select(_ => g()).ToArray();
        }

        public static void Show<T>(T[] hr, T[] dr)
        {
            if (Verbose && (hr.Length <= ShowLimit))
            {
                for (var i = 0; i < hr.Length; i++)
                    Console.WriteLine("hr{0}, dr{0} ===> {1}, {2}", i, hr[i], dr[i]);
            }
        }

        public static void Compare<T>(T[] hr, T[] dr)
        {
            Compare(hr,dr,0.0);
        }

        public static void Compare<T>(T[] hr, T[] dr, dynamic delta)
        {
            for (var i = 0; i < hr.Length; i++)
                Assert.AreEqual(hr[i], dr[i], delta);
        }

        //[GenericScanTests]
        [Test]
        public static void ScanTest()
        {
            SumInts();
            SumIntsGeneric();
            SumDoubles();
            SumDoublesGeneric();
        }
        //[/GenericScanTests]
    }
}
