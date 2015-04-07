using System;
using System.Linq;
using NUnit.Framework;

namespace Tutorial.Cs.examples.generic_scan
{
    class Test
    {
        public Random Rng = new Random();
        public int[] Nums = {1, 2, 8, 128, 100, 1024 };
        public bool Verbose = false;
        public int ShowLimit = 8;

        [Test]
        public void InclusiveSumInts()
        {
            foreach (var n in Nums)
            {
                var values = Gen(() => Rng.Next(), n);
                var dr = ScanApi.Sum(values, true);
                var hr = cpuScan((x,y)=> x+y, values, true);
                Show(hr, dr);
                Compare(hr,dr);
            }
        }

        [Test]
        public void ExclusiveSumInts()
        {
            foreach (var n in Nums)
            {
                var values = Gen(() => Rng.Next(), n);
                var dr = ScanApi.Sum(values, false);
                var hr = cpuScan((x, y) => x + y, values, false);
                Show(hr, dr);
                Compare(hr,dr);
            }
        }
        
        [Test]
        public void InclusiveSumDoubles()
        {
            foreach (var n in Nums)
            {
                var values = Gen(() => Rng.NextDouble(), n);
                var dr = ScanApi.Sum(values, true);
                var hr = cpuScan((x,y) => x + y, values, true);
                Show(hr, dr);
                Compare(hr,dr);
            }
        }

        [Test]
        public void InclusiveSumSingles()
        {
            foreach (var n in Nums)
            {
                var values = Gen(() => (float) Rng.NextDouble(), n);
                var dr = ScanApi.Sum(values, true);
                var hr = cpuScan((x, y) => x + y, values, true);
                Show(hr, dr);
                Compare(hr, dr);
            }            
        }

        [Test]
        public void GenericTests()
        {
            const bool inclusive = true;

            foreach (var n in Nums)
            {
                var v1 = Gen(() => Rng.Next(), n);
                var v2 = Gen(() => Rng.NextDouble(), n);
                var hr1 = cpuScan((x, y) => x + y, v1, inclusive);
                var hr2 = cpuScan((x, y) => x + y, v2, inclusive);
                var dr1 = ScanApi.Scan(v1, (x,y) => x+y, inclusive);
                var dr2 = ScanApi.Scan(v2, (x, y) => x + y, inclusive);
                Compare(hr1, dr1);
                Compare(hr2, dr2, 1e-12);
            }
        }

        public T[] cpuScan<T>(Func<T, T, T> op, T[] input, bool inclusive)
        {
            var result = new T[input.Length + 1];
            result[0] = default(T);
            for (var i = 1; i < result.Length; i++)
                result[i] = op(result[i - 1], input[i - 1]);
            return inclusive ? result.Skip(1).ToArray() : result;
        }

        public T[] Gen<T>(Func<T> g, int n)
        {
            return Enumerable.Range(0, n).Select(_ => g()).ToArray();
        }

        public void Show<T>(T[] hr, T[] dr)
        {
            if (Verbose && (hr.Length <= ShowLimit))
            {
                for (var i = 0; i < hr.Length; i++)
                    Console.WriteLine("hr{0}, dr{0} ===> {1}, {2}", i, hr[i], dr[i]);
            }
        }

        public void Compare<T>(T[] hr, T[] dr)
        {
            Compare(hr,dr,0.0);
        }

        public void Compare<T>(T[] hr, T[] dr, dynamic delta)
        {
            for (var i = 0; i < hr.Length; i++)
                Assert.AreEqual(hr[i], dr[i], delta);
        }
    }
}
