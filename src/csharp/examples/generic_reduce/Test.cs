using System;
using System.Linq;
using NUnit.Framework;

namespace Tutorial.Cs.examples.generic_reduce
{
    class Test
    {
        public Random Rng = new Random();
        public int[] Nums = { 1, 2, 8, 128, 100, 1024 };
        public bool Verbose = false;
        public int ShowLimit = 8;

        [Test]
        public void SumInts()
        {
            foreach (var n in Nums)
            {
                var values = Gen(() => Rng.Next(), n);
                var dr = ReduceApi.Sum(values);
                var hr = cpuReduce((x, y) => x + y, values);
                Assert.AreEqual(hr,dr);
            }
        }

        [Test]
        public void SumDoubles()
        {
            foreach (var n in Nums)
            {
                var values = Gen(() => Rng.NextDouble(), n);
                var dr = ReduceApi.Sum(values);
                var hr = cpuReduce((x, y) => x + y, values);
                Assert.AreEqual(hr, dr, 1e-11);
            }
        }

        [Test]
        public void SumSingles()
        {
            foreach (var n in Nums)
            {
                var values = Gen(() => (float)Rng.NextDouble(), n);
                var dr = ReduceApi.Sum(values);
                var hr = cpuReduce((x, y) => x + y, values);
                Assert.AreEqual(hr, dr, 1e-3);
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
                var hr1 = cpuReduce((x, y) => x + y, v1);
                var hr2 = cpuReduce((x, y) => x + y, v2);
                var dr1 = ReduceApi.Reduce(v1, (x, y) => x + y);
                var dr2 = ReduceApi.Reduce(v2, (x, y) => x + y);
                Assert.AreEqual(hr1, dr1);
                Assert.AreEqual(hr2, dr2, 1e-11);
            }
        }

        public T cpuReduce<T>(Func<T, T, T> op, T[] input)
        {
            var r = input[0];
            for (var i = 1; i < input.Length; i++)
                r = op(r, input[i]);
            return r;
        }

        public T[] Gen<T>(Func<T> g, int n)
        {
            return Enumerable.Range(0, n).Select(_ => g()).ToArray();
        }

        

    }
}
