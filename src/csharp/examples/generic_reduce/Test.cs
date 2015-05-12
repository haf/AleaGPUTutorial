using System;
using System.Linq;
using Alea.CUDA.Utilities;
using NUnit.Framework;

namespace Tutorial.Cs.examples.generic_reduce
{
    public class Test
    {
        public static Random Rng = new Random();
        public static int[] Nums = { 1, 2, 8, 128, 100, 1024 };
        public static bool Verbose = false;
        public static int ShowLimit = 8;

        //[GenericReduceSumInts]
        public static void SumInts()
        {
            foreach (var n in Nums)
            {
                var values = Gen(Rng.Next, n);
                var dr = ReduceApi.Sum(values);
                var hr = cpuReduce((x, y) => x + y, values);
                Assert.AreEqual(hr,dr);
            }
        }
        //[/GenericReduceSumInts]

        //[GenericReduceSumDoubles]
        public static void SumDoubles()
        {
            foreach (var n in Nums)
            {
                var values = Gen(Rng.NextDouble, n);
                var dr = ReduceApi.Sum(values);
                var hr = cpuReduce((x, y) => x + y, values);
                Assert.AreEqual(hr, dr, 1e-11);
            }
        }
        //[/GenericReduceSumDoubles]

        //[GenericReduceScalarProdDoubles]
        public static void ScalarProdDoubles()
        {
            foreach (var n in Nums)
            {
                var values1 = Gen(Rng.NextDouble, n);
                var values2 = Gen(Rng.NextDouble, n);
                var dr = ReduceApi.ScalarProd(values1, values2);
                var hr = cpuScalarProd((x, y) => x + y, (x, y) => x*y, values1, values2);
                Assert.AreEqual(hr, dr, 1e-11);
            }
        }
        //[/GenericReduceScalarProdDoubles]

        //[GenericReduceMaxDoubles]
        public static void MaxDoubles()
        {
            foreach (var n in Nums)
            {
                var values = Gen(Rng.NextDouble, n);
                //var dr = ReduceApi.Reduce(() => double.NegativeInfinity, Math.Max, values);
                var dr = ReduceApi.Reduce(LibDevice.__neginf<double>, Math.Max, values);
                var hr = cpuReduce(Math.Max, values);
                Assert.AreEqual(hr, dr, 1e-11);
            }
        }
        //[/GenericReduceMaxDoubles]

        //[GenericReduceMinDoubles]
        public static void MinDoubles()
        {
            foreach (var n in Nums)
            {
                var values = Gen(Rng.NextDouble, n);
                //var dr = ReduceApi.Reduce(() => double.PositiveInfinity, Math.Min, values);
                var dr = ReduceApi.Reduce(LibDevice.__posinf<double>, Math.Min, values);
                var hr = cpuReduce(Math.Min, values);
                Assert.AreEqual(hr, dr, 1e-11);
            }
        }
        //[/GenericReduceMinDoubles]

        public static T cpuReduce<T>(Func<T, T, T> op, T[] input)
        {
            var r = input[0];
            for (var i = 1; i < input.Length; i++)
                r = op(r, input[i]);
            return r;
        }

        public static T cpuScalarProd<T>(Func<T, T, T> add, Func<T, T, T> mult, T[] values1, T[] values2)
        {
            return 
                cpuReduce(  add,
                            Enumerable.Range(0, values1.Length)
                            .Select(i => mult(values1[i], values2[i])).ToArray());
        }

        public static T[] Gen<T>(Func<T> g, int n)
        {
            return Enumerable.Range(0, n).Select(_ => g()).ToArray();
        }

        //[GenericReduceTests]
        [Test]
        public static void ReduceTest()
        {
            SumInts();
            SumDoubles();
            ScalarProdDoubles();
            MaxDoubles();
            MinDoubles();
        }
        //[/GenericReduceTests]
    }
}
