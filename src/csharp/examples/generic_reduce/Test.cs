using System;
using System.Linq;
using Alea.CUDA;
using Microsoft.FSharp.Core;
using NUnit.Framework;
using OpenTK.Graphics.ES20;
using SharpDX.Direct3D9;

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
            Func<double, double, double> sum = (x, y) => x + y;
            foreach (int n in nums)
            {
                var values = Gen(n);
                var dr = ReduceApi.Sum(values);
                var hr = values.Aggregate(sum);
                Assert.AreEqual(dr, hr);
            }
        }
    }
}
