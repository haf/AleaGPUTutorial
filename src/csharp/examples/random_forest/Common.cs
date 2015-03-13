using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Remoting.Messaging;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using Alea.CUDA;
using Alea.IL;
using NUnit.Framework;

namespace Tutorial.Cs.examples.randomForest.Common
{
    public static class MinMax<T> where T : IComparable
    {
        public static Tuple<T, int> MinAndArgMin(T[] source)
        {
            if (source.Length == 0)
            {
                throw new ArgumentException("empty array", "source");
            }
            var acci = 0;
            var accv = source[0];
            for(var i = 0; i < source.Length; i++)
            {
                if (source[i].CompareTo(accv) >= 0) continue;
                accv = source[i];
                acci = i;
            }
            return new Tuple<T, int>(accv, acci);
        }

        public static Tuple<T, int> MaxAndArgMax(T[] source)
        {
            if (source.Length == 0)
            {
                throw new ArgumentException("empty array", "source");
            }
            var acci = 0;
            var accv = source[0];
            for (var i = 0; i < source.Length; i++)
            {
                if (source[i].CompareTo(accv) <= 0) continue;
                accv = source[i];
                acci = i;
            }
            return new Tuple<T, int>(accv, acci);
        }

        //[Test]
        //public static void MinAndArgMinTest()
        //{
        //    Int32[] source = { 1, 2, 3, 4, 5 };
        //    var minMinIndex = MinAndArgMin(source);
        //    Assert.AreEqual(minMinIndex.Item1, minMinIndex.Item2);

        //}

        //[Test]
        //public static void MaxAndArgMaxTest()
        //{
        //    Int32[] source = {2, 3, 1, 4, 5, 1, 2, 5};
        //    var maxMaxIndex = MaxAndArgMax(source);
        //    Assert.AreEqual(maxMaxIndex.Item1, maxMaxIndex.Item2);
        //}
    }

    //public static class Seq<T> where T : IComparable
    //{
    //    public static List<List<T>> ToChunks(int n, List<T> s)
    //    {
    //        var pos = &0;
    //        var buffer = new T[n];
    //        for (var x = 0; x < n; x++)
    //        {
    //            buffer[pos] = s[x];
    //            if (pos == n - 1)
    //            {
    //                pos = 0;
    //            }
    //            else
    //            {
    //                pos++;
    //            }
    //        }
    //        if (pos > 0)
    //        {
                
    //        }
    //    }
    //}

    public static class Collections<T>
    {
        public class BlockingObjectPool<T>
        {
            private readonly int _size;
            private readonly System.Collections.Concurrent.BlockingCollection<T> _pool;
            
            public BlockingObjectPool(T[] obj)
            {
                this._size = obj.Length;
                this._pool = new System.Collections.Concurrent.BlockingCollection<T>(_size);
                foreach (var o in obj)
                {
                    _pool.Add(o);
                }
            }

            public T Acquire()
            {
                var obj = _pool.Take();
                return (obj);
            }

            public void Release(T obj)
            {
                _pool.Add(obj);
            }

            public int Size()
            {
                return(_size);
            }
        }
    }
}
