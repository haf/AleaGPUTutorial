
using System;
using System.Collections.Generic;
using System.Configuration.Internal;
using System.IO;
using System.Runtime.InteropServices;
using Alea.IL.CompilerServices;

namespace Tutorial.Cs.examples.RandomForest.DataModel
{
    //using Label = Int32;
    //using FeatureValue = Double;
    using Sample = List<double>;
    using Domain = List<double>;
    using Domains = List<List<double>>;
    using Labels = List<int>;
    using LabeledSample = Tuple<List<double>, int>;
    using LabelHistogram = Tuple<List<int>, int>;
    using Weights = List<int>;
    using Indices = List<int>;

    public class Split
    {
        public int Feature;
        public double Threshold;

        public Split(int feature, double threshold)
        {
            Feature = feature;
            Threshold = threshold;
        }

        public override string ToString()
        {
            return String.Concat("( F: ", this.Feature.ToString(), ", T: ", this.Threshold.ToString(), ")");
        }
    }

    //http://stackoverflow.com/questions/66893/tree-data-structure-in-c-sharp
    public class Tree
    {
        private int label;
        private Split split;
        private Tree low;
        private Tree high;

        //private LinkedList<NTree<T>> children;

        //public NTree(T data)
        //{
        //    data = data;
        //    children = new LinkedList<NTree<T>>();
        //}

        //public void AddChild(T data)
        //{
        //    children.AddFirst(new NTree<T>(data));
        //}

        //public NTree<T> GetChild(int i)
        //{
        //    foreach (NTree<T> n in children)
        //        if (--i == 0)
        //            return n;
        //    return null;
        //}

        //public void Traverse(NTree<T> node, TreeVisitor<T> visitor)
        //{
        //    visitor(node.data);
        //    foreach (NTree<T> kid in node.children)
        //        Traverse(kid, visitor);
        //}
        public override string ToString()
        {
            return String.Concat("( ", low.ToString(), ", ", split.ToString(), ", ", high.ToString(), ")");
        }
    }

    public class Model
    {
        private List<Tree> trees;
        private int numClasses;
    }

}
