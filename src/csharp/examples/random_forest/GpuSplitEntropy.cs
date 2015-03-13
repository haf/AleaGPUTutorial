using System;

using Alea.CUDA;
using Alea.CUDA.Unbound;
using Alea.CUDA.Utilities;
using Microsoft.FSharp.Core;
using Tutorial.Fs.examples.RandomForest.Cuda;

namespace Tutorial.Cs.examples.RandomForest
{
    class GpuSplitEntropy
    {
        class MatrixRowScanPrimitive
        {
            private readonly BlockRangeScan<int> _blockRangeScan;

            public MatrixRowScanPrimitive(Alea.CUDA.DeviceArch arch, AddressSize addressSize, int blockThreads)
            {
                this._blockRangeScan =
                    DeviceScanPolicy.Create<int>(arch, addressSize, new FSharpOption<int>(blockThreads)).BlockRangeScan;
            }

            [ReflectedDefinition]
            public void BlockRangeScan(int blockOffset, int blockEnd, deviceptr<int> inputs)
            {
                var tempStorage = _blockRangeScan.TempStorage.AllocateShared();
                var plus = Microsoft.FSharp.Core.FuncConvert.ToFSharpFunc((int y) =>
                    {
                        var result = Microsoft.FSharp.Core.FuncConvert.ToFSharpFunc((int x) => x + y);
                        return result;
                    });

                _blockRangeScan.ConsumeRangeConsecutiveInclusive(tempStorage, new Iterator<int>(inputs), new Iterator<int>(inputs), plus, blockOffset, blockEnd);
            }

            public int BlockThreads()
            {
                return (_blockRangeScan.BlockThreads);
            }
        }

        class EntropyOptimizationOptions
        {
            public int AbsMinWeight;
            public int RelMinDivisor;
            public int RelMinBound;
            public int Decimals;
            public Func<int, bool[]> FeatureSelector;

            public EntropyOptimizationOptions(int absMinWeight, int realMinDivisor, int relMinBound, int decimals, Func<int, bool[]> featureSelector )
            {
                AbsMinWeight = absMinWeight;
                RelMinDivisor = realMinDivisor;
                RelMinBound = relMinBound;
                Decimals = decimals;
                FeatureSelector = featureSelector;
            }

            public EntropyOptimizationOptions()
            {
                AbsMinWeight = 1;
                RelMinDivisor = 10;
                RelMinBound = 25;
                Decimals = 6;
                FeatureSelector = (n) =>
                {
                    var array = new bool[Decimals];
                    for (var i = 0; i < n; i++)
                    {
                        array[i] = true;
                    }
                    return (array);
                };

            }

            public int MinWeight(int numClasses, int total)
            {
                var relativeMinWeight = System.Math.Min(total/(numClasses*this.RelMinDivisor), this.RelMinBound);
                return (Math.Max(this.AbsMinWeight, relativeMinWeight));
            }

            static bool[] SquareRootFeatureSelector(Random rnd, int n)
            {
                var k = (int) Math.Sqrt(n);
                var idcs = Tutorial.Fs.examples.RandomForest.Array.randomSubIndices(rnd, n, k);
                var mask = new bool[n];
                for (var i = 0; i < n; i++)
                {
                    mask[i] = false;
                }
                foreach (var t in idcs)
                {
                    mask[t] = true;
                }
                return (mask);
            }
        }

        class EntropyOptimizer:GPUModule
        {
            private GPUModuleResource<MatrixRowScanPrimitive> _primitive;
            private MultiChannelReduce.MatrixRowOptimizer _minimizer;

            public EntropyOptimizer(GPUModuleTarget target) : base(target)
            {
                //OptimizedClosures.FSharpFunc<CompileOptions,TemplateBuilder>.FromConverer(options => return (Alea.CUDA.Compilation.cuda  { return MatrixRowScanPrimitive(options.MinimalArch, options.AddressSize, None) }))
                //Func<Context, T> a = options => return (Alea.CUDA.Compilation.cuda  { return MatrixRowScanPrimitive(options.MinimalArch, options.AddressSize, None) })


                //type Template<'T>(f:IRModuleBuildingContext -> 'T) =
                //    member this.Invoke b = f b

                //var a = Microsoft.FSharp.Core.FuncConvert.ToFSharpFunc((CompileOptions options) =>
                //    {
                //        var b = Alea.CUDA.Compilation.cuda
                //        {
                //            return (MatrixRowScanPrimitive(options.MinimalArch, options.AddressSize, None));
                //        }
                //        return(b);
                //    });

                //var f = FSharpFunc.FromConverter(ctx =>  MatrixRowScanPrimitive(options.MinimalArch, options.AddressSize, None) )  
                //new Template<blablabla>(f)
                //_primitive = Func < CompileOptions > cuda { return MatrixRowScanPrimitive(options.MinimalArch, options.AddressSize, None) }
                _minimizer = new MultiChannelReduce.MatrixRowOptimizer(target);
            }
        }
    }
}
