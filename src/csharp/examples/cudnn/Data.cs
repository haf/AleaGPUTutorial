﻿using System;
using System.IO;
using System.Linq;

namespace Tutorial.Cs.examples.cudnn
{
    //[CudnnMnistData]
    public static class Data
    {
        public const string FirstImage = "one_28x28.pgm";
        public const string SecondImage = "three_28x28.pgm";
        public const string ThirdImage = "five_28x28.pgm";

        public const string Conv1Bin = "conv1.bin";
        public const string Conv1BiasBin = "conv1.bias.bin";
        public const string Conv2Bin = "conv2.bin";
        public const string Conv2BiasBin = "conv2.bias.bin";

        public const string Ip1Bin = "ip1.bin";
        public const string Ip1BiasBin = "ip1.bias.bin";
        public const string Ip2Bin = "ip2.bin";
        public const string Ip2BiasBin = "ip2.bias.bin";

        public const int ImageH = 28;
        public const int ImageW = 28;

        private static readonly string[] FilesToCheck = {"Alea.Tutorial.sln", "build.bat", "build.fsx"};

        private static bool IsSolutionDir(string dir)
        {
            return FilesToCheck.All(file => File.Exists(Path.Combine(dir, file)));
        }

        private static string FindSolutionDir(string dir)
        {
            return IsSolutionDir(dir) ? dir : FindSolutionDir(Directory.GetParent(dir).FullName);
        }

        private static string GetDataDir()
        {
            return Path.Combine(FindSolutionDir("./"), "src/csharp/examples/cudnn/data");
        }

        public static string GetPath(string fname)
        {
            return Path.Combine(GetDataDir(), fname);
        }

        public static float[] ReadBinaryFile(string fname)
        {
            var b = File.ReadAllBytes(fname);
            var length = b.Length / 4;
            var a = new float[length];
            for (var i = 0; i < length; ++i)
            {
                a[i] = BitConverter.ToSingle(b, i * 4);
            }
            return a;
        }

        public static byte[] LoadImage(string fname)
        {
            var p = GetPath(fname);
            return File.ReadAllBytes(p).Skip(52).ToArray();
        }
    }
    //[/CudnnMnistData]
}
