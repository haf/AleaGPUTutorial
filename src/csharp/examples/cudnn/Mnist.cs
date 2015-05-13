using System;
using Alea.CUDA;
using NUnit.Framework;

namespace Tutorial.Cs.examples.cudnn
{
    public class Test
    {
        //[CudnnMnistTest]
        [Test]
        public static void MnistTest()
        {
            var worker = Worker.Default;
            
            using (var network = new Network(worker))
            {
                var conv1 = Layer.Conv1(worker);
                var conv2 = Layer.Conv2(worker);
                var ip1 = Layer.Ip1(worker);
                var ip2 = Layer.Ip2(worker);

                Console.WriteLine("Classifying....");
                var i1 = network.ClassifyExample(Data.FirstImage, conv1, conv2, ip1, ip2);
                var i2 = network.ClassifyExample(Data.SecondImage, conv1, conv2, ip1, ip2);
                var i3 = network.ClassifyExample(Data.ThirdImage, conv1, conv2, ip1, ip2);

                Console.WriteLine("\n==========================================================\n");
                Console.WriteLine("Result of Classification: {0}, {1}, {2}", i1, i2, i3);

                if ((i1 != 1) || (i2 != 3) || (i3 != 5))
                    Console.WriteLine("Test Failed!!");
                else
                    Console.WriteLine("Test Passed!!");

                Console.WriteLine("\n==========================================================\n");
                Assert.AreEqual(i1, 1);
                Assert.AreEqual(i2, 3);
                Assert.AreEqual(i3, 5);
            }
        }
        //[/CudnnMnistTest]
    }
}
