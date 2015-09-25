using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;

namespace Tutorial.Cs.examples.cublas
{
    class Util
    {
        public static void FallBack(Action action)
        {
            if (Alea.CUDA.PlatformUtil.Instance.OperatingSystem.IsMacOSX)
            {
                Assert.Inconclusive("CUBLAS destroy has some issues in macosx when deinit it in finalizer.");
            }
            else
            {
                try
                {
                    action();
                }
                catch (TypeInitializationException ex)
                {
                    if (ex.InnerException.GetType().FullName == "System.DllNotFoundException")
                    {
                        Assert.Inconclusive(
                            "Native libraries cannot be found, please setup your environment, or use app.config.");
                    }
                    else throw;
                }
            }
        }
    }
}
