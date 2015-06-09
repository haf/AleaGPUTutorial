# Alea GPU Tutorial

This project contains tutorial and samples of [Alea GPU](http://quantalea.com) compiler. It uses literal programming with [F# Formatting](http://tpetricek.github.io/FSharp.Formatting/) to code samples and then generates the documentation directly from the source code. The generated documentation can be found here: [Alea GPU Tutorial](http://quantalea.com/static/app/tutorial/index.html).

## How to build

Before building the solution, please make sure you have a proper JIT license installed.

If you have one and only one GeForce GPU device attached to your host, then you can follow these steps to verify and install a free community license:

- go to your solution folder and restore packages:
  - `cd MySolutionFolder`
  - `.paket\paket.bootstrapper.exe`
  - `.paket\paket.exe restore`
- in your solution folder, run the following command to verify an installed license for the current user:
  - `packages\Alea.CUDA\tools\LicenseManager.exe list`
  - verify the output, you need a valid compilation license which supports GeForce GPUs
- if you don't have a community license, follow these steps to install one for free:
  - go to [QuantAlea website](http://quantalea.com/accounts/login/), sign in or sign up for an account
  - then go to [My Licenses page](http://quantalea.com/licenses/), to get your free community license code
  - in your solution folder, run the following command to install your community license:
    - `packages\Alea.CUDA\tools\LicenseManager.exe install -l %your_license_code%`
  - verify the installed license again via:
    - `packages\Alea.CUDA\tools\LicenseManager.exe list`

For more details on licensing, please reference:

- [License introduction](http://quantalea.com/static/app/tutorial/quick_start/licensing_and_deployment.html)
- [License Manager manual](http://quantalea.com/static/app/manual/compilation-license_manager.html)
- [Licensing page](http://quantalea.com/licensing/)

This project uses [Paket](http://fsprojects.github.io/Paket/) for package management.

To build on Windows, open a command prompt and navigate to your solution folder. Now simply run `build.bat` (on Linux and OsX run `build.sh`) and the script will execute the following steps:

- download the latest version of `paket.exe`;
- run `paket.exe restore` to restore the packages listed in the `paket.lock` file;
- build projects;
- run tests;
- generate documentation (only on Windows);

The build script accepts the arguments _NoTests_ and _NoDocs_ which can be used like so:

- run `build.bat NoTests` to skip running the tests;
- run `build.bat NoDocs` to skip documentation generation;
- run `build.bat NoTests NoDocs` to only build the projects;

Running the tests and generating the documentation are both fairly lengthly processes (especially running the tests).  Usually you will want to use `build.bat NoTests NoDocs` for your general troubleshooting and save the full build for doing a final verification of your work.

Once you have all of the projects built you can:

- check `docs\output\index.html` for the generated documentation;
- execute `release\Tutorial.FS.exe <name>` to run example `<name>` written in F#.
- execute `release\Tutorial.CS.exe <name>` to run example `<name>` written in C#.
- execute `release\Tutorial.VB.exe <name>` to run example `<name>` written in VB.
- execute `release\Tutorial.FS.exe`, `release\Tutorial.CS.exe` or `release\Tutorial.VB.exe` to see more examples.
- Explore the source code with Visual Studio and run unit tests.

Before building within Visual Studio, it is recommended that you restore the packages prior to opening the solution. This is due to a known issue of using Fody with F# projects.  You can find further details about this issue in the [installation manual (especially the Remarks section)](http://quantalea.com/static/app/manual/compilation-installation.html)). 

To build within Visual Studio, please follow following steps:

- navigate to your solution folder and restore packages:
  - `cd MySolutionFolder`
  - `.paket\paket.bootstrapper.exe`
  - `.paket\paket.exe restore`
- open then solution with Visual Studio and then build it using the `Release` configuration
- set the debug argument to one example, such as `examplenbodysimulation`
- run/debug the tutorial program

## How to collaborate

We use the light [git-flow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) workflow for collaboration. Using the light git-flow means that the main development branch is `master` (as opposed to develop), and that all feature branches should be rebased along with the creation of a merge commit onto the `master` branch. This can also be done by a GitHub pull request.

## Using FSharp.Core

Since [Alea GPU](http://quantalea.com) uses F#, the F# runtime [FSharp.Core](http://www.nuget.org/packages/FSharp.Core/) is needed. In this solution we use the [FSharp.Core](http://www.nuget.org/packages/FSharp.Core/) NuGet package for all projects, regardless of the project being written in F#, C# or VB.

## How to upgrade packages

To upgrade packages, follow these steps:

- edit the `paket.dependencies` file by changing the packages or their versions; alternatively, you can use `paket install` (see [here](http://fsprojects.github.io/Paket/paket-install.html))
- execute `paket update --redirects` (see [here](http://fsprojects.github.io/Paket/paket-update.html))
- execute `package add nuget <name> -i` to add the `<name>` package to your project (see [here](http://fsprojects.github.io/Paket/paket-add.html))
- commit your changed files (`paket.lock`, `paket.dependencies`) and your modified project files.

If you rebase your branch onto the master branch and the packages have been upgraded, follow these steps:

- shutdown Visual Studio if the project is opened in Visual Studio
- do rebasing on your branch
- run `.paket\paket.exe restore` or simply run `build.bat`
- open the Visual Studio again

The reason why we need to close Visual Studio is because AleaGPU uses the Fody plugin.  Fody is a build plugin and Visual Studio will use it in the build process; if the Fody package has been upgraded, it cannot be written to the `packages` folder.

