# Alea GPU Tutorial

This project contains tutorial and samples of [Alea GPU](http://quantalea.com) compiler. It uses literal programming with [F# Formatting](http://tpetricek.github.io/FSharp.Formatting/) to code samples and then generates document directly from source code. The generated document can be found at [Alea GPU Tutorial](http://quantalea.com/static/app/tutorial/index.html).

## How to build

This project uses [Paket](http://fsprojects.github.io/Paket/) for package management.

To build on Windows, simply run `build.bat` from command-line under the solution folder. This script will execute the following steps:

- download latest `paket.exe` from Internet;
- run `paket.exe restore` to restore the packages listed in `paket.lock` file;
- build projects and generate documents;

Then you can:

- check `docs\output\index.html` for the generated document;
- execute `release\Tutorial.FS.exe examplesnbodysimulation` to run NBody simulation written in F#.
- execute `release\Tutorial.CS.exe examplesnbodysimulation` to run NBody simulation written in C#.
- execute `release\Tutorial.FS.exe` or `release\Tutorial.CS.exe` to see more examples.
- Explore the source code with Visual Studio and run unit tests.

## How to collaborate

We use light [git-flow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) for collaborate. That means, the main development branch is `master`, and all feature branches should be rebased and then create merge commit onto `master` branch. This can also be done by GitHub pull request.

## Using FSharp.Core

Since [Alea GPU](http://quantalea.com) uses F#, thus F# runtime [FSharp.Core](http://www.nuget.org/packages/FSharp.Core/) is needed. In this solution, we use the [FSharp.Core](http://www.nuget.org/packages/FSharp.Core/) NuGet package for all projects, wether it is written in F#, C# or VB.

## How to upgrade packages

To upgrade packages, follow these steps:

- edit `paket.dependencies` files, edit the package or their versions; alternatively, you can use `paket install` (see [here](http://fsprojects.github.io/Paket/paket-install.html))
- execute `paket update --redirects` (see [here](http://fsprojects.github.io/Paket/paket-update.html))
- execute `package add nuget ... -i` to add that pacakge to your project, if you are adding new packages (see [here](http://fsprojects.github.io/Paket/paket-add.html))
- commit changed files, such as `paket.lock`, `paket.dependencies` and your modified project files.

If you rebase your branch onto master branch which packages have been upgraded, follow these steps:

- shutdown Visual Studio if this project is opened in Visual Studio
- do rebasing on your branch
- run `.paket\paket.exe restore` or simply run `build.bat`
- then open the Visual Studio again

The reason why we need close Visual Studio, is because Alea GPU uses Fody plugin, which is a build plugin, and Visual Studio will use it in the process, so if Fody package is upgraded, it cannot be written to `packages` folder.

