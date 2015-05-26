(*** hide ***)
[<AutoOpen>]
module Tutorial.Fs.examples.cudnn.Data
open System
open System.IO

(*** define:CudnnMnistData ***)
let findSolutionDir (startDir:string) =
    let filesToCheck = [ "Alea.Tutorial.sln"
                         "build.bat"
                         "build.fsx" ]

    let isSolutionDir (dir:string) =
        filesToCheck |> List.forall (fun file -> File.Exists(Path.Combine(dir, file)))

    let rec find (dir:string) =
        if isSolutionDir dir then dir
        else find (Directory.GetParent(dir).FullName)

    find startDir

let datadir = Path.Combine(findSolutionDir "./", "src/fsharp/examples/cudnn/data")

let getPath fname = Path.Combine(datadir, fname)

let readBinaryFile fname size = 
    let b = File.ReadAllBytes(fname)
    [for i in [0..4..b.Length-4] do yield BitConverter.ToSingle(b,i)]
    |> Seq.toArray

let loadImage fname = File.ReadAllBytes(getPath fname).[52..]

let ImageH = 28
let ImageW = 28

let FirstImage = "one_28x28.pgm"
let SecondImage = "three_28x28.pgm"
let ThirdImage = "five_28x28.pgm"

let Conv1Bin = "conv1.bin"
let Conv1BiasBin = "conv1.bias.bin"
let Conv2Bin = "conv2.bin"
let Conv2BiasBin = "conv2.bias.bin"

let Ip1Bin = "ip1.bin"
let Ip1BiasBin = "ip1.bias.bin"
let Ip2Bin = "ip2.bin"
let Ip2BiasBin = "ip2.bias.bin"

