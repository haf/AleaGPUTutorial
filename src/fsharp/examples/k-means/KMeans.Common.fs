[<AutoOpen>]
module KMeans.Common

open System
open System.IO
open System.Drawing
open System.Windows.Forms
open Alea.CUDA

let readRGB file =
    let imageData = File.ReadAllBytes file
    use stream = new MemoryStream(imageData)
    use image = new Bitmap(stream)
    Array2D.init image.Width image.Height (fun i j -> image.GetPixel(i, j))

let showImage title pixels =
    let form = new Form()
    let picture = new PictureBox()
    let image = new Bitmap(Array2D.length1 pixels, Array2D.length2 pixels)
    Array2D.iteri (fun i j col -> image.SetPixel(i, j, col)) pixels
    picture.Image <- image
    picture.SizeMode <- PictureBoxSizeMode.AutoSize
    form.Controls.Add picture
    form.Size <- picture.Size
    form.Text <- title
    form.Show()

let mutable doPrint = true
let mutable timing = 0.0

let mutable threshold = 0.01
let mutable maxIters = 100
let mutable seed = 42

// use a worker created on current thread to avoid thread context switching
let worker = lazy Worker.CreateOnCurrentThread(Device.Default)