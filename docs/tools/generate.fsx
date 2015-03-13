// --------------------------------------------------------------------------------------
// Builds the tutorial documentation 
// (the generated documentation is stored in the 'docs/output' directory)
// --------------------------------------------------------------------------------------

#I @"..\..\packages\FSharp.Compiler.Service\lib\net40"
#I @"..\..\packages\FSharp.Formatting\lib\net40"
#I @"..\..\packages\RazorEngine\lib\net40"
#I @"..\..\packages\Microsoft.AspNet.Razor\lib\net40"
#I @"..\..\packages\FAKE\tools"
#r "FakeLib.dll"
#r "System.Web.Razor.dll"
#r "RazorEngine.dll"
#r "FSharp.Markdown.dll"
#r "FSharp.Literate.dll"
#r "FSharp.CodeFormat.dll"
#r "FSharp.MetadataFormat.dll"

#load "formatting.fsx"
#load "directories.fsx"

open Fake
open Fake.FileHelper
open FSharp.MetadataFormat

open DocExtensions.Formatting
open DocExtensions.Directories

#if RELEASE
let root = "/static/app/tutorial"
#else
let root = "file://" + (__SOURCE_DIRECTORY__ @@ "..\\output")
#endif
printfn "document root = %s" root

// Binaries that have XML documentation (in a corresponding generated XML file)
let referenceBinaries = [ ]

// Paths with template/source/output locations
let src        = __SOURCE_DIRECTORY__ @@ "..\\..\\src"
let bin        = __SOURCE_DIRECTORY__ @@ "..\\..\\release"
let packages   = __SOURCE_DIRECTORY__ @@ "..\\..\\packages"

let output     = __SOURCE_DIRECTORY__ @@ "..\\output"
let tutorial   = __SOURCE_DIRECTORY__ @@ "..\\tutorial"
let files      = __SOURCE_DIRECTORY__ @@ "..\\files"
let templates  = __SOURCE_DIRECTORY__ @@ "templates"

let formatting = packages @@ "FSharp.Formatting"

let docTemplate = templates @@ "template.html"
let fsharpDir = src @@ "fsharp"

let info = Map.ofList [ "root", root; "project-name", "GPU tutorial" ] 

// Where to look for *.csproj templates (in this order)
let layoutRoots =
  [ templates; formatting @@ "templates"
    formatting @@ "templates/reference" ]

// Copy static files and CSS + JS from F# Formatting
let copyFiles () =
  let content = output @@ "content"
  ensureDirectory content
  
  CopyRecursive files content true |> Log "Copying file: "
  CopyRecursive (templates @@ "styles") content true 
    |> Log "Copying styles and scripts: "

// Build API reference from XML comments
let buildReference () =
  CleanDir (output @@ "reference")
  for lib in referenceBinaries do
    MetadataFormat.Generate
      ( bin @@ lib, output @@ "reference", layoutRoots, 
        parameters = Map.toList info,
        sourceFolder = __SOURCE_DIRECTORY__.Substring(0, __SOURCE_DIRECTORY__.Length - @"\docs\tools".Length ) )

// References should be absolute and use '\' as separator, if they start with '..' or separator is '/' the tooltips are not shown
// They are passed to the compiler because original code files are .fs 
let references = [
    packages @@ @"Alea.CUDA\lib\net40\Alea.CUDA.dll"
    packages @@ @"Alea.CUDA.IL\lib\net40\Alea.CUDA.IL.dll"
    packages @@ @"Alea.IL\lib\net40\Alea.IL.dll"
]

let buildDocumentation() =
    let compilerOptions = System.String.Join(" ",  List.map ((+) "-r:") references)
    let dirSettings = 
        { SamplesRootDir = "..\\..\\src"
          TutorialDir = tutorial
          OutputDir = output
          DocTemplate = docTemplate
          Info = info
          LayoutRoots = layoutRoots }

    buildDocs dirSettings compilerOptions [| CSHARP; FSHARP; VB |]

copyFiles()
buildDocumentation()
