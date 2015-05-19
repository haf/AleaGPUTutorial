module DocExtensions.Directories

#if INTERACTIVE
#I "../../packages/FSharp.Compiler.Service/lib/net40"
#I "../../packages/FSharp.Formatting/lib/net40"
#r @"../../packages/RazorEngine/lib/net40/RazorEngine.dll"
#r @"../../packages/FAKE/tools/FakeLib.dll"
#r @"FSharp.Markdown.dll"
#r @"FSharp.CodeFormat.dll"
#r @"FSharp.Literate.dll"

#load "formatting.fsx"

System.IO.Directory.SetCurrentDirectory __SOURCE_DIRECTORY__
#endif 

open System
open System.IO
open System.Collections.Generic
open System.Diagnostics

open Fake
open FSharp.Literate

open DocExtensions.Formatting

let htmlEncode (text:string) =
    text.Replace("&", "&amp;")
        .Replace("<", "&lt;")
        .Replace(">", "&gt;")
        .Replace("\"", "&quot;")

let normalizeDocumentName (name:string) = 
    if String.IsNullOrWhiteSpace name then 0, "", ""
    else
        let idx = name.IndexOf('.')
        let order = name.Substring(0, idx)
        let order = Int32.Parse(order)
        let name = name.Substring(idx + 1, name.Length - idx - 1)
        let filename = name.Replace(' ', '_')
        let filename = filename.Replace('.', '_')
        let filename = filename.Replace(",", "")
        let filename = filename.Replace("#", "sharp")
        let filename = filename.ToLower()
        order, name, filename

type [<AbstractClass>] Document(dirSettings: DirectorySettings, parent: Document option, originalDocName: string, ?originalFileName: string) =
    let order, name, docName = normalizeDocumentName originalDocName

    let originalFileName, outputFileName =
        match originalFileName with
        | Some fileName -> fileName, docName + ".html"
        | _ -> "", "index.html"

    member this.IsRoot = parent.IsNone
    member this.Name = name
    member this.Order = order 
    member this.Parent = match parent with Some parent -> parent | None -> failwith "This is root doc"

    // Normalized relative directory path
    member this.Dir =     
        match parent, String.IsNullOrWhiteSpace originalFileName with
        | Some parent, true -> parent.Dir @@ docName
        | Some parent, false -> parent.Dir
        | _ -> ""

    member this.UrlName = dirSettings.Info.["root"] @@ this.Dir @@ outputFileName

    member this.TutorialDir =
        match parent, String.IsNullOrWhiteSpace originalFileName with
        | Some parent, true -> parent.TutorialDir @@ originalDocName
        | Some parent, false -> parent.TutorialDir
        | _ -> dirSettings.TutorialDir
    
    member this.TutorialPath = this.TutorialDir @@ originalFileName

    member this.OutputDir = dirSettings.OutputDir @@ this.Dir
    member this.OutputPath = this.OutputDir @@ outputFileName

    abstract Dump : unit -> unit
    abstract Build : unit -> unit

type Folder(dirSettings: DirectorySettings, parent:Document option, folderName:string) =
    inherit Document(dirSettings, parent, folderName)

    let documents = List<Document>()

    member this.AddDocument(doc) = documents.Add(doc)
    member this.Documents = documents |> Seq.toArray |> Array.sortBy (fun doc -> doc.Order)
    override this.Dump() = this.Documents |> Array.iter (fun doc -> doc.Dump())
    override this.Build() = documents |> Seq.iter (fun doc -> doc.Build())   

    member this.GenNavList(urlroot:string, child:Document) =
        let strs = List<string>()

        if not this.IsRoot then
            let parent = this.Parent :?> Folder
            strs.Add(parent.GenNavList(urlroot, this))

        strs.Add(sprintf "<li class=\"nav-header\">%s</li>" this.Name)

        this.Documents |> Array.iter (fun doc -> doc.Name |> function
            | "Index" | "index" -> ()
            | name when name = child.Name -> strs.Add(sprintf "<li class=\"active\"><a href=\"%s\">%s</a></li>" doc.UrlName doc.Name)
            | name -> strs.Add(sprintf "<li><a href=\"%s\">%s</a></li>" doc.UrlName doc.Name))

        strs |> String.concat "\n"

    member this.GenIndex(urlroot:string, child:Document) =
        let strs = List<string>()

        if child.Order = 0 then
            strs.Add("<ul>")
            this.Documents |> Array.iter (fun doc -> doc.Name |> function
                | "Index" | "index" -> ()
                | name -> strs.Add(sprintf "<li><a href=\"%s\">%s</a></li>" doc.UrlName doc.Name))
            strs.Add("</ul>")
        strs |> String.concat "\n"

type [<AbstractClass>] Page(dirSettings: DirectorySettings, parent:Document option, pageName:string) =
    inherit Document(dirSettings, parent, Path.GetFileNameWithoutExtension pageName, pageName)

    override this.Dump() = printfn "%s -> %s" this.TutorialPath this.OutputPath

let getReplacements (dirSettings: DirectorySettings) (doc: #Document) (scriptOutput: string) =
    let root = dirSettings.Info.["root"]
    let parentFolder = doc.Parent :?> Folder
    dirSettings.Info
    |> Map.add "nav-list" (parentFolder.GenNavList(root, doc))
    |> Map.add "index" (parentFolder.GenIndex(root, doc))
    |> Map.add "script-output" scriptOutput
    |> Map.toList

type MarkdownPage(dirSettings: DirectorySettings, parent:Document option, pageName:string) =
    inherit Page(dirSettings, parent, pageName)

    override this.Build() =
        printfn "Generating %s ..." this.UrlName
        let replacements = getReplacements dirSettings this ""
        Literate.ProcessMarkdown(this.TutorialPath, dirSettings.DocTemplate, this.OutputPath, OutputKind.Html, replacements = replacements, lineNumbers = true)

type ScriptPage(dirSettings: DirectorySettings, parent:Document option, pageName:string) =
    inherit Page(dirSettings, parent, pageName)

    override this.Build() =
        printfn "Generating %s ..." this.UrlName

        let exitcode, stdout, stderr =
            use p = new Process()
            p.StartInfo.FileName <- "fsi"
            p.StartInfo.Arguments <- sprintf "-O \"%s\"" this.TutorialPath
            p.StartInfo.WorkingDirectory <- this.TutorialDir
            p.StartInfo.CreateNoWindow <- false
            p.StartInfo.UseShellExecute <- false
            p.StartInfo.RedirectStandardOutput <- true
            p.StartInfo.RedirectStandardError <- true
            if not (p.Start()) then failwithf "Fail to run %s" this.TutorialPath
            let stdout = p.StandardOutput.ReadToEnd()
            let stderr = p.StandardError.ReadToEnd()
            p.WaitForExit()
            p.ExitCode, stdout, stderr

        if exitcode <> 0 then
            printfn "%s" stderr
            failwithf "Fail to run %s" this.TutorialPath
        let scriptOutput = sprintf "<h2>Script Output</h2><pre lang=\"text\">%s</pre>" (htmlEncode(stdout))

        let replacements = getReplacements dirSettings this scriptOutput
        Literate.ProcessScriptFile(this.TutorialPath, dirSettings.DocTemplate, this.OutputPath, OutputKind.Html, replacements = replacements, lineNumbers = true)


type SamplePage(dirSettings: DirectorySettings, compilerOptions: string, parent: Document option, pageName: string, langs: string[]) =
    inherit Page(dirSettings, parent, pageName)

    override this.Build() =
        let sampleName = (File.ReadAllText this.TutorialPath)
        let replacements = getReplacements dirSettings this ""
        processFile dirSettings replacements compilerOptions this.OutputPath this.Dir sampleName langs

let createDocument (dirSettings: DirectorySettings) (compilerOptions: string) langs =
    let rec create (parent:Document option) (docName:string) =
        let folder = Folder(dirSettings, parent, docName)
        Directory.CreateDirectory folder.OutputDir |> ignore // ensure that directory is created

        let parent = Some (folder :> Document)
        let tutorial = DirectoryInfo folder.TutorialPath

        for file in tutorial.GetFiles() do
            let fileName = file.Name
            let doc =
                let ext = Path.GetExtension(fileName)
                match ext with
                | ".md" -> MarkdownPage(dirSettings, parent, fileName) :> Document
                | ".fsx" -> ScriptPage(dirSettings, parent, fileName) :> Document
                | ".ref" -> SamplePage(dirSettings, compilerOptions, parent, fileName, langs) :> Document
                | ext -> failwithf "Unknown ext %s" ext
            folder.AddDocument(doc)

        for dir in tutorial.GetDirectories() do
            let doc = create parent dir.Name
            folder.AddDocument(doc)
        folder :> Document

    let rootdoc = create None "00.Alea GPU Tutorial"
    rootdoc

let buildDocs dirSettings compilerOptions langs =
    let rootdoc = createDocument dirSettings compilerOptions langs
    rootdoc.Build()

let buildDocsLive dirSettings langs =
    let run _ =
        try 
            let rootdoc = createDocument dirSettings "" langs
            rootdoc.Build()
        with
        | e -> printfn "%A" e

    let watcher = new FileSystemWatcher(dirSettings.TutorialDir)
    watcher.IncludeSubdirectories <- true
    watcher.Changed.Add run
    watcher.EnableRaisingEvents <- true
    run()

    printf "waiting for changes, press any key to quit..."
    Console.ReadKey() |> ignore
