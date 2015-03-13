module DocExtensions.Formatting

#if INTERACTIVE
#I "../../packages/FSharp.Compiler.Service/lib/net40"
#I "../../packages/FSharp.Formatting/lib/net40"
#r @"../../packages/RazorEngine/lib/net40/RazorEngine.dll"
#r @"FSharp.Markdown.dll"
#r @"FSharp.CodeFormat.dll"
#r @"FSharp.Literate.dll"

System.IO.Directory.SetCurrentDirectory __SOURCE_DIRECTORY__
#endif 

open System.IO

open FSharp.CodeFormat
open FSharp.Literate
open FSharp.Literate.Matching
open FSharp.Markdown

/// Language names (input/output subfolders should be the same)
[<Literal>]
let FSHARP = "fsharp"
[<Literal>]
let FSHARP_QUOT = "fsharp-quotations"
[<Literal>]
let CSHARP = "csharp"
[<Literal>]
let VB = "vb"

/// Directory settings to define tutorial structure
type DirectorySettings = 
    {
        SamplesRootDir: string // Directory paths are tool-relative
        TutorialDir: string
        OutputDir: string
        DocTemplate: string
        LayoutRoots: string list
        Info: Map<string, string>
    }

/// Map language names to file extensions
let fileExt lang =
    match lang with
    | FSHARP | FSHARP_QUOT -> ".fs"
    | CSHARP -> ".cs"
    | VB -> ".vb"
    | _ -> failwithf "unsupported language: %s" lang

/// Path to a language-specific file given its name and root directory
let inline langSpecificPath root dir name lang = Path.Combine(root, lang, dir, name + fileExt lang)

/// Set of functions used to generate actual documents from F# template
module internal CodeReplacement =
    open System.Reflection

    // Check if a paragraph should be replaced for a concrete language - marked as (*** define:... ***)
    let inline (|CodePlaceholder|_|) paragraph =
        match paragraph with
        | LiterateParagraph (LiterateCode (lines, ({Visibility = NamedCode name}))) ->
            Some (lines, name)
        | _ -> None

    // Choose all headings till the first sample code block
    let rec takeHeaders hs = function
        | [] -> List.rev hs, []
        | (CodePlaceholder _ :: _) as ps -> List.rev hs, ps
        | (Heading _ as h) :: ps -> takeHeaders (h::hs) ps
        | _ :: ps-> takeHeaders hs ps

    // Collect everything but headings till the first sample code block
    let rec findCode reqs = function
        | [] -> List.rev reqs, None, []
        | Heading _ :: ps -> findCode reqs ps
        | (CodePlaceholder code) :: ps -> List.rev reqs, Some code, ps
        | reqParagraph :: ps -> findCode (reqParagraph::reqs) ps 

    // Embed F# code - as if it wasn't inside a 'define' section
    let inline embedFsCode lines = 
        EmbedParagraphs (LiterateCode (lines, {Evaluate = true; OutputName = None; Visibility = VisibleCode}))
    
    // Add reference to a concrete language sample section (e.g. [lang=csharp,file=Sample.cs,key=test])
    let inline embedCustomCode lang path key = 
        CodeBlock (sprintf "[lang=%s,file=%s,key=%s]" lang path key)

    // Merge F# template with another F# file (used for FSHARP_QUOT): take template comments and actual code
    let rec merge template replacement = seq {
        match takeHeaders [] template with
        | hs, CodePlaceholder (_, sampleName) :: template ->
            yield! hs
            match findCode [] replacement with
            | reqs, Some (lines, name), replacement' ->
                yield! reqs
                if name <> sampleName then
                    System.Diagnostics.Debug.WriteLine("Skipping code section: sample '{0}' was expected, but got '{1}'", sampleName, name)
                    yield! merge template replacement
                else
                    yield embedFsCode lines
                    yield! merge template replacement'
            | [], _, _ -> yield! merge template []
            | _ ->
                System.Diagnostics.Debug.WriteLine("Skipping code section: sample '{0}' wasn't found", sampleName)
                yield! merge template [] 

        | hs, _ -> yield! hs
    }

    let embedCode path lines key = function
        | FSHARP -> embedFsCode lines
        | CSHARP -> embedCustomCode CSHARP path key
        | VB -> embedCustomCode VB path key
        | lang -> failwithf "unsupported language: %s" lang
    
    // Create an HTML container for the snippets
    let createSnippets filePath lines key langs =
        let link i lang =
            let name = match lang with 
                       | FSHARP -> "F#"
                       | CSHARP -> "C#"
                       | VB     -> "VB"
                       | _      -> lang
            sprintf """<div class="tablink-%d">%s</div>""" i name
        
        let snippet i lang =
            let code = embedCode (filePath lang) lines key lang
            [ InlineBlock (sprintf """<div class="tab-%d">""" i)
              code
              InlineBlock "</div>" ]

        let controlStart = """<div class="container-fluid"><div id="tabs" class="row-fluid"><div id="anim-holder">"""
        let links = System.String.Join(" ", Array.mapi link langs)
        let snippets = Seq.mapi snippet langs |> Seq.collect id
        let controlEnd = "</div></div></div>"

        [ yield InlineBlock controlStart
          yield InlineBlock links
          yield! snippets
          yield InlineBlock controlEnd ]

    // Fill template paragraphs with an actual implementation
    let replaceWithLang paragraphs filePath langs = 
        if File.Exists (filePath FSHARP) then
            let langs = langs |> Array.filter (fun lang -> File.Exists (filePath lang))

            paragraphs
            |> List.collect (function
                | CodePlaceholder (lines, key) ->
                    createSnippets filePath lines key langs
                | paragraph -> [paragraph])
        else
            []
    
    // Use 'formatCode' method from internal type 'FSharp.Literate.Transformations'
    let formatCodeSnippetsMethod =         
        let flags = BindingFlags.InvokeMethod ||| BindingFlags.NonPublic ||| BindingFlags.Static
        let transformationsType = typeof<FSharp.Literate.LiterateDocument>.Assembly.GetType "FSharp.Literate.Transformations"

        transformationsType.GetMethods flags |> Array.find (fun m -> m.Name.StartsWith "formatCodeSnippets")
    
    let inline formatCodeSnippets fileName compilerOptions doc = 
        let ctx = { FormatAgent = CodeFormat.CreateAgent(); CompilerOptions = Some compilerOptions; Evaluator = None; DefinedSymbols = None } 
        formatCodeSnippetsMethod.Invoke(null, [|fileName; ctx; doc|]) :?> LiterateDocument

/// Generate HTML 
// - dirSettings: key information about the directories (roots, template paths etc)
// - replacements: set of replacements, including dirSettings info and path-specific ones
// - compilerOptions: required compiler options, e.g. dll references
// - outputPath: path to the output (may be absolute)
// - sampleDir: path to the sample directory (relative to the root)
// - sampleName: name of the sample files without extension
// - langs: the list of languages used with template
// The lang-specific subfolder names are supposed to be the same as language names, like FSHARP
let processFile (dirSettings: DirectorySettings) replacements compilerOptions outputPath sampleDir sampleName langs = 
    let templatePath = langSpecificPath dirSettings.SamplesRootDir sampleDir sampleName FSHARP
    let templateDoc = Literate.ParseScriptFile(templatePath, compilerOptions = compilerOptions)
    
    let ctx: ProcessingContext = { TemplateFile = Some dirSettings.DocTemplate 
                                   Replacements = replacements
                                   GenerateLineNumbers = false
                                   IncludeSource = false
                                   Prefix = "fs"
                                   OutputKind = OutputKind.Html
                                   GenerateHeaderAnchors = false
                                   LayoutRoots = dirSettings.LayoutRoots }

    match CodeReplacement.replaceWithLang templateDoc.Paragraphs (langSpecificPath dirSettings.SamplesRootDir sampleDir sampleName) langs with
    | [] -> ()
    | newParagraphs ->
        let fsRelativePath = 
            let innerDirs = Seq.filter ((=)Path.DirectorySeparatorChar) sampleDir |> Seq.length
            let up = String.replicate (innerDirs + 1) (".." + string Path.DirectorySeparatorChar)
            Path.Combine(up, sampleDir, sampleName + fileExt FSHARP)

        let formattedDoc = 
            templateDoc.With(paragraphs = newParagraphs, sourceFile = fsRelativePath)
            |> CodeReplacement.formatCodeSnippets fsRelativePath compilerOptions

        Templating.processFile formattedDoc outputPath ctx
