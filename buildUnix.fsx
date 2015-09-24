#r @"packages/FAKE/tools/FakeLib.dll"

open System
open Fake

let resultsDir = "release"
let docDir = "docs/output"

Target "Clean" (fun _ ->
    DeleteDirs [resultsDir; docDir]
)

Target "Build" (fun _ ->
    !! "/**/*.*sproj"
    |> MSBuildRelease resultsDir "Build"
    |> Log "Build-Output: "
)

Target "Tests" (fun _ ->
    !! "/**/*.exe"
    |> SetBaseDir resultsDir
    //|> NUnitParallel (fun defaults -> { defaults with Framework = "net-4.5"})
    |> NUnit (fun defaults -> { defaults with Framework = "4.5"
                                              TimeOut = (TimeSpan.FromDays 1.0)  } )
)

Target "Default" DoNothing

"Clean"
    ==> "Build"
    =?> ("Tests", not <| hasBuildParam "NoTests")
    ==> "Default"

RunTargetOrDefault "Default"
