#r @"packages/FAKE/tools/FakeLib.dll"

open Fake

let resultsDir = "release"
let docDir = "docs/output"

Target "Clean" (fun _ ->
    DeleteDirs [resultsDir; docDir]
)

Target "Build" (fun _ ->
    !! "/**/*.*proj"
    |> MSBuildRelease resultsDir "Build"
    |> Log "Build-Output: "
)

Target "Tests" (fun _ ->
    !! "/**/Test.*.dll"
    ++ "/**/Test.*.exe"
    |> SetBaseDir resultsDir
    |> NUnitParallel (fun defaults -> { defaults with Framework = "net-4.5"})
)

Target "BuildDocs" (fun _ ->
    let arg = if hasBuildParam "Release" then "--define:RELEASE" else "--define:DEBUG"
    if not <| executeFSIWithArgs "docs/tools" "generate.fsx" [arg] [] then failwith "documentation building failed.")

Target "Default" DoNothing

"Clean"
    ==> "Build"
    =?> ("Tests", not <| hasBuildParam "NoTests")
    ==> "BuildDocs"
    ==> "Default"

RunTargetOrDefault "Default"
