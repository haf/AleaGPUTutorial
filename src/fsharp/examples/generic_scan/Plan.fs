(*** hide ***)
module Tutorial.Fs.examples.genericScan.Plan

open Tutorial.Fs.examples

let WARP_SIZE = genericReduce.Plan.WARP_SIZE
let LOG_WARP_SIZE = genericReduce.Plan.LOG_WARP_SIZE

(**
The Plan type for generic scan is the same used for generic reduce.
*)
type Plan = genericReduce.Plan.Plan
type Planner = genericReduce.Plan.Planner

