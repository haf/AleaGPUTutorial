(*** hide ***)
module Tutorial.Fs.examples.genericScan.Plan

open Tutorial.Fs.examples

let WARP_SIZE = genericReduce.Plan.WARP_SIZE
let LOG_WARP_SIZE = genericReduce.Plan.LOG_WARP_SIZE

type Plan = genericReduce.Plan.Plan
type Planner = genericReduce.Plan.Planner

