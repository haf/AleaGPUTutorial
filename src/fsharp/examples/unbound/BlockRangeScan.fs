(*** hide ***)
module Tutorial.Fs.examples.unbound.BlockRageScan

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound
open NUnit.Framework
open FsUnit

(**
Matrix row scan resource.

Using block range scan from Alea Unbound. 
*)
type MatrixRowScanPrimitive<'T>(arch:DeviceArch, op:Expr<'T -> 'T -> 'T>) =

    let blockRangeScan = DeviceScanPolicy.Create(arch, PlatformUtil.Instance.ProcessBitness, None).BlockRangeScan

    [<ReflectedDefinition>]
    member this.BlockRangeScan blockOffset blockEnd (inputs:deviceptr<_>) =
        let tempStorage = blockRangeScan.TempStorage.AllocateShared()
        blockRangeScan.ConsumeRangeConsecutiveInclusive
            tempStorage (Iterator inputs) (Iterator inputs) (__eval op) blockOffset blockEnd

    member this.BlockThreads = blockRangeScan.BlockThreads

(**
Matrix row scan module.
*)
type MatrixRowScanModule<'T>(target, op:Expr<'T -> 'T -> 'T>) as this =
    inherit GPUModule(target)

    let primitive = 
        fun (options:CompileOptions) ->
            cuda { return MatrixRowScanPrimitive(options.MinimalArch, op) }
        |> this.GPUDefineResource

    [<Kernel;ReflectedDefinition>]
    member this.Kernel numCols (inputs:deviceptr<_>) =
        let blockOffset = blockIdx.x * numCols
        primitive.Resource.BlockRangeScan blockOffset (blockOffset + numCols) inputs

    member this.Apply(numRows:int, numCols:int, inputs:deviceptr<_>) =
        let lp = LaunchParam(numRows, primitive.Resource.BlockThreads)
        this.GPULaunch <@ this.Kernel @> lp numCols inputs

    member this.Apply(data:'T[,]) =
        let numRows, numCols = Array2D.length1 data, Array2D.length2 data
        let data = Array2D.toArrayRowMajor data |> this.GPUWorker.Malloc
        this.Apply(numRows, numCols, data.Ptr)
        data.Gather() |> Array2D.ofArrayRowMajor numRows numCols

[<ReflectedDefinition>]
let parallelSum (a:int2) (b:int2) =
    int2(a.x + b.x, a.y + b.y)

[<AOTCompile>]
type MatrixRowScanInt2Sum(target) = 
    inherit MatrixRowScanModule<int2>(target, <@ parallelSum @>)
    static let instance = lazy new MatrixRowScanInt2Sum(GPUModuleTarget.DefaultWorker)
    static member DefaultInstance = instance.Value

let generateTestData numRows numCols =
    let rng = new Random()
    Array.init numRows (fun _ -> Array.init numCols (fun _ -> rng.Next(0, 10), rng.Next(0, 10)))   
    |> array2D
    |> Array2D.map (fun (x, y) -> int2 (x, y))

let scanRows (op:'T -> 'T -> 'T) (inputs:'T[,]) =
    let m = Array2D.numRows inputs
    let n = Array2D.numCols inputs
    let results = Array2D.zeroCreate m n
    for i = 0 to m-1 do
        results.[i, 0] <- inputs.[i, 0]
        for j = 1 to n-1 do
            results.[i, j] <- op results.[i, j-1] inputs.[i, j]
    results

[<Test>]
let blockRangeScanTest () =   
    let worker = Worker.Default
    let inputs = generateTestData 4 8
    let expected = scanRows parallelSum inputs

    let scan = MatrixRowScanInt2Sum.DefaultInstance
    let outputs = scan.Apply inputs

    printfn "inputs = %A" inputs
    printfn "outputs = %A" outputs
    printfn "expected = %A" expected

    outputs |> should equal expected

// TODO fixit and move: temporary work

type ExpandWeightsModule(target) =
    inherit GPUModule(target)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel n numWeights (numPerms:int) (permutations:deviceptr<int>) (weights:deviceptr<int>) (permutedWeights:deviceptr<int>) =
        let permutationsShared = __shared__.ExternArray<int>()  
        let tid = threadIdx.x 
        let bid = blockIdx.x
        let permutationId = blockIdx.y
        let blockSize = blockDim.x 
        let pi = blockSize*bid + tid
        let ld = numWeights*numPerms
        
        if pi < n then
            permutationsShared.[tid] <- permutations.[permutationId*n + pi]
        
        __syncthreads()

        let mutable pi = 0
        let mutable permutedWeightOffset = ld*blockSize*bid + permutationId*numWeights
        while pi < blockSize do
            let p = permutationsShared.[pi]
            let mutable wi = tid
            while wi < numWeights do
                permutedWeights.[permutedWeightOffset + wi] <- weights.[numWeights*p + wi]
                wi <- wi + blockSize   
            permutedWeightOffset <- permutedWeightOffset + ld
            pi <- pi + 1

    member this.Apply(n:int, numWeights:int, numPerms:int, permutations:deviceptr<_>, weights:deviceptr<_>, permutedWeights:deviceptr<_>) =
        let blockSize = 16
        if n % blockSize <> 0 then failwithf "number of samples %d must be a multiple of block size %d" n blockSize
        let blocks = dim3(n/blockSize, numPerms)
        let sharedSize = blockSize * __sizeof<int>()
        let lp = LaunchParam(blocks, dim3 blockSize, sharedSize)
        this.GPULaunch <@ this.Kernel @> lp n numWeights numPerms permutations weights permutedWeights

    member this.Apply(permutations:int[,], weights:int[,]) =
        let numPerms = Array2D.length1 permutations
        let numWeights, n = Array2D.length1 weights, Array2D.length2 weights
        let permutations = Array2D.toArrayRowMajor permutations |> this.GPUWorker.Malloc
        let weights = Array2D.toArrayColumnMajor weights |> this.GPUWorker.Malloc
        printfn "perms = %A" (permutations.Gather())
        let permutedWeights = this.GPUWorker.Malloc<int>(numWeights*numPerms*n)
        this.Apply(n, numWeights, numPerms, permutations.Ptr, weights.Ptr, permutedWeights.Ptr)
        permutedWeights.Gather() |> Array2D.ofArrayColumnMajor (numWeights*numPerms) n
    
    [<Kernel;ReflectedDefinition>]
    member this.KernelSimple n numWeights (numPerms:int) (permutations:deviceptr<int>) (weights:deviceptr<int>) (permutedWeights:deviceptr<int>) =
        let weightId = blockIdx.y
        let permutationId = blockIdx.z
        let stride = gridDim.x * blockDim.x  
        let weightOffset = n*numWeights
        
        let mutable pi = blockIdx.x * blockDim.x + threadIdx.x 
        while pi < n do
            permutedWeights.[weightOffset*permutationId + n*weightId + pi] <- weights.[n*weightId + permutations.[n*permutationId + pi]]
            pi <- pi + stride

    member this.ApplySimple(n:int, numWeights:int, numPerms:int, permutations:deviceptr<_>, weights:deviceptr<_>, permutedWeights:deviceptr<_>) =
        let blockSize = 128
        let blocks = dim3(divup n blockSize, numWeights, numPerms)
        let lp = LaunchParam(blocks, dim3 blockSize)
        this.GPULaunch <@ this.KernelSimple @> lp n numWeights numPerms permutations weights permutedWeights

    member this.ApplySimple(permutations:int[,], weights:int[,]) =
        let numPerms = Array2D.length1 permutations
        let numWeights, n = Array2D.length1 weights, Array2D.length2 weights
        let permutations = Array2D.toArrayRowMajor permutations |> this.GPUWorker.Malloc
        let weights = Array2D.toArrayRowMajor weights |> this.GPUWorker.Malloc
        let permutedWeights = this.GPUWorker.Malloc<int>(numWeights*numPerms*n)
        this.ApplySimple(n, numWeights, numPerms, permutations.Ptr, weights.Ptr, permutedWeights.Ptr)
        permutedWeights.Gather() |> Array2D.ofArrayRowMajor (numWeights*numPerms) n

let expandWeights (permutations:int[,]) (weights:int[,]) =
    if Array2D.length2 permutations <> Array2D.length2 weights then failwith "invalid dimensions"
    let nw = Array2D.length1 weights
    let n = Array2D.length2 weights
    let k = Array2D.length1 permutations
    let permutedWeights = Array2D.zeroCreate (nw*k) n
    for l = 0 to k - 1 do
        for i = 0 to nw - 1 do     
            for j = 0 to n - 1 do
                permutedWeights.[l*nw + i, j] <- weights.[i, permutations.[l, j]]
    permutedWeights

let generateWeights numWeights n =
    let rng = new Random()
    Array.init numWeights (fun l -> Array.init n (fun i -> (l+1)*100 + i))   
    |> array2D

let generatePermutations numPerm n =
    let rng = new Random()
    Array.init numPerm (fun _ -> Array.init n (fun _ -> rng.Next(0, n-1)))   
    |> array2D

[<Test>]
let expandWeightsTest () =
    let expandWeightsModule = new ExpandWeightsModule(GPUModuleTarget.DefaultWorker)
    let blockSize = 16
    let n = 2*blockSize
    let permutations = generatePermutations 2 n
    let weights = generateWeights 8 n

    printfn "permutations = \n%A" permutations
    printfn "weights = \n%A" weights 
    
    let expandedWeights = expandWeights permutations weights
    let expandedWeightsGPU = expandWeightsModule.ApplySimple(permutations, weights)

    printfn "expandedWeights = \n%A" expandedWeights        
    printfn "expandedWeightsGPU = \n%A" expandedWeightsGPU

    let expandedWeightsGPU' = expandWeightsModule.Apply(permutations, weights)
    printfn "expandedWeightsGPU' = \n%A" expandedWeightsGPU'