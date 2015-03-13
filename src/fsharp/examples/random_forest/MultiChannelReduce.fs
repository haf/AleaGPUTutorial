module Tutorial.Fs.examples.RandomForest.Cuda.MultiChannelReduce

open Tutorial.Fs.examples.RandomForest.Cublas

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound

[<Struct;Align(8)>]
type ValueAndIndex =
    val Value : float
    val Index : int
    
    [<ReflectedDefinition>]
    new (value, index) = { Value = value; Index = index }
    
    override this.ToString() =
        sprintf "Value: %f Index: %d" this.Value this.Index
with
    [<ReflectedDefinition>]
    static member inline Opt sign (a:ValueAndIndex) (b:ValueAndIndex) = 
        let comparison = if (a.Value > b.Value) then sign else -sign
        if comparison > 0 then a else b

    [<ReflectedDefinition>]
    static member inline OPT_INFINITY sign =
        ValueAndIndex(System.Double.NegativeInfinity * (float sign), -1)

    [<ReflectedDefinition>]
    static member ofDouble (value : float) index = 
        new ValueAndIndex(value, index)

    [<ReflectedDefinition>]
    static member ofSingle (value : float32) index = 
        new ValueAndIndex(float value, index)

[<Literal>]
let private BLOCK_SIZE = 1024

type private MinOrMax =
    | Min
    | Max

    member this.Sign = 
        match this with
        | Min -> -1
        | Max -> 1

    member this.OptFun a b = ValueAndIndex.Opt this.Sign a b

    member this.DefaultValue = ValueAndIndex.OPT_INFINITY this.Sign

(**
Matrix row scan resource.

Using block range scan from Alea Unbound. 
*)
type private MultiChannelReducePrimitive(arch:DeviceArch) =

    let blockReduce = BlockReduce.WarpReductions<ValueAndIndex>(dim3 BLOCK_SIZE, arch)

    [<ReflectedDefinition>]
    member inline internal this.OptAndArgOpt (output:deviceptr<ValueAndIndex>)  numOutputCols 
        sign (matrix:deviceptr<_>) numCols numValid (indices:deviceptr<int>) hasIndices =
        __static_assert(__is_static_constant hasIndices)

        let optFun = ValueAndIndex.Opt sign
        let defaultValue = ValueAndIndex.OPT_INFINITY sign

        let tempStorage = blockReduce.TempStorage.AllocateShared()
        let tid = threadIdx.x
        let colIdx = blockIdx.y * blockDim.x + tid

        let reduceInput = 
            if colIdx < numValid then 
                let matrixIdx = blockIdx.x * numCols + colIdx 
                let index = if hasIndices then indices.[matrixIdx] else colIdx
                ValueAndIndex.ofDouble (__gfloat matrix.[matrixIdx]) index
            else 
                defaultValue
        let aggregate = blockReduce.Reduce(tempStorage, reduceInput, optFun)

        if tid = 0 then
            output.[blockIdx.x * numOutputCols  + blockIdx.y] <- aggregate 

[<AOTCompile>]
type MatrixRowOptimizer(target) as this =
    inherit GPUModule(target)

    let primitive = 
        fun (options:CompileOptions) ->
            cuda { return MultiChannelReducePrimitive(options.MinimalArch) }
        |> this.GPUDefineResource

    let checkDimension (a : Matrix<_>) (b : Matrix<_>) =
        if a.NumRows <> b.NumRows || a.NumCols <> b.NumCols then
            failwith "matrix dimensions must agree"

    let mutable output : Matrix<_> option = None

    let mallocOutput numRows numCols = 
        let newMatrix () = output <- Some (new Matrix<_>(this.GPUWorker, numRows, numCols))
        let emptyDisp () = ()
        let disposer = 
            match output with
            | None -> 
                newMatrix()
                emptyDisp
            | Some matrix ->
                if (matrix.NumRows <> numRows) || (matrix.NumCols <> numCols) then
                    newMatrix()
                    fun () -> matrix.Dispose()
                else 
                    emptyDisp

        output.Value, disposer

    member this.MinAndArgMin(values : Matrix<_>) =
        this.MinAndArgMin(values.DeviceData, values.NumRows, values.NumCols, values.NumCols, None)

    /// Computes the minimum and the index of the minimum along the rows of a matrix
    member this.MinAndArgMin(values : Matrix<_>, indices : Matrix<_>, numValid) =
        checkDimension values indices
        this.MinAndArgMin(values.DeviceData, values.NumRows, values.NumCols, numValid, Some indices.DeviceData)

    /// Computes the minimum and the index of the minimum along the rows of a matrix
    member this.MinAndArgMin (matrix : DeviceMemory<_>, numRows, numCols, numValid, indices : DeviceMemory<_> option) =
        this.OptAndArgOpt MinOrMax.Min matrix numRows numCols numValid indices

    member this.MaxAndArgMax(values : Matrix<_>) =
        this.MaxAndArgMax(values.DeviceData, values.NumRows, values.NumCols, values.NumCols, None)

    /// Computes the maximum and the index of the maximum along the rows of a matrix
    member this.MaxAndArgMax(values : Matrix<_>, indices : Matrix<_>, numValid) =
        checkDimension values indices
        this.MaxAndArgMax(values.DeviceData, values.NumRows, values.NumCols, numValid, Some indices.DeviceData)

    /// Computes the maximum and the index of the maximum along the rows of a matrix
    member this.MaxAndArgMax (matrix : DeviceMemory<_>, numRows, numCols, numValid, indices : DeviceMemory<_> option) =
        this.OptAndArgOpt MinOrMax.Max matrix numRows numCols numValid indices

    [<ReflectedDefinition;Kernel>]
    member private this.OptAndArgOptKernelSingle output numOutCols sign (matrix : deviceptr<float32>)  numCols numValid =
        primitive.Resource.OptAndArgOpt output numOutCols sign
            matrix numCols numValid (__null()) false

    [<ReflectedDefinition;Kernel>]
    member private this.OptAndArgOptKernelDouble output numOutCols sign (matrix : deviceptr<float>) numCols numValid =
        primitive.Resource.OptAndArgOpt output numOutCols sign
            matrix numCols numValid (__null()) false

    [<ReflectedDefinition;Kernel>]
    member private this.OptAndArgOptKernelSingleWithIdcs output numOutCols sign (matrix : deviceptr<float32>) 
        numCols numValid indices =
        primitive.Resource.OptAndArgOpt output numOutCols sign
            matrix numCols numValid indices true

    [<ReflectedDefinition;Kernel>]
    member private this.OptAndArgOptKernelDoubleWithIdcs output numOutCols sign (matrix : deviceptr<float>) 
        numCols numValid indices =
        primitive.Resource.OptAndArgOpt output numOutCols sign
            matrix numCols numValid indices true

    member private this.OptAndArgOpt (minOrMax : MinOrMax) (matrix : DeviceMemory<'T>) numRows numCols numValid
        (indices : DeviceMemory<_> option) =
        if (numCols < numValid) then invalidArg "numValid" "numValid must be less or equal then numCols"

        let numBlocks = divup numValid BLOCK_SIZE
        let gridSize = dim3(numRows, numBlocks)
        let lp = LaunchParam(gridSize, dim3 BLOCK_SIZE)
        let numOutputCols = (divup numCols BLOCK_SIZE)
        let reducedOut, disposer = mallocOutput numRows numOutputCols

        let outputPtr = reducedOut.DeviceData.Ptr
        let sign = minOrMax.Sign

        match  indices, box matrix with
        | Some idcs, (:? DeviceMemory<float32> as matrix) -> 
            this.GPULaunch <@ this.OptAndArgOptKernelSingleWithIdcs @> lp outputPtr numOutputCols sign
                matrix.Ptr numCols numValid idcs.Ptr
        | Some idcs, (:? DeviceMemory<float> as matrix) -> 
            this.GPULaunch <@ this.OptAndArgOptKernelDoubleWithIdcs @> lp outputPtr numOutputCols sign
                matrix.Ptr numCols numValid idcs.Ptr
        | None, (:? DeviceMemory<float32> as matrix) -> 
            this.GPULaunch <@ this.OptAndArgOptKernelSingle @> lp outputPtr numOutputCols sign
                matrix.Ptr numCols numValid 
        | None, (:? DeviceMemory<float> as matrix) -> 
            this.GPULaunch <@ this.OptAndArgOptKernelDouble @> lp outputPtr numOutputCols sign
                matrix.Ptr numCols numValid 
        | _, _ -> failwith "Unsupported value type"

        let valuesAndIndices = reducedOut.ToArray()
        disposer()
        let optima = Array.init numRows (fun _ -> minOrMax.DefaultValue)
        for rowIdx = 0 to numRows - 1 do
            let rowOffset = rowIdx * numOutputCols
            for colIdx = 0 to numBlocks - 1 do
                optima.[rowIdx] <- minOrMax.OptFun optima.[rowIdx] valuesAndIndices.[rowOffset + colIdx]
        optima |> Array.map (fun x -> (x.Value, x.Index))

    override this.Dispose(disposing) =
        if disposing then
            match output with
            | Some matrix -> matrix.Dispose()
            | None -> ()

        base.Dispose(disposing)