(*** hide ***)
module Tutorial.Fs.examples.RandomForest.Array

open Tutorial.Fs.examples.RandomForest.DataModel

(**
# Array

Functionality to find non-zero indices, and store them in an array.
*)
let findNextNonZeroIdx (v : _[]) startIdx =
    let mutable i = startIdx
    while i < v.Length && v.[i] = 0 do
        i <- i + 1
    if (i >= v.Length) then None
    else Some i

(**
Returns indices of non-zero indices in an array.
*)
let findNonZeroIdcs count (mask : _[]) =
    let dst = Array.zeroCreate count
    let mutable maskIdx = 0
    for i = 0 to count - 1 do
        let next = findNextNonZeroIdx mask maskIdx
        match next with
        | Some idx ->
            maskIdx <- idx + 1
            dst.[i] <- idx
        | None -> failwith "insufficient number non zero elements"
    dst

(**
Samples are sorted according to their value. We still want to get the initial weights, i.e. what this method provides.
*)
let projectIdcs (indices : _[]) (weights : _[]) = 
    Array.init indices.Length (fun i -> weights.[indices.[i]])

(**
Apply `projectIdcs` to all features.
*)
let expandWeights indicesPerFeature weights =
    indicesPerFeature |> Array.Parallel.map (fun indices -> projectIdcs indices weights)

let permByIdcs (indices : _[]) (weights : _[]) = 
    weights |> Array.permute (fun i -> indices.[i])

let findNonZeroWeights (weightsPerFeature : Weights[]) =
    let countNonZero =
        weightsPerFeature.[0]
        |> Seq.filter (fun w -> w <> 0)
        |> Seq.length

    let nonZeroIdcsPerFeature = weightsPerFeature |> Array.Parallel.map (findNonZeroIdcs countNonZero)
    nonZeroIdcsPerFeature

(**
Shuffle the elements in an array.
The function `rnd` is a random-number provider, a function taking an int `l` and returning a random number between 0 and `l`.
As a default you may use `DataModel.getRngFunction` for `rnd`.
*)
let shuffle (rnd : int -> int) arr = arr |> Seq.sortBy (fun _ -> rnd(System.Int32.MaxValue))

(**
Returns array of length `k` with unique random integer numbers from 0 to `n` - 1.
The function `rnd` is a random-number provider, a function taking an int `l` and returning a random number between 0 and `l`.
As a default you may use `DataModel.getRngFunction` for `rnd`.
*)
let randomSubIndices rnd n k =
    seq { 0..n - 1 } |> shuffle rnd |> Seq.take k |> Seq.toArray

(**
Selects randomly `k` elements (without replacement) of the array `x`.
The function `rnd` is a random-number provider, a function taking an int `l` and returning a random number between 0 and `l`.
As a default you may use `DataModel.getRngFunction` for `rnd`.
*)
let randomlySplitUpArray rnd k x =
    let shuffledSequence =
        x |> shuffle rnd |> Seq.toArray
    shuffledSequence.[0..k - 1], shuffledSequence.[k..]

(**
Returns value and position of the minimum in a sequence.
In case of multiple minima it returns the last occurring.
This function has been written in non-functional style in
order to get higher performance.
*)
let inline minAndArgMin (source : seq<'T>) : 'T*int =
    use e = source.GetEnumerator()
    if not (e.MoveNext()) then invalidArg "source" "empty sequence"
    let mutable i = 0
    let mutable accv = e.Current
    let mutable acci = i
    while e.MoveNext() do
        i <- i + 1
        let curr = e.Current
        if curr <= accv then
            accv <- curr
            acci <- i
    (accv, acci)

(** 
Returns value and position of the maximum in a sequence.
In case of multiple maxima it returns the last occurring.
This function has been written in non-functional style in
order to get higher performance.
*)
let inline maxAndArgMax (source : seq<'T>) : 'T * int =
    use e = source.GetEnumerator()
    if not (e.MoveNext()) then invalidArg "source" "empty sequence"
    let mutable i = 0
    let mutable accv = e.Current
    let mutable acci = i
    while e.MoveNext() do
        i <- i + 1
        let curr = e.Current
        if curr >= accv then
            accv <- curr
            acci <- i
    (accv, acci)