module Tutorial.Fs.examples.RandomForest.Array

open Tutorial.Fs.examples.RandomForest.DataModel

let findNextNonZeroIdx (v : _ []) startIdx =
    let mutable i = startIdx
    while i < v.Length && v.[i] = 0 do
        i <- i + 1
    if (i >= v.Length) then None
    else Some i

let findNonZeroIdcs count (mask : _ []) =
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

let projectIdcs (indices : _ []) (weights : _ []) = Array.init indices.Length (fun i -> weights.[indices.[i]])
let permByIdcs (indices : _ []) (weights : _ []) = weights |> Array.permute (fun i -> indices.[i])
let expandWeights indicesPerFeature weights =
    indicesPerFeature |> Array.Parallel.map (fun indices -> projectIdcs indices weights)

let findNonZeroWeights (weightsPerFeature : Weights []) =
    let countNonZero =
        weightsPerFeature.[0]
        |> Seq.filter (fun w -> w <> 0)
        |> Seq.length

    let nonZeroIdcsPerFeature = weightsPerFeature |> Array.Parallel.map (findNonZeroIdcs countNonZero)
    nonZeroIdcsPerFeature

let shuffle (rnd : System.Random) arr = arr |> Seq.sortBy (fun _ -> rnd.NextDouble())

/// Returns array of length `k` with uniqe random integer numbers from 0 to `n` - 1.
let randomSubIndices (rnd : System.Random) n k =
    seq { 0..n - 1 } |> shuffle rnd |> Seq.take k |> Seq.toArray

let randomlySplitUpArray x (rnd : System.Random) k =
    let shuffledSequence =
        x |> shuffle rnd |> Seq.toArray
    shuffledSequence.[0..k - 1], shuffledSequence.[k..]

/// Returns value and position of minimum in a sequence.
/// In case of multiple minima it returns the last occuring.
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

/// Returns value and position of maximum in a sequence.
/// In case of multiple maxima it returns the last occuring.
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