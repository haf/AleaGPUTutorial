module Tutorial.Fs.examples.RandomForest.Common

module MinMax = 

    /// Returns value and position of minimum in a sequence. 
    /// In case of multiple minima it returns the first occuring.
    let inline minAndArgMin (source: seq<'T>) : ('T * int) =
        source |> Seq.mapi (fun i e -> e,i)
               |> Seq.minBy (fun e -> fst e)

    /// Returns value and position of maximum in a sequence. 
    /// In case of multiple maxima it returns the first occuring.
    let inline maxAndArgMax (source: seq<'T>) : ('T * int) =
        source |> Seq.mapi (fun i e -> e,i)
               |> Seq.maxBy (fun e -> fst e)
