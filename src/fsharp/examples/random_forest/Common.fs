module Tutorial.Fs.examples.RandomForest.Common

module MinMax = 

    /// Returns value and position of minimum in a sequence. 
    /// In case of multiple minima it returns the first occuring.
    let inline minAndArgMin (source: seq<'T>) : ('T * int) =
        use e = source.GetEnumerator() 
        if not (e.MoveNext()) then invalidArg "source" "empty sequence"
        let mutable i = 0

        let mutable accv = e.Current
        let mutable acci = i

        while e.MoveNext() do
            i <- i + 1
            let curr = e.Current
            if curr < accv then
                accv <- curr
                acci <- i

        (accv, acci)

    /// Returns value and position of maximum in a sequence. 
    /// In case of multiple maxima it returns the first occuring.
    let inline maxAndArgMax (source: seq<'T>) : ('T * int) =
        use e = source.GetEnumerator() 
        if not (e.MoveNext()) then invalidArg "source" "empty sequence"
        let mutable i = 0

        let mutable accv = e.Current
        let mutable acci = i

        while e.MoveNext() do
            i <- i + 1
            let curr = e.Current
            if curr > accv then
                accv <- curr
                acci <- i

        (accv, acci)
    
module Seq =

    let toChunks n (s:seq<'t>) = seq {
        let pos = ref 0
        let buffer = Array.zeroCreate<'t> n
 
        for x in s do
            buffer.[!pos] <- x
            if !pos = n - 1 then
                yield buffer |> Array.copy
                pos := 0
            else
                incr pos
 
        if !pos > 0 then
            yield Array.sub buffer 0 !pos
    }

    let bufferedAsync (bufSize : int) (s:seq<'T>) = 

        let queue = new System.Collections.Concurrent.BlockingCollection<_>(bufSize)

        async {
            s |> Seq.iter (fun batch -> queue.Add (Some batch))
            queue.Add None 
        } |> Async.Start

        let rec consumer () = 
            match queue.Take() with
            | None -> 
                queue.Dispose()
                Seq.empty
            | Some x -> seq { yield x; yield! consumer ()}

        consumer()

module Collections =
    
    type BlockingObjectPool<'T> (objects : 'T[]) =
        
        let size = objects.Length
        let pool = new System.Collections.Concurrent.BlockingCollection<'T>(size)

        do for obj in objects do pool.Add(obj)

        member this.Acquire() = 
            let obj = pool.Take()
            obj

        member this.Release(obj) = 

            pool.Add(obj)

        member this.Size = size
