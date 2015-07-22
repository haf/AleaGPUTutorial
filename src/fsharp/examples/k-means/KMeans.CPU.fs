module KMeans.CPU

open System
open System.Drawing
open System.Diagnostics
open KMeans

(**
    This is a k-means implementation. The k-means algorithm is implemented according to the (description available at onmyphd.com)[http://www.onmyphd.com/?p=k-means.clustering].
    The general idea is, that one alternates a cluster assignment step with an adjust centroids step.

    For the cluster assignment step, we calculate the distance from every point to every centroid. And assign the point to the centroid with the smallest metric. In this particular
    implementation of the k-means, we use a cartesian distance as the metric. Or, more precisely, we use the square thereof, because the squaring is a monotonuous operation and
    as such the arg max that we are interested in, will give us the same results. But since we now do not need to calculate the square root, we are more computationally efficient.

    Once the cluster assignment is done, we calculate the mean of the points for each cluster and define this mean to be the new cluster centroid. This second operation is the
    adjust centroids operation.

    When those two steps are completed, we calculate the delta of how many points have changed their cluster assignments in this step. If this delta is small enough, we claim we are
    done. Otherwise, we make a new iteration, with an upper bound on how many iterations we allow at most. 

    Throughout the run of the algorithm, it is possible, that there are some clusters without any assigned points. A strategy needs to be made to make sure, how to deal with empty
    clusters. One possible way of dealing with this problem, is just deleting empty clusters, thus reducing the number of clusters after the algorithm terminates. Another possible
    way of dealing with this problem, is just re-initializing the centroid for removed clusters by means of random values. The last way of "dealing" with the problem is ignoring it
    altogether and just leave empty clusters at their previous position. If no point is assigned to any one of those vectors, then it behaves like deleting the cluster. Otherwise,
    the centroid is adjusted and can be used in the next iteration. This last approach yields a worse convergence, because the clusters a re-introduced, they have a tendency to 
    move around a lot and increase the delta.

    Typically, one would want to normalize the data to a range like, say, [0..1) With this approach, a random point is just simply a vector of IID [0..1) variables.

    As for the termination criteria; in principle, the k-means algorithm should terminate exactly in a finite amount of steps, but giving an upper bound on the number of iterations
    and lower bound on precision is a good idea in practice, because the convergence can take quite  a few steps.

    The initial centroids should be chosen randomly, because identical centroid would lead to identical distances of points to their respective centroids, thus yielding no new
    degrees of freedom. "Random" here can mean some points from the dataset sampled at random, or actually just random values of centroid vectors.

    This implementation of the k-means takes an array of n-dimensional feature vectors and an array of n-dimensional initial centroids. The respective lengths and number of
    dimensions are the way of passing in the input size.

    Also, this implementation is not complete in the sense, that it is not generic with respect to the types allowed and because it does not deal with the special cases, where some 
    clusters become empty.
*)
let kmeans data initialCentroids =
    let nPoints = Array.length data
    let nDim = Array.get data 0 |> Array.length
    let inline ``dist^2`` (x:_[]) (y:_[]) =
        let mutable res = 0.0f
        for i = 0 to nDim - 1 do
            let diff = x.[i]-y.[i]
            res <- res + diff * diff
        res

    /// assign each point to a centroid. Essentially an arg min
    let assignCluster centroids data =
        let centroids = centroids |> Array.mapi (fun idx v -> idx, v)
            
        data 
        |> Array.Parallel.map (fun x ->
            centroids
            |> Array.minBy (fun (_, y) -> ``dist^2`` x y)
            |> fst)

    /// adjust the centroids by data: assign each point to a centroid and then calculate the n-dimensional mean.
    let adjustCentroids data centroids =
        let clusterIndices = assignCluster centroids data
        let centroids =
            let nClusters = Array.length centroids
            let centroids = Array.copy centroids
            let pointsPerCentroid = Array.zeroCreate nClusters

            for i = 0 to data.Length - 1 do
                let clusterIdx = clusterIndices.[i]
                for j = 0 to nDim - 1 do
                    centroids.[clusterIdx].[j] <- centroids.[clusterIdx].[j] + data.[i].[j]
                pointsPerCentroid.[clusterIdx] <- pointsPerCentroid.[clusterIdx] + 1

            centroids
            |> Array.mapi (fun idx x ->
                Array.map (fun xi -> xi / float32 pointsPerCentroid.[idx]) x)

        centroids, clusterIndices        

    /// alternatively iterate assignment and cluster adjustment, until either the maximum number
    /// of iterations is reached, or the cluster assignment is stable enough (delta small enough)
    let rec loop data centroids maxIter (oldClusterIndices : _ []) =
        if maxIter <= 0 then 
            centroids
        else
            let centroids, clusterIndices = adjustCentroids data centroids
            
            let delta =
                let pct = 1.0 / float nPoints
                let mutable res = 0.0
                for i = 0 to oldClusterIndices.Length - 1 do
                    if oldClusterIndices.[i] <> clusterIndices.[i] then
                        res <- res + pct
                res

            if doPrint then printfn "iterations to go: %d\tdelta = %2.2f%%" maxIter (delta * 100.0)
            if delta <= threshold then
                centroids
            else
                loop data centroids (maxIter - 1) clusterIndices

    let watch = Stopwatch.StartNew()

    let centroids = loop data initialCentroids maxIters (Array.create nPoints -1)

    let ret = assignCluster centroids data, centroids

    watch.Stop()
    timing <- watch.Elapsed.TotalSeconds

    ret

let clusteredRGB useForgy origRGB k =
    let rng = Random(seed)

    let rawRGB = Array.init (Array2D.length1 origRGB * Array2D.length2 origRGB) (fun _ -> Array.zeroCreate 3)

    let linIdx i j = i * (Array2D.length2 origRGB) + j

    origRGB
    |> Array2D.iteri (fun i j (col:Color) ->
        let linIdx = linIdx i j
        rawRGB.[linIdx].[0] <- float32 col.R
        rawRGB.[linIdx].[1] <- float32 col.G
        rawRGB.[linIdx].[2] <- float32 col.B)

    let initialClusters =
        { 0..Array.length rawRGB - 1 }
        |> Seq.sortBy (fun _ -> rng.Next())
        |> Seq.take k
        |> Seq.map (fun idx -> [|rawRGB.[idx].[0]; rawRGB.[idx].[1]; rawRGB.[idx].[2]|])
        |> Array.ofSeq

    let initialClusters =
        if useForgy then
            Array.init initialClusters.Length (fun i -> Array.init 3 (fun j -> initialClusters.[i].[j]))
        else
            Array.init initialClusters.Length (fun i -> Array.init 3 (fun j -> float32 (rng.Next 255)))

    let (clusterAssignments, newClusters) = kmeans rawRGB initialClusters
    
    let newRawRGB = clusterAssignments |> Array.map (fun i -> newClusters.[i])

    let newRGB = Array2D.zeroCreate (Array2D.length1 origRGB) (Array2D.length2 origRGB)
    newRGB
    |> Array2D.iteri (fun i j _ -> 
        let linIdx = linIdx i j        
        let r, g, b = 
            match newRawRGB.[linIdx] with
            | [| r; g; b |] -> r, g, b
            | _ -> failwith "bug rgb"
        newRGB.[i, j] <- Color.FromArgb(int r, int g, int b))

    newRGB
