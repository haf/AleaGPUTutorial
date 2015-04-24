(*** hide ***)
module Tutorial.Fs.examples.RandomForest.DataModel

type Label = int

type FeatureValue = float

(**
Array of `FeatureValue`s for a given sample,
e.g.:

    [| sepalLength; sepalWidth; petalLength; petalWidth |]

*)
type Sample = FeatureValue[]

(**
Array of `FeatureValue`s for a given feature but many samples,
e.g:

    [| sepalLength_1; sepalLength_2; ...; sepalLength_n |]

for n samples.
*)
type FeatureArray = FeatureValue[]

(**
Array of `FeatureArray`s,
e.g:

    |                                                                  |
    | |             |  |            |  |             |  |            | |
    | |sepalLength_1|  |sepalWidth_1|  |petalLength_1|  |sepalWidth_1| |
    | |sepalLength_2|  |sepalWidth_2|  |petalLength_2|  |sepalWidth_2| |
    | | ...         |  | ...        |  | ...         |  | ...        | |
    | |sepalLength_m|  |sepalWidth_m|  |petalLength_m|  |sepalWidth_m| |
    | |             |; |            |; |             |; |            | |
    |                                                                  |

m samples with 4 features each.
*)
type FeatureArrays = FeatureArray[]

type Labels = Label[]

(**
Sample with attached Label,
e.g:

    ([| sepalLength; sepalWidth; petalLength; petalWidth |], 1)

a sample with label 1.
*)
type LabeledSample = Sample * Label

type LabelHistogram = int[] * int // counts per class and sum

type Weights = int[]

type Indices = int[]

type Split =
    { Feature : int
      Threshold : FeatureValue }

    static member Create feature threshold =
        { Feature = feature
          Threshold = threshold }

    override this.ToString() = sprintf "{F:%d, T:%.3f}" this.Feature this.Threshold

type Tree =
    | Leaf of Label
    | Node of low : Tree * split : Split * high : Tree
    override this.ToString() =
        match this with
        | Leaf label -> sprintf "%s" (label.ToString())
        | Node(low, split, high) -> sprintf "(%s %s %s)" (low.ToString()) (split.ToString()) (high.ToString())

type Model =
    | RandomForest of trees : Tree[] * numClasses : int

(** 
We abstract from the random number generator, demanding only a function taking an integer `l` and returning a 
random integer between 0 and `l`. The method `getRngFunction` is a factory providing such a function using `System.Random`.
*)
let getRngFunction seed = 
    let rng = System.Random(seed)
    (fun l -> rng.Next(l))