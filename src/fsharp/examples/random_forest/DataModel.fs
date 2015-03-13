namespace Tutorial.Fs.examples.RandomForest.DataModel

type Label = int
type FeatureValue = float
type Sample = FeatureValue[]
type Domain = FeatureValue[]
type Domains = Domain[]
type Labels = Label[]
type LabeledSample = Sample * Label
type LabelHistogram = int[] * int // counts per class and sum
type Weights = int[]
type Indices = int[]

type Split = 
    {
        Feature : int
        Threshold : FeatureValue
    }

    static member Create feature threshold =
        {
            Feature = feature
            Threshold = threshold
        }
        
    override this.ToString () = sprintf "{F:%d, T:%.3f}" this.Feature this.Threshold

type Tree = 
    | Leaf of Label
    | Node of low : Tree * split : Split * high : Tree

    override this.ToString () =
        match this with
        | Leaf label -> sprintf "%s" (label.ToString())
        | Node (low, split, high) -> sprintf "(%s %s %s)" (low.ToString()) (split.ToString()) (high.ToString())

type Model =
    | RandomForest of trees : Tree[] * numClasses : int