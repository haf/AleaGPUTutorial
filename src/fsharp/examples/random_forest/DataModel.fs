namespace Tutorial.Fs.examples.RandomForest.DataModel

type Label = int
type FeatureValue = float

/// Array of `FeatureValue`s for a given sample (but might have many features).
/// e.g.: | height; age; ... size | for one sample.
type Sample = FeatureValue[]

/// Array of `FeatureValue`s for a given feature but many Samples.
/// e.g: | height1; height2; ...; heightm | for m samples.
type Domain = FeatureValue[]

/// Array of `Domain`s, 
/// e.g:
/// |                                 |
/// | |       |  |    |       |     | |
/// | |height1|  |age1|       |size1| |
/// | |height2|  |age2|       |size2| |
/// | | ...   |  | ...|       | ... | |
/// | |heightm|  |agem|       |sizem| |
/// | |       |; |    |; ...; |     | |
/// |                                 |
///
/// for m samples with n features each.
type Domains = Domain[]
type Labels = Label[]
/// Sample with attatched Label,
/// e.g.: (| height; age; ... size |, 1) a sample with label 1.
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