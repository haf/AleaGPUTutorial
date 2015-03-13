(**
Functionality common to the different implementations.
*)
[<AutoOpen>]
module Tutorial.Fs.examples.NBody.Common

open Alea.CUDA
open FsUnit

(**
Abstract simulator type.
*)
type ISimulator =
    abstract Description : string
    abstract Integrate : newPos:deviceptr<float4> -> oldPos:deviceptr<float4> -> vel:deviceptr<float4> -> numBodies:int -> deltaTime:float32 -> softeningSquared:float32 -> damping:float32 -> unit

(**
Abstract simulation testing type.
*)
type ISimulatorTester =
    abstract Description : string
    abstract Integrate : pos:float4[] -> vel:float4[] -> numBodies:int -> deltaTime:float32 -> softeningSquared:float32 -> damping:float32 -> steps:int -> unit

let float4Zero = float4(0.0f, 0.0f, 0.0f, 0.0f)

(**
Initialize particles randomly in a cube with random velocities but zero momentum.
*)
let initializeBodies1 clusterScale velocityScale numBodies = 
    let rng = System.Random(42)
    let random a b = rng.NextDouble()*a - b |> float32
    let scale = clusterScale * max 1.0f (float32 numBodies/1024.0f)
    let vscale = velocityScale*scale

    let randomPos _ = scale * random 1.0 0.5
    let randomVel _ = vscale * random 1.0 0.5

    let pos = Array.init numBodies (fun _ -> float4(randomPos(), randomPos(), randomPos() + 50.0f, 1.0f))
    let vel = Array.init numBodies (fun _ -> float4(randomVel(), randomVel(), randomVel(), 1.0f))
    let totalMomentum = vel |> Array.fold (fun (acc:float4) e -> 
        float4(acc.x + e.x*e.w, acc.y + e.y*e.w, acc.z + e.z*e.w, acc.w + e.w)) float4Zero

    pos, vel |> Array.map (fun e ->
        float4(e.x - totalMomentum.x/totalMomentum.w/e.w,
               e.y - totalMomentum.y/totalMomentum.w/e.w,
               e.z - totalMomentum.z/totalMomentum.w/e.w,
               e.w)) // initialize with total momentum = 0

(**
Initialize particles randomly in two cubes with angular momentum, s.t. some galaxy similar patterns emerge. Total momentum is set to zero.
*)
let initializeBodies2 clusterScale velocityScale numBodies = 
    let rng = System.Random(42)
    let random a b = rng.NextDouble()*a - b |> float32
    let scale = clusterScale * max 1.0f (float32 numBodies/1024.0f)
    let vscale = velocityScale*scale

    let randomPos _ = scale * random 1.0 0.5
    let randomVel _ = vscale * random 1.0 0.5

    let pos = Array.init numBodies (fun i -> 
        if i < numBodies/2 then float4(randomPos() + 0.5f*scale, randomPos(), randomPos() + 50.0f, 1.0f)
        else float4(randomPos() - 0.5f*scale, randomPos(), randomPos() + 50.0f, 1.0f) )
    let vel = Array.init numBodies (fun i -> 
        if i < numBodies/2 then float4(randomVel(), randomVel() + 0.01f*vscale*pos.[i].x*pos.[i].x, randomVel(), 1.0f)
        else float4(randomVel(), randomVel() - 0.01f*vscale*pos.[i].x*pos.[i].x, randomVel(), 1.0f) )
    let totalMomentum = vel |> Array.fold (fun (acc:float4) e -> 
        float4(acc.x + e.x*e.w, acc.y + e.y*e.w, acc.z + e.z*e.w, acc.w + e.w)) float4Zero

    pos, vel |> Array.map (fun e -> 
        float4(e.x - totalMomentum.x/totalMomentum.w/e.w, 
               e.y - totalMomentum.y/totalMomentum.w/e.w, 
               e.z - totalMomentum.z/totalMomentum.w/e.w, 
               e.w)) // initialize with total momentum = 0

(**
Initialize particles randomly in two cubes with random velocities angular momentum, s.t. some galaxy similar patterns emerge.
Give different particles random mass. Do not set a seed for the random number generator, hence simulations will have different behavour each call. Total momentum is set to zero.
*)
let initializeBodies3 clusterScale velocityScale numBodies = 
    let rng = System.Random()
    let random a b = rng.NextDouble()*a - b |> float32
    let scale = clusterScale * max 1.0f (float32 numBodies/1024.0f)
    let vscale = velocityScale*scale

    let randomPos _ = scale * random 1.0 0.5
    let randomVel _ = vscale * random 1.0 0.5
    let randomMass _ =
        let maxv = 1.3
        let minv = 0.7
        (rng.NextDouble() * (maxv - minv) + minv) |> float32

    let pos = Array.init numBodies (fun i -> 
        if i < numBodies/2 then float4(randomPos() + 0.5f*scale, randomPos(), randomPos() + 50.0f, randomMass())
        else float4(randomPos() - 0.5f*scale, randomPos(), randomPos() + 50.0f, randomMass()) )
    let vel = Array.init numBodies (fun i -> 
        if i < numBodies/2 then float4(randomVel(), randomVel() + 0.01f*vscale*pos.[i].x*pos.[i].x, randomVel(), 1.0f)
        else float4(randomVel(), randomVel() - 0.01f*vscale*pos.[i].x*pos.[i].x, randomVel(), 1.0f) )
    let totalMomentum = vel |> Array.fold (fun (acc:float4) e -> 
        float4(acc.x + e.x*e.w, acc.y + e.y*e.w, acc.z + e.z*e.w, acc.w + e.w)) float4Zero

    pos, vel |> Array.map (fun e -> 
        float4(e.w*e.x - totalMomentum.x/totalMomentum.w/e.w,
               e.w*e.y - totalMomentum.y/totalMomentum.w/e.w, 
               e.w*e.z - totalMomentum.z/totalMomentum.w/e.w, 
               e.w)) // initialize with total momentum = 0

(**
Testing if two simulations behave the same up to some tolerance.
*)
let test (expectedSimulator:ISimulatorTester) (actualSimulator:ISimulatorTester) numBodies =
    let clusterScale = 1.0f
    let velocityScale = 1.0f
    let deltaTime = 0.001f
    let softeningSquared = 0.00125f
    let damping = 0.9995f
    let steps = 5

    printfn "Testing %A against %A with %d bodies..." actualSimulator.Description expectedSimulator.Description numBodies

    let verify (expected:float4[]) (actual:float4[]) (tol:float) =
        (expected, actual) ||> Array.iter2 (fun expected actual ->
            actual.x |> should (equalWithin tol) expected.x
            actual.y |> should (equalWithin tol) expected.y
            actual.z |> should (equalWithin tol) expected.z
            actual.w |> should (equalWithin tol) expected.w )

    let test (tol:float) (initializeBodies:int -> float4[] * float4[]) =
        let expectedPos, expectedVel = initializeBodies numBodies
        let actualPos = Array.copy expectedPos
        let actualVel = Array.copy expectedVel
        for i = 1 to steps do
            expectedSimulator.Integrate expectedPos expectedVel numBodies deltaTime softeningSquared damping 1
            actualSimulator.Integrate actualPos actualVel numBodies deltaTime softeningSquared damping 1
            verify expectedPos actualPos tol
            verify expectedVel actualVel tol
    
    initializeBodies1 clusterScale velocityScale |> test 1e-5
    initializeBodies3 clusterScale velocityScale |> test 1e-4

(**
Measure the performence of a simulation method.
*)
let performance (simulator:ISimulatorTester) numBodies =
    let clusterScale = 1.0f
    let velocityScale = 1.0f
    let deltaTime = 0.001f
    let softeningSquared = 0.00125f
    let damping = 0.9995f
    let steps = 10

    printfn "Perfomancing %A with %d bodies..." simulator.Description numBodies

    let test (initializeBodies:int -> float4[] * float4[]) =
        let pos, vel = initializeBodies numBodies
        simulator.Integrate pos vel numBodies deltaTime softeningSquared damping steps
    
    initializeBodies1 clusterScale velocityScale |> test

(**
Calculating the acceleration $a_{ij}$ particle at position `bj` exerts on particle with position `bi`, and adds it to the acceleration `ai` (resulting in total acceleration on particle with position `bi`, after calling this method for all particles `bj`) which will be returned.
It is used for the CPU as well as for both GPU implementations.

The acceleration is calculated as:

$$$
\begin{equation}
    a_{ij} = \frac{f_{ij}}{m_i} = \frac{r_{ij} m_j}{\sqrt{\left||r_{ij}\right||^2 + \varepsilon^2}^3},
\end{equation}

note, we use artificial units where $G$ is set to 1.

CUDA datatypes `float3` (acceleration) and `float4` (position, 4-th element is the particles mass) are used. 
The function `__nv_rsqrtf` calculates $f(x) = \frac{1}{\sqrt{x}}$ faster and with less registers than calling the two functions seperately.
*)
[<ReflectedDefinition>]
let bodyBodyInteraction softeningSquared (ai:float3) (bi:float4) (bj:float4) =
    // r_ij  [3 FLOPS]
    let r = float3(bj.x - bi.x, bj.y - bi.y, bj.z - bi.z)

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    let distSqr = r.x*r.x + r.y*r.y + r.z*r.z + softeningSquared

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    let invDist = __nv_rsqrtf distSqr
    let invDistCube =  invDist * invDist * invDist
    
    // s = m_j * invDistCube [1 FLOP]
    let s = bj.w * invDistCube

    // a_i =  a_i + s * r_ij [6 FLOPS]
    float3(ai.x + r.x*s, ai.y + r.y*s, ai.z + r.z*s)

