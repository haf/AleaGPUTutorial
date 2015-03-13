(*** hide ***)
module Tutorial.Fs.examples.heatPde.Solver

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound.LinAlg.TriDiag.CUDA
open NUnit.Framework
open FsUnit

(**
# Solve Two Dimensional Heat Equation 

We solve the two dimensional heat equation with an ADI scheme and the tridiagonal parallel cyclic reduction
algorithm. We need first a few helper functions.

Create a homogeneous grid between a and b of n points.
*)
let inline homogeneousGrid (real:RealTraits<'T>) (n:int) (a:'T) (b:'T) =
    let dx = (b - a) / (real.Of (n - 1))
    let x = Array.init n (fun i -> a + (real.Of i) * dx)
    x, dx

(**
Create an exponentially grid up to tstop of step size not larger than dt, with nc condensing points in the first interval.
*)
let inline exponentiallyCondensedGrid (real:RealTraits<'T>) (nc:int) (tstart:'T) (tstop:'T) (dt:'T) =
    if abs (tstop - tstart) < __epsilon() then
        [|tstart|]
    else
        let n = int(ceil (tstop-tstart)/dt)
        let dt' = (tstop-tstart) / (real.Of n)
        let dt'' = dt' / (real.Of (1 <<< (nc+1)))
        let tg1 = [0..nc] |> Seq.map (fun n -> tstart + (real.Of (1 <<< n))*dt'')
        let tg2 = [1..n] |> Seq.map (fun n -> tstart + (real.Of n)*dt')
        Seq.concat [Seq.singleton tstart; tg1; tg2] |> Seq.toArray

(**
Solves ny-2 systems of dimension nx in the x-coordinate direction.
*)
[<ReflectedDefinition>]
let inline xSweep (boundary:'T -> 'T -> 'T -> 'T) (sourceFunction:'T -> 'T -> 'T -> 'T) (nx:int) (ny:int) 
           (x:deviceptr<'T>) (y:deviceptr<'T>) (Cx:'T) (Cy:'T) (dt:'T) (t0:'T) (t1:'T) (u0:deviceptr<'T>) (u1:deviceptr<'T>) =
    let shared = __shared__.ExternArray<'T>() |> __array_to_ptr
    let h = shared
    let d = h + nx
    let l = d + nx
    let u = l + nx

    let mutable xi = 0G
    let mutable yj = 0G

    let mstride = ny

    let mutable j = blockIdx.x
    while j < ny do  
        yj <- y.[j]

        if j = 0 || j = ny-1 then

            let mutable i = threadIdx.x
            while i < nx do  
                xi <- x.[i]
                u1.[i*mstride+j] <- boundary t1 xi yj 
                i <- i + blockDim.x

            __syncthreads()

        else

            let mutable i = threadIdx.x
            while i < nx do
                xi <- x.[i]

                if i = 0 then
                    d.[i] <- 1G
                    u.[i] <- 0G
                    h.[i] <- boundary t1 xi yj
                else if i = nx-1 then
                    l.[i] <- 0G
                    d.[i] <- 1G
                    h.[i] <- boundary t1 xi yj
                else
                    l.[i] <- -Cx
                    d.[i] <- 2G + 2G*Cx
                    u.[i] <- -Cx
                    h.[i] <- 2G*u0.[i*mstride+j] +
                             Cy*(u0.[i*mstride+(j-1)] - 2G*u0.[i*mstride+j] + u0.[i*mstride+(j+1)]) +
                             dt*(sourceFunction t1 xi yj)

                i <- i + blockDim.x

            __syncthreads()

            blockSolveSmem nx l d u h

            i <- threadIdx.x
            while i < nx do  
                u1.[i*mstride+j] <- h.[i]
                i <- i + blockDim.x

            __syncthreads()

        j <- j + gridDim.x

(**
Solves nx-2 systems of dimension ny in the y-coordinate direction. 
*)
[<ReflectedDefinition>]
let inline ySweep (boundary:'T -> 'T -> 'T -> 'T) (sourceFunction:'T -> 'T -> 'T -> 'T) (nx:int) (ny:int) 
           (x:deviceptr<'T>) (y:deviceptr<'T>) (Cx:'T) (Cy:'T) (dt:'T) (t0:'T) (t1:'T) (u0:deviceptr<'T>) (u1:deviceptr<'T>) =
    let shared = __shared__.ExternArray<'T>() |> __array_to_ptr
    let h = shared
    let d = h + ny
    let l = d + ny
    let u = l + ny

    let mutable xi = 0G
    let mutable yj = 0G

    let mstride = ny

    let mutable i = blockIdx.x
    while i < nx do

        xi <- x.[i]

        if i = 0 || i = nx-1 then

            let mutable j = threadIdx.x
            while j < ny do
                yj <- y.[j]
                u1.[i*mstride+j] <- boundary t1 xi yj
                j <- j + blockDim.x

            __syncthreads()
        
        else

            let mutable j = threadIdx.x
            while j < ny do  
                yj <- y.[j]

                if j = 0 then
                    d.[j] <- 1G
                    u.[j] <- 0G
                    h.[j] <- boundary t1 xi yj
                else if j = ny-1 then
                    l.[j] <- 0G
                    d.[j] <- 1G
                    h.[j] <- boundary t1 xi yj
                else
                    l.[j] <- -Cy
                    d.[j] <- 2G + 2G*Cy
                    u.[j] <- -Cy
                    h.[j] <- 2G*u0.[i*mstride+j] +
                             Cx*(u0.[(i-1)*mstride+j] - 2G*u0.[i*mstride+j] + u0.[(i+1)*mstride+j]) +
                             dt*(sourceFunction t1 xi yj)

                j <- j + blockDim.x

            __syncthreads()

            blockSolveSmem ny l d u h

            j <- threadIdx.x
            while j < ny do 
                u1.[i*mstride+j] <- h.[j]
                j <- j + blockDim.x

            __syncthreads()

        i <- i + gridDim.x

(**
Solver interface for implementation with multiple kernel calls, two for each time step.
*)
type ISolver<'T when 'T:unmanaged> =
    abstract TimeGrid : 'T[]  
    abstract XGrid : 'T[]  
    abstract YGrid : 'T[]  
    abstract Launch : deviceptr<'T> -> deviceptr<'T> -> deviceptr<'T> -> deviceptr<'T> -> unit

(**
Builds a ISolver interface.
*)
let inline build (real:RealTraits<'T>)
          (initCondExpr:Expr<'T -> 'T -> 'T -> 'T>) 
          (boundaryExpr:Expr<'T -> 'T -> 'T -> 'T>) 
          (sourceExpr:Expr<'T -> 'T -> 'T -> 'T>) = cuda {

    let! initCondKernel =     
        <@ fun nx ny t (x:deviceptr<'T>) (y:deviceptr<'T>) (u:deviceptr<'T>) ->
            let initCond = %initCondExpr
            let i = blockIdx.x*blockDim.x + threadIdx.x
            let j = blockIdx.y*blockDim.y + threadIdx.y
            let mstride = ny
            if i < nx && j < ny then u.[i*mstride+j] <- initCond t x.[i] y.[j] @> |> Compiler.DefineKernel

    let! xSweepKernel =     
        <@ fun nx ny (x:deviceptr<'T>) (y:deviceptr<'T>) Cx Cy dt t0 t1 (u0:deviceptr<'T>) (u1:deviceptr<'T>) ->     
            let boundary = %boundaryExpr
            let source = %sourceExpr     
            xSweep boundary source nx ny x y Cx Cy dt t0 t1 u0 u1 @> |> Compiler.DefineKernel

    let! ySweepKernel =     
        <@ fun nx ny (x:deviceptr<'T>) (y:deviceptr<'T>) Cx Cy dt t0 t1 (u0:deviceptr<'T>) (u1:deviceptr<'T>) ->          
            let boundary = %boundaryExpr
            let source = % sourceExpr     
            ySweep boundary source nx ny x y Cx Cy dt t0 t1 u0 u1 @> |> Compiler.DefineKernel

    return fun (program:Program) ->
        let worker = program.Worker
        let initCondKernel = program.Apply(initCondKernel)
        let xSweepKernel = program.Apply(xSweepKernel)
        let ySweepKernel = program.Apply(ySweepKernel)

        fun (tstart:'T) (tstop:'T) (dt:'T) (nx:int) (xMin:'T) (xMax:'T) (ny:int) (yMin:'T) (yMax:'T) (diffusionCoeff:'T) ->
            let timeGrid = exponentiallyCondensedGrid real 5 tstart tstop dt
            let xgrid, dx = homogeneousGrid real nx xMin xMax
            let ygrid, dy = homogeneousGrid real ny yMin yMax
            let nu = nx * ny
            let lp0 = LaunchParam(dim3(divup nx 16, divup ny 16), dim3(16, 16))
            let lpx = LaunchParam(ny, nx, 4*nx*sizeof<'T>)
            let lpy = LaunchParam(nx, ny, 4*ny*sizeof<'T>)

            let launch (x:deviceptr<'T>) (y:deviceptr<'T>) (u0:deviceptr<'T>) (u1:deviceptr<'T>) =

                let initCondKernelFunc = initCondKernel.Launch lp0 
                let xSweepKernelFunc = xSweepKernel.Launch lpx
                let ySweepKernelFunc = ySweepKernel.Launch lpy

                initCondKernelFunc nx ny tstart x y u0

                if timeGrid.Length > 1 then
                    let step (t0, t1) =
                        let dt = t1 - t0
                        let Cx = diffusionCoeff * dt / (dx * dx)
                        let Cy = diffusionCoeff * dt / (dy * dy)
                        xSweepKernelFunc nx ny x y Cx Cy dt t0 (t0 + __half() * dt) u0 u1
                        ySweepKernelFunc nx ny x y Cx Cy dt (t0 + __half() * dt) t1 u1 u0

                    let timeIntervals = timeGrid |> Seq.pairwise |> Seq.toArray
                    timeIntervals |> Array.iter step

            { new ISolver<'T> with
                member this.TimeGrid = timeGrid
                member this.XGrid = xgrid
                member this.YGrid = ygrid
                member this.Launch x y u0 u1 = launch x y u0 u1  
            } 
    }

(**
Solve heat equation

$$$
\begin{equation}
    u_t = u_{xx} + u_{yy} + f(t, x, y)
\end{equation}

with boundary condition $b(t, x, y)$ and source function $f(t, x, y)$.

The x coordinate is mapped to the rows and the y coordinate to the columns
of the solution matrix. The solution matrix is stored in row major format.

This solver calls per time step two kernels, one for the x-sweep and one for for the y-sweep.
*)
let inline solver (real:RealTraits<'T>) init boundary source = cuda {
    let! solver = build real init boundary source
    
    return Entry(fun (program:Program) ->
        let worker = program.Worker
        let solver = solver program
        fun (tstart:'T) (tstop:'T) (dt:'T) (nx:int) (xMin:'T) (xMax:'T) (ny:int) (yMin:'T) (yMax:'T) (diffusionCoeff:'T) ->
            let solver = solver tstart tstop dt nx xMin xMax ny yMin yMax diffusionCoeff
            let t = solver.TimeGrid 
            let x = solver.XGrid
            let y = solver.YGrid
            let nxy = x.Length * y.Length
            
            use x' = program.Worker.Malloc(x)
            use y' = program.Worker.Malloc(y)
            use u0 = program.Worker.Malloc<'T>(nxy)
            use u1 = program.Worker.Malloc<'T>(nxy)

            solver.Launch x'.Ptr y'.Ptr u0.Ptr u1.Ptr

            x, y, u0.Gather()) 
    }

(**
Solver interface for implementation with a single kernel.
*)
type ISolverFused<'T when 'T:unmanaged> =
    abstract TimeGrid : 'T[]  
    abstract XGrid : 'T[]  
    abstract YGrid : 'T[]  
    abstract Launch : deviceptr<'T> -> deviceptr<'T> -> deviceptr<'T> -> deviceptr<'T> -> deviceptr<'T> -> unit

let inline buildFused (real:RealTraits<'T>)
               (initCondExpr:Expr<'T -> 'T -> 'T -> 'T>) 
               (boundaryExpr:Expr<'T -> 'T -> 'T -> 'T>) 
               (sourceExpr:Expr<'T -> 'T -> 'T -> 'T>) = cuda {

    let! initCondKernel =     
        <@ fun nx ny t (x:deviceptr<'T>) (y:deviceptr<'T>) (u:deviceptr<'T>) ->
            let initCond = %initCondExpr
            let i = blockIdx.x*blockDim.x + threadIdx.x
            let j = blockIdx.y*blockDim.y + threadIdx.y
            let mstride = ny
            if i < nx && j < ny then u.[i*mstride+j] <- initCond t x.[i] y.[j] @> |> Compiler.DefineKernel

    let! stepKernel =     
        <@ fun k nt nx ny (t:deviceptr<'T>) (x:deviceptr<'T>) dx (y:deviceptr<'T>) dy (u0:deviceptr<'T>) (u1:deviceptr<'T>) ->     
            let boundary = %boundaryExpr
            let source = %sourceExpr  
            
            let mutable i = 0
            while i < nt - 1 do
                let t0 = t.[i]
                let t1 = t.[i + 1]
                let dt = t1 - t0
                let Cx = k * dt / (dx * dx)
                let Cy = k * dt / (dy * dy)
              
                xSweep boundary source nx ny x y Cx Cy dt t0 (t0 + __half() * dt) u0 u1 

                __syncthreads()
                
                ySweep boundary source nx ny x y Cx Cy dt (t0 + __half() * dt) t1 u0 u1

                __syncthreads()

                i <- i + 1

        @> |> Compiler.DefineKernel

    return fun (program:Program) ->
        let worker = program.Worker
        let initCondKernel = program.Apply(initCondKernel)
        let stepKernel = program.Apply(stepKernel)

        fun (tstart:'T) (tstop:'T) (dt:'T) (nx:int) (xMin:'T) (xMax:'T) (ny:int) (yMin:'T) (yMax:'T) (diffusionCoeff:'T) ->
            let timeGrid = exponentiallyCondensedGrid real 5 tstart tstop dt
            let xgrid, dx = homogeneousGrid real nx xMin xMax
            let ygrid, dy = homogeneousGrid real ny yMin yMax
            let nu = nx * ny
            let lp0 = LaunchParam(dim3(divup nx 16, divup ny 16), dim3(16, 16))
            let n = max nx ny
            let lp1 = LaunchParam(n, n, 4*n*sizeof<'T>)

            let launch (t:deviceptr<'T>) (x:deviceptr<'T>) (y:deviceptr<'T>) (u0:deviceptr<'T>) (u1:deviceptr<'T>) =

                let initCondKernelFunc = initCondKernel.Launch lp0 
                initCondKernelFunc nx ny tstart x y u0

                if timeGrid.Length > 1 then
                    let stepKernelFunc = stepKernel.Launch lp1
                    stepKernelFunc diffusionCoeff timeGrid.Length nx ny t x dx y dy u0 u1

            { new ISolverFused<'T> with
                member this.TimeGrid = timeGrid
                member this.XGrid = xgrid
                member this.YGrid = ygrid
                member this.Launch t x y u0 u1 = launch t x y u0 u1  
            } 
    }

(**
Solve heat equation

$$$
\begin{equation}
    u_t = u_{xx} + u_{yy} + f(t, x, y),
\end{equation}

with boundary condition $b(t, x, y)$ and source function $f(t, x, y)$.

The x coordinate is mapped to the rows and the y coordinate to the columns
of the solution matrix. The solution matrix is stored in row major format.

This solver calls a single kernel, which does the time stepping. It is adviced to 
use approximately the same number of grid points in x and y direction.
*)
let inline solverFused (real:RealTraits<'T>) init boundary source = cuda {
    let! solver = buildFused real init boundary source
    
    return Entry(fun (program:Program) ->
        let worker = program.Worker
        let solver = solver program
        fun (tstart:'T) (tstop:'T) (dt:'T) (nx:int) (xMin:'T) (xMax:'T) (ny:int) (yMin:'T) (yMax:'T) k ->
            let solver = solver tstart tstop dt nx xMin xMax ny yMin yMax k
            let t = solver.TimeGrid 
            let x = solver.XGrid
            let y = solver.YGrid
            let nxy = x.Length * y.Length
            
            use t' = program.Worker.Malloc(t)
            use x' = program.Worker.Malloc(x)
            use y' = program.Worker.Malloc(y)
            use u0 = program.Worker.Malloc<'T>(nxy)
            use u1 = program.Worker.Malloc<'T>(nxy)

            solver.Launch t'.Ptr x'.Ptr y'.Ptr u0.Ptr u1.Ptr

            x, y, u0.Gather()) 
    }

(**
Solve heat equation in the unit box.
*)
let inline unitBoxSolver (real:RealTraits<'T>) tstop nx ny init boundary source =
    let diffusionCoeff = 1G
    let tstart = 0G
    let xMin = 0G
    let xMax = 1G
    let yMin = 0G
    let yMax = 1G
    let dt = real.Of 0.01

    let solver = solver real init boundary source |> Compiler.load Worker.Default       
    solver.Run tstart tstop dt nx xMin xMax ny yMin yMax diffusionCoeff  

(*** hide ***)
let inline maxErr (b:'T[]) (b':'T[]) = Array.map2 (fun bi bi' -> abs (bi - bi')) b b' |> Array.max

(**
Solve the heat equation for initial conditions

$$$
\begin{equation}
    exp(-t) sin(\pi x) cos(\pi x),
\end{equation}

and source function

$$$
\begin{equation}
    exp(-t) sin(\pi x) cos(\pi x) (2 \pi^2 - 1),
\end{equation}

which allow a closed form solution 

$$$
\begin{equation}
    exp(-t) sin(\pi x) cos(\pi x),
\end{equation}

and test the exactness of the result. 
*)
[<Test>]
let heatPdeTest () =
    let real = RealTraits.Real64
    
    let uexact t x y = exp(-t) * sin(__pi()*x) * cos(__pi()*y)
    let init = <@ fun t x y -> exp(-t) * sin(__pi()*x) * cos(__pi()*y) @>
    let boundary = <@ fun t x y -> exp(-t) * sin(__pi()*x) * cos(__pi()*y) @>
    let source = <@ fun t x y -> exp(-t) * sin(__pi()*x) * cos(__pi()*y) * (2.0*__pi()*__pi() - 1.0) @>

    let nx = 128
    let ny = 128   
    let tstop = 1.0   
    let x, y, u = unitBoxSolver real tstop nx ny init boundary source 
    
    let ue = Array.zeroCreate (x.Length*y.Length)
    let mstride = ny
    for i = 0 to x.Length-1 do
        for j = 0 to y.Length-1 do
            ue.[i*mstride+j] <- uexact tstop x.[i] y.[j]
     
    printfn "max error = %A" (maxErr u ue)   
        
    u |> should (equalWithin 1e-3) ue

