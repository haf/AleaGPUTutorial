(*** hide ***)
module Tutorial.Fs.examples.deviceQuery

open Alea.CUDA

(**
# Device Query
*)

(**
Query various device properties.
*)
let query (dev:Device) =
    printfn "\nDevice %d: \"%s\"" dev.ID dev.Name

    printfn "  CUDA Capability Major/Minor version number:    %d.%d" dev.Arch.Major dev.Arch.Minor

    printfn "  Total amount of global memory:                 %.0f MBytes (%u bytes)" (float(uint64(dev.TotalMemory)) / 1048576.0) (uint64(dev.TotalMemory))

    let multiProcessorCount = dev.Attributes.MULTIPROCESSOR_COUNT
    let cudaCoresPerSM = dev.Cores / multiProcessorCount
    printfn "  (%2d) Multiprocessors x (%3d) CUDA Cores/MP:    %d CUDA Cores" multiProcessorCount cudaCoresPerSM dev.Cores

    let clockRate = dev.Attributes.CLOCK_RATE |> float
    printfn "  GPU Clock rate:                                %.0f MHz (%0.2f GHz)" (clockRate * 1e-3) (clockRate * 1e-6)

    let memoryClock = dev.Attributes.MEMORY_CLOCK_RATE |> float
    printfn "  Memory Clock rate:                             %.0f MHz" (memoryClock * 1e-3)

    let memBusWidth = dev.Attributes.GLOBAL_MEMORY_BUS_WIDTH
    printfn "  Memory Bus Width:                              %d-bit" memBusWidth

    let L2CacheSize = dev.Attributes.L2_CACHE_SIZE
    printfn "  L2 Cache Size:                                 %d bytes" L2CacheSize

    printfn "  Max Texture Dimension Sizes:                   1D=(%d) 2D=(%d,%d) 3D=(%d,%d,%d)"
        dev.Attributes.MAXIMUM_TEXTURE1D_WIDTH
        dev.Attributes.MAXIMUM_TEXTURE2D_WIDTH
        dev.Attributes.MAXIMUM_TEXTURE2D_HEIGHT
        dev.Attributes.MAXIMUM_TEXTURE3D_WIDTH
        dev.Attributes.MAXIMUM_TEXTURE3D_HEIGHT
        dev.Attributes.MAXIMUM_TEXTURE3D_DEPTH

    printfn "  Max Layered Texture Size (dim) x layers:       1D=(%d) x %d, 2D=(%d,%d) x %d"
        dev.Attributes.MAXIMUM_TEXTURE2D_LAYERED_WIDTH
        dev.Attributes.MAXIMUM_TEXTURE2D_LAYERED_LAYERS
        dev.Attributes.MAXIMUM_TEXTURE2D_LAYERED_WIDTH
        dev.Attributes.MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
        dev.Attributes.MAXIMUM_TEXTURE2D_LAYERED_LAYERS

    let totalConstantMemory = dev.Attributes.TOTAL_CONSTANT_MEMORY
    printfn "  Total amount of constant memory:               %d bytes" totalConstantMemory

    let sharedMemPerBlock = dev.Attributes.MAX_SHARED_MEMORY_PER_BLOCK
    printfn "  Total amount of shared memory per block:       %d bytes" sharedMemPerBlock

    let regsPerBlock = dev.Attributes.MAX_REGISTERS_PER_BLOCK
    printfn "  Total number of registers available per block: %d" regsPerBlock

    let warpSize = dev.Attributes.WARP_SIZE
    printfn "  Warp size:                                     %d" warpSize

    let maxThreadsPerMultiProcessor = dev.Attributes.MAX_THREADS_PER_MULTIPROCESSOR
    printfn "  Maximum number of threads per multiprocessor:  %d" maxThreadsPerMultiProcessor

    let maxThreadsPerBlock = dev.Attributes.MAX_THREADS_PER_BLOCK
    printfn "  Maximum number of threads per blocks:          %d" maxThreadsPerBlock

    printfn "  Maximum sizes of each dimension of a block:    %d x %d x %d"
        dev.Attributes.MAX_BLOCK_DIM_X
        dev.Attributes.MAX_BLOCK_DIM_Y
        dev.Attributes.MAX_BLOCK_DIM_Z

    printfn "  Maximum sizes of each dimension of a grid:     %d x %d x %d"
        dev.Attributes.MAX_GRID_DIM_X
        dev.Attributes.MAX_GRID_DIM_Y
        dev.Attributes.MAX_GRID_DIM_Z

    let textureAlign = dev.Attributes.TEXTURE_ALIGNMENT
    printfn "  Texture alignment:                             %d bytes" textureAlign

    let memPitch = dev.Attributes.MAX_PITCH
    printfn "  Maximum memory pitch:                          %d bytes" memPitch

    let gpuOverlap = dev.Attributes.GPU_OVERLAP
    let gpuOverlap = if gpuOverlap <> 0 then "Yes" else "No"
    let asyncEngineCount = dev.Attributes.ASYNC_ENGINE_COUNT
    printfn "  Concurrent copy and kernel execution:          %s with %d copy engine(s)" gpuOverlap asyncEngineCount

    let kernelExecTimeoutEnabled = dev.Attributes.KERNEL_EXEC_TIMEOUT
    let kernelExecTimeoutEnabled = if kernelExecTimeoutEnabled <> 0 then "Yes" else "No"
    printfn "  Run time limit on kernels                      %s" kernelExecTimeoutEnabled

    let integrated = dev.Attributes.INTEGRATED
    let integrated = if integrated <> 0 then "Yes" else "No"
    printfn "  Integrated GPU shareing Host Memory:           %s" integrated

    let canMapHostMemory = dev.Attributes.CAN_MAP_HOST_MEMORY
    let canMapHostMemory = if canMapHostMemory <> 0 then "Yes" else "No"
    printfn "  Support host page-locked memory mapping:       %s" canMapHostMemory

    let concurrentKernels = dev.Attributes.CONCURRENT_KERNELS
    let concurrentKernels = if concurrentKernels <> 0 then "Yes" else "No"
    printfn "  Concurrent kernel execution:                   %s" concurrentKernels

    let surfaceAlignment = dev.Attributes.SURFACE_ALIGNMENT
    let surfaceAlignment = if surfaceAlignment <> 0 then "Yes" else "No"
    printfn "  Alignment requirement for Surfaces:            %s" surfaceAlignment

    let eccEnabled = dev.Attributes.ECC_ENABLED
    let eccEnabled = if eccEnabled <> 0 then "Enabled" else "Disabled"
    printfn "  Device has ECC support:                        %s" eccEnabled

    let tccDriver = dev.Attributes.TCC_DRIVER
    let tccDriver = if tccDriver <> 0 then "TCC (Tesla Compute Cluster Driver)" else "WDDM (Windows Display Driver Model)"
    printfn "  CUDA Device Driver Mode (TCC or WDDM):         %s" tccDriver

    let unifiedAddressing = dev.Attributes.UNIFIED_ADDRESSING
    let unifiedAddressing = if unifiedAddressing <> 0 then "Yes" else "No"
    printfn "  Device supports Unified Addressing (UVA):      %s" unifiedAddressing

    printfn "  Device PCI Bus ID / PCI location ID:           %d / %d"
        dev.Attributes.PCI_BUS_ID
        dev.Attributes.PCI_DEVICE_ID

    let sComputeMode =
        [|
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)"
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)"
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)"
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)"
            "Unknown"
        |]
    let computeMode = dev.Attributes.COMPUTE_MODE
    printfn "  Compute Mode:"
    printfn "     < %s >" sComputeMode.[computeMode]

(**
List CUDA driver version and query device properties for all devices of Fermi architecture or newer.
*)
let deviceQuery () = 
    let driverVersion = CUDADriver.Version
    printfn "CUDA Driver Version: %d.%d" (driverVersion/1000) ((driverVersion%100)/10)

    Device.Devices |> Array.iter query
