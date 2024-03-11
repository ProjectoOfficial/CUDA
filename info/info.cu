/*
    This program is provided as is without any guarantees or warranty.
    By using this program the user accepts the full responsibility for any
    and all damages that may occur. The author is not responsible for any
    consequences of the use of this program.

    * This program prints information about the GPU device.
    * The program uses the CUDA runtime API to query the device properties.

    * The program takes an optional argument, which is the device id.
    * If no argument is provided, the program will use the default device (device 0).

    * The program can be compiled using the following command:
        * nvcc info.cu -o info
    * to run the program, use the following command:
        * ./info <device_id>

    @Author: Daniel Rossi
    @Date: 2023-03-11
    @License: MIT
    @Version: 1.0
    @
*/

#include <stdio.h>
#include <cuda_runtime.h>

void info(int device){
    printf("CUDA version: %d.%d\n", CUDART_VERSION / 1000, (CUDART_VERSION % 100) / 10);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Using device %d: %s\n", device, prop.name);
    printf("GPU compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Total global memory: %lu bytes\n", prop.totalGlobalMem);
    printf("Total constant memory: %lu bytes\n", prop.totalConstMem);
    printf("Shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Clock rate: %d kHz\n", prop.clockRate);
    printf("Memory clock rate: %d kHz\n", prop.memoryClockRate);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("L2 cache size: %d bytes\n", prop.l2CacheSize);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Registers per multiprocessor: %d\n", prop.regsPerMultiprocessor);
    printf("Device has ECC support: %d\n", prop.ECCEnabled);
    printf("Device has unified addressing: %d\n", prop.unifiedAddressing);
    printf("Device has host memory mapping: %d\n", prop.canMapHostMemory);
    printf("Device has error correction: %d\n", prop.ECCEnabled);
    printf("Device has async engine count: %d\n", prop.asyncEngineCount);
    printf("Device has concurrent kernels: %d\n", prop.concurrentKernels);
    printf("Device has PCI bus ID: %d\n", prop.pciBusID);
    printf("Device has PCI device ID: %d\n", prop.pciDeviceID);
    printf("Device has PCI domain ID: %d\n", prop.pciDomainID);
    printf("Device has tcc driver: %d\n", prop.tccDriver);
    printf("Device has memory clock rate: %d kHz\n", prop.memoryClockRate);
    printf("Device has memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("Device has memory bandwidth: %f GB/s\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("Device has L2 cache size: %d bytes\n", prop.l2CacheSize);
    printf("Device has max memory pitch: %lu bytes\n", prop.memPitch);
    printf("Device has texture alignment: %lu bytes\n", prop.textureAlignment);
    printf("Device has texture pitch alignment: %lu bytes\n", prop.texturePitchAlignment);
    printf("Device has GPU overlap: %d\n", prop.deviceOverlap);
    printf("Device has kernel execution timeout: %d\n", prop.kernelExecTimeoutEnabled);
    printf("Device has integrated GPU: %d\n", prop.integrated);
    printf("Device has can map host memory: %d\n", prop.canMapHostMemory);
    printf("Device has compute mode: %d\n", prop.computeMode);
    printf("Device has max texture 1D size: %d\n", prop.maxTexture1D);
    printf("Device has max texture 1D linear size: %d\n", prop.maxTexture1DLinear);
    printf("Device has max texture 1D mipmapped size: %d\n", prop.maxTexture1DMipmap);
    printf("Device has max texture 2D size: (%d, %d)\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
    printf("Device has max texture 2D linear size: %d\n", prop.maxTexture2DLinear);
    printf("Device has max texture 2D mipmapped size: (%d, %d)\n", prop.maxTexture2DMipmap[0], prop.maxTexture2DMipmap[1]);
    printf("Device has max texture 3D size: (%d, %d, %d)\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
    printf("Device has max texture 3D size: %d\n", prop.maxTexture3D);
}

int main(int argc, char **argv) {
    int device = 0; // Default device id (change if you have more than one GPU)
    if (argc > 1) {
        device = atoi(argv[1]);
    }

    // Set the device
    cudaSetDevice(device);

    info(device);
}