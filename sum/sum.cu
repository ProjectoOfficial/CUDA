/*
    This program is provided as is without any guarantees or warranty.
    By using this program the user accepts the full responsibility for any
    and all damages that may occur. The author is not responsible for any
    consequences of the use of this program.

    * This program adds two arrays of floats using CUDA.
    * 
    * The program first allocates memory for the arrays on the host and device.
    * It then initializes the host arrays with random values.
    * The host arrays are then copied to the device.
    * The add() kernel is then launched on the GPU.
    * The result is then copied back to the host.
    * The program then verifies the result.
    * Finally, the program frees the memory on the device and host.
    
    * The program takes one command line argument, N, which is the size of the arrays.
    * If no argument is provided, the default value of N is 1.

    * The program can be compiled using the following command:
        * nvcc sum.cu -o sum
    * to run the program, use the following command:
        * ./sum <N>

    @Author: Daniel Rossi
    @Date: 2023-03-08
    @License: MIT
    @Version: 1.0
*/

#include <stdio.h>
#include <cuda_runtime.h>

void setup() {
    // Set the random seed
    srand(time(NULL));
    int device = 0; // Default device id (change if you have more than one GPU)

    // Set the device
    cudaSetDevice(device);
}

// Kernel function to add two arrays
__global__ void add(float *a, float *b, float *c, int n) {
    // Get the index of the current element
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if the index is within the array bounds
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

void print_cuda_error(cudaError_t err) {
    if (err != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

int main(int argc, char **argv) {
    setup();
    int N = 1;

    // Parse command line arguments
    if (argc > 1){
        N = atoi(argv[1]);
    }

    printf("N = %d\n", N);

    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    // Allocate memory on the host
    a = (float *) malloc(N * sizeof(float));
    b = (float *) malloc(N * sizeof(float));
    c = (float *) malloc(N * sizeof(float));

    // Allocate memory on the device
    cudaError_t err;
    err = cudaMalloc(&d_a, N * sizeof(float));
    print_cuda_error(err);

    err = cudaMalloc(&d_b, N * sizeof(float));
    print_cuda_error(err);

    err =cudaMalloc(&d_c, N * sizeof(float));
    print_cuda_error(err);

    // Initialize host values
    for (int i = 0; i < N; ++i){
        // Generate random values between 0 and 1 
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }   

    // Copy inputs to device
    err = cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    print_cuda_error(err);
    
    err = cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    print_cuda_error(err);

    // Launch add() kernel on GPU
    add<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    err = cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    print_cuda_error(err);

    // Verify the result
    for (int i = 0; i < N; ++i){
        if (c[i] != (a[i] + b[i])){
            printf("Error: %f + %f != %f\n", a[i], b[i], c[i]);
            break;
        }
    }

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free memory on host
    free(a);
    free(b);
    free(c);

    printf("Done\n");

    return 0;
}