#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void matmul(float *a, float *b, float *c, size_t N){
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = 0; i < N; ++i){
        c[row * N + col] += a[row * N + i] * b [i * N + col];
    }

}

// Generate a matrix filled with zeros
float * zeros(size_t N, size_t M) {
    float *tensor = (float *)malloc(N * M * sizeof(float));
    for (size_t i = 0; i < N * M; i++) {
        tensor[i] = 0;
    }
    return tensor;
}

// Generate a matrix filled with ones
float * ones(size_t N, size_t M) {
    float *tensor = (float *)malloc(N * M * sizeof(float));
    for (size_t i = 0; i < N * M; i++) {
        tensor[i] = 1;
    }
    return tensor;
}

// Generate a random matrix tensor
float * rand(size_t N, size_t M) {
    float *tensor = (float *)malloc(N * M * sizeof(float));
    for (size_t i = 0; i < N * M; i++) {
        tensor[i] = (float)rand() / RAND_MAX;
    }
    return tensor;
}


int main(int argc, char** argv) {
    float *a, *b, *c;
    size_t N = 2;

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    // generate the matrices
    a = rand(N, N);
    b = ones(N, N);
    c = zeros(N, N);

    float *dev_a, *dev_b, *dev_c;
    size_t size = N * N * sizeof(float);

    float start_time, end_time;
    float calc_start_time, calc_end_time;

    start_time = clock();
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, size, cudaMemcpyHostToDevice);

    // This is a three dimensional grid with one block
    // It is used to specify the dimensions of the grid
    // Here we create a 2D grid of blocks with 1 block in the x and y dimensions
    // The grid is used when launching the kernel to specify the number of blocks that should be executed
    dim3 dimGrid(1, 1);

    // Same principle as the grid, but for the block
    // Here we create a 2D block with N threads in the x and y dimensions
    // The block is used when launching the kernel to specify the number of threads that should be executed
    dim3 dimBlock(N, N);

    // Launch the function to perform the matrix multiplication on the GPU
    calc_start_time = clock();
    matmul<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, N);
    calc_end_time = clock();

    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    end_time = clock();

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    if (N <= 10){
        printf("Matrix A:\n");
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                printf("%f ", a[i * N + j]);
            }
            printf("\n");
        }
        printf("\n");

        printf("Matrix B:\n");
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                printf("%f ", b[i * N + j]);
            }
            printf("\n");
        }
        printf("\n");

        printf("Matrix C:\n");
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                printf("%f ", c[i * N + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("Overall GPU time: %f\n", (end_time - start_time) / CLOCKS_PER_SEC);
    printf("Execution only time: %f\n", (calc_end_time - calc_start_time) / CLOCKS_PER_SEC);

    free(a);
    free(b);
    free(c);

    return 0;
}