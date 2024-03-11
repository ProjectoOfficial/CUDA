#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

// enums used to generate matrices of different types
enum {
    ZEROS = 0,
    ONES = 1,
    RAND = 2,
};

void getSharedMemory() {
    cudaDeviceProp prop;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    printf("Shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
}

__global__ void matmul(float *a, float *b, float *c, size_t n, size_t m, size_t p) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y; // obviously rows are along y axis
    size_t col = blockIdx.x * blockDim.x + threadIdx.x; // and columns are along x axis (think about a spreadsheet!)

    float sum = 0;
    if (row < n && col < p){ // we need to check if the current thread is within the matrix boundaries
        for (size_t i = 0; i < m; ++i) {
            sum += a[row * m + i] * b[i * p + col]; // this is the dot product of the row-th row of a and the col-th column of b
        }
        c[row * p + col] = sum; 
    }
}


// generates a matrix of size n x m of three types: zeros, ones, random
float *matrix(size_t n, size_t m, int type) {
    float *mat = (float *)malloc(n * m * sizeof(float));
    for (size_t i = 0; i < n * m; i++) {
        if (type == ZEROS) {
            mat[i] = 0;
        } else if (type == ONES) {
            mat[i] = 1;
        } else if (type == RAND) {
            mat[i] = (float)rand() / RAND_MAX;
        }
    }
    return mat;
}


void print_matrix(char name, float *matrix, size_t n, size_t m){
    printf("Matrix %c:\n", name);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                printf("%f ", matrix[i * m + j]);
            }
            printf("\n");
        }
        printf("\n");
}


void cpu_matmul(float *a, float *b, float *c_cpu, size_t n, size_t m, size_t p){
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            for (size_t k = 0; k < m; ++k) {
                c_cpu[i * p + j] += a[i * m + k] * b[k * p + j];
            }
        }
    }
}


bool equals(float *a, float *b, size_t n, size_t m) {
    for (size_t i = 0; i < n * m; i++) {
        if (abs(a[i] - b[i]) > 1e-3) {
            return false;
        }
    }
    return true;
}


void parse_args(int argc, char **argv, size_t *n, size_t *m, size_t *p) {
    if (argc > 1 && argc <= 2) {
        *n = atoi(argv[1]);
        *m = atoi(argv[1]);
        *p = atoi(argv[1]);
    } else if (argc > 3) {
        *n = atoi(argv[1]);
        *m = atoi(argv[2]);
        *p = atoi(argv[3]);
    } else {
        *n = 3;
        *m = 3;
        *p = 3;
    }
}

int main(int argc, char** argv) {
    float *a, *b, *c;
    size_t n, m, p;
    parse_args(argc, argv, &n, &m, &p);

    getSharedMemory();
    srand(41); // set the seed for random number generation

    // generate the matrices
    a = matrix(n, m, RAND);
    b = matrix(m, p, RAND);
    c = matrix(n, p, ZEROS);

    float *dev_a, *dev_b, *dev_c;

    float start_time, end_time;

    start_time = clock();
    // Allocate memory on the device
    cudaMalloc((void **)&dev_a, n * m * sizeof(float));
    cudaMalloc((void **)&dev_b, m * p * sizeof(float));
    cudaMalloc((void **)&dev_c, n * p * sizeof(float));

    // Copy the input matrices from the host to the device
    cudaMemcpy(dev_a, a, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, m * p * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate the number of block needed along rows and columns to cover each matrix dimension 
    // using blocks of size BLOCK_SIZE
    // BLOCK_SIZE - 1 guarantees that the last block will be filled with the remaining elements
    size_t gridRows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t gridCols = (p + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 dimGrid(gridCols, gridRows); // this is a struct which defines the size of the blocks grid used to perform parallel computations
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // this is a struct which represents the size of a CUDA block within the CUDA kernel
    
    cudaEvent_t start, stop; // these are events used to measure the time of the kernel execution
    float gpuTime = 0.0f;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    matmul<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, n, m, p);
    cudaEventRecord(stop, 0);
    
    cudaEventSynchronize(stop); // Wait for the stop event to complete
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaMemcpy(c, dev_c, n * p * sizeof(float), cudaMemcpyDeviceToHost);
    end_time = clock();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    if (n * p <= 25){
        print_matrix('a', a, n, m);
        print_matrix('b', b, m, p);
        print_matrix('c', c, n, p);
    }

    printf("Overall GPU time: %f s \n", (end_time - start_time) / CLOCKS_PER_SEC);
    printf("GPU time: %f s\n", gpuTime / 1000);

    float *c_cpu = matrix(n, p, ZEROS);
    start_time = clock();
    cpu_matmul(a, b, c_cpu, n, m, p);
    end_time = clock();

    printf("Overall CPU time: %f s\n", (end_time - start_time) / CLOCKS_PER_SEC);
    printf("\n");

    if (n * p <= 25){
        print_matrix('x', c_cpu, n, p);
    }
    printf("Matrices are %s\n", equals(c, c_cpu, n, p) ? "equal" : "different");

    free(a);
    free(b);
    free(c);
    free(c_cpu);

    return 0;
}