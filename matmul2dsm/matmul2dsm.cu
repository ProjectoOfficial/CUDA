/*
    This program is provided as is without any guarantees or warranty.
    By using this program the user accepts the full responsibility for any
    and all damages that may occur. The author is not responsible for any
    consequences of the use of this program.

    * This program performs matrix multiplication using CUDA and shared memory.
    * The matrices are generated using three different methods: zeros, ones, random.
    * The program uses the following functions:
        * getSharedMemory: prints the amount of shared memory per block
        * matmul_sm: the CUDA kernel which performs the matrix multiplication using shared memory
        * matrix: generates a matrix of size n x m of three types: zeros, ones, random
        * print_matrix: prints a matrix of size n x m
        * cpu_matmul: performs the matrix multiplication on the CPU
        * equals: checks if two matrices are equal
        * parse_args: parses the command line arguments
    
    * The program takes three command line arguments:
        * n: the number of rows of the first matrix
        * m: the number of columns of the first matrix and the number of rows of the second matrix
        * p: the number of columns of the second matrix
    * If the number of command line arguments is less than 3, the program uses the default values of 3 for n, m, and p.
    * If the number of command line arguments is 1, the program uses the value of the first argument for n, m, and p.
    
    * The program can be compiled using the following command:
        * nvcc matmul2d.cu -o matmul
    * to run the program, use the following command:
        * ./matmul <n> <m> <p>

    @Author: Daniel Rossi
    @Date: 2023-03-12
    @License: MIT
    @Version: 1.0
*/

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


/*
 * CUDA kernel to perform matrix multiplication using shared memory
 * The shared memory is a memory space that is shared between all threads in a block.
 * It is fast because it is located on-chip, but it is limited in size.
*/
__global__ void matmul_sm(float *a, float *b, float *c, size_t n, size_t m, size_t p) {
    // Calculate the global row and column indices
    size_t row = blockIdx.y * blockDim.y + threadIdx.y; // this is the row index for the current thread
    size_t col = blockIdx.x * blockDim.x + threadIdx.x; // this is the column index for the current thread

    // Allocate shared memory for the tile of matrix A and B
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];
    
    float sum = 0.0f;

    // Iterate over the tiles of matrix A and B
    for (size_t tileIdx = 0; tileIdx < (m + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tileIdx) {
        /*
        * TILEs:
        *  - If we consider a 3x3 matrix A, i can scroll the first row if globalRow = 0 and globalCol = 0, 1, 2;
        *  - to scroll the second row, I need globalRow to be 3 and globalCol to be 0, 1, 2; and so on.
        *  - Thus, the maximum value for globalRow is 3x3 = 9. This is why globalRow cannot be larger than n * m.
        *  - But, since we operate within blocks, the row index for globalRow is given by row, which is the block index 
        *       multiplied by the block size, plus the thread index.
        * 
        *   - For globalCol is a little more complicated. The maximum value for globalCol is 3 in this example because we consider
        *     A to be a 3x3 matrix. Consider now to have BLOCK_SIZE = 2. This means that we have 2x2 tiles, or better, a 4 elements tiles.
        *     If we iterate from 0 to (3 + 2 - 1) / 2 = 2, the index of the first element would be [0 * 2 + 0] = 0, the second would be 
        *     [0 * 2 + 1] = 1, 3rd = [1 * 2 + 0] = 2 and 4th = [1 * 2 + 1] = 3. Thus we are able to scroll the columns of A.
        *     (BLOCK_SIZE of 2 means that each block has 2 threads)
        *     
        *   - For B, the same logic applies, but the maximum value for globalRow is m, and for globalCol is p.
        *     If we consider a 3x3 matrix B, the maximum value for globalRow is 3x3 = 9. We can consider to scroll the rows of B
        *     by incrementing the globalRow index from 0 to 3 and multiplying it by B's height (3). 
        *     
        *   - Why can we do this with tiles of size 4 when the matrix is 3x3? 
        *     Since, after the assingment of the elements to the tiles, we synchronize the threads, we can safely ignore the
        *     elements that are not part of the matrix. In particular, we are performing the overall matmul slicing in pieces 
        *     the matrices. After filling the tiles, we have all we need to calculate the result for the current tile.
        */
        
        size_t globalRow = row * m;
        size_t globalCol = tileIdx * BLOCK_SIZE + threadIdx.x;

        if (globalRow < n * m && globalCol < m) {
            tileA[threadIdx.y][threadIdx.x] = a[globalRow + globalCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        globalRow = tileIdx * BLOCK_SIZE + threadIdx.y;
        globalCol = col;

        if (globalRow < m && globalCol < p) {
            tileB[threadIdx.y][threadIdx.x] = b[(globalRow) * p + globalCol];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize threads to ensure all elements are loaded into shared memory
        __syncthreads();

        // Perform the matrix multiplication for the current tile
        for (size_t k = 0; k < BLOCK_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Synchronize threads to ensure all elements are used in the matrix multiplication
        __syncthreads();
    }

    // Write the result to the output matrix
    if (row < n && col < p) {
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


bool equals(float *gpu, float *cpu, size_t n, size_t m) {
    for (size_t i = 0; i < n * m; i++) {
        if (abs(gpu[i] - cpu[i]) > 1e-3) {
            printf("a[%lu] = %f, b[%lu] = %f\n", i, gpu[i], i, cpu[i]);
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

    srand(41); // set the seed for random number generation

    // generate the matrices
    a = matrix(n, m, ONES);
    b = matrix(m, p, ONES);
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
    matmul_sm<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, n, m, p);
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