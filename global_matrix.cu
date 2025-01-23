#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16
#define BASE_TYPE double

__global__ void matrixMult(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int Acols, int Bcols)
{
    int i0 = Acols * (blockDim.y * blockIdx.y + threadIdx.y);
    int j0 = blockDim.x * blockIdx.x + threadIdx.x;
    BASE_TYPE sum = 0;
    for (int k = 0; k < Acols; k++)
        sum += A[i0 + k] * B[k * Bcols + j0];
    int ind = Bcols * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    C[ind] = sum;
}

int toMultiple(int a, int b)
{
    int mod = a % b;
    if (mod != 0)
    {
        mod = b - mod;
        return a + mod;
    }
    return a;
}

int main()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int Arows = 1000;
    int Acols = 2000;
    int Brows = Acols;
    int Bcols = 1500;

    Arows = toMultiple(Arows, BLOCK_SIZE);
    Acols = toMultiple(Acols, BLOCK_SIZE);
    Brows = toMultiple(Brows, BLOCK_SIZE);
    Bcols = toMultiple(Bcols, BLOCK_SIZE);

    size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
    size_t Bsize = Brows * Bcols * sizeof(BASE_TYPE);
    size_t Csize = Arows * Bcols * sizeof(BASE_TYPE);

    BASE_TYPE *h_A = (BASE_TYPE *)malloc(Asize);
    BASE_TYPE *h_B = (BASE_TYPE *)malloc(Bsize);

    for (int i = 0; i < Arows * Acols; ++i)
        h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
    for (int i = 0; i < Brows * Bcols; ++i)
        h_B[i] = rand() / (BASE_TYPE)RAND_MAX;

    BASE_TYPE *d_A = NULL;
    cudaMalloc((void **)&d_A, Asize);

    BASE_TYPE *d_B = NULL;
    cudaMalloc((void **)&d_B, Bsize);

    BASE_TYPE *d_C = NULL;
    cudaMalloc((void **)&d_C, Csize);

    cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice);

    // параметры запуска ядра
    dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid = dim3(Bcols / BLOCK_SIZE, Arows / BLOCK_SIZE);

    cudaEventRecord(start, 0);
    matrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, Acols, Bcols);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float kernelTime;
    cudaEventElapsedTime(&kernelTime, start, stop);
    printf("KernelTime: %.2f milliseconds\n", kernelTime);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  
    return 0;
}