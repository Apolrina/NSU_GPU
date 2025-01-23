#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 16
#define BASE_TYPE double

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
    printf("Arows = %d\n", Arows);

    Acols = toMultiple(Acols, BLOCK_SIZE);
    printf("Acols = %d\n", Acols);

    Brows = toMultiple(Brows, BLOCK_SIZE);
    printf("Brows = %d\n", Brows);

    Bcols = toMultiple(Bcols, BLOCK_SIZE);
    printf("Bcols = %d\n", Bcols);

    size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
    size_t Bsize = Brows * Bcols * sizeof(BASE_TYPE);
    size_t Csize = Arows * Bcols * sizeof(BASE_TYPE);

    BASE_TYPE *h_A = (BASE_TYPE *)malloc(Asize);
    BASE_TYPE *h_B = (BASE_TYPE *)malloc(Bsize);
    BASE_TYPE *h_C = (BASE_TYPE *)malloc(Csize);


    for (int i = 0; i < Arows * Acols; ++i)
        h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
    for (int i = 0; i < Brows * Bcols; ++i)
        h_B[i] = rand() / (BASE_TYPE)RAND_MAX;

    BASE_TYPE *h_A_col = (BASE_TYPE *)malloc(Asize);
    BASE_TYPE *h_B_col = (BASE_TYPE *)malloc(Bsize);
    BASE_TYPE *h_C_col = (BASE_TYPE *)malloc(Csize);


    for (int i = 0; i < Arows; i++) {
        for (int j = 0; j < Acols; j++) {
            h_A_col[j * Arows + i] = h_A[i * Acols + j];
        }
    }

    
    for (int i = 0; i < Brows; i++) {
        for (int j = 0; j < Bcols; j++) {
            h_B_col[j * Brows + i] = h_B[i * Bcols + j];
        }
    }

    BASE_TYPE *d_A = NULL;
    cudaMalloc((void **)&d_A, Asize);

    BASE_TYPE *d_B = NULL;
    cudaMalloc((void **)&d_B, Bsize);

    BASE_TYPE *d_C = NULL;
    cudaMalloc((void **)&d_C, Csize);

    cudaMemcpy(d_A, h_A_col, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_col, Bsize, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const double alpha = 1.0;
    const double beta = 0.0;

    cudaEventRecord(start, 0);
    cublasDgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                Arows, Bcols, Acols,
                &alpha,
                d_A, Arows,
                d_B, Brows,
                &beta,
                d_C, Arows);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float KernelTime;
    cudaEventElapsedTime(&KernelTime, start, stop);
    printf("KernelTime: %.2f milliseconds\n", KernelTime);

    cudaMemcpy(h_C_col, d_C, Csize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < Arows; i++) {
        for (int j = 0; j < Bcols; j++) {
            h_C[i * Bcols + j] = h_C_col[j * Arows + i];
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_A_col);
    free(h_B_col);
    free(h_C_col);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cublasDestroy(handle);

    return 0;
}