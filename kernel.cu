
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include <curand_kernel.h>
#include <curand.h>
#include <chrono>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    std::cout << "\nError at "<<__FILE__<<":"<<__LINE__<<": "<<x<<"\n"; \
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    std::cout << "\nError at "<<__FILE__<<":"<<__LINE__<<": "<<x<<"\n"; \
    return EXIT_FAILURE;}} while(0)

#define EXP_TIMES 100
#define DATA_BLOCK 1024

using namespace std;

__global__ void moro_inv(float* data, int cnt, float mean, float std) {
    // Each thread will handle one transfer
    size_t Idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (Idx >= cnt) return;

    data[Idx] = normcdfinvf(data[Idx]) * std + mean;
}

__global__ void moro_inv_v2(float* data, int cnt, float mean, float std) {
    // Each thread will handle DATA_BLOCK transfer
    size_t Idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (Idx >= cnt) return;

    float* data_p = &data[Idx * DATA_BLOCK];

    for (int i = 0; i < DATA_BLOCK; i++) {
        data_p[i] = normcdfinvf(data_p[i]) * std + mean;
    }
    
}

int main()
{
    int m = 1 << 12;
    int n = 1 << 12;    // Test with 16M data, generate (4k * 4k) sobol sequence
    cout << "Demension M: " << m << endl;
    cout << "Count N: " << n << endl;

    // Allocate memory
    size_t bytes = m * n * sizeof(float);
    // host
    float *sepr_host = (float*)malloc(bytes);    // Seperate sobol and moro
    float *comb_host = (float*)malloc(bytes);   // Combine sobol and moro
    // device
    float *sepr_dev = NULL;
    float *comb_dev = NULL;
    CUDA_CALL(cudaMalloc((void**)&sepr_dev, bytes));
    CUDA_CALL(cudaMalloc((void**)&comb_dev, bytes));

    // Generator
    int offset = 1024;

    // Single run for correction check and warm-up
    // Seperate 
    // Set up block manually
    int blocksize = 1024;
    dim3 block(blocksize, 1);
    dim3 grid_v1((m * n - 1) / block.x + 1, 1);
    dim3 grid_v2((m * n - 1) / (block.x * DATA_BLOCK) + 1, 1);
    printf("grid_v1 %d grid_v2 %d block %d \n", grid_v1.x, grid_v2.x, block.x);

    chrono::steady_clock::time_point start, end;
    start = chrono::steady_clock::now();
    curandGenerator_t gen_sepr;
    CURAND_CALL(curandCreateGenerator(&gen_sepr, CURAND_RNG_QUASI_SOBOL32));
    CURAND_CALL(curandSetGeneratorOffset(gen_sepr, offset));
    CURAND_CALL(curandSetQuasiRandomGeneratorDimensions(gen_sepr, m));
    CURAND_CALL(curandGenerateUniform(gen_sepr, sepr_dev, n * m));
    //moro_inv << < grid_v1, block >> > (sepr_dev, m * n, 0, 0.5);
    moro_inv_v2 << < grid_v2, block >> > (sepr_dev, m * n, 0, 0.5);
    cudaDeviceSynchronize();
    end = chrono::steady_clock::now();
    chrono::duration<double, std::milli> elapsed = end - start;

    //cout << "Seperated version EXE TIME: " << elapsed.count() << "ms" << endl;

    CUDA_CALL(cudaMemcpy(sepr_host, sepr_dev, bytes, cudaMemcpyDeviceToHost));
    CUDA_CALL(curandDestroyGenerator(gen_sepr));


    // Combined
    start = chrono::steady_clock::now();
    curandGenerator_t gen_comb;
    CURAND_CALL(curandCreateGenerator(&gen_comb, CURAND_RNG_QUASI_SOBOL32));
    CURAND_CALL(curandSetGeneratorOffset(gen_comb, offset));
    CURAND_CALL(curandSetQuasiRandomGeneratorDimensions(gen_comb, m));
    CURAND_CALL(curandGenerateNormal(gen_comb, comb_dev, n * m, 0, 0.5));
    end = chrono::steady_clock::now();
    elapsed = end - start;

    //cout << "Combined version EXE TIME: " << elapsed.count() << "ms" << endl;


    CUDA_CALL(cudaMemcpy(comb_host, comb_dev, bytes, cudaMemcpyDeviceToHost));
    CUDA_CALL(curandDestroyGenerator(gen_comb));

    // Correction check
    bool pass = true;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (abs(comb_host[i * n + j] - sepr_host[i * n + j]) > 1e-6) {
                pass = false;
                cout << i << " "<< j << endl;
                cout << comb_host[i * n + j] << " "<< sepr_host[i * n + j] << endl;
                break;
            }
        }
        if (!pass) break;
    }
    if (!pass) {
        cout << "CORRECTION CHECK: FAILED" << endl;
    }
    else {
        cout << "CORRECTION CHECK: PASS" << endl;

        // loop to get average time
        double sepr_time1 = 0;
        double sepr_time2 = 0;
        double comb_time = 0;

        for (int i = 0; i < EXP_TIMES; i++) {
            
            // Seperated_v1
            // ------------------------
            start = chrono::steady_clock::now();

            CURAND_CALL(curandCreateGenerator(&gen_sepr, CURAND_RNG_QUASI_SOBOL32));
            CURAND_CALL(curandSetGeneratorOffset(gen_sepr, offset));
            CURAND_CALL(curandSetQuasiRandomGeneratorDimensions(gen_sepr, m));
            CURAND_CALL(curandGenerateUniform(gen_sepr, sepr_dev, n * m));
            moro_inv << < grid_v1, block >> > (sepr_dev, m * n, 0, 0.5);
            cudaDeviceSynchronize();
            CUDA_CALL(curandDestroyGenerator(gen_sepr));

            end = chrono::steady_clock::now();
            elapsed = end - start;
            sepr_time1 += elapsed.count();
            // ------------------------

            // Seperated_v2
            // ------------------------
            start = chrono::steady_clock::now();

            CURAND_CALL(curandCreateGenerator(&gen_sepr, CURAND_RNG_QUASI_SOBOL32));
            CURAND_CALL(curandSetGeneratorOffset(gen_sepr, offset));
            CURAND_CALL(curandSetQuasiRandomGeneratorDimensions(gen_sepr, m));
            CURAND_CALL(curandGenerateUniform(gen_sepr, sepr_dev, n * m));
            moro_inv_v2 << < grid_v2, block >> > (sepr_dev, m * n, 0, 0.5);
            cudaDeviceSynchronize();
            CUDA_CALL(curandDestroyGenerator(gen_sepr));

            end = chrono::steady_clock::now();
            elapsed = end - start;
            sepr_time2 += elapsed.count();
            // ------------------------
            

            // Combined
            // ------------------------
            start = chrono::steady_clock::now();
            CURAND_CALL(curandCreateGenerator(&gen_comb, CURAND_RNG_QUASI_SOBOL32));
            CURAND_CALL(curandSetGeneratorOffset(gen_comb, offset));
            CURAND_CALL(curandSetQuasiRandomGeneratorDimensions(gen_comb, m));
            CURAND_CALL(curandGenerateNormal(gen_comb, comb_dev, n * m, 0, 0.5));
            CUDA_CALL(curandDestroyGenerator(gen_comb));
            end = chrono::steady_clock::now();
            elapsed = end - start;
            comb_time += elapsed.count();
            // ------------------------
        }

        cout << "Seperated version 1 average EXE TIME: " << sepr_time1 / EXP_TIMES << "ms" << endl;;
        cout << "Seperated version 2 average EXE TIME: " << sepr_time2 / EXP_TIMES << "ms" << endl;;
        cout << "Combined version average EXE TIME: " << comb_time / EXP_TIMES << "ms" << endl;;
    }  

    free(comb_host);
    free(sepr_host);
    CUDA_CALL(cudaFree(sepr_dev));
    CUDA_CALL(cudaFree(comb_dev));

    return 0;
}