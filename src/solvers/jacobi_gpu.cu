#include "solvers/jacobi.h"
#include "utils/cuda_utils.h"

/*
TODO:
    - Teste de convergencia na gpu (reducao)
    - Otimizações
*/
__global__ void jacobi_kernel(double *A, double *b, double *x, double *x_new, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) {
        double soma = 0.0;
        for (int j = 0; j < n; j++) {
            if (j != i) {
                soma += A[i * n + j] * x[j];
            }
        }
        x_new[i] = (b[i] - soma) / A[i * n + i];
    }
}

SolverResult JacobiGPU::solve(const std::vector<double>& A,
                               const std::vector<double>& b,
                               const std::vector<double>& x0,
                               int n,
                               const SolverConfig& config) {
    SolverResult result;
    double *A_d, *b_d, *x_d, *x_new_d;

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start_alloc, end_alloc;
    cudaEvent_t start_h2d, end_h2d;
    cudaEvent_t start_compute, end_compute;
    cudaEvent_t start_d2h, end_d2h;

    CHECK_CUDA(cudaEventCreate(&start_alloc));
    CHECK_CUDA(cudaEventCreate(&end_alloc));
    CHECK_CUDA(cudaEventCreate(&start_h2d));
    CHECK_CUDA(cudaEventCreate(&end_h2d));
    CHECK_CUDA(cudaEventCreate(&start_compute));
    CHECK_CUDA(cudaEventCreate(&end_compute));
    CHECK_CUDA(cudaEventCreate(&start_d2h));
    CHECK_CUDA(cudaEventCreate(&end_d2h));

    // Alocação
    CHECK_CUDA(cudaEventRecord(start_alloc));

    CHECK_CUDA(cudaMalloc(&A_d, n * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&b_d, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&x_d, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&x_new_d, n * sizeof(double)));

    CHECK_CUDA(cudaEventRecord(end_alloc));

    // H->D
    CHECK_CUDA(cudaEventRecord(start_h2d));

    CHECK_CUDA(cudaMemcpy(A_d, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_d, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x0.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(end_h2d));

    // Computação
    CHECK_CUDA(cudaEventRecord(start_compute));

    for (int k = 0; k < config.max_iterations; k++) {
        jacobi_kernel<<<blocks, threadsPerBlock>>>(A_d, b_d, x_d, x_new_d, n);
        CHECK_CUDA(cudaGetLastError());
        std::swap(x_d, x_new_d);
    }

    CHECK_CUDA(cudaEventRecord(end_compute));

    // D->H
    CHECK_CUDA(cudaEventRecord(start_d2h));
    result.solution.resize(n);
    CHECK_CUDA(cudaMemcpy(result.solution.data(), x_d, n * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(end_d2h));

    // Sincronizar e obter tempos
    CHECK_CUDA(cudaEventSynchronize(end_d2h));

    CHECK_CUDA(cudaEventElapsedTime(&timings_.allocation_ms, start_alloc, end_alloc));
    CHECK_CUDA(cudaEventElapsedTime(&timings_.host_to_device_ms, start_h2d, end_h2d));
    CHECK_CUDA(cudaEventElapsedTime(&timings_.computation_ms, start_compute, end_compute));
    CHECK_CUDA(cudaEventElapsedTime(&timings_.device_to_host_ms, start_d2h, end_d2h));

    // Cleanup
    CHECK_CUDA(cudaFree(A_d));
    CHECK_CUDA(cudaFree(b_d));
    CHECK_CUDA(cudaFree(x_d));
    CHECK_CUDA(cudaFree(x_new_d));

    CHECK_CUDA(cudaEventDestroy(start_alloc));
    CHECK_CUDA(cudaEventDestroy(end_alloc));
    CHECK_CUDA(cudaEventDestroy(start_h2d));
    CHECK_CUDA(cudaEventDestroy(end_h2d));
    CHECK_CUDA(cudaEventDestroy(start_compute));
    CHECK_CUDA(cudaEventDestroy(end_compute));
    CHECK_CUDA(cudaEventDestroy(start_d2h));
    CHECK_CUDA(cudaEventDestroy(end_d2h));

    result.iterations = config.max_iterations;
    result.residual = 0.0;  // TODO: Calculate residual
    result.time_ms = timings_.total();

    return result;
}
