#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>

void imprimir_sistema(const std::vector<double> &A, const std::vector<double> &b, int n) {
    std::cout << "Ax = b" << std::endl;
    
    for (int i = 0; i < n; i++) {
        std::cout << "| ";
        for (int j = 0; j < n; j++) {
            std::cout << std::setw(6) << A[i * n + j] << " ";
        }
        std::cout << "|";
        
        if (i == n / 2) {
            std::cout << " * |x" << i + 1 << "| = ";
        } else {
            std::cout << "   |x" << i + 1 << "|   ";
        }
        
        std::cout << "|" << std::setw(6) << b[i] << " |" << std::endl;
    }
    std::cout << std::endl;
}

void imprimir_vetor(const std::vector<double> &v) {
    std::cout << "[ ";
    for (size_t i = 0; i < v.size(); i++) {
        std::cout << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << " ]" << std::endl;
}

std::vector<double> gauss_jacobi_cpu(const std::vector<double> &A, 
                                  const std::vector<double> &b,
                                  std::vector<double> x,
                                  int n, double tol, int max_iter = 100) {
    std::vector<double> x_new(n);
    
    for (int k = 0; k < max_iter; k++) {
        for (int i = 0; i < n; i++) {
            double soma = 0.0;
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    soma += A[i * n + j] * x[j];
                }
            }
            x_new[i] = (b[i] - soma) / A[i * n + i];
        }
        x = x_new;
    }
    
    return x;
}
// TODO Verificar convergência na GPU com redução 
__global__ void gauss_jacobi_kernel(double *A, double *b, double *x, double *x_new, int n) {
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

std::vector<double> gauss_jacobi_gpu(const std::vector<double> &A_h,
                                     const std::vector<double> &b_h,
                                     const std::vector<double> &x_h,
                                     int n, double tol, int max_iter,
                                     double &tempo_alocacao,
                                     double &tempo_computacao,
                                     double &tempo_transferencia) {
    std::vector<double> x(n);
    double *A_d, *b_d, *x_d, *x_new_d;
    
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    auto start_alloc = std::chrono::high_resolution_clock::now();
    
    cudaMalloc(&A_d, n * n * sizeof(double));
    cudaMalloc(&b_d, n * sizeof(double));
    cudaMalloc(&x_d, n * sizeof(double));
    cudaMalloc(&x_new_d, n * sizeof(double));

    cudaMemcpy(A_d, A_h.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x_h.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    
    auto end_alloc = std::chrono::high_resolution_clock::now();
    tempo_alocacao = std::chrono::duration<double, std::milli>(end_alloc - start_alloc).count();

    auto start_compute = std::chrono::high_resolution_clock::now();
    
    for (int k = 0; k < max_iter; k++) {
        gauss_jacobi_kernel<<<blocks, threadsPerBlock>>>(A_d, b_d, x_d, x_new_d, n);
        std::swap(x_d, x_new_d);
    }
    cudaDeviceSynchronize();
    
    auto end_compute = std::chrono::high_resolution_clock::now();
    tempo_computacao = std::chrono::duration<double, std::milli>(end_compute - start_compute).count();

    auto start_transfer = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(x.data(), x_d, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    auto end_transfer = std::chrono::high_resolution_clock::now();
    tempo_transferencia = std::chrono::duration<double, std::milli>(end_transfer - start_transfer).count();

    cudaFree(A_d);
    cudaFree(b_d);
    cudaFree(x_d);
    cudaFree(x_new_d);

    return x;
}

int main() {
    int n;
    std::vector<double> A, b;

    std::ifstream file("data/matriz2000x2000.txt");

    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo.\n";
        return 1;
    }

    file >> n;

    A.resize(n * n);
    b.resize(n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file >> A[i * n + j];
        }
        file >> b[i];
    }

    if (file.fail()) {
        std::cerr << "Erro ao ler os dados do arquivo.\n";
        return 1;
    }

    file.close();

    std::cout << "Sistema " << n << "x" << n << std::endl;
    std::cout << "========================================" << std::endl;

    std::vector<double> x0(n, 0.0);
    int max_iter = 100;

    auto start = std::chrono::high_resolution_clock::now();
    auto x_cpu = gauss_jacobi_cpu(A, b, x0, n, 1e-12, max_iter);
    auto end = std::chrono::high_resolution_clock::now();
    double tempo_cpu = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "\nCPU:" << std::endl;
    std::cout << "  Tempo total: " << tempo_cpu << " ms" << std::endl;

    double tempo_alloc, tempo_compute, tempo_transfer;
    auto x_gpu = gauss_jacobi_gpu(A, b, x0, n, 1e-12, max_iter, 
                                   tempo_alloc, tempo_compute, tempo_transfer);
    
    std::cout << "\nGPU:" << std::endl;
    std::cout << "  Alocacao + H->D: " << tempo_alloc << " ms" << std::endl;
    std::cout << "  Computacao:      " << tempo_compute << " ms" << std::endl;
    std::cout << "  D->H:            " << tempo_transfer << " ms" << std::endl;
    std::cout << "  Total:           " << tempo_alloc + tempo_compute + tempo_transfer << " ms" << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Speedup (so computacao): " << tempo_cpu / tempo_compute << "x" << std::endl;
    std::cout << "Speedup (total):         " << tempo_cpu / (tempo_alloc + tempo_compute + tempo_transfer) << "x" << std::endl;

    double max_diff = 0.0;
    for (int i = 0; i < n; i++) {
        max_diff = std::max(max_diff, std::abs(x_cpu[i] - x_gpu[i]));
    }
    std::cout << "\nDiferenca max CPU vs GPU: " << max_diff << std::endl;

    return 0;
}