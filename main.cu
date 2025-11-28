#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>

// Eliminação de Gauss (Gauss sem pivoteamento) cuda

// Pivoteamento parcial (Gauss com pivoteamento parcial) cuda

// Pivoteamento completo (Gauss com pivoteamento completo) cuda

// Fatoração LU cuda

// Fatoração de Cholesky cuda

// Método iterativo de Gauss–Jacobi cuda

// Método iterativo de Gauss–Seidel red black cuda

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

// TODO Sassenfeld

std::vector<double> gauss_jacobi_cpu(const std::vector<double> &A, 
                                  const std::vector<double> &b,
                                  std::vector<double> x,
                                  int n, double tol, int max_iter = 100) {
    std::vector<double> x_new(n);
    
    for (int k = 0; k < max_iter; k++) {
        
        for (int i = 0; i < n; i++) {
            double soma = 0.0;
            
            // Soma tudo exceto diagonal 
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    soma += A[i * n + j] * x[j];
                }
            }
            x_new[i] = (b[i] - soma) / A[i * n + i];
        }
        
        // Critério de parada
        /* double max_diff = 0.0;
        for (int i = 0; i < n; i++) {
            max_diff = std::max(max_diff, std::abs(x_new[i] - x[i]));
        }
        if (max_diff < tol) {
            std::cout << "Convergência na iteração k=" << k << std::endl;
            return x_new;
        } */
        
        x = x_new;  // Atualiza só no final
    }
    
    std::cout << "Máximo de iterações atingido." << std::endl;
    return x;
}

// Cada thread calcula x_new[i] da linha i:
// soma os produtos A[i,j] * x[j] para j != i, depois aplica a fórmula de Jacobi
// TODO: shared memory para o vetor x
// TODO: criterio de parada na gpu, usando redução
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

std::vector<double> gauss_jacobi(const std::vector<double> &A_h,
                                  const std::vector<double> &b_h,
                                  const std::vector<double> &x_h,
                                  int n, double tol, int max_iter = 100) {
	// Criar copia local de x para modificar
    std::vector<double> x = x_h;

	// Alocar memoria gpu
    double *A_d, *b_d, *x_d, *x_new_d;
    cudaMalloc(&A_d, n * n * sizeof(double));
    cudaMalloc(&b_d, n * sizeof(double));
    cudaMalloc(&x_d, n * sizeof(double));
    cudaMalloc(&x_new_d, n * sizeof(double));

	// Transferir cpu -> gpu
    cudaMemcpy(A_d, A_h.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x_h.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

	// Host controla as iterações
    for (int k = 0; k < max_iter; k++) {
        // Computa as linhas em paralelo
        gauss_jacobi_kernel<<<blocks, threadsPerBlock>>>(A_d, b_d, x_d, x_new_d, n);
        std::swap(x_d, x_new_d);
/*      // Copiamos x para o host para checar convergencia na cpu
        std::vector<double> x_old(n), x_new_h(n);
        cudaMemcpy(x_old.data(), x_d, n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(x_new_h.data(), x_new_d, n * sizeof(double), cudaMemcpyDeviceToHost);
        
        // Criterio de parada
        double max_diff = 0.0;
        for (int i = 0; i < n; i++) {
            max_diff = std::max(max_diff, std::abs(x_new_h[i] - x_old[i]));
        }

        std::swap(x_d, x_new_d);

        if (max_diff < tol) {
            std::cout << "Convergência na iteração k=" << k << std::endl;
            
        } */
    }
    cudaDeviceSynchronize();

	// Transferir gpu -> cpu
    cudaMemcpy(x.data(), x_d, n * sizeof(double), cudaMemcpyDeviceToHost);

	// Free memoria gpu
    cudaFree(A_d);
    cudaFree(b_d);
    cudaFree(x_d);
    cudaFree(x_new_d);

    return x;
}

int main() {
	int n;
	std::vector<double> A, b;

	std::ifstream file("data/sistema2000x2000.txt");

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

	//imprimir_sistema(A, b, n);

	std::cout << "Método iterativo de Gauss–Jacobi" << std::endl;
	std::vector<double> x0(n, 0.0); // Chute inicial 0

	// CPU
	auto start = std::chrono::high_resolution_clock::now();
	auto x_cpu = gauss_jacobi_cpu(A, b, x0, n, 1e-12);
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "CPU: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
	//imprimir_vetor(x_cpu);

	// GPU
	start = std::chrono::high_resolution_clock::now();
	auto x_gpu = gauss_jacobi(A, b, x0, n, 1e-12);
	end = std::chrono::high_resolution_clock::now();
	std::cout << "GPU: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
	//imprimir_vetor(x_gpu);

	return 0;
}