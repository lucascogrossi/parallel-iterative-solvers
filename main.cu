#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

// Eliminação de Gauss (Gauss sem pivoteamento) cuda

// Pivoteamento parcial (Gauss com pivoteamento parcial) cuda

// Pivoteamento completo (Gauss com pivoteamento completo) cuda

// Fatoração LU cuda

// Fatoração de Cholesky cuda

// Método iterativo de Gauss–Jacobi cuda

// Método iterativo de Gauss–Seidel red black cuda


// TODO Sassenfeld
/*
std::vector<double> gauss_jacobi(const std::vector<double> &A, 
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
        double max_diff = 0.0;
        for (int i = 0; i < n; i++) {
            max_diff = std::max(max_diff, std::abs(x_new[i] - x[i]));
        }
        if (max_diff < tol) {
            std::cout << "Convergência na iteração k=" << k << std::endl;
            return x_new;
        }
        
        x = x_new;  // Atualiza só no final
    }
    
    std::cout << "Máximo de iterações atingido." << std::endl;
    return x;
}
*/

std::vector<double> gauss_jacobi(const std::vector<double> &A, 
                                  const std::vector<double> &b,
                                  std::vector<double> x,
                                  int n, double tol, int max_iter = 100) {
	// Alocar memoria gpu

	// Transferir cpu -> gpu

	// Computar

	// Transferir gpu -> cpu

	// Free memoria gpu
}
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


int main() {
	int n;
	std::vector<double> A;
	std::vector<double> b;

	std::ifstream file("data/teste.txt");

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

	imprimir_sistema(A, b, n);

	std::cout << "Método iterativo de Gauss–Jacobi" << std::endl;
	std::vector<double> x0(n, 0.0); // Chute inicial 0
	auto x = gauss_jacobi(A, b, x0, n, 5e-2);
	imprimir_vetor(x);

	return 0;
}	

