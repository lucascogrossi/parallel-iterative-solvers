#include "utils/matrix_io.h"
#include <fstream>
#include <iostream>

bool load_linear_system(const std::string& filename, LinearSystem& system) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo: " << filename << std::endl;
        return false;
    }

    file >> system.n;

    system.A.resize(system.n * system.n);
    system.b.resize(system.n);

    for (int i = 0; i < system.n; i++) {
        for (int j = 0; j < system.n; j++) {
            file >> system.A[i * system.n + j];
        }
        file >> system.b[i];
    }

    if (file.fail()) {
        std::cerr << "Erro ao ler os dados do arquivo: " << filename << std::endl;
        return false;
    }

    file.close();
    return true;
}

bool save_solution(const std::string& filename,
                   const std::vector<double>& x,
                   int n) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Erro ao abrir arquivo para escrita: " << filename << std::endl;
        return false;
    }

    for (int i = 0; i < n; i++) {
        file << x[i] << "\n";
    }

    file.close();
    return true;
}
