#include "utils/print.h"
#include <iostream>
#include <iomanip>

void print_system(const std::vector<double>& A,
                  const std::vector<double>& b,
                  int n) {
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

void print_vector(const std::vector<double>& v) {
    std::cout << "[ ";
    for (size_t i = 0; i < v.size(); i++) {
        std::cout << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << " ]" << std::endl;
}

void print_vector(const std::vector<double>& v, int n) {
    std::cout << "[ ";
    for (int i = 0; i < n && i < static_cast<int>(v.size()); i++) {
        std::cout << v[i];
        if (i < n - 1) std::cout << ", ";
    }
    std::cout << " ]" << std::endl;
}
