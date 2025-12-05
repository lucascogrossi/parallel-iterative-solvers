#ifndef MATRIX_IO_H
#define MATRIX_IO_H

#include <vector>
#include <string>

struct LinearSystem {
    std::vector<double> A;
    std::vector<double> b;
    int n;
};

bool load_linear_system(const std::string& filename, LinearSystem& system);

bool save_solution(const std::string& filename,
                   const std::vector<double>& x,
                   int n);

#endif // MATRIX_IO_H
