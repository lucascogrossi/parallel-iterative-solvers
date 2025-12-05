#ifndef PRINT_H
#define PRINT_H

#include <vector>

void print_system(const std::vector<double>& A,
                  const std::vector<double>& b,
                  int n);

void print_vector(const std::vector<double>& v);

void print_vector(const std::vector<double>& v, int n);

#endif // PRINT_H
