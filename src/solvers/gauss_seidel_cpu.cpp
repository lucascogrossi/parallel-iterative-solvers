#include "solvers/jacobi.h"
#include "utils/timer.h"

SolverResult JacobiCPU::solve(const std::vector<double>& A,
                               const std::vector<double>& b,
                               const std::vector<double>& x0,
                               int n,
                               const SolverConfig& config) {
    Timer timer;
    SolverResult result;

    std::vector<double> x = x0;
    std::vector<double> x_new(n);

    int k;
    for (k = 0; k < config.max_iterations; k++) {
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

    result.solution = x;
    result.iterations = k;
    result.residual = 0.0;  // TODO: Calculate residual
    result.time_ms = timer.elapsed_ms();

    return result;
}
