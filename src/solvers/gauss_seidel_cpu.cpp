#include "solvers/gauss_seidel.h"
#include "utils/timer.h"

SolverResult GaussSeidelCPU::solve(const std::vector<double>& A,
                                    const std::vector<double>& b,
                                    const std::vector<double>& x0,
                                    int n,
                                    const SolverConfig& config) {
    Timer timer;
    SolverResult result;

    std::vector<double> x = x0;
    std::vector<double> x_old(n);

    int k;
    for (k = 0; k < config.max_iterations; k++) {
        x_old = x;

        for (int i = 0; i < n; i++) {
            double soma = 0.0;

            // Parte antes de i -> usa valores já atualizados
            for (int j = 0; j < i; j++) {
                soma += A[i * n + j] * x[j];
            }

            // Parte depois de i -> usa valores antigos
            for (int j = i + 1; j < n; j++) {
                soma += A[i * n + j] * x_old[j];
            }

            x[i] = (b[i] - soma) / A[i * n + i];
        }
    }

    result.solution = x;
    result.iterations = k;
    result.residual = 0.0;  // TODO: Calculate residual
    result.time_ms = timer.elapsed_ms();

    return result;
}