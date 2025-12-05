#ifndef SOLVER_BASE_H
#define SOLVER_BASE_H

#include <vector>
#include <string>

struct SolverResult {
    std::vector<double> solution;
    int iterations;
    double residual;
    double time_ms;
};

struct SolverConfig {
    int max_iterations = 100;
    double tolerance = 1e-12;
    bool verbose = false;
};

class IterativeSolver {
public:
    virtual ~IterativeSolver() = default;

    virtual SolverResult solve(const std::vector<double>& A,
                               const std::vector<double>& b,
                               const std::vector<double>& x0,
                               int n,
                               const SolverConfig& config) = 0;

    virtual std::string name() const = 0;
};

#endif // SOLVER_BASE_H
