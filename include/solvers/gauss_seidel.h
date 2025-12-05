#ifndef GS_H
#define GS_H

#include "solver_base.h"

class GaussSeidelCPU : public IterativeSolver {
public:
    SolverResult solve(const std::vector<double>& A,
                       const std::vector<double>& b,
                       const std::vector<double>& x0,
                       int n,
                       const SolverConfig& config) override;

    std::string name() const override { return "Gauss-Seidel (CPU)"; }
};

#endif