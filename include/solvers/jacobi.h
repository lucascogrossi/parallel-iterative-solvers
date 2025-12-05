#ifndef JACOBI_H
#define JACOBI_H

#include "solver_base.h"

class JacobiCPU : public IterativeSolver {
public:
    SolverResult solve(const std::vector<double>& A,
                       const std::vector<double>& b,
                       const std::vector<double>& x0,
                       int n,
                       const SolverConfig& config) override;

    std::string name() const override { return "Jacobi (CPU)"; }
};

struct GPUTimings {
    float allocation_ms = 0.0f;
    float host_to_device_ms = 0.0f;
    float computation_ms = 0.0f;
    float device_to_host_ms = 0.0f;

    float total() const {
        return allocation_ms + host_to_device_ms + computation_ms + device_to_host_ms;
    }
};

class JacobiGPU : public IterativeSolver {
public:
    SolverResult solve(const std::vector<double>& A,
                       const std::vector<double>& b,
                       const std::vector<double>& x0,
                       int n,
                       const SolverConfig& config) override;

    std::string name() const override { return "Jacobi (GPU)"; }

    const GPUTimings& get_timings() const { return timings_; }

private:
    GPUTimings timings_;
};

#endif // JACOBI_H
