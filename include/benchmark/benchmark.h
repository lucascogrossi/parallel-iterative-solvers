#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "../solvers/solver_base.h"
#include <vector>
#include <string>
#include <memory>

struct BenchmarkResult {
    std::string solver_name;
    std::string matrix_file;
    int matrix_size;
    double time_ms;
    int iterations;
    double residual;
    double max_error;  // Compared to reference solution
};

class BenchmarkSuite {
public:
    void add_solver(std::shared_ptr<IterativeSolver> solver);
    void add_matrix(const std::string& filepath);

    void set_config(const SolverConfig& config);
    void set_reference_solver(std::shared_ptr<IterativeSolver> solver);

    std::vector<BenchmarkResult> run();

    void print_results(const std::vector<BenchmarkResult>& results);
    void save_results_csv(const std::string& filename,
                          const std::vector<BenchmarkResult>& results);

private:
    std::vector<std::shared_ptr<IterativeSolver>> solvers_;
    std::vector<std::string> matrices_;
    SolverConfig config_;
    std::shared_ptr<IterativeSolver> reference_solver_;
};

#endif // BENCHMARK_H
