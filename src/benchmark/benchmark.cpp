#include "benchmark/benchmark.h"
#include "solvers/jacobi.h"
#include "utils/matrix_io.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <map>

void BenchmarkSuite::add_solver(std::shared_ptr<IterativeSolver> solver) {
    solvers_.push_back(solver);
}

void BenchmarkSuite::add_matrix(const std::string& filepath) {
    matrices_.push_back(filepath);
}

void BenchmarkSuite::set_config(const SolverConfig& config) {
    config_ = config;
}

void BenchmarkSuite::set_reference_solver(std::shared_ptr<IterativeSolver> solver) {
    reference_solver_ = solver;
}

std::vector<BenchmarkResult> BenchmarkSuite::run() {
    std::vector<BenchmarkResult> results;

    for (const auto& matrix_file : matrices_) {
        LinearSystem system;
        if (!load_linear_system(matrix_file, system)) {
            std::cerr << "Falha ao carregar: " << matrix_file << std::endl;
            continue;
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "Matriz: " << matrix_file << " (" << system.n << "x" << system.n << ")" << std::endl;
        std::cout << "========================================" << std::endl;

        std::vector<double> x0(system.n, 0.0);
        std::vector<double> reference_solution;

        // Run reference solver if specified
        if (reference_solver_) {
            std::cout << "Executando solver de referencia: " << reference_solver_->name() << std::endl;
            auto ref_result = reference_solver_->solve(system.A, system.b, x0, system.n, config_);
            reference_solution = ref_result.solution;
        }

        // Run all solvers
        for (const auto& solver : solvers_) {
            std::cout << "\nExecutando: " << solver->name() << "..." << std::endl;

            auto result = solver->solve(system.A, system.b, x0, system.n, config_);

            BenchmarkResult bench_result;
            bench_result.solver_name = solver->name();
            bench_result.matrix_file = matrix_file;
            bench_result.matrix_size = system.n;
            bench_result.time_ms = result.time_ms;
            bench_result.iterations = result.iterations;
            bench_result.residual = result.residual;

            // Calculate error against reference
            if (!reference_solution.empty()) {
                double max_error = 0.0;
                for (int i = 0; i < system.n; i++) {
                    max_error = std::max(max_error, std::abs(result.solution[i] - reference_solution[i]));
                }
                bench_result.max_error = max_error;
                std::cout << "  Erro max (vs referencia): " << max_error << std::endl;
            } else {
                bench_result.max_error = 0.0;
            }

            std::cout << "  Tempo: " << result.time_ms << " ms" << std::endl;
            std::cout << "  Iteracoes: " << result.iterations << std::endl;

            // Show detailed GPU timings if it's a GPU solver
            auto gpu_solver = std::dynamic_pointer_cast<JacobiGPU>(solver);
            if (gpu_solver) {
                const auto& timings = gpu_solver->get_timings();
                std::cout << "    - cudaMalloc:  " << timings.allocation_ms << " ms" << std::endl;
                std::cout << "    - cudaMemcpyHostToDevice:        " << timings.host_to_device_ms << " ms" << std::endl;
                std::cout << "    - Kernel: " << timings.computation_ms << " ms" << std::endl;
                std::cout << "    - cudaMemcpyDeviceToHost:        " << timings.device_to_host_ms << " ms" << std::endl;
            }

            results.push_back(bench_result);
        }
    }

    return results;
}

void BenchmarkSuite::print_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n\n========================================" << std::endl;
    std::cout << "RESUMO DOS BENCHMARKS" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Group results by matrix size
    std::map<int, std::vector<BenchmarkResult>> grouped;
    for (const auto& r : results) {
        grouped[r.matrix_size].push_back(r);
    }

    // Print grouped by matrix size
    for (const auto& [size, group] : grouped) {
        std::cout << "Matriz " << size << "x" << size << ":" << std::endl;
        std::cout << std::string(77, '-') << std::endl;

        std::cout << std::left << std::setw(25) << "Solver"
                  << std::right << std::setw(12) << "Tempo (ms)"
                  << std::setw(10) << "Iter"
                  << std::setw(15) << "Erro Max"
                  << std::setw(12) << "Speedup" << std::endl;
        std::cout << std::string(77, '-') << std::endl;

        // Find CPU baseline time for speedup calculation
        double cpu_time = 0.0;
        for (const auto& r : group) {
            if (r.solver_name.find("CPU") != std::string::npos) {
                cpu_time = r.time_ms;
                break;
            }
        }

        // Print results
        for (const auto& r : group) {
            std::cout << std::left << std::setw(25) << r.solver_name
                      << std::right << std::setw(12) << std::fixed << std::setprecision(3) << r.time_ms
                      << std::setw(10) << r.iterations
                      << std::setw(15) << std::scientific << std::setprecision(2) << r.max_error;

            // Calculate and print speedup
            if (cpu_time > 0.0 && r.solver_name.find("GPU") != std::string::npos) {
                double speedup = cpu_time / r.time_ms;
                std::cout << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x";
            } else {
                std::cout << std::setw(12) << "-";
            }

            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

}

void BenchmarkSuite::save_results_csv(const std::string& filename,
                                       const std::vector<BenchmarkResult>& results) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir arquivo CSV: " << filename << std::endl;
        return;
    }

    file << "Solver,MatrixFile,MatrixSize,TimeMS,Iterations,Residual,MaxError,Speedup\n";

    // Group by matrix size to calculate speedup
    std::map<int, std::vector<BenchmarkResult>> grouped;
    for (const auto& r : results) {
        grouped[r.matrix_size].push_back(r);
    }

    for (const auto& [size, group] : grouped) {
        // Find CPU baseline
        double cpu_time = 0.0;
        for (const auto& r : group) {
            if (r.solver_name.find("CPU") != std::string::npos) {
                cpu_time = r.time_ms;
                break;
            }
        }

        // Write results with speedup
        for (const auto& r : group) {
            file << r.solver_name << ","
                 << r.matrix_file << ","
                 << r.matrix_size << ","
                 << r.time_ms << ","
                 << r.iterations << ","
                 << r.residual << ","
                 << r.max_error << ",";

            if (cpu_time > 0.0 && r.solver_name.find("GPU") != std::string::npos) {
                file << (cpu_time / r.time_ms);
            } else {
                file << "-";
            }
            file << "\n";
        }
    }

    file.close();
    std::cout << "\nResultados salvos em: " << filename << std::endl;
}
