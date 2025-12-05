#include "solvers/jacobi.h"
#include "utils/matrix_io.h"
#include "benchmark/benchmark.h"
#include <iostream>
#include <memory>

int main() {
    BenchmarkSuite suite;

    // Configuração dos solvers
    SolverConfig config;
    config.max_iterations = 100;
    config.tolerance = 1e-12;
    config.verbose = false;

    // Criar solvers
    auto jacobi_cpu = std::make_shared<JacobiCPU>();
    auto jacobi_gpu = std::make_shared<JacobiGPU>();

    // Definir solver de referência (CPU)
    suite.set_reference_solver(jacobi_cpu);

    // Adicionar solvers para benchmark
    suite.add_solver(jacobi_cpu);
    suite.add_solver(jacobi_gpu);

    // Adicionar matrizes
    suite.add_matrix("data/small/matriz3x3.txt");
    suite.add_matrix("data/medium/matriz500x500.txt");
    suite.add_matrix("data/large/matriz2000x2000.txt");

    // Configurar e executar
    suite.set_config(config);
    auto results = suite.run();

    // Imprimir resultados
    suite.print_results(results);

    // Salvar em CSV
    suite.save_results_csv("results/benchmark_results.csv", results);

    return 0;
}
