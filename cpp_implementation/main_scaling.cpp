
#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "sl3_hecke.h"

// Simple JSON writer helper
void write_hessenberg_json(int L, Complex n_val, std::string op_type, int dim, const std::vector<std::vector<Complex>>& H, double time_elapsed) {
    std::cout << "{" << std::endl;
    std::cout << "  \"L\": " << L << "," << std::endl;
    std::cout << "  \"n_real\": " << n_val.real() << "," << std::endl;
    std::cout << "  \"n_imag\": " << n_val.imag() << "," << std::endl;
    std::cout << "  \"operator\": \"" << op_type << "\"," << std::endl;
    std::cout << "  \"dim\": " << dim << "," << std::endl;
    std::cout << "  \"time_elapsed\": " << time_elapsed << "," << std::endl;
    std::cout << "  \"hessenberg_matrix\": [" << std::endl;
    for (size_t i = 0; i < H.size(); ++i) {
        std::cout << "    [";
        for (size_t j = 0; j < H[i].size(); ++j) {
            std::cout << "{\"real\": " << std::setprecision(16) << H[i][j].real()
                      << ", \"imag\": " << H[i][j].imag() << "}";
            if (j < H[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]";
        if (i < H.size() - 1) std::cout << ",";
        std::cout << std::endl;
    }
    std::cout << "  ]" << std::endl;
    std::cout << "}" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <L> <n_real> <n_imag> <operator_H_or_T> [k_arnoldi]" << std::endl;
        return 1;
    }

    int L = std::stoi(argv[1]);
    double n_real = std::stod(argv[2]);
    double n_imag = std::stod(argv[3]);
    std::string op_type = argv[4];
    int k_arnoldi = 50;
    if (argc >= 6) {
        k_arnoldi = std::stoi(argv[5]);
    }

    Complex n_val(n_real, n_imag);

    // Initialize system
    Sl3Hecke system(L, n_val);

    // Adjust k if dim is small
    if (system.basis.size() <= k_arnoldi) {
        k_arnoldi = system.basis.size();
    }

    char op_char = op_type[0];

    // Timer start
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run Arnoldi
    auto H_mat = system.arnoldi(k_arnoldi, -1, op_char); // -1 for random start

    // Timer stop
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Output JSON to stdout
    write_hessenberg_json(L, n_val, op_type, system.basis.size(), H_mat, elapsed.count());

    return 0;
}
