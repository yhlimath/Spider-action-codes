
#include <iostream>
#include <vector>
#include <complex>
#include "sl3_hecke.h"

int main() {
    std::cout << "Sanity Check for C++ Implementation" << std::endl;
    std::cout << "===================================" << std::endl;

    // L=2
    int L = 2;
    Complex n_val = 1.0; // Use n=1 for check

    std::cout << "Initializing system with L=" << L << ", n=" << n_val << std::endl;
    Sl3Hecke system(L, n_val);

    std::cout << "Dimension: " << system.basis.size() << std::endl;

    // Check Basis
    std::cout << "\nBasis Vectors:" << std::endl;
    for (int i = 0; i < system.basis.size(); ++i) {
        std::cout << i << ": [";
        for (size_t j = 0; j < system.basis[i].size(); ++j) {
            std::cout << system.basis[i][j] << (j < system.basis[i].size() - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
    }

    // Check e_1 action on first basis vector
    std::cout << "\nAction of e_1 on basis[0]:" << std::endl;
    auto res = system.apply_e(system.basis[0], 1);
    for (const auto& p : res) {
        std::cout << "  Coeff: " << p.first << ", String: [";
        for (size_t j = 0; j < p.second.size(); ++j) std::cout << p.second[j] << (j < p.second.size() - 1 ? ", " : "");
        std::cout << "]" << std::endl;
    }

    // Build H matrix explicitly for checking
    std::cout << "\nExplicit H Matrix (L=" << L << "):" << std::endl;
    std::vector<std::vector<Complex>> H_mat(system.basis.size(), std::vector<Complex>(system.basis.size(), 0.0));

    for (int j = 0; j < system.basis.size(); ++j) {
        SparseVector v;
        v[system.basis[j]] = 1.0;
        SparseVector w = system.apply_H(v);

        for (const auto& pair : w) {
            if (system.basis_map.count(pair.first)) {
                int i = system.basis_map[pair.first];
                H_mat[i][j] = pair.second;
            }
        }
    }

    for (int i = 0; i < system.basis.size(); ++i) {
        for (int j = 0; j < system.basis.size(); ++j) {
            std::cout << H_mat[i][j] << "\t";
        }
        std::cout << std::endl;
    }

    // Build T matrix explicitly
    std::cout << "\nExplicit T Matrix (L=" << L << "):" << std::endl;
    std::vector<std::vector<Complex>> T_mat(system.basis.size(), std::vector<Complex>(system.basis.size(), 0.0));

    for (int j = 0; j < system.basis.size(); ++j) {
        SparseVector v;
        v[system.basis[j]] = 1.0;
        SparseVector w = system.apply_T(v);

        for (const auto& pair : w) {
            if (system.basis_map.count(pair.first)) {
                int i = system.basis_map[pair.first];
                T_mat[i][j] = pair.second;
            }
        }
    }

    for (int i = 0; i < system.basis.size(); ++i) {
        for (int j = 0; j < system.basis.size(); ++j) {
            std::cout << T_mat[i][j] << "\t";
        }
        std::cout << std::endl;
    }

    return 0;
}
