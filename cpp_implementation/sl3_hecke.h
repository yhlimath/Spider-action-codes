#ifndef SL3_HECKE_H
#define SL3_HECKE_H

#include <vector>
#include <map>
#include <iostream>
#include <cmath>
#include <string>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <complex>

// Constants and Typedefs
using Complex = std::complex<double>;

// Polynomial class to handle coefficients like 3*n^2 + 1
// We assume coefficients are integers for simplicity, or complex?
// The user mentions "C(q)-vector space" but Arnoldi is usually numerical.
// However, the action of e generates polynomials in n.
// For the final matrix H or T, we need to evaluate n to a number (complex).
// So for the core logic, we can either work symbolically (Polynomial) or numerically (Complex).
// Given "speed up", numerical evaluation during action is fastest if n is fixed.
// If we want symbolic matrices, we need Polynomial.
// The user asked for "algorithm that computes the action of e on paths... Python is inefficient... move to C".
// And "parts that apply the Arnoldi method". Arnoldi is numerical.
// So I will implement the action numerically for a given n_val.

// String representation: vector of ints (1, 0, -1)
using String = std::vector<int>;

// Hash function for String to use in unordered_map
struct StringHash {
    size_t operator()(const String& s) const {
        size_t seed = 0;
        for (int i : s) {
            seed ^= std::hash<int>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Sparse Vector: Map from String to Coefficient (Complex)
// Using unordered_map for O(1) average access (hashing)
using SparseVector = std::unordered_map<String, Complex, StringHash>;

class Sl3Hecke {
public:
    int L;
    Complex n_val;
    std::vector<String> basis;
    std::unordered_map<String, int, StringHash> basis_map; // String to Index

    Sl3Hecke(int L, Complex n_val);

    // Core string manipulation functions (static or members)
    static std::vector<String> generate_all_valid_strings(int n);
    static void generate_valid_strings_recursive(std::vector<int>& sequence, int n, int count_1, int count_0, int count_neg1, std::vector<String>& results);

    // Helper logic for e action
    static std::vector<std::tuple<int, int, int>> form_triplets_with_positions(const String& s);
    static String bend_string(const String& s);
    static String bending_power(const String& s, int p);
    static String inverse_bending_power(const String& s, int p);
    static String generate_triplet_at_position(const String& s, int p);
    static std::tuple<int, int, int> find_last_triplet(const String& s);

    // Decomposition logic: returns (decomposed_string, list of operations)
    // Operation: type 't' (triplet removal) or 'e' (generator application)
    // We represent operation as pair {type_char, pos}
    struct Operation {
        char type; // 't' or 'e'
        int pos;
    };
    static std::pair<String, std::vector<Operation>> string_decomposition(const String& s);

    // The generator action e_i
    // Returns a list of (coefficient, resulting_string)
    // Coefficient is Complex (evaluated polynomial)
    std::vector<std::pair<Complex, String>> apply_e(const String& s, int i);

    // Apply e_i to a sparse vector
    SparseVector apply_e_vector(const SparseVector& v, int i);

    // Apply H = sum e_i
    SparseVector apply_H(const SparseVector& v);

    // Apply T = prod e_even * prod e_odd
    SparseVector apply_T(const SparseVector& v);

    // Arnoldi Iteration
    // Returns Hessenberg matrix (dense vector of vectors)
    // start_idx: index in basis to start with
    std::vector<std::vector<Complex>> arnoldi(int k, int start_idx, char operator_type);
};

#endif
