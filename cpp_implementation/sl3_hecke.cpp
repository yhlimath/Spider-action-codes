#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <complex>
#include <map>

// Include the header
#include "sl3_hecke.h"

// Define constants
const double PI = 3.14159265358979323846;

// Helper function for n polynomial evaluation
// We don't have a Polynomial class, so we compute coefficients and evaluate.
// The coefficients from string decomposition logic are simple polynomials (power of n).
// We'll compute them on the fly.

Sl3Hecke::Sl3Hecke(int L, Complex n_val) : L(L), n_val(n_val) {
    basis = generate_all_valid_strings(L);
    for (int i = 0; i < basis.size(); ++i) {
        basis_map[basis[i]] = i;
    }
}

std::vector<String> Sl3Hecke::generate_all_valid_strings(int n) {
    std::vector<String> results;
    std::vector<int> sequence;
    generate_valid_strings_recursive(sequence, n, 0, 0, 0, results);
    return results;
}

void Sl3Hecke::generate_valid_strings_recursive(std::vector<int>& sequence, int n, int count_1, int count_0, int count_neg1, std::vector<String>& results) {
    if (sequence.size() == 3 * n) {
        results.push_back(sequence);
        return;
    }
    if (count_1 < n) {
        sequence.push_back(1);
        generate_valid_strings_recursive(sequence, n, count_1 + 1, count_0, count_neg1, results);
        sequence.pop_back();
    }
    if (count_0 < count_1 && count_0 < n) {
        sequence.push_back(0);
        generate_valid_strings_recursive(sequence, n, count_1, count_0 + 1, count_neg1, results);
        sequence.pop_back();
    }
    if (count_neg1 < count_0 && count_neg1 < n) {
        sequence.push_back(-1);
        generate_valid_strings_recursive(sequence, n, count_1, count_0, count_neg1 + 1, results);
        sequence.pop_back();
    }
}

// Helper: form triplets with positions
std::vector<std::tuple<int, int, int>> Sl3Hecke::form_triplets_with_positions(const String& s) {
    std::vector<std::tuple<int, int, int>> triplets;
    std::vector<std::pair<int, int>> indexed_sequence;
    for (int i = 0; i < s.size(); ++i) {
        indexed_sequence.push_back({i, s[i]});
    }

    while (!indexed_sequence.empty()) {
        int index_1 = -1;
        int max_i = -1;
        // Find max index for value 1
        for (int i = 0; i < indexed_sequence.size(); ++i) {
            if (indexed_sequence[i].second == 1) {
                if (indexed_sequence[i].first > max_i) {
                    max_i = indexed_sequence[i].first;
                    index_1 = i; // Store index in indexed_sequence
                }
            }
        }
        if (index_1 == -1) break; // No 1 found

        int pos_1 = indexed_sequence[index_1].first;

        int index_0 = -1;
        for (int i = 0; i < indexed_sequence.size(); ++i) {
            if (indexed_sequence[i].second == 0 && indexed_sequence[i].first > pos_1) {
                index_0 = i;
                break;
            }
        }
        if (index_0 == -1) break; // No 0 found after 1

        int pos_0 = indexed_sequence[index_0].first;

        int index_neg1 = -1;
        for (int i = 0; i < indexed_sequence.size(); ++i) {
            if (indexed_sequence[i].second == -1 && indexed_sequence[i].first > pos_0) {
                index_neg1 = i;
                break;
            }
        }
        if (index_neg1 == -1) break; // No -1 found after 0

        int pos_neg1 = indexed_sequence[index_neg1].first;

        triplets.emplace_back(pos_1, pos_0, pos_neg1);

        // Remove used elements from indexed_sequence
        // To remove efficiently and keep order relative to original string?
        // We just filter out the used indices.
        std::vector<std::pair<int, int>> next_sequence;
        for (const auto& p : indexed_sequence) {
            if (p.first != pos_1 && p.first != pos_0 && p.first != pos_neg1) {
                next_sequence.push_back(p);
            }
        }
        indexed_sequence = next_sequence;
    }
    return triplets;
}

String Sl3Hecke::bend_string(const String& s) {
    String new_s = s;
    std::vector<std::pair<int, int>> indexed_sequence;
    for (int i = 0; i < s.size(); ++i) indexed_sequence.push_back({i, s[i]});

    int start_1_index = -1;
    for (const auto& p : indexed_sequence) {
        if (p.second == 1) {
            start_1_index = p.first;
            break; // Find first 1
        }
    }
    if (start_1_index == -1) return s;

    auto triplets = form_triplets_with_positions(s);
    int index_1 = -1, index_0 = -1, index_neg1 = -1;
    bool found = false;

    for (const auto& t : triplets) {
        if (std::get<0>(t) == start_1_index) {
            index_1 = std::get<0>(t);
            index_0 = std::get<1>(t);
            index_neg1 = std::get<2>(t);
            found = true;
            break;
        }
    }

    if (!found) return s;

    // Apply bend: 1 -> removed, 0 -> 1, -1 -> 0, append -1
    // Careful with indices shifting if we modify in place.
    // Construct new string manually.
    String result;
    for (int i = 0; i < s.size(); ++i) {
        if (i == index_1) continue; // Remove 1
        else if (i == index_0) result.push_back(1); // 0 becomes 1
        else if (i == index_neg1) result.push_back(0); // -1 becomes 0
        else result.push_back(s[i]);
    }
    result.push_back(-1); // Append -1
    return result;
}

String Sl3Hecke::bending_power(const String& s, int p) {
    String res = s;
    for (int i = 0; i < p; ++i) {
        res = bend_string(res);
    }
    return res;
}

String Sl3Hecke::inverse_bending_power(const String& s, int p) {
    return bending_power(s, s.size() - p);
}

String Sl3Hecke::generate_triplet_at_position(const String& s, int p) {
    String res;
    if (p > s.size()) {
        res = s;
        res.push_back(1); res.push_back(0); res.push_back(-1);
    } else {
        // Insert at p-1 (1-based index p)
        res.insert(res.end(), s.begin(), s.begin() + p - 1);
        res.push_back(1); res.push_back(0); res.push_back(-1);
        res.insert(res.end(), s.begin() + p - 1, s.end());
    }
    return res;
}

std::tuple<int, int, int> Sl3Hecke::find_last_triplet(const String& s) {
    // Find last 1
    int last_1_index = -1;
    for (int i = s.size() - 1; i >= 0; --i) {
        if (s[i] == 1) {
            last_1_index = i;
            break;
        }
    }
    if (last_1_index == -1) return {-1, -1, -1};

    int last_0_index = -1;
    for (int i = last_1_index + 1; i < s.size(); ++i) {
        if (s[i] == 0) {
            last_0_index = i;
            break;
        }
    }
    if (last_0_index == -1) return {-1, -1, -1};

    int last_neg1_index = -1;
    for (int i = last_0_index + 1; i < s.size(); ++i) {
        if (s[i] == -1) {
            last_neg1_index = i;
            break;
        }
    }
    if (last_neg1_index == -1) return {-1, -1, -1};

    return {last_1_index, last_0_index, last_neg1_index};
}

std::pair<String, std::vector<Sl3Hecke::Operation>> Sl3Hecke::string_decomposition(const String& s) {
    String s_decomposed = s;
    std::vector<Operation> operations;

    while (s_decomposed.size() > 6) {
        auto triplet = find_last_triplet(s_decomposed);
        int l1 = std::get<0>(triplet);
        int l0 = std::get<1>(triplet);
        int ln1 = std::get<2>(triplet);

        if (l1 == -1) break;

        if (l0 == l1 + 1 && ln1 == l0 + 1) {
            // Consecutive triplet: remove
            s_decomposed.erase(s_decomposed.begin() + ln1);
            s_decomposed.erase(s_decomposed.begin() + l0);
            s_decomposed.erase(s_decomposed.begin() + l1);
            operations.push_back({'t', l1 + 1}); // 1-based index
        } else {
            // Check -1s between 1 and 0
            bool all_neg1 = true;
            for (int i = l1 + 1; i < l0; ++i) {
                if (s_decomposed[i] != -1) { all_neg1 = false; break; }
            }

            if (all_neg1) {
                int start_neg1 = l1 + 1;
                int end_neg1 = l0 - 1;
                // Swap 1 and block of -1s
                std::swap(s_decomposed[l1], s_decomposed[end_neg1]); // Move 1 to end of block
                // Actually cycle: 1, -1, ..., -1 -> -1, ..., -1, 1
                // Wait, logic in Python:
                // s_decomposed[last_1_index], s_decomposed[end_neg1_string] = s_decomposed[end_neg1_string], s_decomposed[last_1_index]
                // This swaps the 1 with the last -1. The intermediate -1s stay put.
                // Result: -1, -1, ..., 1. Correct.

                for (int i = start_neg1; i <= end_neg1; ++i) {
                    operations.push_back({'e', i}); // 0-based in loop, but 'e' usually takes 1-based?
                    // Python code used: operations.append(('e', i)) where i comes from range(start, end+1).
                    // Indices in Python list are 0-based. The e function takes 1-based index?
                    // Let's check Python code e(S, i): bent_s = bending_power(s, i-1).
                    // So Python e function expects 1-based index i.
                    // The range loop gives 0-based index i.
                    // Python: for i in range(start_neg1_string, end_neg1_string + 1): operations.append(('e', i))
                    // This stores the index directly.
                    // When replaying: e(..., position).
                    // So we should store the index as is.
                }
            }

            // Check 0s between 0 and -1
            bool all_0 = true;
            for (int i = l0 + 1; i < ln1; ++i) {
                if (s_decomposed[i] != 0) { all_0 = false; break; }
            }

            if (all_0) {
                int start_0 = l0 + 1;
                int end_0 = ln1 - 1;
                // Swap -1 and block of 0s
                std::swap(s_decomposed[ln1], s_decomposed[start_0]);

                for (int i = end_0; i >= start_0; --i) { // decreasing order
                    operations.push_back({'e', i + 1}); // Python: i+1
                }
            }

            // If neither case matched (non-consecutive but not simple blocks), what happens?
            // Python code had "pass". The loop might get stuck or proceed to next iteration if string changes?
            // If string didn't change, we break to avoid infinite loop.
            if (!all_neg1 && !all_0) {
                 // The Python code relies on finding *some* operation.
                 // If we find a triplet but can't reduce it, we might be stuck.
                 // But for "valid strings" generated, maybe this case is covered?
                 // Let's assume logic covers valid cases.
                 break;
            }
        }
    }
    return {s_decomposed, operations};
}

// e action implementation
std::vector<std::pair<Complex, String>> Sl3Hecke::apply_e(const String& s, int i) {
    std::vector<std::pair<Complex, String>> results;

    // Apply e_i
    String bent_s = bending_power(s, i - 1);
    auto decomp = string_decomposition(bent_s);
    String s_decomposed = decomp.first;
    auto operations = decomp.second;

    // Check patterns
    // Convert String to vector for comparison
    // Patterns: [1, 0, 1, 0, -1, -1], [1, 0, 1, -1, 0, -1], [1, 0, -1, 1, 0, -1]
    bool pattern1 = (s_decomposed == String{1, 0, 1, 0, -1, -1});
    bool pattern2 = (s_decomposed == String{1, 0, 1, -1, 0, -1});
    bool pattern3 = (s_decomposed == String{1, 0, -1, 1, 0, -1});

    if (pattern1 || pattern2 || pattern3) {
        // Coeff * n
        results.push_back({n_val, s});
    } else if (s_decomposed == String{1, 1, 0, -1, 0, -1}) {
        // Jesper's conjecture case
        auto triplets = form_triplets_with_positions(bent_s);
        std::tuple<int, int, int> target_triplet = {-1, -1, -1};

        // Find second 1
        int first_1 = -1;
        for(int k=0; k<bent_s.size(); ++k) if(bent_s[k]==1) { first_1 = k; break; }

        int second_1 = -1;
        if(first_1 != -1) {
            for(int k=first_1+1; k<bent_s.size(); ++k) if(bent_s[k]==1) { second_1 = k; break; }
        }

        if (second_1 != -1) {
            for(const auto& t : triplets) {
                if (std::get<0>(t) == second_1) {
                    target_triplet = t;
                    break;
                }
            }
        }

        if (std::get<0>(target_triplet) != -1) {
            String s_jescon = bent_s;
            int idx1 = std::get<0>(target_triplet);
            int idx0 = std::get<1>(target_triplet);
            std::swap(s_jescon[idx1], s_jescon[idx0]);

            String res = inverse_bending_power(s_jescon, i - 1);
            results.push_back({1.0, res});
        }

    } else if (s_decomposed == String{1, 1, 0, 0, -1, -1}) {
        // Splitting case
        std::vector<String> new_strings = {
            {1, 0, -1, 1, 0, -1},
            {1, 0, 1, 0, -1, -1}
        };

        // Apply operations in reverse
        for (int k = operations.size() - 1; k >= 0; --k) {
            char type = operations[k].type;
            int pos = operations[k].pos;

            std::vector<String> next_strings;
            if (type == 't') {
                for (const auto& ns : new_strings) {
                    next_strings.push_back(generate_triplet_at_position(ns, pos));
                }
                new_strings = next_strings;
            } else if (type == 'e') {
                // Recursive call?
                // Need to apply e(ns, pos) and collect results
                // This is tricky because we are inside the class method but need to call it recursively
                // But `apply_e` is static logic essentially if n_val is passed? No, n_val is member.
                // We can call `this->apply_e`.
                // Note: coefficients propagate.
                // The coefficient for intermediate steps is 1.
                for (const auto& ns : new_strings) {
                    auto sub_results = this->apply_e(ns, pos);
                    for (const auto& p : sub_results) {
                        // Coefficient matters?
                        // In Python: `temp_input = [(1, new_s)]`, `temp_result = e(temp_input, ...)`
                        // `new_strings = [string for _, string in temp_result]`
                        // It seems coefficients are IGNORED in the recursive step for string generation?
                        // Let's check Python code:
                        // `new_strings = [string for _, string in temp_result]`
                        // Yes, coefficients are discarded/assumed 1 during the structural decomposition reverse application?
                        // Wait, "coefficients should still be 1". The splitting case generates new structural strings.
                        // The coefficient comes from the final result.
                        next_strings.push_back(p.second);
                    }
                }
                new_strings = next_strings;
            }
        }

        for (const auto& ns : new_strings) {
            String res = inverse_bending_power(ns, i - 1);
            results.push_back({1.0, res});
        }
    }

    return results;
}

// Arnoldi and Matrix Ops

SparseVector Sl3Hecke::apply_H(const SparseVector& v) {
    SparseVector w;
    int num_gen = 3 * L - 1;

    // Iterate over input vector elements
    for (const auto& pair : v) {
        const String& s = pair.first;
        Complex coeff = pair.second;

        // Sum e_k(s)
        for (int k = 1; k <= num_gen; ++k) {
            auto res = apply_e(s, k);
            for (const auto& r : res) {
                // r.first is coeff from e_k, r.second is string
                w[r.second] += coeff * r.first;
            }
        }
    }
    return w;
}

SparseVector Sl3Hecke::apply_T(const SparseVector& v) {
    SparseVector current = v;
    int num_gen = 3 * L - 1;

    // Odd generators
    for (int k = 1; k <= num_gen; k += 2) {
        SparseVector next;
        for (const auto& pair : current) {
            auto res = apply_e(pair.first, k);
            for (const auto& r : res) {
                next[r.second] += pair.second * r.first;
            }
        }
        current = next;
    }

    // Even generators
    for (int k = 2; k <= num_gen; k += 2) {
        SparseVector next;
        for (const auto& pair : current) {
            auto res = apply_e(pair.first, k);
            for (const auto& r : res) {
                next[r.second] += pair.second * r.first;
            }
        }
        current = next;
    }

    return current;
}

// Continuation of sl3_hecke.cpp

std::vector<std::vector<Complex>> Sl3Hecke::arnoldi(int k, int start_idx, char operator_type) {
    int dim = basis.size();
    if (k > dim) k = dim;

    // Q matrix (orthogonal vectors)
    // We store Q as a list of SparseVectors for efficiency in application?
    // Or dense vectors? Dimension can be large.
    // But Arnoldi vectors are usually dense.
    // Let's use dense vectors std::vector<Complex> for Q columns.

    std::vector<std::vector<Complex>> Q(k + 1, std::vector<Complex>(dim, 0.0));
    std::vector<std::vector<Complex>> h(k, std::vector<Complex>(k + 1, 0.0)); // h is (k+1) x k
    // Transposed storage for h usually easier: h[j][i]

    // Start vector
    if (start_idx < 0 || start_idx >= dim) start_idx = 0;
    Q[0][start_idx] = 1.0;

    for (int j = 0; j < k; ++j) {
        // Convert Q[j] to SparseVector for efficient application
        SparseVector v_sparse;
        for(int i=0; i<dim; ++i) {
            if (std::abs(Q[j][i]) > 1e-12) {
                v_sparse[basis[i]] = Q[j][i];
            }
        }

        SparseVector w_sparse;
        if (operator_type == 'H') {
            w_sparse = apply_H(v_sparse);
        } else {
            w_sparse = apply_T(v_sparse);
        }

        // Convert back to dense w
        std::vector<Complex> w(dim, 0.0);
        for(const auto& pair : w_sparse) {
            // Find index. This is slow if we look up every time.
            // We have basis_map.
            if (basis_map.count(pair.first)) {
                w[basis_map[pair.first]] = pair.second;
            }
        }

        // Orthogonalize (MGS)
        for (int i = 0; i <= j; ++i) {
            Complex dot = 0;
            for(int d=0; d<dim; ++d) dot += std::conj(Q[i][d]) * w[d];
            h[j][i] = dot;

            for(int d=0; d<dim; ++d) w[d] -= h[j][i] * Q[i][d];
        }

        double norm_w = 0;
        for(int d=0; d<dim; ++d) norm_w += std::norm(w[d]);
        norm_w = std::sqrt(norm_w);

        if (j + 1 < k + 1) h[j][j+1] = norm_w; // h[j][j+1] is valid if k+1 columns? No h is (k, k+1)
        // h[j] is vector of size k+1.

        if (norm_w < 1e-12) {
            // Arnoldi breakdown at step j.
            // We have a (j+1) x (j+1) Hessenberg matrix ideally (projected on j+1 vectors).
            // h is size k x (k+1).
            // We filled columns 0..j.
            // Row indices go up to j+1 (due to norm_w stored at h[j][j+1]).

            // Return square matrix (j+1) x (j+1)
            int sub_k = j + 1;
            std::vector<std::vector<Complex>> H_breakdown(sub_k, std::vector<Complex>(sub_k, 0.0));

            for(int row=0; row<sub_k; ++row) {
                for(int col=0; col<sub_k; ++col) {
                    // h stores columns. h[col] is vector.
                    // We need h[col][row].
                    if(col < k && row < h[col].size())
                        H_breakdown[row][col] = h[col][row];
                }
            }
            return H_breakdown;
        }

        if (j < k) { // if j=k-1, we don't need Q[k] unless we continue
             // if j < k-1 ? No, we need Q[j+1] for next step.
             // If j = k-1, we are done with loop.
             // But we usually compute h[k-1][k] (residual norm).
             if (j < k - 1)
                for(int d=0; d<dim; ++d) Q[j+1][d] = w[d] / norm_w;
        }
    }

    // Return square matrix k x k for eigenvalues?
    // h is vector<vector> of size k x (k+1).
    // Let's create a proper k x k square matrix to return.

    std::vector<std::vector<Complex>> H_square(k, std::vector<Complex>(k, 0.0));
    for(int j=0; j<k; ++j) {
        for(int i=0; i<k; ++i) {
            if (i < h[j].size()) H_square[j][i] = h[j][i];
        }
    }

    // We return H_square. But wait, if k > dim (breakdown handled inside), we might have smaller.
    // But here we are at full k.
    // Also, my h indexing is h[col][row].
    // Standard matrix indexing is M[row][col].
    // So H_square[row][col] should be h[col][row].

    std::vector<std::vector<Complex>> H_final(k, std::vector<Complex>(k, 0.0));
    for(int row=0; row<k; ++row) {
        for(int col=0; col<k; ++col) {
            H_final[row][col] = h[col][row];
        }
    }

    return H_final;
}
