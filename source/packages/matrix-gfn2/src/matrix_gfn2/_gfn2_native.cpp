#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <queue>
#include <tuple>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#elif defined(MATRIX_GFN2_NATIVE_LAPACK)
#include <cblas.h>
#endif

#if defined(MATRIX_GFN2_NATIVE_LAPACK) && !defined(__APPLE__)
extern "C" {
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda,
             int* info);
void dsygst_(const int* itype, const char* uplo, const int* n, double* a,
             const int* lda, double* b, const int* ldb, int* info);
void dsyevr_(const char* jobz, const char* range, const char* uplo,
             const int* n, double* a, const int* lda, const double* vl,
             const double* vu, const int* il, const int* iu,
             const double* abstol, int* m, double* w, double* z,
             const int* ldz, int* isuppz, double* work, const int* lwork,
             int* iwork, const int* liwork, int* info);
void dsyev_(const char* jobz, const char* uplo, const int* n, double* a,
            const int* lda, double* w, double* work, const int* lwork,
            int* info);
void dsyevd_(const char* jobz, const char* uplo, const int* n, double* a,
             const int* lda, double* w, double* work, const int* lwork,
             int* iwork, const int* liwork, int* info);
}
#endif

namespace {

template <typename Apply>
bool native_lowest_davidson_action(
    Apply&& apply, const std::vector<double>& diagonal,
    int dimension, int roots, double tolerance, int maximum_iterations,
    int maximum_subspace, std::vector<double>& eigenvalues,
    std::vector<double>& eigenvectors, int& operator_batches) {
    const int initial = roots;
    std::vector<int> order(dimension);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
        return diagonal[left] < diagonal[right];
    });
    std::vector<double> basis(static_cast<std::size_t>(dimension)*maximum_subspace, 0.0);
    std::vector<double> products(static_cast<std::size_t>(dimension)*maximum_subspace, 0.0);
    for (int column = 0; column < initial; ++column)
        basis[order[column] + column*dimension] = 1.0;
    apply(basis.data(), initial, products.data());
    operator_batches = 1;
    int columns = initial;
    eigenvalues.assign(roots, 0.0);
    eigenvectors.assign(static_cast<std::size_t>(dimension)*roots, 0.0);
    std::vector<double> ritz(static_cast<std::size_t>(dimension)*roots);
    std::vector<double> sigma(static_cast<std::size_t>(dimension)*roots);
    std::vector<double> residual(static_cast<std::size_t>(dimension)*roots);
    for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
        std::vector<double> projected(static_cast<std::size_t>(columns)*columns, 0.0);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
            columns, columns, dimension, 1.0, basis.data(), dimension,
            products.data(), dimension, 0.0, projected.data(), columns);
        for (int column = 0; column < columns; ++column)
            for (int row = 0; row < column; ++row) {
                const double average = 0.5*(projected[row + column*columns]
                    + projected[column + row*columns]);
                projected[row + column*columns] = average;
                projected[column + row*columns] = average;
            }
        std::vector<double> projected_values(columns);
        char jobz = 'V', uplo = 'L'; int info = 0, lwork = -1;
        double work_query = 0.0;
        dsyev_(&jobz, &uplo, &columns, projected.data(), &columns,
            projected_values.data(), &work_query, &lwork, &info);
        if (info != 0) return false;
        lwork = std::max(1, static_cast<int>(work_query));
        std::vector<double> work(lwork);
        dsyev_(&jobz, &uplo, &columns, projected.data(), &columns,
            projected_values.data(), work.data(), &lwork, &info);
        if (info != 0) return false;
        std::copy_n(projected_values.data(), roots, eigenvalues.data());
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            dimension, roots, columns, 1.0, basis.data(), dimension,
            projected.data(), columns, 0.0, ritz.data(), dimension);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            dimension, roots, columns, 1.0, products.data(), dimension,
            projected.data(), columns, 0.0, sigma.data(), dimension);
        bool converged = true;
        std::vector<double> corrections;
        corrections.reserve(static_cast<std::size_t>(dimension)*roots);
        int correction_count = 0;
        for (int root = 0; root < roots; ++root) {
            double norm_squared = 0.0;
            for (int row = 0; row < dimension; ++row) {
                const double value = sigma[row + root*dimension]
                    - eigenvalues[root]*ritz[row + root*dimension];
                residual[row + root*dimension] = value;
                norm_squared += value*value;
            }
            if (std::sqrt(norm_squared) <= tolerance) continue;
            converged = false;
            std::vector<double> trial(dimension);
            for (int row = 0; row < dimension; ++row) {
                double denominator = eigenvalues[root] - diagonal[row];
                if (std::abs(denominator) < 1.0e-10)
                    denominator = std::copysign(1.0e-10, denominator + 1.0e-300);
                trial[row] = residual[row + root*dimension]/denominator;
            }
            for (int column = 0; column < columns; ++column) {
                const double projection = cblas_ddot(
                    dimension, basis.data() + column*dimension, 1, trial.data(), 1);
                cblas_daxpy(dimension, -projection,
                    basis.data() + column*dimension, 1, trial.data(), 1);
            }
            for (int previous = 0; previous < correction_count; ++previous) {
                const auto* vector = corrections.data() + previous*dimension;
                const double projection = cblas_ddot(dimension, vector, 1, trial.data(), 1);
                cblas_daxpy(dimension, -projection, vector, 1, trial.data(), 1);
            }
            const double norm = cblas_dnrm2(dimension, trial.data(), 1);
            if (norm > 1.0e-11) {
                for (auto& value : trial) value /= norm;
                corrections.insert(corrections.end(), trial.begin(), trial.end());
                ++correction_count;
            }
        }
        if (converged) { eigenvectors = ritz; return true; }
        if (correction_count == 0) break;
        if (columns + correction_count > maximum_subspace) {
            std::copy(ritz.begin(), ritz.end(), basis.begin());
            columns = roots;
            apply(basis.data(), columns, products.data());
        } else {
            std::copy(corrections.begin(), corrections.end(),
                basis.begin() + static_cast<std::size_t>(columns)*dimension);
            apply(
                basis.data() + static_cast<std::size_t>(columns)*dimension,
                correction_count,
                products.data() + static_cast<std::size_t>(columns)*dimension);
            columns += correction_count;
        }
        ++operator_batches;
    }
    eigenvectors = ritz;
    return false;
}

bool native_lowest_davidson(
    const std::vector<double>& matrix, const std::vector<double>& diagonal,
    int dimension, int roots, double tolerance, int maximum_iterations,
    int maximum_subspace, std::vector<double>& eigenvalues,
    std::vector<double>& eigenvectors, int& operator_batches) {
    auto apply = [&](const double* input, int columns, double* output) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            dimension, columns, dimension, 1.0, matrix.data(), dimension,
            input, dimension, 0.0, output, dimension);
    };
    return native_lowest_davidson_action(
        apply, diagonal, dimension, roots, tolerance, maximum_iterations,
        maximum_subspace, eigenvalues, eigenvectors, operator_batches);
}

struct CellKey {
    std::int64_t x, y, z;
    bool operator==(const CellKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct CellHash {
    std::size_t operator()(const CellKey& key) const {
        std::size_t value = std::hash<std::int64_t>{}(key.x);
        value ^= std::hash<std::int64_t>{}(key.y) + 0x9e3779b9 + (value<<6) + (value>>2);
        value ^= std::hash<std::int64_t>{}(key.z) + 0x9e3779b9 + (value<<6) + (value>>2);
        return value;
    }
};

struct SparseMatrixNative {
    npy_intp size = 0;
    std::vector<std::int32_t> rows;
    std::vector<std::int32_t> columns;
    std::vector<double> values;
};

SparseMatrixNative sparse_scaled_add_native(
    const SparseMatrixNative& left, const SparseMatrixNative& right,
    double left_scale, double right_scale, double threshold, double& discarded) {
    SparseMatrixNative result;
    result.size = left.size;
    std::size_t i = 0, j = 0;
    auto key_less = [](std::int64_t ar, std::int64_t ac, std::int64_t br, std::int64_t bc) {
        return ar < br || (ar == br && ac < bc);
    };
    discarded = 0.0;
    while (i < left.values.size() || j < right.values.size()) {
        std::int64_t row, column;
        double value;
        if (j >= right.values.size() || (i < left.values.size() &&
            key_less(left.rows[i], left.columns[i], right.rows[j], right.columns[j]))) {
            row = left.rows[i]; column = left.columns[i]; value = left_scale * left.values[i++];
        } else if (i >= left.values.size() ||
            key_less(right.rows[j], right.columns[j], left.rows[i], left.columns[i])) {
            row = right.rows[j]; column = right.columns[j]; value = right_scale * right.values[j++];
        } else {
            row = left.rows[i]; column = left.columns[i];
            value = left_scale * left.values[i++] + right_scale * right.values[j++];
        }
        if (row == column || std::abs(value) >= threshold) {
            result.rows.push_back(row); result.columns.push_back(column); result.values.push_back(value);
        } else {
            discarded += (row == column ? 1.0 : 2.0) * value * value;
        }
    }
    discarded = std::sqrt(discarded);
    return result;
}

template <typename Index>
SparseMatrixNative sparse_square_native(
    const SparseMatrixNative& matrix,
    const Index* allowed_left, const Index* allowed_right,
    npy_intp allowed_count, double threshold, double& discarded) {
    using Entry = std::pair<std::int64_t, double>;
    std::vector<std::vector<Entry>> adjacency(matrix.size);
    for (std::size_t index = 0; index < matrix.values.size(); ++index) {
        const auto row = matrix.rows[index], column = matrix.columns[index];
        const double value = matrix.values[index];
        adjacency[row].push_back({column, value});
        if (row != column) adjacency[column].push_back({row, value});
    }
    for (auto& row : adjacency) {
        std::sort(row.begin(), row.end(), [](const Entry& a, const Entry& b) {
            return a.first < b.first;
        });
    }
    SparseMatrixNative result;
    result.size = matrix.size;
    result.rows.reserve(allowed_count); result.columns.reserve(allowed_count);
    result.values.reserve(allowed_count);
    double discarded_squared = 0.0;
    for (npy_intp pair = 0; pair < allowed_count; ++pair) {
        const auto left = allowed_left[pair], right = allowed_right[pair];
        const auto& a = adjacency[left]; const auto& b = adjacency[right];
        std::size_t ia = 0, ib = 0;
        double value = 0.0;
        while (ia < a.size() && ib < b.size()) {
            if (a[ia].first < b[ib].first) ++ia;
            else if (b[ib].first < a[ia].first) ++ib;
            else { value += a[ia].second * b[ib].second; ++ia; ++ib; }
        }
        if (left == right || std::abs(value) >= threshold) {
            result.rows.push_back(left); result.columns.push_back(right); result.values.push_back(value);
        } else {
            discarded_squared += (left == right ? 1.0 : 2.0) * value * value;
        }
    }
    discarded = std::sqrt(discarded_squared);
    return result;
}

SparseMatrixNative sparse_product_native(
    const SparseMatrixNative& left_matrix,
    const SparseMatrixNative& right_matrix,
    const std::int32_t* allowed_left, const std::int32_t* allowed_right,
    npy_intp allowed_count, double threshold, double& discarded,
    int native_threads = 1) {
    using Entry = std::pair<std::int64_t, double>;
    std::vector<std::vector<Entry>> left_rows(left_matrix.size);
    std::vector<std::vector<Entry>> right_rows(right_matrix.size);
    auto expand = [](const SparseMatrixNative& matrix,
                     std::vector<std::vector<Entry>>& rows) {
        for (std::size_t index = 0; index < matrix.values.size(); ++index) {
            const auto row = matrix.rows[index], column = matrix.columns[index];
            const double value = matrix.values[index];
            rows[row].push_back({column, value});
            if (row != column) rows[column].push_back({row, value});
        }
        for (auto& row : rows) {
            std::sort(row.begin(), row.end(), [](const Entry& a, const Entry& b) {
                return a.first < b.first;
            });
        }
    };
    expand(left_matrix, left_rows);
    expand(right_matrix, right_rows);
    auto dot = [](const std::vector<Entry>& a, const std::vector<Entry>& b) {
        std::size_t ia = 0, ib = 0;
        double value = 0.0;
        while (ia < a.size() && ib < b.size()) {
            if (a[ia].first < b[ib].first) ++ia;
            else if (b[ib].first < a[ia].first) ++ib;
            else { value += a[ia].second * b[ib].second; ++ia; ++ib; }
        }
        return value;
    };
    native_threads = std::max(1, std::min(
        native_threads, static_cast<int>(std::max<npy_intp>(1, allowed_count))));
    std::vector<SparseMatrixNative> partial(native_threads);
    std::vector<double> discarded_squared(native_threads, 0.0);
    const npy_intp chunk =
        (allowed_count + native_threads - 1) / native_threads;
    auto evaluate_range = [&](int thread) {
        const npy_intp begin = static_cast<npy_intp>(thread) * chunk;
        const npy_intp end = std::min(allowed_count, begin + chunk);
        auto& local = partial[thread];
        local.size = left_matrix.size;
        if (begin >= end) return;
        const auto reserve_count = static_cast<std::size_t>(end - begin);
        local.rows.reserve(reserve_count);
        local.columns.reserve(reserve_count);
        local.values.reserve(reserve_count);
        double local_discarded = 0.0;
        for (npy_intp pair = begin; pair < end; ++pair) {
            const auto left = allowed_left[pair], right = allowed_right[pair];
            // Newton--Schulz operates on commuting symmetric matrix
            // polynomials.  Truncation can introduce a small commutator, so
            // retain the symmetric Jordan product.
            const double value = 0.5 * (
                dot(left_rows[left], right_rows[right])
                + dot(right_rows[left], left_rows[right]));
            if (left == right || std::abs(value) >= threshold) {
                local.rows.push_back(left);
                local.columns.push_back(right);
                local.values.push_back(value);
            } else {
                local_discarded += (left == right ? 1.0 : 2.0) * value * value;
            }
        }
        discarded_squared[thread] = local_discarded;
    };
    std::vector<std::thread> workers;
    workers.reserve(native_threads > 1 ? native_threads - 1 : 0);
    for (int thread = 1; thread < native_threads; ++thread)
        workers.emplace_back(evaluate_range, thread);
    evaluate_range(0);
    for (auto& worker : workers) worker.join();

    SparseMatrixNative result;
    result.size = left_matrix.size;
    std::size_t result_count = 0;
    for (const auto& local : partial) result_count += local.values.size();
    result.rows.reserve(result_count);
    result.columns.reserve(result_count);
    result.values.reserve(result_count);
    double total_discarded_squared = 0.0;
    for (int thread = 0; thread < native_threads; ++thread) {
        auto& local = partial[thread];
        result.rows.insert(result.rows.end(), local.rows.begin(), local.rows.end());
        result.columns.insert(
            result.columns.end(), local.columns.begin(), local.columns.end());
        result.values.insert(
            result.values.end(), local.values.begin(), local.values.end());
        total_discarded_squared += discarded_squared[thread];
    }
    discarded = std::sqrt(total_discarded_squared);
    return result;
}

struct SparseCsrNative {
    std::vector<std::int64_t> offsets;
    std::vector<std::int32_t> columns;
    std::vector<double> values;
};

SparseCsrNative symmetric_csr_native(
    const SparseMatrixNative& matrix,
    const std::int64_t* topology_offsets = nullptr,
    const std::int32_t* topology_neighbors = nullptr,
    const std::int32_t* topology_pairs = nullptr) {
    SparseCsrNative csr;
    csr.offsets.assign(matrix.size + 1, 0);
    if (topology_offsets && topology_neighbors && topology_pairs) {
        csr.offsets.assign(topology_offsets, topology_offsets + matrix.size + 1);
        const auto count = csr.offsets.back();
        csr.columns.assign(topology_neighbors, topology_neighbors + count);
        csr.values.resize(count);
        for (std::int64_t index = 0; index < count; ++index)
            csr.values[index] = matrix.values[topology_pairs[index]];
        return csr;
    }
    for (std::size_t index = 0; index < matrix.values.size(); ++index) {
        ++csr.offsets[matrix.rows[index] + 1];
        if (matrix.rows[index] != matrix.columns[index])
            ++csr.offsets[matrix.columns[index] + 1];
    }
    for (npy_intp row = 0; row < matrix.size; ++row)
        csr.offsets[row + 1] += csr.offsets[row];
    csr.columns.resize(csr.offsets.back());
    csr.values.resize(csr.offsets.back());
    auto cursor = csr.offsets;
    for (std::size_t index = 0; index < matrix.values.size(); ++index) {
        const auto row = matrix.rows[index], column = matrix.columns[index];
        auto target = cursor[row]++;
        csr.columns[target] = column; csr.values[target] = matrix.values[index];
        if (row != column) {
            target = cursor[column]++;
            csr.columns[target] = row; csr.values[target] = matrix.values[index];
        }
    }
    std::vector<std::pair<std::int32_t, double>> entries;
    for (npy_intp row = 0; row < matrix.size; ++row) {
        entries.clear();
        for (auto index = csr.offsets[row]; index < csr.offsets[row + 1]; ++index)
            entries.push_back({csr.columns[index], csr.values[index]});
        std::sort(entries.begin(), entries.end());
        for (std::size_t local = 0; local < entries.size(); ++local) {
            const auto index = csr.offsets[row] + static_cast<std::int64_t>(local);
            csr.columns[index] = entries[local].first;
            csr.values[index] = entries[local].second;
        }
    }
    return csr;
}

struct CongruenceWorkspace {
    std::vector<std::int64_t> pair_offsets;
    std::vector<std::int32_t> pair_indices;
    std::vector<double> pair_values;
    std::vector<double> accumulator;
    std::vector<std::int64_t> marks;
    std::vector<std::int32_t> touched;
    std::vector<std::int64_t> block_offsets;
    std::vector<std::int32_t> block_columns;
    std::vector<double> block_values;
};

npy_intp congruence_block_size(npy_intp size, npy_intp pair_count) {
    if (const char* configured = std::getenv("MATRIX_GFN2_CONGRUENCE_BLOCK")) {
        const auto value = std::strtoll(configured, nullptr, 10);
        if (value > 0) return std::min<npy_intp>(size, value);
    }
    const double pairs_per_row = static_cast<double>(pair_count)
        / std::max<npy_intp>(size, 1);
    if (size <= 1024 || pairs_per_row <= 32.0) return size;
    if (size <= 4096) return 1024;
    if (size <= 16384) return 512;
    return 256;
}

SparseMatrixNative sparse_congruence_native(
    const SparseMatrixNative& transformation,
    const SparseMatrixNative& matrix,
    const std::int32_t* allowed_left, const std::int32_t* allowed_right,
    npy_intp allowed_count, double threshold, double& discarded,
    const std::int64_t* topology_offsets = nullptr,
    const std::int32_t* topology_neighbors = nullptr,
    const std::int32_t* topology_pairs = nullptr,
    int native_threads = 1) {
    const auto transform_csr = symmetric_csr_native(
        transformation, topology_offsets, topology_neighbors, topology_pairs);
    const auto matrix_csr = symmetric_csr_native(matrix);
    static thread_local CongruenceWorkspace arena;
    arena.pair_offsets.assign(transformation.size + 1, 0);
    for (npy_intp pair = 0; pair < allowed_count; ++pair)
        ++arena.pair_offsets[allowed_left[pair] + 1];
    for (npy_intp row = 0; row < transformation.size; ++row)
        arena.pair_offsets[row + 1] += arena.pair_offsets[row];
    arena.pair_indices.resize(allowed_count);
    auto pair_cursor = arena.pair_offsets;
    for (npy_intp pair = 0; pair < allowed_count; ++pair)
        arena.pair_indices[pair_cursor[allowed_left[pair]]++] = pair;
    arena.pair_values.assign(allowed_count, 0.0);
    const auto& pair_offsets = arena.pair_offsets;
    const auto& pair_indices = arena.pair_indices;
    auto& pair_values = arena.pair_values;
    const auto row_block_size = congruence_block_size(
        transformation.size, allowed_count);
    std::atomic<npy_intp> next_block{0};
    auto process_blocks = [&](CongruenceWorkspace& workspace) {
        workspace.accumulator.assign(transformation.size, 0.0);
        workspace.marks.assign(transformation.size, -1);
        while (true) {
            const auto block_begin = next_block.fetch_add(row_block_size);
            if (block_begin >= transformation.size) break;
            const auto block_end = std::min(
                block_begin + row_block_size, transformation.size);
            workspace.block_offsets.assign(block_end - block_begin + 1, 0);
            workspace.block_columns.clear(); workspace.block_values.clear();
            for (npy_intp row = block_begin; row < block_end; ++row) {
                workspace.touched.clear();
                for (auto xi = transform_csr.offsets[row];
                     xi < transform_csr.offsets[row + 1]; ++xi) {
                    const auto k = transform_csr.columns[xi];
                    const auto xik = transform_csr.values[xi];
                    for (auto ai = matrix_csr.offsets[k];
                         ai < matrix_csr.offsets[k + 1]; ++ai) {
                        const auto column = matrix_csr.columns[ai];
                        if (workspace.marks[column] != row) {
                            workspace.marks[column] = row;
                            workspace.touched.push_back(column);
                        }
                        workspace.accumulator[column] += xik * matrix_csr.values[ai];
                    }
                }
                std::sort(workspace.touched.begin(), workspace.touched.end());
                for (const auto column : workspace.touched) {
                    if (workspace.accumulator[column] != 0.0) {
                        workspace.block_columns.push_back(column);
                        workspace.block_values.push_back(workspace.accumulator[column]);
                    }
                    workspace.accumulator[column] = 0.0;
                }
                workspace.block_offsets[row - block_begin + 1] =
                    static_cast<std::int64_t>(workspace.block_columns.size());
                for (auto pi = pair_offsets[row];
                     pi < pair_offsets[row + 1]; ++pi) {
                    const auto pair = pair_indices[pi];
                    const auto right = allowed_right[pair];
                    auto bi = workspace.block_offsets[row - block_begin];
                    const auto bi_end = workspace.block_offsets[row - block_begin + 1];
                    auto xi = transform_csr.offsets[right];
                    const auto xi_end = transform_csr.offsets[right + 1];
                    double value = 0.0;
                    while (bi < bi_end && xi < xi_end) {
                        const auto bc = workspace.block_columns[bi];
                        const auto xc = transform_csr.columns[xi];
                        if (bc < xc) ++bi;
                        else if (xc < bc) ++xi;
                        else {
                            value += workspace.block_values[bi]
                                * transform_csr.values[xi];
                            ++bi; ++xi;
                        }
                    }
                    pair_values[pair] = value;
                }
            }
        }
    };
    native_threads = std::max(1, std::min(native_threads,
        static_cast<int>((transformation.size + row_block_size - 1) / row_block_size)));
    std::vector<std::thread> workers;
    workers.reserve(native_threads > 1 ? native_threads - 1 : 0);
    for (int thread = 1; thread < native_threads; ++thread)
        workers.emplace_back([&]() {
            CongruenceWorkspace local;
            process_blocks(local);
        });
    process_blocks(arena);
    for (auto& worker : workers) worker.join();
    SparseMatrixNative result;result.size=matrix.size;
    result.rows.reserve(allowed_count);result.columns.reserve(allowed_count);result.values.reserve(allowed_count);
    double discarded_squared=0.0;
    for(npy_intp pair=0;pair<allowed_count;++pair){
        const auto left=allowed_left[pair],right=allowed_right[pair];
        const double value = pair_values[pair];
        if(left==right||std::abs(value)>=threshold){
            result.rows.push_back(left);result.columns.push_back(right);result.values.push_back(value);
        }else discarded_squared+=(left==right?1.0:2.0)*value*value;
    }
    discarded=std::sqrt(discarded_squared);return result;
}

PyObject* sparse_arrays(const SparseMatrixNative& matrix) {
    npy_intp dims[1] = {static_cast<npy_intp>(matrix.values.size())};
    auto* rows = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, dims, NPY_INT32, 0));
    auto* columns = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, dims, NPY_INT32, 0));
    auto* values = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, dims, NPY_DOUBLE, 0));
    if (!rows || !columns || !values) {
        Py_XDECREF(rows); Py_XDECREF(columns); Py_XDECREF(values); return PyErr_NoMemory();
    }
    std::copy(matrix.rows.begin(), matrix.rows.end(), static_cast<std::int32_t*>(PyArray_DATA(rows)));
    std::copy(matrix.columns.begin(), matrix.columns.end(), static_cast<std::int32_t*>(PyArray_DATA(columns)));
    std::copy(matrix.values.begin(), matrix.values.end(), static_cast<double*>(PyArray_DATA(values)));
    return Py_BuildValue("NNN", rows, columns, values);
}

int odd_double_factorial(int value) {
    int result = 1;
    for (; value > 0; value -= 2) result *= value;
    return result;
}

double primitive_normalization(double exponent, const std::int64_t* angular) {
    const int sum = static_cast<int>(angular[0] + angular[1] + angular[2]);
    const double denominator = std::sqrt(
        static_cast<double>(odd_double_factorial(2 * angular[0] - 1))
        * odd_double_factorial(2 * angular[1] - 1)
        * odd_double_factorial(2 * angular[2] - 1));
    return std::pow(2.0 * exponent / M_PI, 0.75)
        * std::pow(4.0 * exponent, 0.5 * sum) / denominator;
}

[[maybe_unused]] void axis_moment_bundle(
    int lp, int rp, double lc, double rc, double la, double ra,
    double relative, double* base, double* raised, double* lowered) {
    // Shifted second-derivative recurrences can raise d functions twice;
    // retain the full workspace for this general helper.  The hot first-
    // derivative path uses the bounded helper below.
    double table[8][8];
    const double total = la + ra;
    const double product = (la*lc + ra*rc)/total;
    const int maximum_left = lp + 3;
    // The common three-dimensional Gaussian product exponential is applied
    // once by the caller.  Every recurrence entry on one axis shares it.
    table[0][0] = std::sqrt(M_PI/total);
    for (int left = 1; left <= maximum_left; ++left) {
        table[left][0] = (product-lc)*table[left-1][0];
        if (left > 1)
            table[left][0] += (left-1)*table[left-2][0]/(2.0*total);
    }
    for (int right = 1; right <= rp; ++right) {
        table[0][right] = (product-rc)*table[0][right-1];
        if (right > 1)
            table[0][right] += (right-1)*table[0][right-2]/(2.0*total);
    }
    for (int left = 1; left <= maximum_left; ++left) {
        for (int right = 1; right <= rp; ++right) {
            table[left][right] = (product-rc)*table[left][right-1]
                + left*table[left-1][right-1]/(2.0*total);
            if (right > 1)
                table[left][right] += (right-1)*table[left][right-2]/(2.0*total);
        }
    }
    const auto moments = [&](int left, double* output) {
        const double s0 = table[left][rp];
        const double s1 = table[left+1][rp];
        const double s2 = table[left+2][rp];
        output[0] = s0;
        output[1] = relative*s0 + s1;
        output[2] = relative*relative*s0 + 2.0*relative*s1 + s2;
    };
    moments(lp, base);
    moments(lp+1, raised);
    if (lp > 0) moments(lp-1, lowered);
    else std::fill(lowered, lowered+3, 0.0);
}

void axis_moment_bundle_first_derivative(
    int lp, int rp, double lc, double rc, double la, double ra,
    double relative, double* base, double* raised, double* lowered) {
    // Unshifted GFN2 orbitals stop at d functions.  First derivatives need
    // at most l+3=5 on the left and l=2 on the right, so only 18 recurrence
    // entries are reachable in the energy/gradient hot path.
    // Every reachable entry is assigned by the recurrences below before it
    // is read.  Avoid zero-filling the complete stack workspace for each
    // Cartesian axis and primitive pair: that fixed initialization was a
    // measurable fraction of the H/C/N/O molecular integral path.
    const double total = la + ra;
    const double product = (la*lc + ra*rc)/total;
    const int maximum_left = lp + 3;
    double table[6][3];
    table[0][0] = std::sqrt(M_PI/total);
    for (int left = 1; left <= maximum_left; ++left) {
        table[left][0] = (product-lc)*table[left-1][0];
        if (left > 1)
            table[left][0] += (left-1)*table[left-2][0]/(2.0*total);
    }
    for (int right = 1; right <= rp; ++right) {
        table[0][right] = (product-rc)*table[0][right-1];
        if (right > 1)
            table[0][right] += (right-1)*table[0][right-2]/(2.0*total);
    }
    for (int left = 1; left <= maximum_left; ++left) {
        for (int right = 1; right <= rp; ++right) {
            table[left][right] = (product-rc)*table[left][right-1]
                + left*table[left-1][right-1]/(2.0*total);
            if (right > 1)
                table[left][right] += (right-1)*table[left][right-2]/(2.0*total);
        }
    }
    const auto moments = [&](int left, double* output) {
        const double s0 = table[left][rp];
        const double s1 = table[left+1][rp];
        const double s2 = table[left+2][rp];
        output[0] = s0;
        output[1] = relative*s0 + s1;
        output[2] = relative*relative*s0 + 2.0*relative*s1 + s2;
    };
    moments(lp, base);
    moments(lp+1, raised);
    if (lp > 0) moments(lp-1, lowered);
    else std::fill(lowered, lowered+3, 0.0);
}

struct ShiftedAxisMomentCache {
    // Two nuclear derivatives can shift either endpoint by at most two powers.
    // Cache the one-dimensional recurrences once per primitive pair, instead
    // of recomputing them for every moment and Cartesian derivative component.
    double values[3][5][5][3] = {};

    ShiftedAxisMomentCache(
        const std::int64_t* left_angular, const std::int64_t* right_angular,
        const double* left_center, const double* right_center,
        double left_exponent, double right_exponent, const double* origin) {
        for (int axis = 0; axis < 3; ++axis) {
            // All shifted moments on this axis share one Hermite table.
            // Computing a separate recurrence for each of the 25 possible
            // endpoint-shift pairs dominated the large-system Hessian action.
            const double total = left_exponent + right_exponent;
            const double product = (
                left_exponent*left_center[axis]
                + right_exponent*right_center[axis])/total;
            const int maximum_left = static_cast<int>(left_angular[axis]) + 4;
            const int maximum_right = static_cast<int>(right_angular[axis]) + 2;
            double table[8][8];
            table[0][0] = std::sqrt(M_PI/total);
            for (int left = 1; left <= maximum_left; ++left) {
                table[left][0] = (product-left_center[axis])*table[left-1][0];
                if (left > 1)
                    table[left][0] += (left-1)*table[left-2][0]/(2.0*total);
            }
            for (int right = 1; right <= maximum_right; ++right) {
                table[0][right] = (product-right_center[axis])*table[0][right-1];
                if (right > 1)
                    table[0][right] += (right-1)*table[0][right-2]/(2.0*total);
            }
            for (int left = 1; left <= maximum_left; ++left) {
                for (int right = 1; right <= maximum_right; ++right) {
                    table[left][right] = (product-right_center[axis])
                        * table[left][right-1]
                        + left*table[left-1][right-1]/(2.0*total);
                    if (right > 1)
                        table[left][right] += (right-1)*table[left][right-2]
                            /(2.0*total);
                }
            }
            const double relative = left_center[axis] - origin[axis];
            for (int ls = -2; ls <= 2; ++ls) {
                const int lp = static_cast<int>(left_angular[axis]) + ls;
                if (lp < 0) continue;
                for (int rs = -2; rs <= 2; ++rs) {
                    const int rp = static_cast<int>(right_angular[axis]) + rs;
                    if (rp < 0) continue;
                    const double s0 = table[lp][rp];
                    const double s1 = table[lp+1][rp];
                    const double s2 = table[lp+2][rp];
                    values[axis][ls+2][rs+2][0] = s0;
                    values[axis][ls+2][rs+2][1] = relative*s0 + s1;
                    values[axis][ls+2][rs+2][2] = relative*relative*s0
                        + 2.0*relative*s1 + s2;
                }
            }
        }
    }
};

struct PrimitiveMomentDerivatives {
    double base[3][3] = {};
    double first[2][3][3] = {};
    double second[2][2][3][3] = {};

    PrimitiveMomentDerivatives(
        const ShiftedAxisMomentCache& cache,
        const std::int64_t* left_angular, const std::int64_t* right_angular,
        double left_exponent, double right_exponent) {
        for (int axis = 0; axis < 3; ++axis) {
            const double l = static_cast<double>(left_angular[axis]);
            const double r = static_cast<double>(right_angular[axis]);
            const auto integral = [&](int ls, int rs, int moment) {
                return cache.values[axis][ls+2][rs+2][moment];
            };
            for (int moment = 0; moment < 3; ++moment) {
                const double central = integral(0, 0, moment);
                base[axis][moment] = central;
                first[0][axis][moment] = 2.0*left_exponent
                    * integral(1, 0, moment) - l*integral(-1, 0, moment);
                first[1][axis][moment] = 2.0*right_exponent
                    * integral(0, 1, moment) - r*integral(0, -1, moment);
                second[0][0][axis][moment] = 4.0*left_exponent*left_exponent
                    * integral(2, 0, moment)
                    - 2.0*left_exponent*(2.0*l+1.0)*central
                    + l*(l-1.0)*integral(-2, 0, moment);
                second[1][1][axis][moment] = 4.0*right_exponent*right_exponent
                    * integral(0, 2, moment)
                    - 2.0*right_exponent*(2.0*r+1.0)*central
                    + r*(r-1.0)*integral(0, -2, moment);
                const double mixed = 4.0*left_exponent*right_exponent
                    * integral(1, 1, moment)
                    - 2.0*left_exponent*r*integral(1, -1, moment)
                    - 2.0*right_exponent*l*integral(-1, 1, moment)
                    + l*r*integral(-1, -1, moment);
                second[0][1][axis][moment] = mixed;
                second[1][0][axis][moment] = mixed;
            }
        }
    }
};

double primitive_moment_second_derivative(
    const PrimitiveMomentDerivatives& derivatives, const int* moment,
    int first_endpoint, int first_axis, int second_endpoint,
    int second_axis) {
    double value = 1.0;
    for (int axis = 0; axis < 3; ++axis) {
        if (axis == first_axis && axis == second_axis)
            value *= derivatives.second[first_endpoint][second_endpoint]
                [axis][moment[axis]];
        else if (axis == first_axis)
            value *= derivatives.first[first_endpoint][axis][moment[axis]];
        else if (axis == second_axis)
            value *= derivatives.first[second_endpoint][axis][moment[axis]];
        else
            value *= derivatives.base[axis][moment[axis]];
    }
    return value;
}

PyObject* multipole_pair_batch(PyObject*, PyObject* args) {
    PyObject *centers_o, *angular_o, *exponents_o, *coefficients_o;
    PyObject *counts_o, *left_o, *right_o, *norms_o, *origin_o;
    int native_threads = 1;
    int normalized_coefficients = 0;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOOO|ii", &centers_o, &angular_o, &exponents_o,
            &coefficients_o, &counts_o, &left_o, &right_o, &norms_o, &origin_o,
            &native_threads, &normalized_coefficients)) {
        return nullptr;
    }
    auto* centers = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(centers_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* angular = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(angular_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* exponents = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(exponents_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* coefficients = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(coefficients_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* counts = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(counts_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* pair_left = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(left_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    auto* pair_right = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(right_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    auto* norms = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(norms_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* origin = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(origin_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!centers || !angular || !exponents || !coefficients || !counts ||
        !pair_left || !pair_right || !norms || !origin) {
        Py_XDECREF(centers); Py_XDECREF(angular); Py_XDECREF(exponents);
        Py_XDECREF(coefficients); Py_XDECREF(counts); Py_XDECREF(pair_left);
        Py_XDECREF(pair_right); Py_XDECREF(norms); Py_XDECREF(origin);
        return nullptr;
    }
    const npy_intp pair_count = PyArray_SIZE(pair_left);
    npy_intp values_dims[2] = {pair_count, 13};
    npy_intp derivatives_dims[3] = {pair_count, 13, 3};
    auto* values = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2, values_dims, NPY_DOUBLE, 0));
    auto* dl = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(3, derivatives_dims, NPY_DOUBLE, 0));
    auto* dr = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(3, derivatives_dims, NPY_DOUBLE, 0));
    if (!values || !dl || !dr) {
        Py_XDECREF(values); Py_XDECREF(dl); Py_XDECREF(dr);
        Py_DECREF(centers); Py_DECREF(angular); Py_DECREF(exponents);
        Py_DECREF(coefficients); Py_DECREF(counts); Py_DECREF(pair_left);
        Py_DECREF(pair_right); Py_DECREF(norms); Py_DECREF(origin);
        return PyErr_NoMemory();
    }
    const auto* c = static_cast<const double*>(PyArray_DATA(centers));
    const auto* a = static_cast<const std::int64_t*>(PyArray_DATA(angular));
    const auto* e = static_cast<const double*>(PyArray_DATA(exponents));
    const auto* coef = static_cast<const double*>(PyArray_DATA(coefficients));
    const auto* nprim = static_cast<const std::int64_t*>(PyArray_DATA(counts));
    const auto* pl = static_cast<const std::int32_t*>(PyArray_DATA(pair_left));
    const auto* pr = static_cast<const std::int32_t*>(PyArray_DATA(pair_right));
    const auto* norm = static_cast<const double*>(PyArray_DATA(norms));
    const auto* org = static_cast<const double*>(PyArray_DATA(origin));
    auto* out = static_cast<double*>(PyArray_DATA(values));
    auto* out_dl = static_cast<double*>(PyArray_DATA(dl));
    auto* out_dr = static_cast<double*>(PyArray_DATA(dr));
    const npy_intp max_primitives = PyArray_DIM(exponents, 1);
    const npy_intp orbital_count = PyArray_DIM(angular, 0);
    std::vector<npy_intp> shell_id(static_cast<std::size_t>(orbital_count));
    std::vector<npy_intp> shell_representative;
    shell_representative.reserve(static_cast<std::size_t>(orbital_count));
    for (npy_intp orbital = 0; orbital < orbital_count; ++orbital) {
        bool same_shell = orbital > 0 && nprim[orbital] == nprim[orbital-1];
        for (int axis = 0; same_shell && axis < 3; ++axis)
            same_shell = c[3*orbital+axis] == c[3*(orbital-1)+axis];
        for (npy_intp primitive = 0;
                same_shell && primitive < nprim[orbital]; ++primitive) {
            same_shell = e[orbital*max_primitives+primitive]
                    == e[(orbital-1)*max_primitives+primitive]
                && coef[orbital*max_primitives+primitive]
                    == coef[(orbital-1)*max_primitives+primitive];
        }
        if (!same_shell) shell_representative.push_back(orbital);
        shell_id[orbital] = static_cast<npy_intp>(shell_representative.size()-1);
    }
    const npy_intp shell_count = static_cast<npy_intp>(shell_representative.size());
    const std::size_t primitive_stride = static_cast<std::size_t>(
        max_primitives*max_primitives);
    // Retain the large shell-pair workspaces on the calling thread.  Their
    // contents are overwritten for every geometry, while capacity survives
    // resident trajectory calls and avoids repeated multi-megabyte allocation.
    thread_local std::vector<double> shell_pair_prefactors;
    thread_local std::vector<double> shell_pair_axis_tables;
    thread_local std::vector<std::size_t> shell_pair_index;
    if (normalized_coefficients) {
        std::unordered_map<std::uint64_t, std::size_t> shell_pair_slots;
        shell_pair_slots.reserve(static_cast<std::size_t>(pair_count));
        std::vector<std::pair<npy_intp, npy_intp>> unique_shell_pairs;
        unique_shell_pairs.reserve(static_cast<std::size_t>(pair_count));
        shell_pair_index.resize(static_cast<std::size_t>(pair_count));
        for (npy_intp pair = 0; pair < pair_count; ++pair) {
            const auto left_shell = shell_id[pl[pair]];
            const auto right_shell = shell_id[pr[pair]];
            const auto key = static_cast<std::uint64_t>(left_shell)
                *static_cast<std::uint64_t>(shell_count)
                +static_cast<std::uint64_t>(right_shell);
            const auto slot = unique_shell_pairs.size();
            const auto [found, inserted] = shell_pair_slots.emplace(key, slot);
            if (inserted)
                unique_shell_pairs.emplace_back(
                    shell_representative[left_shell],
                    shell_representative[right_shell]);
            shell_pair_index[static_cast<std::size_t>(pair)] = found->second;
        }
        const std::size_t unique_pair_count = unique_shell_pairs.size();
        shell_pair_prefactors.resize(unique_pair_count*primitive_stride);
        constexpr std::size_t axis_table_stride = 3*5*2;
        shell_pair_axis_tables.resize(
            unique_pair_count*primitive_stride*axis_table_stride);
        for (std::size_t shell_pair = 0;
                shell_pair < unique_pair_count; ++shell_pair) {
            const auto [left, right] = unique_shell_pairs[shell_pair];
            double center_distance2 = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                const double delta = c[3*left+axis]-c[3*right+axis];
                center_distance2 += delta*delta;
            }
            auto* cached = shell_pair_prefactors.data()
                + static_cast<std::size_t>(shell_pair)*primitive_stride;
            for (npy_intp li = 0; li < nprim[left]; ++li) {
                const double le = e[left*max_primitives+li];
                for (npy_intp ri = 0; ri < nprim[right]; ++ri) {
                    const double re = e[right*max_primitives+ri];
                    const double reduced = le*re/(le+re);
                    cached[li*max_primitives+ri] =
                        coef[left*max_primitives+li]
                        *coef[right*max_primitives+ri]
                        *std::exp(-reduced*center_distance2);
                    auto* cached_tables = shell_pair_axis_tables.data()
                        +(static_cast<std::size_t>(shell_pair)*primitive_stride
                            +li*max_primitives+ri)*axis_table_stride;
                    for (int axis = 0; axis < 3; ++axis) {
                        auto* table = cached_tables+axis*10;
                        const double total = le+re;
                        const double product = (le*c[3*left+axis]
                            +re*c[3*right+axis])/total;
                        table[0] = std::sqrt(M_PI/total);
                        for (int left_power = 1; left_power <= 4; ++left_power) {
                            table[left_power*2] = (product-c[3*left+axis])
                                *table[(left_power-1)*2];
                            if (left_power > 1)
                                table[left_power*2] += (left_power-1)
                                    *table[(left_power-2)*2]/(2.0*total);
                        }
                        table[1] = (product-c[3*right+axis])*table[0];
                        for (int left_power = 1; left_power <= 4; ++left_power)
                            table[left_power*2+1] =
                                (product-c[3*right+axis])*table[left_power*2]
                                +left_power*table[(left_power-1)*2]/(2.0*total);
                    }
                }
            }
        }
    }
    // The shell-pair tables are retained in thread-local storage across
    // geometries, but this call's workers must read the tables built by the
    // calling thread.  Referring to the thread_local names inside the worker
    // lambda would instead select each worker's empty TLS instance.
    const double* shell_pair_prefactor_data = shell_pair_prefactors.data();
    const double* shell_pair_axis_table_data = shell_pair_axis_tables.data();
    const std::size_t* shell_pair_index_data = shell_pair_index.data();
    constexpr int moments[13][3] = {
        {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1},
        {2,0,0}, {1,1,0}, {1,0,1}, {1,1,0}, {0,2,0},
        {0,1,1}, {1,0,1}, {0,1,1}, {0,0,2}
    };
    constexpr int unique_moments[10] = {0,1,2,3,4,5,6,8,9,12};
    auto evaluate_range = [&](npy_intp begin, npy_intp end) {
    for (npy_intp pair = begin; pair < end; ++pair) {
        const auto left = pl[pair];
        const auto right = pr[pair];
        const auto* left_a = a + 3 * left;
        const auto* right_a = a + 3 * right;
        double center_distance2 = 0.0;
        double left_relative[3];
        for (int axis = 0; axis < 3; ++axis) {
            const double delta = c[3*left+axis]-c[3*right+axis];
            center_distance2 += delta*delta;
            left_relative[axis] = c[3*left+axis]-org[axis];
        }
        for (int li = 0; li < nprim[left]; ++li) {
            const double le = e[left * max_primitives + li];
            const double ln = normalized_coefficients
                ? 1.0 : primitive_normalization(le, left_a);
            for (int ri = 0; ri < nprim[right]; ++ri) {
                const double re = e[right * max_primitives + ri];
                double prefactor;
                if (normalized_coefficients) {
                    const auto shell_pair =
                        shell_pair_index_data[static_cast<std::size_t>(pair)];
                    prefactor = shell_pair_prefactor_data[
                        static_cast<std::size_t>(shell_pair)*primitive_stride
                        +li*max_primitives+ri];
                } else {
                    prefactor = coef[left * max_primitives + li]
                        * coef[right * max_primitives + ri] * ln
                        * primitive_normalization(re, right_a);
                    const double reduced = le*re/(le+re);
                    prefactor *= std::exp(-reduced*center_distance2);
                }
                double base[3][3], raised[3][3], lowered[3][3] = {};
                const bool sp_pair = normalized_coefficients
                    && left_a[0]+left_a[1]+left_a[2] <= 1
                    && right_a[0]+right_a[1]+right_a[2] <= 1;
                if (sp_pair) {
                    constexpr std::size_t axis_table_stride = 3*5*2;
                    const auto shell_pair =
                        shell_pair_index_data[static_cast<std::size_t>(pair)];
                    const auto* cached_tables = shell_pair_axis_table_data
                        +(static_cast<std::size_t>(shell_pair)*primitive_stride
                            +li*max_primitives+ri)*axis_table_stride;
                    for (int axis = 0; axis < 3; ++axis) {
                        const auto* table = cached_tables+axis*10;
                        const int lp = static_cast<int>(left_a[axis]);
                        const int rp = static_cast<int>(right_a[axis]);
                        const auto moments = [&](int left_power, double* output) {
                            const double s0=table[left_power*2+rp];
                            const double s1=table[(left_power+1)*2+rp];
                            const double s2=table[(left_power+2)*2+rp];
                            output[0]=s0;
                            output[1]=left_relative[axis]*s0+s1;
                            output[2]=left_relative[axis]*left_relative[axis]*s0
                                +2.0*left_relative[axis]*s1+s2;
                        };
                        moments(lp,base[axis]);
                        moments(lp+1,raised[axis]);
                        if(lp>0)moments(lp-1,lowered[axis]);
                    }
                } else {
                    for (int axis = 0; axis < 3; ++axis)
                        axis_moment_bundle_first_derivative(
                            left_a[axis], right_a[axis], c[3*left+axis],
                            c[3*right+axis], le, re, left_relative[axis], base[axis],
                            raised[axis], lowered[axis]);
                }
                for (int unique = 0; unique < 10; ++unique) {
                    const int m = unique_moments[unique];
                    const int m0 = moments[m][0], m1 = moments[m][1], m2 = moments[m][2];
                    out[pair*13+m] += prefactor * base[0][m0] * base[1][m1] * base[2][m2];
                    for (int axis = 0; axis < 3; ++axis) {
                        const int mp = moments[m][axis];
                        const double other = axis == 0 ? base[1][m1]*base[2][m2]
                            : axis == 1 ? base[0][m0]*base[2][m2]
                            : base[0][m0]*base[1][m1];
                        out_dl[(pair*13+m)*3+axis] += prefactor * other
                            * (2.0*le*raised[axis][mp] - left_a[axis]*lowered[axis][mp]);
                    }
                }
            }
        }
        const double inverse_norm = 1.0 / std::sqrt(norm[left] * norm[right]);
        for (int unique = 0; unique < 10; ++unique) {
            const int m = unique_moments[unique];
            out[pair*13+m] *= inverse_norm;
            for (int axis = 0; axis < 3; ++axis) out_dl[(pair*13+m)*3+axis] *= inverse_norm;
        }
        constexpr int duplicate_moments[3][2] = {{7,5}, {10,6}, {11,9}};
        for (const auto& duplicate : duplicate_moments) {
            const int target = duplicate[0], source = duplicate[1];
            out[pair*13+target] = out[pair*13+source];
            for (int axis = 0; axis < 3; ++axis)
                out_dl[(pair*13+target)*3+axis] =
                    out_dl[(pair*13+source)*3+axis];
        }
        for (int axis = 0; axis < 3; ++axis) out_dr[(pair*13)*3+axis] = -out_dl[(pair*13)*3+axis];
        for (int component = 0; component < 3; ++component) {
            for (int axis = 0; axis < 3; ++axis) {
                out_dr[(pair*13+1+component)*3+axis] =
                    (component == axis ? out[pair*13] : 0.0)
                    - out_dl[(pair*13+1+component)*3+axis];
            }
        }
        for (int first = 0; first < 3; ++first) for (int second = 0; second < 3; ++second) {
            const int index = 4 + 3*first + second;
            for (int axis = 0; axis < 3; ++axis) {
                out_dr[(pair*13+index)*3+axis] =
                    (first == axis ? out[pair*13+1+second] : 0.0)
                    + (second == axis ? out[pair*13+1+first] : 0.0)
                    - out_dl[(pair*13+index)*3+axis];
            }
        }
    }};
    native_threads = std::max(1, std::min(native_threads, static_cast<int>(pair_count)));
    Py_BEGIN_ALLOW_THREADS
    std::vector<std::thread> workers;
    workers.reserve(native_threads > 1 ? native_threads - 1 : 0);
    const npy_intp chunk = (pair_count + native_threads - 1) / native_threads;
    for (int thread = 1; thread < native_threads; ++thread) {
        const auto begin = std::min<npy_intp>(pair_count, thread * chunk);
        const auto end = std::min<npy_intp>(pair_count, begin + chunk);
        workers.emplace_back(evaluate_range, begin, end);
    }
    evaluate_range(0, std::min<npy_intp>(pair_count, chunk));
    for (auto& worker : workers) worker.join();
    Py_END_ALLOW_THREADS
    Py_DECREF(centers); Py_DECREF(angular); Py_DECREF(exponents);
    Py_DECREF(coefficients); Py_DECREF(counts); Py_DECREF(pair_left);
    Py_DECREF(pair_right); Py_DECREF(norms); Py_DECREF(origin);
    return Py_BuildValue("NNN", values, dl, dr);
}

struct IntegralContextNative {
    PyArrayObject* angular = nullptr;
    PyArrayObject* exponents = nullptr;
    PyArrayObject* coefficients = nullptr;
    PyArrayObject* counts = nullptr;
    PyArrayObject* left = nullptr;
    PyArrayObject* right = nullptr;
    PyArrayObject* norms = nullptr;
    PyArrayObject* spherical_offsets = nullptr;
    PyArrayObject* spherical_cartesian = nullptr;
    PyArrayObject* spherical_coefficients = nullptr;
    PyArrayObject* spherical_left = nullptr;
    PyArrayObject* spherical_right = nullptr;
    std::vector<std::int64_t> spherical_plan_offsets;
    std::vector<std::int32_t> spherical_plan_source;
    std::vector<double> spherical_plan_weight;
    std::vector<std::uint8_t> spherical_plan_swap;
    bool spherical_plan_identity = false;
    PyArrayObject* core = nullptr;
    PyArrayObject* overlap = nullptr;
    std::vector<double> overlap_cholesky;
    std::vector<double> generalized_eigh_matrix;
    std::vector<double> generalized_eigh_factor;
    std::vector<double> generalized_eigh_values;
    std::vector<double> generalized_eigh_work;
    std::vector<int> generalized_eigh_iwork;
    PyArrayObject* pair_overlap = nullptr;
    PyArrayObject* pair_dipole = nullptr;
    PyArrayObject* pair_second = nullptr;
    PyArrayObject* centers = nullptr;
    PyArrayObject* atom = nullptr;
    PyArrayObject* shell = nullptr;
    PyArrayObject* reference = nullptr;
    PyArrayObject* valence = nullptr;
    PyArrayObject* core_atom = nullptr;
    PyArrayObject* core_level = nullptr;
    PyArrayObject* core_shift = nullptr;
    PyArrayObject* core_polynomial = nullptr;
    PyArrayObject* core_slater = nullptr;
    PyArrayObject* core_angular = nullptr;
    PyArrayObject* core_radii = nullptr;
    PyArrayObject* core_en = nullptr;
    std::vector<PyArrayObject*> scf_arrays;
    PyObject* last_observables = nullptr;
    PyObject* last_anisotropic = nullptr;
    PyObject* last_d4 = nullptr;
    double last_scf_energy = 0.0;
    ~IntegralContextNative() {
        Py_XDECREF(angular); Py_XDECREF(exponents); Py_XDECREF(coefficients);
        Py_XDECREF(counts); Py_XDECREF(left); Py_XDECREF(right); Py_XDECREF(norms);
        Py_XDECREF(spherical_offsets); Py_XDECREF(spherical_cartesian);
        Py_XDECREF(spherical_coefficients); Py_XDECREF(spherical_left);
        Py_XDECREF(spherical_right);
        Py_XDECREF(core); Py_XDECREF(overlap); Py_XDECREF(pair_overlap);
        Py_XDECREF(pair_dipole); Py_XDECREF(pair_second); Py_XDECREF(centers);
        Py_XDECREF(atom); Py_XDECREF(shell); Py_XDECREF(reference); Py_XDECREF(valence);
        Py_XDECREF(core_atom); Py_XDECREF(core_level); Py_XDECREF(core_shift);
        Py_XDECREF(core_polynomial); Py_XDECREF(core_slater);
        Py_XDECREF(core_angular); Py_XDECREF(core_radii); Py_XDECREF(core_en);
        for (auto* array : scf_arrays) Py_XDECREF(array);
        Py_XDECREF(last_observables); Py_XDECREF(last_anisotropic);
        Py_XDECREF(last_d4);
    }
};

PyObject* dense_scc_observables_native(PyObject*, PyObject*);
PyObject* dense_scc_fock_native(PyObject*, PyObject*);
PyObject* anisotropic_energy_potentials_native(PyObject*, PyObject*);
PyObject* d4_two_body_pairs_native(PyObject*, PyObject*);
PyObject* sparse_multipole_congruence_native(PyObject*, PyObject*);

enum ScfArrayIndex : std::size_t {
    SCF_INTERACTION, SCF_THIRD,
    SCF_ANIS_DIPOLE_XC, SCF_ANIS_QUADRUPOLE_XC,
    SCF_ANIS_LEFT, SCF_ANIS_RIGHT, SCF_ANIS_DISPLACEMENT,
    SCF_ANIS_F3, SCF_ANIS_F5, SCF_ANIS_DISTANCE2,
    SCF_D4_FREQUENCY, SCF_D4_WEIGHTS, SCF_D4_WEIGHT_DERIVATIVES,
    SCF_D4_REFERENCE_CHARGES, SCF_D4_REFERENCES, SCF_D4_MASK,
    SCF_D4_EFFECTIVE, SCF_D4_HARDNESS, SCF_D4_ROWS, SCF_D4_COLUMNS,
    SCF_D4_RATIO, SCF_D4_PAIR_FACTOR, SCF_ARRAY_COUNT
};

constexpr const char* integral_context_capsule_name =
    "matrix_gfn2.IntegralContext.v1";

void integral_context_destructor(PyObject* capsule) {
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (context) delete context;
    else PyErr_Clear();
}

PyObject* create_integral_context_native(PyObject*, PyObject* args) {
    PyObject *angular_o, *exponents_o, *coefficients_o, *counts_o;
    PyObject *left_o, *right_o, *norms_o;
    if (!PyArg_ParseTuple(args, "OOOOOOO", &angular_o, &exponents_o,
            &coefficients_o, &counts_o, &left_o, &right_o, &norms_o))
        return nullptr;
    auto* context = new (std::nothrow) IntegralContextNative();
    if (!context) return PyErr_NoMemory();
    context->angular = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        angular_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    context->exponents = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        exponents_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    context->coefficients = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        coefficients_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    context->counts = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        counts_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    context->left = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        left_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    context->right = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        right_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    context->norms = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        norms_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const bool valid = context->angular && context->exponents
        && context->coefficients && context->counts && context->left
        && context->right && context->norms
        && PyArray_NDIM(context->angular) == 2
        && PyArray_DIM(context->angular, 1) == 3
        && PyArray_NDIM(context->exponents) == 2
        && PyArray_NDIM(context->coefficients) == 2
        && PyArray_SIZE(context->exponents) == PyArray_SIZE(context->coefficients)
        && PyArray_SIZE(context->counts) == PyArray_DIM(context->angular, 0)
        && PyArray_SIZE(context->norms) == PyArray_DIM(context->angular, 0)
        && PyArray_SIZE(context->left) == PyArray_SIZE(context->right);
    if (!valid) {
        delete context;
        PyErr_SetString(PyExc_ValueError, "invalid persistent integral context arrays");
        return nullptr;
    }
    return PyCapsule_New(
        context, integral_context_capsule_name, integral_context_destructor);
}

PyObject* integral_context_evaluate_native(PyObject*, PyObject* args) {
    PyObject *capsule, *centers, *origin;
    int native_threads = 1;
    if (!PyArg_ParseTuple(args, "OOO|i", &capsule, &centers, &origin,
            &native_threads)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context) return nullptr;
    auto* thread_object = PyLong_FromLong(native_threads);
    auto* normalized_object = PyLong_FromLong(1);
    if (!thread_object || !normalized_object) {
        Py_XDECREF(thread_object); Py_XDECREF(normalized_object);
        return nullptr;
    }
    auto* forwarded = PyTuple_Pack(
        11, centers, context->angular, context->exponents,
        context->coefficients, context->counts, context->left, context->right,
        context->norms, origin, thread_object, normalized_object);
    Py_DECREF(thread_object); Py_DECREF(normalized_object);
    if (!forwarded) return nullptr;
    auto* result = multipole_pair_batch(nullptr, forwarded);
    Py_DECREF(forwarded);
    return result;
}

PyObject* context_configure_spherical_native(PyObject*, PyObject* args) {
    PyObject *capsule, *offsets_o, *cartesian_o, *coefficients_o;
    PyObject *left_o, *right_o;
    if (!PyArg_ParseTuple(args, "OOOOOO", &capsule, &offsets_o, &cartesian_o,
            &coefficients_o, &left_o, &right_o)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context) return nullptr;
    std::array<PyArrayObject*, 5> converted = {
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            offsets_o, NPY_INT64, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            cartesian_o, NPY_INT64, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            coefficients_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            left_o, NPY_INT64, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            right_o, NPY_INT64, NPY_ARRAY_IN_ARRAY))};
    const bool valid = std::all_of(converted.begin(), converted.end(),
            [](const auto* item) { return item != nullptr; })
        && PyArray_NDIM(converted[0]) == 1 && PyArray_SIZE(converted[0]) >= 1
        && PyArray_SIZE(converted[1]) == PyArray_SIZE(converted[2])
        && PyArray_SIZE(converted[3]) == PyArray_SIZE(converted[4]);
    if (!valid) {
        for (auto* item : converted) Py_XDECREF(item);
        if (!PyErr_Occurred()) PyErr_SetString(
            PyExc_ValueError, "invalid persistent spherical transformation arrays");
        return nullptr;
    }
    const auto* offsets = static_cast<const std::int64_t*>(
        PyArray_DATA(converted[0]));
    const auto spherical_size = PyArray_SIZE(converted[0]) - 1;
    bool topology_valid = offsets[0] == 0
        && offsets[spherical_size] == PyArray_SIZE(converted[1]);
    for (npy_intp index = 0; topology_valid && index < spherical_size; ++index)
        topology_valid = offsets[index] <= offsets[index + 1];
    if (!topology_valid) {
        for (auto* item : converted) Py_DECREF(item);
        PyErr_SetString(PyExc_ValueError, "invalid persistent spherical offsets");
        return nullptr;
    }
    const auto same_array = [](PyArrayObject* left, PyArrayObject* right) {
        return left && right && PyArray_TYPE(left) == PyArray_TYPE(right)
            && PyArray_NDIM(left) == PyArray_NDIM(right)
            && PyArray_SIZE(left) == PyArray_SIZE(right)
            && std::memcmp(PyArray_DATA(left), PyArray_DATA(right),
                static_cast<std::size_t>(PyArray_NBYTES(left))) == 0;
    };
    if (same_array(context->spherical_offsets, converted[0])
        && same_array(context->spherical_cartesian, converted[1])
        && same_array(context->spherical_coefficients, converted[2])
        && same_array(context->spherical_left, converted[3])
        && same_array(context->spherical_right, converted[4])) {
        for (auto* item : converted) Py_DECREF(item);
        Py_RETURN_NONE;
    }
    std::array<PyArrayObject**, 5> targets = {
        &context->spherical_offsets, &context->spherical_cartesian,
        &context->spherical_coefficients, &context->spherical_left,
        &context->spherical_right};
    for (std::size_t index = 0; index < converted.size(); ++index) {
        Py_XDECREF(*targets[index]);
        *targets[index] = converted[index];
    }
    const auto cartesian_size = PyArray_DIM(context->angular, 0);
    const auto source_count = PyArray_SIZE(context->left);
    const auto* source_left = static_cast<const std::int32_t*>(
        PyArray_DATA(context->left));
    const auto* source_right = static_cast<const std::int32_t*>(
        PyArray_DATA(context->right));
    const auto* cartesian = static_cast<const std::int64_t*>(
        PyArray_DATA(context->spherical_cartesian));
    const auto* coefficient = static_cast<const double*>(
        PyArray_DATA(context->spherical_coefficients));
    const auto* target_left = static_cast<const std::int64_t*>(
        PyArray_DATA(context->spherical_left));
    const auto* target_right = static_cast<const std::int64_t*>(
        PyArray_DATA(context->spherical_right));
    const auto target_count = PyArray_SIZE(context->spherical_left);
    std::unordered_map<std::int64_t, std::int32_t> lookup;
    lookup.reserve(static_cast<std::size_t>(source_count*1.4)+1);
    for (npy_intp pair = 0; pair < source_count; ++pair)
        lookup[static_cast<std::int64_t>(source_left[pair])*cartesian_size
            + source_right[pair]] = static_cast<std::int32_t>(pair);
    context->spherical_plan_offsets.assign(
        static_cast<std::size_t>(target_count)+1, 0);
    context->spherical_plan_source.clear();
    context->spherical_plan_weight.clear();
    context->spherical_plan_swap.clear();
    for (npy_intp target = 0; target < target_count; ++target) {
        for (auto li = offsets[target_left[target]];
                li < offsets[target_left[target]+1]; ++li) {
            const auto a = cartesian[li];
            for (auto ri = offsets[target_right[target]];
                    ri < offsets[target_right[target]+1]; ++ri) {
                const auto b = cartesian[ri];
                const auto high = std::max(a,b), low = std::min(a,b);
                const auto found = lookup.find(high*cartesian_size+low);
                if (found == lookup.end()) continue;
                context->spherical_plan_source.push_back(found->second);
                context->spherical_plan_weight.push_back(
                    coefficient[li]*coefficient[ri]);
                context->spherical_plan_swap.push_back(a < b ? 1 : 0);
            }
        }
        context->spherical_plan_offsets[target+1] =
            static_cast<std::int64_t>(context->spherical_plan_source.size());
    }
    context->spherical_plan_identity = target_count == source_count;
    for (npy_intp target = 0;
            context->spherical_plan_identity && target < target_count; ++target) {
        const auto begin = context->spherical_plan_offsets[target];
        const auto end = context->spherical_plan_offsets[target+1];
        context->spherical_plan_identity = end == begin+1
            && context->spherical_plan_source[begin] == target
            && context->spherical_plan_swap[begin] == 0
            && context->spherical_plan_weight[begin] == 1.0;
    }
    Py_RETURN_NONE;
}

PyObject* context_evaluate_spherical_native(PyObject*, PyObject* args) {
    PyObject *capsule, *centers, *origin;
    int native_threads = 1;
    if (!PyArg_ParseTuple(args, "OOO|i", &capsule, &centers, &origin,
            &native_threads)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context) return nullptr;
    if (!context->spherical_offsets || !context->spherical_cartesian
        || !context->spherical_coefficients || !context->spherical_left
        || !context->spherical_right) {
        PyErr_SetString(PyExc_RuntimeError,
            "native GFN2 context has no spherical transformation");
        return nullptr;
    }
    auto* thread_object = PyLong_FromLong(native_threads);
    auto* normalized_object = PyLong_FromLong(1);
    if (!thread_object || !normalized_object) {
        Py_XDECREF(thread_object); Py_XDECREF(normalized_object);
        return nullptr;
    }
    auto* integral_args = PyTuple_Pack(
        11, centers, context->angular, context->exponents,
        context->coefficients, context->counts, context->left, context->right,
        context->norms, origin, thread_object, normalized_object);
    Py_DECREF(thread_object); Py_DECREF(normalized_object);
    if (!integral_args) return nullptr;
    auto* cartesian_result = multipole_pair_batch(nullptr, integral_args);
    Py_DECREF(integral_args);
    if (!cartesian_result) return nullptr;
    // s/p-only bases have an exactly identity Cartesian-to-real-spherical
    // map.  Reuse the three pair-packed arrays directly instead of allocating
    // and traversing an identical transformed working set.  The identity is
    // certified when the persistent plan is compiled; d shells retain the
    // general congruence below.
    if (context->spherical_plan_identity) return cartesian_result;
    PyObject *values = nullptr, *left_derivatives = nullptr;
    PyObject *right_derivatives = nullptr;
    if (!PyArg_ParseTuple(cartesian_result, "OOO", &values, &left_derivatives,
            &right_derivatives)) {
        Py_DECREF(cartesian_result);
        return nullptr;
    }
    auto* input_values = reinterpret_cast<PyArrayObject*>(values);
    auto* input_left = reinterpret_cast<PyArrayObject*>(left_derivatives);
    auto* input_right = reinterpret_cast<PyArrayObject*>(right_derivatives);
    const npy_intp target_count = PyArray_SIZE(context->spherical_left);
    npy_intp value_dims[2] = {target_count, 13};
    npy_intp derivative_dims[3] = {target_count, 13, 3};
    auto* output_values = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, value_dims, NPY_DOUBLE, 0));
    auto* output_left = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(3, derivative_dims, NPY_DOUBLE, 0));
    auto* output_right = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(3, derivative_dims, NPY_DOUBLE, 0));
    if (!output_values || !output_left || !output_right) {
        Py_XDECREF(output_values); Py_XDECREF(output_left); Py_XDECREF(output_right);
        Py_DECREF(cartesian_result);
        return PyErr_NoMemory();
    }
    const auto* source_values = static_cast<const double*>(
        PyArray_DATA(input_values));
    const auto* source_left = static_cast<const double*>(PyArray_DATA(input_left));
    const auto* source_right = static_cast<const double*>(PyArray_DATA(input_right));
    auto* transformed = static_cast<double*>(PyArray_DATA(output_values));
    auto* transformed_left = static_cast<double*>(PyArray_DATA(output_left));
    auto* transformed_right = static_cast<double*>(PyArray_DATA(output_right));
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp target = 0; target < target_count; ++target) {
        const auto begin = context->spherical_plan_offsets[target];
        const auto end = context->spherical_plan_offsets[target+1];
        for (auto item = begin; item < end; ++item) {
            const auto source = context->spherical_plan_source[item];
            const auto weight = context->spherical_plan_weight[item];
            const bool swap = context->spherical_plan_swap[item] != 0;
            const auto* dl = swap ? source_right : source_left;
            const auto* dr = swap ? source_left : source_right;
            for (int component = 0; component < 13; ++component) {
                transformed[target*13+component] +=
                    weight*source_values[source*13+component];
                for (int axis = 0; axis < 3; ++axis) {
                    const auto destination = (target*13+component)*3+axis;
                    const auto source_index = (source*13+component)*3+axis;
                    transformed_left[destination] += weight*dl[source_index];
                    transformed_right[destination] += weight*dr[source_index];
                }
            }
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(cartesian_result);
    return Py_BuildValue("NNN", output_values, output_left, output_right);
}

PyObject* context_spherical_directional_seconds_native(PyObject*, PyObject* args) {
    PyObject *capsule, *centers_o, *origin_o, *direction_o;
    int native_threads = 1;
    if (!PyArg_ParseTuple(args, "OOOO|i", &capsule, &centers_o, &origin_o,
            &direction_o, &native_threads)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context) return nullptr;
    if (!context->spherical_left || !context->spherical_right
        || !context->spherical_offsets) {
        PyErr_SetString(PyExc_RuntimeError,
            "native GFN2 context has no spherical transformation");
        return nullptr;
    }
    auto* centers = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        centers_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* origin = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        origin_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* direction = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        direction_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const npy_intp orbital_count = PyArray_DIM(context->angular, 0);
    const bool valid = centers && origin && direction
        && PyArray_NDIM(centers) == 2
        && PyArray_DIM(centers, 0) == orbital_count
        && PyArray_DIM(centers, 1) == 3
        && PyArray_SIZE(origin) == 3
        && (PyArray_NDIM(direction) == 2 || PyArray_NDIM(direction) == 3)
        && PyArray_DIM(direction, PyArray_NDIM(direction)-2) == orbital_count
        && PyArray_DIM(direction, PyArray_NDIM(direction)-1) == 3;
    if (!valid) {
        Py_XDECREF(centers); Py_XDECREF(origin); Py_XDECREF(direction);
        if (!PyErr_Occurred()) PyErr_SetString(
            PyExc_ValueError, "invalid directional AO second-derivative arrays");
        return nullptr;
    }
    const bool batched = PyArray_NDIM(direction) == 3;
    const npy_intp batch_count = batched ? PyArray_DIM(direction, 0) : 1;
    const npy_intp source_count = PyArray_SIZE(context->left);
    npy_intp source_dims[4] = {batch_count, source_count, 13, 3};
    auto* source_left = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(4, source_dims, NPY_DOUBLE, 0));
    auto* source_right = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(4, source_dims, NPY_DOUBLE, 0));
    if (!source_left || !source_right) {
        Py_XDECREF(source_left); Py_XDECREF(source_right);
        Py_DECREF(centers); Py_DECREF(origin); Py_DECREF(direction);
        return PyErr_NoMemory();
    }
    const auto* angular = static_cast<const std::int64_t*>(
        PyArray_DATA(context->angular));
    const auto* exponents = static_cast<const double*>(
        PyArray_DATA(context->exponents));
    const auto* coefficients = static_cast<const double*>(
        PyArray_DATA(context->coefficients));
    const auto* counts = static_cast<const std::int64_t*>(
        PyArray_DATA(context->counts));
    const auto* pair_left = static_cast<const std::int32_t*>(
        PyArray_DATA(context->left));
    const auto* pair_right = static_cast<const std::int32_t*>(
        PyArray_DATA(context->right));
    const auto* norms = static_cast<const double*>(PyArray_DATA(context->norms));
    const auto* center = static_cast<const double*>(PyArray_DATA(centers));
    const auto* origin_values = static_cast<const double*>(PyArray_DATA(origin));
    const auto* vector = static_cast<const double*>(PyArray_DATA(direction));
    auto* left_output = static_cast<double*>(PyArray_DATA(source_left));
    auto* right_output = static_cast<double*>(PyArray_DATA(source_right));
    // A Cartesian Hessian column moves only one atom.  The second derivative
    // of a two-center integral is exactly zero when neither center moves.
    std::vector<unsigned char> displaced(orbital_count, 0);
    for (npy_intp batch = 0; batch < batch_count; ++batch)
        for (npy_intp orbital = 0; orbital < orbital_count; ++orbital) {
            const auto offset = (batch*orbital_count+orbital)*3;
            displaced[orbital] |= vector[offset] != 0.0
                || vector[offset+1] != 0.0 || vector[offset+2] != 0.0;
        }
    const npy_intp maximum_primitives = PyArray_DIM(context->exponents, 1);
    constexpr int moments[13][3] = {
        {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1},
        {2,0,0}, {1,1,0}, {1,0,1}, {1,1,0}, {0,2,0},
        {0,1,1}, {1,0,1}, {0,1,1}, {0,0,2}
    };
    auto evaluate_range = [&](npy_intp begin, npy_intp end) {
        for (npy_intp pair = begin; pair < end; ++pair) {
            const auto left = pair_left[pair], right = pair_right[pair];
            if (!displaced[left] && !displaced[right]) continue;
            const auto* left_angular = angular + 3*left;
            const auto* right_angular = angular + 3*right;
            const auto* left_center = center + 3*left;
            const auto* right_center = center + 3*right;
            for (int li = 0; li < counts[left]; ++li) {
                const double left_exponent = exponents[left*maximum_primitives+li];
                for (int ri = 0; ri < counts[right]; ++ri) {
                    const double right_exponent = exponents[right*maximum_primitives+ri];
                    double prefactor = coefficients[left*maximum_primitives+li]
                        * coefficients[right*maximum_primitives+ri];
                    const double reduced = left_exponent*right_exponent
                        /(left_exponent+right_exponent);
                    double distance2 = 0.0;
                    for (int axis = 0; axis < 3; ++axis) {
                        const double delta = left_center[axis]-right_center[axis];
                        distance2 += delta*delta;
                    }
                    prefactor *= std::exp(-reduced*distance2);
                    const ShiftedAxisMomentCache moment_cache(
                        left_angular, right_angular, left_center, right_center,
                        left_exponent, right_exponent, origin_values);
                    const PrimitiveMomentDerivatives moment_derivatives(
                        moment_cache, left_angular, right_angular,
                        left_exponent, right_exponent);
                    for (int component = 0; component < 13; ++component) {
                        for (int output_axis = 0; output_axis < 3; ++output_axis) {
                            for (int direction_axis = 0; direction_axis < 3;
                                    ++direction_axis) {
                                const double left_left =
                                    primitive_moment_second_derivative(
                                    moment_derivatives, moments[component],
                                    0, output_axis, 0, direction_axis);
                                const double left_right =
                                    primitive_moment_second_derivative(
                                    moment_derivatives, moments[component],
                                    0, output_axis, 1, direction_axis);
                                const double right_left =
                                    primitive_moment_second_derivative(
                                    moment_derivatives, moments[component],
                                    1, output_axis, 0, direction_axis);
                                const double right_right =
                                    primitive_moment_second_derivative(
                                    moment_derivatives, moments[component],
                                    1, output_axis, 1, direction_axis);
                                for (npy_intp batch = 0; batch < batch_count; ++batch) {
                                    const auto direction_offset = batched
                                        ? batch*orbital_count*3 : 0;
                                    const double left_direction = vector[
                                        direction_offset+left*3+direction_axis];
                                    const double right_direction = vector[
                                        direction_offset+right*3+direction_axis];
                                    const auto index =
                                        ((batch*source_count+pair)*13+component)*3
                                        +output_axis;
                                    left_output[index] += prefactor*(
                                        left_direction*left_left
                                        +right_direction*left_right);
                                    right_output[index] += prefactor*(
                                        left_direction*right_left
                                        +right_direction*right_right);
                                }
                            }
                        }
                    }
                }
            }
            const double inverse_norm = 1.0/std::sqrt(norms[left]*norms[right]);
            for (npy_intp batch = 0; batch < batch_count; ++batch)
                for (int component = 0; component < 13; ++component)
                    for (int axis = 0; axis < 3; ++axis) {
                        const auto index =
                            ((batch*source_count+pair)*13+component)*3+axis;
                        left_output[index] *= inverse_norm;
                        right_output[index] *= inverse_norm;
                    }
        }
    };
    native_threads = std::max(1, std::min(native_threads,
        static_cast<int>(source_count)));
    Py_BEGIN_ALLOW_THREADS
    std::vector<std::thread> workers;
    const npy_intp chunk = source_count > 0
        ? (source_count+native_threads-1)/native_threads : 0;
    for (int thread = 1; thread < native_threads; ++thread) {
        const auto begin = std::min<npy_intp>(source_count, thread*chunk);
        workers.emplace_back(evaluate_range, begin,
            std::min<npy_intp>(source_count, begin+chunk));
    }
    evaluate_range(0, std::min<npy_intp>(source_count, chunk));
    for (auto& worker : workers) worker.join();
    Py_END_ALLOW_THREADS

    const npy_intp target_count = PyArray_SIZE(context->spherical_left);
    npy_intp target_dims[4] = {batch_count, target_count, 13, 3};
    auto* target_left = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(4, target_dims, NPY_DOUBLE, 0));
    auto* target_right = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(4, target_dims, NPY_DOUBLE, 0));
    if (!target_left || !target_right) {
        Py_XDECREF(target_left); Py_XDECREF(target_right);
        Py_DECREF(source_left); Py_DECREF(source_right);
        Py_DECREF(centers); Py_DECREF(origin); Py_DECREF(direction);
        return PyErr_NoMemory();
    }
    auto* transformed_left = static_cast<double*>(PyArray_DATA(target_left));
    auto* transformed_right = static_cast<double*>(PyArray_DATA(target_right));
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp target = 0; target < target_count; ++target) {
        const auto begin = context->spherical_plan_offsets[target];
        const auto end = context->spherical_plan_offsets[target+1];
        for (auto item = begin; item < end; ++item) {
            const auto source = context->spherical_plan_source[item];
            const auto weight = context->spherical_plan_weight[item];
            const bool swap = context->spherical_plan_swap[item] != 0;
            const auto* dl = swap ? right_output : left_output;
            const auto* dr = swap ? left_output : right_output;
            for (npy_intp batch = 0; batch < batch_count; ++batch)
                for (int component = 0; component < 13; ++component)
                    for (int axis = 0; axis < 3; ++axis) {
                        const auto destination =
                            ((batch*target_count+target)*13+component)*3+axis;
                        const auto source_index =
                            ((batch*source_count+source)*13+component)*3+axis;
                        transformed_left[destination] += weight*dl[source_index];
                        transformed_right[destination] += weight*dr[source_index];
                    }
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(source_left); Py_DECREF(source_right);
    Py_DECREF(centers); Py_DECREF(origin); Py_DECREF(direction);
    return Py_BuildValue("NN", target_left, target_right);
}

PyObject* context_update_dense_model_native(PyObject*, PyObject* args) {
    PyObject *capsule, *core_o, *overlap_o, *pair_overlap_o, *dipole_o;
    PyObject *second_o, *centers_o, *atom_o, *shell_o, *reference_o, *valence_o;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOO", &capsule, &core_o, &overlap_o,
            &pair_overlap_o, &dipole_o, &second_o, &centers_o, &atom_o,
            &shell_o, &reference_o, &valence_o)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context) return nullptr;
    std::array<PyArrayObject**, 10> targets = {
        &context->core, &context->overlap, &context->pair_overlap,
        &context->pair_dipole, &context->pair_second, &context->centers,
        &context->atom, &context->shell, &context->reference, &context->valence};
    std::array<PyObject*, 10> sources = {core_o, overlap_o, pair_overlap_o,
        dipole_o, second_o, centers_o, atom_o, shell_o, reference_o, valence_o};
    std::array<int, 10> types = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NPY_DOUBLE, NPY_DOUBLE, NPY_INT64, NPY_INT64, NPY_DOUBLE, NPY_DOUBLE};
    std::array<PyArrayObject*, 10> converted = {};
    for (std::size_t index = 0; index < converted.size(); ++index) {
        converted[index] = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            sources[index], types[index], NPY_ARRAY_IN_ARRAY));
        if (!converted[index]) {
            for (auto* item : converted) Py_XDECREF(item);
            return nullptr;
        }
    }
    const npy_intp n = PyArray_DIM(converted[0], 0);
    const npy_intp npair = n*(n+1)/2;
    const bool valid = PyArray_NDIM(converted[0]) == 2
        && PyArray_DIM(converted[0], 1) == n
        && PyArray_SIZE(converted[1]) == n*n
        && PyArray_SIZE(converted[2]) == npair
        && PyArray_SIZE(converted[3]) == npair*3
        && PyArray_SIZE(converted[4]) == npair*9
        && PyArray_SIZE(converted[5]) == n*3
        && PyArray_SIZE(converted[6]) == n
        && PyArray_SIZE(converted[7]) == n;
    if (!valid) {
        for (auto* item : converted) Py_DECREF(item);
        PyErr_SetString(PyExc_ValueError, "incompatible persistent dense model arrays");
        return nullptr;
    }
    for (std::size_t index = 0; index < converted.size(); ++index) {
        Py_XDECREF(*targets[index]);
        *targets[index] = converted[index];
    }
    context->overlap_cholesky.resize(static_cast<std::size_t>(n)*n);
    const auto* overlap_values = static_cast<const double*>(
        PyArray_DATA(context->overlap));
    for (npy_intp row = 0; row < n; ++row)
        for (npy_intp column = 0; column < n; ++column)
            context->overlap_cholesky[row + column*n] =
                overlap_values[row*n + column];
    int dimension = static_cast<int>(n), leading = std::max(1, dimension);
    int factor_info = 0; char lower = 'L';
    Py_BEGIN_ALLOW_THREADS
    dpotrf_(&lower, &dimension, context->overlap_cholesky.data(),
            &leading, &factor_info);
    Py_END_ALLOW_THREADS
    if (factor_info != 0) {
        context->overlap_cholesky.clear();
        PyErr_Format(PyExc_RuntimeError,
            "native overlap Cholesky factorization failed (info=%d)", factor_info);
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* context_configure_core_native(PyObject*, PyObject* args) {
    PyObject *capsule, *atom_o, *level_o, *shift_o, *polynomial_o;
    PyObject *slater_o, *angular_o, *radii_o, *en_o;
    if (!PyArg_ParseTuple(args, "OOOOOOOOO", &capsule, &atom_o, &level_o,
            &shift_o, &polynomial_o, &slater_o, &angular_o, &radii_o, &en_o))
        return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context) return nullptr;
    std::array<PyArrayObject**, 8> targets = {&context->core_atom,
        &context->core_level, &context->core_shift, &context->core_polynomial,
        &context->core_slater, &context->core_angular, &context->core_radii,
        &context->core_en};
    std::array<PyObject*, 8> sources = {atom_o, level_o, shift_o, polynomial_o,
        slater_o, angular_o, radii_o, en_o};
    std::array<int, 8> types = {NPY_INT64, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NPY_DOUBLE, NPY_INT64, NPY_DOUBLE, NPY_DOUBLE};
    std::array<PyArrayObject*, 8> converted = {};
    for (std::size_t index = 0; index < converted.size(); ++index) {
        converted[index] = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            sources[index], types[index], NPY_ARRAY_IN_ARRAY));
        if (!converted[index]) {
            for (auto* item : converted) Py_XDECREF(item);
            return nullptr;
        }
    }
    const auto norb = PyArray_SIZE(converted[0]);
    const bool valid = PyArray_SIZE(converted[1]) == norb
        && PyArray_SIZE(converted[2]) == norb
        && PyArray_SIZE(converted[3]) == norb
        && PyArray_SIZE(converted[4]) == norb
        && PyArray_SIZE(converted[5]) == norb
        && PyArray_SIZE(converted[6]) == PyArray_SIZE(converted[7]);
    if (!valid) {
        for (auto* item : converted) Py_DECREF(item);
        PyErr_SetString(PyExc_ValueError, "incompatible persistent core arrays");
        return nullptr;
    }
    for (std::size_t index = 0; index < converted.size(); ++index) {
        Py_XDECREF(*targets[index]);
        *targets[index] = converted[index];
    }
    Py_RETURN_NONE;
}

PyObject* context_build_core_native(PyObject*, PyObject* args) {
    PyObject *capsule, *overlap_o, *coordinates_o, *coordination_o;
    if (!PyArg_ParseTuple(args, "OOOO", &capsule, &overlap_o, &coordinates_o,
            &coordination_o)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context) return nullptr;
    if (!context->core_atom || !context->core_level || !context->core_shift
        || !context->core_polynomial || !context->core_slater
        || !context->core_angular || !context->core_radii || !context->core_en) {
        PyErr_SetString(PyExc_RuntimeError, "native GFN2 context has no core parameters");
        return nullptr;
    }
    auto* overlap = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        overlap_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* coordinates = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        coordinates_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* coordination = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        coordination_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const npy_intp n = PyArray_SIZE(context->core_atom);
    if (!overlap || !coordinates || !coordination
        || PyArray_SIZE(overlap) != n*n
        || PyArray_NDIM(coordinates) != 2 || PyArray_DIM(coordinates, 1) != 3
        || PyArray_SIZE(coordination) != PyArray_DIM(coordinates, 0)) {
        Py_XDECREF(overlap); Py_XDECREF(coordinates); Py_XDECREF(coordination);
        PyErr_SetString(PyExc_ValueError, "incompatible native core geometry");
        return nullptr;
    }
    npy_intp dims[2] = {n, n};
    auto* result = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(
        2, dims, NPY_DOUBLE, 0));
    if (!result) {
        Py_DECREF(overlap); Py_DECREF(coordinates); Py_DECREF(coordination);
        return nullptr;
    }
    const auto* s = static_cast<const double*>(PyArray_DATA(overlap));
    const auto* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* cn = static_cast<const double*>(PyArray_DATA(coordination));
    const auto* atom = static_cast<const std::int64_t*>(PyArray_DATA(context->core_atom));
    const auto* base = static_cast<const double*>(PyArray_DATA(context->core_level));
    const auto* shift = static_cast<const double*>(PyArray_DATA(context->core_shift));
    const auto* polynomial = static_cast<const double*>(PyArray_DATA(context->core_polynomial));
    const auto* slater = static_cast<const double*>(PyArray_DATA(context->core_slater));
    const auto* angular = static_cast<const std::int64_t*>(PyArray_DATA(context->core_angular));
    const auto* radii = static_cast<const double*>(PyArray_DATA(context->core_radii));
    const auto* en = static_cast<const double*>(PyArray_DATA(context->core_en));
    auto* output = static_cast<double*>(PyArray_DATA(result));
    constexpr double angular_scale[4][4] = {
        {1.85,2.04,2.00,2.00}, {2.04,2.23,2.00,2.00},
        {2.00,2.00,2.23,2.00}, {2.00,2.00,2.00,2.00}};
    std::vector<double> levels(n);
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < n; ++i) {
        levels[i] = base[i] - shift[i]*cn[atom[i]];
        output[i*n+i] = levels[i];
    }
    for (npy_intp j = 0; j < n; ++j) for (npy_intp i = 0; i < j; ++i) {
        const auto ai = atom[i], aj = atom[j];
        if (ai == aj) continue;
        double distance2 = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double delta = xyz[ai*3+axis] - xyz[aj*3+axis];
            distance2 += delta*delta;
        }
        const double distance = std::sqrt(distance2);
        const double exponent_factor = std::sqrt(
            2.0*std::sqrt(slater[i]*slater[j])/(slater[i]+slater[j]));
        const double root = std::sqrt(distance/(0.5*(radii[ai]+radii[aj])));
        const double poly = (1.0+polynomial[i]*root)*(1.0+polynomial[j]*root);
        const double delta_en = en[ai]-en[aj];
        const double value = 0.5*angular_scale[angular[i]][angular[j]]
            *(levels[i]+levels[j])*s[j*n+i]*exponent_factor
            *(1.0+0.02*delta_en*delta_en)*poly;
        output[j*n+i] = value; output[i*n+j] = value;
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(overlap); Py_DECREF(coordinates); Py_DECREF(coordination);
    return reinterpret_cast<PyObject*>(result);
}

PyObject* context_prepare_dense_geometry_native(PyObject*, PyObject* args) {
    PyObject *capsule, *cartesian_centers_o, *origin_o, *coordinates_o;
    PyObject *coordination_radii_o, *atom_o, *shell_o, *reference_o, *valence_o;
    int native_threads = 1;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOi", &capsule,
            &cartesian_centers_o, &origin_o, &coordinates_o,
            &coordination_radii_o, &atom_o, &shell_o, &reference_o,
            &valence_o, &native_threads)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context || !context->spherical_left || !context->spherical_right)
        return nullptr;
    auto f64 = [](PyObject* object) { return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)); };
    auto i64 = [](PyObject* object) { return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object, NPY_INT64, NPY_ARRAY_IN_ARRAY)); };
    auto* coordinates = f64(coordinates_o);
    auto* radii = f64(coordination_radii_o);
    auto* atom = i64(atom_o); auto* shell = i64(shell_o);
    auto* reference = f64(reference_o); auto* valence = f64(valence_o);
    const npy_intp natom = coordinates && PyArray_NDIM(coordinates) == 2
        ? PyArray_DIM(coordinates, 0) : 0;
    const npy_intp norb = atom ? PyArray_SIZE(atom) : 0;
    const npy_intp npair = norb*(norb+1)/2;
    const bool valid = coordinates && radii && atom && shell && reference && valence
        && PyArray_NDIM(coordinates) == 2 && PyArray_DIM(coordinates, 1) == 3
        && PyArray_SIZE(radii) == natom && PyArray_SIZE(valence) == natom
        && PyArray_SIZE(shell) == norb
        && PyArray_SIZE(context->spherical_left) == npair
        && PyArray_SIZE(context->spherical_right) == npair;
    auto cleanup_inputs = [&]() { Py_XDECREF(coordinates); Py_XDECREF(radii);
        Py_XDECREF(atom); Py_XDECREF(shell); Py_XDECREF(reference);
        Py_XDECREF(valence); };
    if (!valid) {
        cleanup_inputs();
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError,
            "incompatible native dense geometry arrays");
        return nullptr;
    }
    auto* thread_o = PyLong_FromLong(native_threads);
    auto* integral_args = thread_o ? PyTuple_Pack(4, capsule,
        cartesian_centers_o, origin_o, thread_o) : nullptr;
    Py_XDECREF(thread_o);
    auto* integrals = integral_args
        ? context_evaluate_spherical_native(nullptr, integral_args) : nullptr;
    Py_XDECREF(integral_args);
    if (!integrals) { cleanup_inputs(); return nullptr; }
    auto* values = reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(integrals, 0));
    auto* left_derivatives = PyTuple_GET_ITEM(integrals, 1);
    auto* right_derivatives = PyTuple_GET_ITEM(integrals, 2);
    if (PyArray_NDIM(values) != 2 || PyArray_DIM(values, 0) != npair
        || PyArray_DIM(values, 1) != 13) {
        Py_DECREF(integrals); cleanup_inputs();
        PyErr_SetString(PyExc_RuntimeError,
            "native spherical integral stream is not complete");
        return nullptr;
    }
    npy_intp matrix_dims[2] = {norb,norb};
    npy_intp pair_dims[1] = {npair}, dipole_dims[2] = {npair,3};
    npy_intp second_dims[3] = {npair,3,3};
    npy_intp center_dims[2] = {norb,3}, atom_dims[1] = {natom};
    auto* overlap = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, matrix_dims, NPY_DOUBLE, 0));
    auto* pair_overlap = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, pair_dims, NPY_DOUBLE, 0));
    auto* pair_dipole = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(2, dipole_dims, NPY_DOUBLE, 0));
    auto* pair_second = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(3, second_dims, NPY_DOUBLE, 0));
    auto* centers = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(2, center_dims, NPY_DOUBLE, 0));
    auto* coordination = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, atom_dims, NPY_DOUBLE, 0));
    if (!overlap || !pair_overlap || !pair_dipole || !pair_second
        || !centers || !coordination) {
        Py_XDECREF(overlap); Py_XDECREF(pair_overlap); Py_XDECREF(pair_dipole);
        Py_XDECREF(pair_second); Py_XDECREF(centers); Py_XDECREF(coordination);
        Py_DECREF(integrals); cleanup_inputs(); return PyErr_NoMemory();
    }
    const auto* rows = static_cast<const std::int64_t*>(
        PyArray_DATA(context->spherical_left));
    const auto* columns = static_cast<const std::int64_t*>(
        PyArray_DATA(context->spherical_right));
    const auto* packed = static_cast<const double*>(PyArray_DATA(values));
    const auto* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* radius = static_cast<const double*>(PyArray_DATA(radii));
    const auto* orbital_atom = static_cast<const std::int64_t*>(PyArray_DATA(atom));
    auto* dense_s = static_cast<double*>(PyArray_DATA(overlap));
    auto* packed_s = static_cast<double*>(PyArray_DATA(pair_overlap));
    auto* packed_mu = static_cast<double*>(PyArray_DATA(pair_dipole));
    auto* packed_second = static_cast<double*>(PyArray_DATA(pair_second));
    auto* orbital_centers = static_cast<double*>(PyArray_DATA(centers));
    auto* cn = static_cast<double*>(PyArray_DATA(coordination));
    bool invalid_atom = false;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp pair = 0; pair < npair; ++pair) {
        const auto row = rows[pair], column = columns[pair];
        const double s = packed[pair*13];
        dense_s[row*norb+column] = dense_s[column*norb+row] = s;
        packed_s[pair] = s;
        for (int component = 0; component < 3; ++component)
            packed_mu[pair*3+component] = packed[pair*13+1+component];
        for (int component = 0; component < 9; ++component)
            packed_second[pair*9+component] = packed[pair*13+4+component];
    }
    for (npy_intp orbital = 0; orbital < norb; ++orbital) {
        const auto center = orbital_atom[orbital];
        if (center < 0 || center >= natom) { invalid_atom = true; continue; }
        for (int axis = 0; axis < 3; ++axis)
            orbital_centers[orbital*3+axis] = xyz[center*3+axis];
    }
    for (npy_intp left = 1; left < natom; ++left)
        for (npy_intp right = 0; right < left; ++right) {
            double distance2 = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                const double delta = xyz[left*3+axis]-xyz[right*3+axis];
                distance2 += delta*delta;
            }
            const double distance = std::sqrt(distance2);
            const double scale1 = 4.0*(radius[left]+radius[right])/3.0;
            const double scale2 = scale1+2.0;
            const double value = 1.0/(1.0+std::exp(-10.0*(scale1/distance-1.0)))
                /(1.0+std::exp(-20.0*(scale2/distance-1.0)));
            cn[left] += value; cn[right] += value;
        }
    Py_END_ALLOW_THREADS
    if (invalid_atom) {
        Py_DECREF(overlap); Py_DECREF(pair_overlap); Py_DECREF(pair_dipole);
        Py_DECREF(pair_second); Py_DECREF(centers); Py_DECREF(coordination);
        Py_DECREF(integrals); cleanup_inputs();
        PyErr_SetString(PyExc_ValueError, "orbital atom index is out of range");
        return nullptr;
    }
    auto* core_args = PyTuple_Pack(4, capsule, overlap, coordinates, coordination);
    auto* core = core_args ? context_build_core_native(nullptr, core_args) : nullptr;
    Py_XDECREF(core_args);
    if (!core) {
        Py_DECREF(overlap); Py_DECREF(pair_overlap); Py_DECREF(pair_dipole);
        Py_DECREF(pair_second); Py_DECREF(centers); Py_DECREF(coordination);
        Py_DECREF(integrals); cleanup_inputs(); return nullptr;
    }
    auto* update_args = PyTuple_Pack(11, capsule, core, overlap, pair_overlap,
        pair_dipole, pair_second, centers, atom, shell, reference, valence);
    auto* updated = update_args
        ? context_update_dense_model_native(nullptr, update_args) : nullptr;
    Py_XDECREF(update_args);
    if (!updated) {
        Py_DECREF(core); Py_DECREF(overlap); Py_DECREF(pair_overlap);
        Py_DECREF(pair_dipole); Py_DECREF(pair_second); Py_DECREF(centers);
        Py_DECREF(coordination); Py_DECREF(integrals); cleanup_inputs();
        return nullptr;
    }
    Py_DECREF(updated); Py_DECREF(pair_overlap); Py_DECREF(pair_dipole);
    Py_DECREF(pair_second); Py_DECREF(centers); cleanup_inputs();
    Py_INCREF(context->spherical_left); Py_INCREF(context->spherical_right);
    Py_INCREF(values); Py_INCREF(left_derivatives); Py_INCREF(right_derivatives);
    Py_DECREF(integrals);
    return Py_BuildValue("NNNNNNNN", context->spherical_left,
        context->spherical_right, values, left_derivatives, right_derivatives,
        overlap, core, coordination);
}

PyObject* context_h0_derivatives_native(PyObject*, PyObject* args) {
    PyObject *capsule, *rows_o, *columns_o, *overlap_o, *left_ds_o;
    PyObject *right_ds_o, *coordinates_o, *coordination_o;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &capsule, &rows_o, &columns_o,
            &overlap_o, &left_ds_o, &right_ds_o, &coordinates_o,
            &coordination_o)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context || !context->core_atom || !context->core_level
        || !context->core_shift || !context->core_polynomial
        || !context->core_slater || !context->core_angular
        || !context->core_radii || !context->core_en) {
        PyErr_SetString(PyExc_RuntimeError,
            "native GFN2 context has no core derivative parameters");
        return nullptr;
    }
    auto i32 = [](PyObject* object) { return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object, NPY_INT32, NPY_ARRAY_IN_ARRAY)); };
    auto f64 = [](PyObject* object) { return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)); };
    auto* rows = i32(rows_o); auto* columns = i32(columns_o);
    auto* overlap = f64(overlap_o); auto* left_ds = f64(left_ds_o);
    auto* right_ds = f64(right_ds_o); auto* coordinates = f64(coordinates_o);
    auto* coordination = f64(coordination_o);
    const npy_intp pair_count = rows ? PyArray_SIZE(rows) : 0;
    const npy_intp natoms = coordinates && PyArray_NDIM(coordinates) == 2
        ? PyArray_DIM(coordinates, 0) : 0;
    const npy_intp norbitals = PyArray_SIZE(context->core_atom);
    const bool valid = rows && columns && overlap && left_ds && right_ds
        && coordinates && coordination
        && PyArray_SIZE(columns) == pair_count
        && PyArray_SIZE(overlap) == pair_count
        && PyArray_SIZE(left_ds) == pair_count*3
        && PyArray_SIZE(right_ds) == pair_count*3
        && PyArray_NDIM(coordinates) == 2 && PyArray_DIM(coordinates, 1) == 3
        && PyArray_SIZE(coordination) == natoms;
    auto release = [&]() { Py_XDECREF(rows); Py_XDECREF(columns);
        Py_XDECREF(overlap); Py_XDECREF(left_ds); Py_XDECREF(right_ds);
        Py_XDECREF(coordinates); Py_XDECREF(coordination); };
    if (!valid) {
        release();
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError,
            "invalid native H0 derivative arrays");
        return nullptr;
    }
    npy_intp vector_dims[1] = {pair_count};
    npy_intp derivative_dims[2] = {pair_count, 3};
    auto* left_output = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, derivative_dims, NPY_DOUBLE, 0));
    auto* right_output = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, derivative_dims, NPY_DOUBLE, 0));
    auto* cn_left_output = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, vector_dims, NPY_DOUBLE, 0));
    auto* cn_right_output = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, vector_dims, NPY_DOUBLE, 0));
    if (!left_output || !right_output || !cn_left_output || !cn_right_output) {
        Py_XDECREF(left_output); Py_XDECREF(right_output);
        Py_XDECREF(cn_left_output); Py_XDECREF(cn_right_output);
        release(); return PyErr_NoMemory();
    }
    const auto* row = static_cast<const std::int32_t*>(PyArray_DATA(rows));
    const auto* column = static_cast<const std::int32_t*>(PyArray_DATA(columns));
    const auto* s = static_cast<const double*>(PyArray_DATA(overlap));
    const auto* dsl = static_cast<const double*>(PyArray_DATA(left_ds));
    const auto* dsr = static_cast<const double*>(PyArray_DATA(right_ds));
    const auto* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* cn = static_cast<const double*>(PyArray_DATA(coordination));
    const auto* atom = static_cast<const std::int64_t*>(PyArray_DATA(context->core_atom));
    const auto* base = static_cast<const double*>(PyArray_DATA(context->core_level));
    const auto* shift = static_cast<const double*>(PyArray_DATA(context->core_shift));
    const auto* polynomial = static_cast<const double*>(PyArray_DATA(context->core_polynomial));
    const auto* slater = static_cast<const double*>(PyArray_DATA(context->core_slater));
    const auto* angular = static_cast<const std::int64_t*>(PyArray_DATA(context->core_angular));
    const auto* radii = static_cast<const double*>(PyArray_DATA(context->core_radii));
    const auto* en = static_cast<const double*>(PyArray_DATA(context->core_en));
    auto* dl = static_cast<double*>(PyArray_DATA(left_output));
    auto* dr = static_cast<double*>(PyArray_DATA(right_output));
    auto* cnl = static_cast<double*>(PyArray_DATA(cn_left_output));
    auto* cnr = static_cast<double*>(PyArray_DATA(cn_right_output));
    constexpr double angular_scale[4][4] = {
        {1.85,2.04,2.00,2.00}, {2.04,2.23,2.00,2.00},
        {2.00,2.00,2.23,2.00}, {2.00,2.00,2.00,2.00}};
    std::vector<double> level(static_cast<std::size_t>(norbitals));
    for (npy_intp orbital = 0; orbital < norbitals; ++orbital)
        level[orbital] = base[orbital]-shift[orbital]*cn[atom[orbital]];
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp pair = 0; pair < pair_count; ++pair) {
        const auto lp = row[pair], rp = column[pair];
        if (lp == rp) { cnl[pair] = -shift[lp]; continue; }
        const auto la = atom[lp], ra = atom[rp];
        if (la == ra) continue;
        double displacement[3], distance2 = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            displacement[axis] = xyz[la*3+axis]-xyz[ra*3+axis];
            distance2 += displacement[axis]*displacement[axis];
        }
        const double distance = std::sqrt(distance2);
        const double radius = 0.5*(radii[la]+radii[ra]);
        const double root = std::sqrt(distance/radius);
        const double left_poly = 1.0+polynomial[lp]*root;
        const double right_poly = 1.0+polynomial[rp]*root;
        const double poly = left_poly*right_poly;
        const double radial = (polynomial[lp]*right_poly
            + polynomial[rp]*left_poly)/(2.0*std::sqrt(distance*radius));
        const double exponent = std::sqrt(2.0*std::sqrt(slater[lp]*slater[rp])
            /(slater[lp]+slater[rp]));
        const double delta_en = en[la]-en[ra];
        const double prefactor = 0.5*angular_scale[angular[lp]][angular[rp]]
            *exponent*(1.0+0.02*delta_en*delta_en);
        const double common = prefactor*(level[lp]+level[rp]);
        for (int axis = 0; axis < 3; ++axis) {
            const auto offset = pair*3+axis;
            const double radial_vector = radial*displacement[axis]/distance;
            dl[offset] = common*(poly*dsl[offset]+s[pair]*radial_vector);
            dr[offset] = common*(poly*dsr[offset]-s[pair]*radial_vector);
        }
        cnl[pair] = -prefactor*s[pair]*poly*shift[lp];
        cnr[pair] = -prefactor*s[pair]*poly*shift[rp];
    }
    Py_END_ALLOW_THREADS
    release();
    return Py_BuildValue("NNNN", left_output, right_output,
        cn_left_output, cn_right_output);
}

bool context_has_dense_model(const IntegralContextNative* context) {
    return context->core && context->overlap && context->pair_overlap
        && context->pair_dipole && context->pair_second && context->centers
        && context->atom && context->shell && context->reference && context->valence;
}

PyObject* context_dense_observables_native(PyObject*, PyObject* args) {
    PyObject *capsule, *density;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &density)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context) return nullptr;
    if (!context_has_dense_model(context)) {
        PyErr_SetString(PyExc_RuntimeError, "native GFN2 context has no dense model");
        return nullptr;
    }
    auto* forwarded = PyTuple_Pack(9, density, context->pair_overlap,
        context->pair_dipole, context->pair_second, context->centers,
        context->atom, context->shell, context->reference, context->valence);
    if (!forwarded) return nullptr;
    auto* result = dense_scc_observables_native(nullptr, forwarded);
    Py_DECREF(forwarded);
    return result;
}

PyObject* context_dense_fock_native(PyObject*, PyObject* args) {
    PyObject *capsule, *shell_potential, *qpot, *mupot, *thetapot, *d4;
    if (!PyArg_ParseTuple(args, "OOOOOO", &capsule, &shell_potential, &qpot,
            &mupot, &thetapot, &d4)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context) return nullptr;
    if (!context_has_dense_model(context)) {
        PyErr_SetString(PyExc_RuntimeError, "native GFN2 context has no dense model");
        return nullptr;
    }
    auto* forwarded = PyTuple_Pack(13, context->core, context->overlap,
        context->pair_overlap, context->pair_dipole, context->pair_second,
        context->centers, context->atom, context->shell, shell_potential,
        qpot, mupot, thetapot, d4);
    if (!forwarded) return nullptr;
    auto* result = dense_scc_fock_native(nullptr, forwarded);
    Py_DECREF(forwarded);
    return result;
}

PyObject* context_configure_scf_native(PyObject*, PyObject* args) {
    PyObject *capsule, *configuration;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &configuration)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context) return nullptr;
    if (!PyDict_Check(configuration)) {
        PyErr_SetString(PyExc_TypeError, "native SCF configuration must be a dictionary");
        return nullptr;
    }
    constexpr std::array<const char*, SCF_ARRAY_COUNT> names = {
        "interaction", "third_order", "anis_dipole_xc", "anis_quadrupole_xc",
        "anis_left", "anis_right", "anis_displacement", "anis_f3", "anis_f5",
        "anis_distance2", "d4_frequency_weights", "d4_weights",
        "d4_weight_derivatives", "d4_reference_charges", "d4_references",
        "d4_mask", "d4_effective", "d4_hardness", "d4_rows", "d4_columns",
        "d4_ratio", "d4_pair_factor"};
    constexpr std::array<int, SCF_ARRAY_COUNT> types = {
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NPY_INT64, NPY_INT64, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
        NPY_DOUBLE, NPY_DOUBLE, NPY_INT64, NPY_INT64, NPY_DOUBLE, NPY_DOUBLE};
    std::vector<PyArrayObject*> converted(SCF_ARRAY_COUNT, nullptr);
    for (std::size_t index = 0; index < SCF_ARRAY_COUNT; ++index) {
        auto* value = PyDict_GetItemString(configuration, names[index]);
        if (!value) {
            for (auto* array : converted) Py_XDECREF(array);
            PyErr_Format(PyExc_KeyError, "missing native SCF array: %s", names[index]);
            return nullptr;
        }
        converted[index] = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            value, types[index], NPY_ARRAY_IN_ARRAY));
        if (!converted[index]) {
            for (auto* array : converted) Py_XDECREF(array);
            return nullptr;
        }
    }
    const auto nshell = PyArray_SIZE(converted[SCF_THIRD]);
    const auto natom = PyArray_SIZE(converted[SCF_ANIS_DIPOLE_XC]);
    const auto anis_pairs = PyArray_SIZE(converted[SCF_ANIS_LEFT]);
    const auto d4_pairs = PyArray_SIZE(converted[SCF_D4_ROWS]);
    const bool valid = PyArray_SIZE(converted[SCF_INTERACTION]) == nshell*nshell
        && PyArray_SIZE(converted[SCF_ANIS_QUADRUPOLE_XC]) == natom
        && PyArray_SIZE(converted[SCF_ANIS_RIGHT]) == anis_pairs
        && PyArray_SIZE(converted[SCF_ANIS_DISPLACEMENT]) == anis_pairs*3
        && PyArray_SIZE(converted[SCF_ANIS_F3]) == anis_pairs
        && PyArray_SIZE(converted[SCF_ANIS_F5]) == anis_pairs
        && PyArray_SIZE(converted[SCF_ANIS_DISTANCE2]) == anis_pairs
        && PyArray_SIZE(converted[SCF_D4_EFFECTIVE]) == natom
        && PyArray_SIZE(converted[SCF_D4_HARDNESS]) == natom
        && PyArray_SIZE(converted[SCF_D4_COLUMNS]) == d4_pairs
        && PyArray_SIZE(converted[SCF_D4_RATIO]) == d4_pairs
        && PyArray_SIZE(converted[SCF_D4_PAIR_FACTOR]) == d4_pairs;
    if (!valid) {
        for (auto* array : converted) Py_XDECREF(array);
        PyErr_SetString(PyExc_ValueError, "incompatible persistent native SCF arrays");
        return nullptr;
    }
    for (auto* array : context->scf_arrays) Py_XDECREF(array);
    context->scf_arrays = std::move(converted);
    Py_RETURN_NONE;
}

PyObject* context_generalized_eigh_native(PyObject*, PyObject* args) {
    PyObject *capsule, *fock_o;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &fock_o)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context) return nullptr;
    if (!context->overlap) {
        PyErr_SetString(PyExc_RuntimeError,
            "native GFN2 context has no overlap matrix");
        return nullptr;
    }
#ifndef MATRIX_GFN2_NATIVE_LAPACK
    PyErr_SetString(PyExc_RuntimeError,
        "native generalized eigensolver was built without LAPACK");
    return nullptr;
#else
    auto* fock = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        fock_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!fock || PyArray_NDIM(fock) != 2
        || PyArray_DIM(fock, 0) != PyArray_DIM(fock, 1)
        || PyArray_SIZE(context->overlap) != PyArray_SIZE(fock)) {
        Py_XDECREF(fock);
        PyErr_SetString(PyExc_ValueError,
            "incompatible native generalized eigenproblem");
        return nullptr;
    }
    const npy_intp n_numpy = PyArray_DIM(fock, 0);
    if (n_numpy > std::numeric_limits<int>::max()) {
        Py_DECREF(fock);
        PyErr_SetString(PyExc_OverflowError, "eigenproblem is too large for LAPACK LP64");
        return nullptr;
    }
    int n = static_cast<int>(n_numpy);
    int lda = std::max(1, n), ldb = std::max(1, n), itype = 1;
    char jobz = 'V', uplo = 'L';
    context->generalized_eigh_matrix.resize(static_cast<std::size_t>(n)*n);
    context->generalized_eigh_factor = context->overlap_cholesky;
    context->generalized_eigh_values.resize(n);
    auto& a = context->generalized_eigh_matrix;
    auto& factor = context->generalized_eigh_factor;
    auto& values = context->generalized_eigh_values;
    const auto* f = static_cast<const double*>(PyArray_DATA(fock));
    // LAPACK is column-major.  The matrices are symmetric, nevertheless copy
    // explicitly so the returned eigenvectors can be transposed unambiguously.
    for (int row = 0; row < n; ++row) for (int column = 0; column < n; ++column) {
        a[row + column*n] = f[row*n + column];
    }
    if (context->overlap_cholesky.size() != static_cast<std::size_t>(n)*n) {
        Py_DECREF(fock);
        PyErr_SetString(PyExc_RuntimeError, "native overlap factor is unavailable");
        return nullptr;
    }
    int info = 0;
    Py_BEGIN_ALLOW_THREADS
    dsygst_(&itype, &uplo, &n, a.data(), &lda, factor.data(), &ldb, &info);
    Py_END_ALLOW_THREADS
    if (info != 0) {
        Py_DECREF(fock);
        PyErr_Format(PyExc_RuntimeError,
            "native LAPACK generalized reduction failed (info=%d)", info);
        return nullptr;
    }
    if (context->generalized_eigh_work.empty()
        || context->generalized_eigh_iwork.empty()) {
        int query_lwork = -1, query_liwork = -1;
        double work_query = 0.0;
        int iwork_query = 0;
        Py_BEGIN_ALLOW_THREADS
        dsyevd_(&jobz, &uplo, &n, a.data(), &lda, values.data(),
                &work_query, &query_lwork, &iwork_query, &query_liwork, &info);
        Py_END_ALLOW_THREADS
        if (info != 0) {
            Py_DECREF(fock);
            PyErr_Format(PyExc_RuntimeError,
                "native LAPACK symmetric workspace query failed (info=%d)", info);
            return nullptr;
        }
        context->generalized_eigh_work.resize(
            std::max(1, static_cast<int>(work_query)));
        context->generalized_eigh_iwork.resize(std::max(1, iwork_query));
    }
    auto& work = context->generalized_eigh_work;
    auto& iwork = context->generalized_eigh_iwork;
    int lwork = static_cast<int>(work.size());
    int liwork = static_cast<int>(iwork.size());
    Py_BEGIN_ALLOW_THREADS
    dsyevd_(&jobz, &uplo, &n, a.data(), &lda, values.data(),
            work.data(), &lwork, iwork.data(), &liwork, &info);
    if (info == 0)
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans,
            CblasNonUnit, n, n, 1.0, factor.data(), ldb, a.data(), lda);
    Py_END_ALLOW_THREADS
    Py_DECREF(fock);
    if (info != 0) {
        PyErr_Format(PyExc_RuntimeError,
            "native LAPACK symmetric eigensolver failed (info=%d)", info);
        return nullptr;
    }
    npy_intp value_dims[1] = {n_numpy};
    npy_intp vector_dims[2] = {n_numpy, n_numpy};
    auto* value_array = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, value_dims, NPY_DOUBLE, 0));
    auto* vector_array = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(2, vector_dims, NPY_DOUBLE, 0));
    if (!value_array || !vector_array) {
        Py_XDECREF(value_array); Py_XDECREF(vector_array);
        return PyErr_NoMemory();
    }
    std::copy(values.begin(), values.end(),
        static_cast<double*>(PyArray_DATA(value_array)));
    auto* vectors = static_cast<double*>(PyArray_DATA(vector_array));
    for (int row = 0; row < n; ++row) for (int column = 0; column < n; ++column)
        vectors[row*n + column] = a[row + column*n];
    return Py_BuildValue("NN", value_array, vector_array);
#endif
}

PyObject* d4_two_body_pairs_native(PyObject*, PyObject* args) {
    PyObject *charges_o, *frequency_o, *weights_o, *weight_derivatives_o;
    PyObject *reference_charges_o, *references_o, *mask_o, *effective_o;
    PyObject *hardness_o, *rows_o, *columns_o, *ratio_o, *pair_factor_o;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOO", &charges_o, &frequency_o,
            &weights_o, &weight_derivatives_o, &reference_charges_o,
            &references_o, &mask_o, &effective_o, &hardness_o, &rows_o,
            &columns_o, &ratio_o, &pair_factor_o)) return nullptr;
    auto f64 = [](PyObject* value) {
        return reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            value, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    };
    auto i64 = [](PyObject* value) {
        return reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            value, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    };
    std::array<PyArrayObject*, 11> values = {
        f64(charges_o), f64(frequency_o), f64(weights_o),
        f64(weight_derivatives_o), f64(reference_charges_o),
        f64(references_o), f64(mask_o), f64(effective_o), f64(hardness_o),
        f64(ratio_o), f64(pair_factor_o)};
    auto* rows = i64(rows_o);
    auto* columns = i64(columns_o);
    auto cleanup = [&]() {
        for (auto* value : values) Py_XDECREF(value);
        Py_XDECREF(rows); Py_XDECREF(columns);
    };
    for (auto* value : values) if (!value) { cleanup(); return nullptr; }
    if (!rows || !columns) { cleanup(); return nullptr; }
    const npy_intp natom = PyArray_SIZE(values[0]);
    const npy_intp nfrequency = PyArray_SIZE(values[1]);
    if (PyArray_NDIM(values[2]) != 2
        || PyArray_DIM(values[2], 0) != natom) {
        cleanup(); PyErr_SetString(PyExc_ValueError, "invalid native D4 weights");
        return nullptr;
    }
    const npy_intp nreference = PyArray_DIM(values[2], 1);
    const npy_intp npair = PyArray_SIZE(rows);
    const bool valid = PyArray_SIZE(columns) == npair
        && PyArray_SIZE(values[3]) == natom*nreference
        && PyArray_SIZE(values[4]) == natom*nreference
        && PyArray_SIZE(values[5]) == natom*nreference*nfrequency
        && PyArray_SIZE(values[6]) == natom*nreference
        && PyArray_SIZE(values[7]) == natom
        && PyArray_SIZE(values[8]) == natom
        && PyArray_SIZE(values[9]) == npair
        && PyArray_SIZE(values[10]) == npair;
    if (!valid) {
        cleanup(); PyErr_SetString(PyExc_ValueError, "incompatible native D4 arrays");
        return nullptr;
    }
    const auto* charges = static_cast<const double*>(PyArray_DATA(values[0]));
    const auto* frequency = static_cast<const double*>(PyArray_DATA(values[1]));
    const auto* weights = static_cast<const double*>(PyArray_DATA(values[2]));
    const auto* dweights = static_cast<const double*>(PyArray_DATA(values[3]));
    const auto* reference_charges = static_cast<const double*>(PyArray_DATA(values[4]));
    const auto* references = static_cast<const double*>(PyArray_DATA(values[5]));
    const auto* mask = static_cast<const double*>(PyArray_DATA(values[6]));
    const auto* effective = static_cast<const double*>(PyArray_DATA(values[7]));
    const auto* hardness = static_cast<const double*>(PyArray_DATA(values[8]));
    const auto* pair_rows = static_cast<const std::int64_t*>(PyArray_DATA(rows));
    const auto* pair_columns = static_cast<const std::int64_t*>(PyArray_DATA(columns));
    const auto* ratio = static_cast<const double*>(PyArray_DATA(values[9]));
    const auto* pair_factor = static_cast<const double*>(PyArray_DATA(values[10]));
    for (npy_intp atom = 0; atom < natom; ++atom) {
        if (effective[atom] + charges[atom] <= 0.0) {
            cleanup(); PyErr_SetString(PyExc_ValueError,
                "D4 effective nuclear charge plus partial charge must be positive");
            return nullptr;
        }
    }
    std::vector<double> alpha(natom*nfrequency, 0.0);
    std::vector<double> charge_alpha(natom*nfrequency, 0.0);
    std::vector<double> coordination_alpha(natom*nfrequency, 0.0);
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp atom = 0; atom < natom; ++atom) {
        const double denominator = effective[atom] + charges[atom];
        for (npy_intp reference = 0; reference < nreference; ++reference) {
            const auto state = atom*nreference + reference;
            if (mask[state] == 0.0) continue;
            const double numerator = effective[atom] + reference_charges[state];
            const double exponential = std::exp(
                2.0*hardness[atom]*(1.0 - numerator/denominator));
            const double scale = std::exp(3.0*(1.0 - exponential));
            const double scale_derivative = scale*(-3.0*exponential)
                *(2.0*hardness[atom]*numerator/(denominator*denominator));
            for (npy_intp frequency_index = 0;
                 frequency_index < nfrequency; ++frequency_index) {
                const double polarizability = references[
                    state*nfrequency + frequency_index];
                const auto target = atom*nfrequency + frequency_index;
                alpha[target] += weights[state]*scale*polarizability;
                charge_alpha[target] += weights[state]*scale_derivative*polarizability;
                coordination_alpha[target] += dweights[state]*scale*polarizability;
            }
        }
    }
    Py_END_ALLOW_THREADS
    npy_intp atom_dims[1] = {natom}, pair_dims[1] = {npair};
    auto* charge_derivative = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, atom_dims, NPY_DOUBLE, 0));
    auto* coordination_derivative = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, atom_dims, NPY_DOUBLE, 0));
    auto* c6_array = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, pair_dims, NPY_DOUBLE, 0));
    auto* c8_array = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, pair_dims, NPY_DOUBLE, 0));
    auto* coordination_left_array = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, pair_dims, NPY_DOUBLE, 0));
    auto* coordination_right_array = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, pair_dims, NPY_DOUBLE, 0));
    if (!charge_derivative || !coordination_derivative || !c6_array || !c8_array
        || !coordination_left_array || !coordination_right_array) {
        cleanup(); Py_XDECREF(charge_derivative); Py_XDECREF(coordination_derivative);
        Py_XDECREF(c6_array); Py_XDECREF(c8_array);
        Py_XDECREF(coordination_left_array); Py_XDECREF(coordination_right_array);
        return PyErr_NoMemory();
    }
    auto* dq = static_cast<double*>(PyArray_DATA(charge_derivative));
    auto* dcn = static_cast<double*>(PyArray_DATA(coordination_derivative));
    auto* c6 = static_cast<double*>(PyArray_DATA(c6_array));
    auto* c8 = static_cast<double*>(PyArray_DATA(c8_array));
    auto* cnleft = static_cast<double*>(PyArray_DATA(coordination_left_array));
    auto* cnright = static_cast<double*>(PyArray_DATA(coordination_right_array));
    constexpr double factor = 3.0/M_PI;
    double energy = 0.0;
    double maximum_self_c6 = 0.0;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp atom = 0; atom < natom; ++atom) {
        double self_c6 = 0.0;
        for (npy_intp frequency_index = 0;
             frequency_index < nfrequency; ++frequency_index) {
            const double polarizability = alpha[atom*nfrequency + frequency_index];
            self_c6 += factor*frequency[frequency_index]
                *polarizability*polarizability;
        }
        maximum_self_c6 = std::max(maximum_self_c6, self_c6);
    }
    for (npy_intp pair = 0; pair < npair; ++pair) {
        const auto left_atom = pair_rows[pair], right_atom = pair_columns[pair];
        double pair_c6 = 0.0, charge_left = 0.0, charge_right = 0.0;
        double coordination_left = 0.0, coordination_right = 0.0;
        for (npy_intp frequency_index = 0;
             frequency_index < nfrequency; ++frequency_index) {
            const double weighted = factor*frequency[frequency_index];
            const auto left_index = left_atom*nfrequency + frequency_index;
            const auto right_index = right_atom*nfrequency + frequency_index;
            pair_c6 += weighted*alpha[left_index]*alpha[right_index];
            charge_left += weighted*charge_alpha[left_index]*alpha[right_index];
            charge_right += weighted*charge_alpha[right_index]*alpha[left_index];
            coordination_left += weighted*coordination_alpha[left_index]*alpha[right_index];
            coordination_right += weighted*coordination_alpha[right_index]*alpha[left_index];
        }
        c6[pair] = pair_c6;
        c8[pair] = pair_c6*ratio[pair];
        cnleft[pair] = coordination_left;
        cnright[pair] = coordination_right;
        energy -= pair_c6*pair_factor[pair];
        dq[left_atom] -= charge_left*pair_factor[pair];
        dq[right_atom] -= charge_right*pair_factor[pair];
        dcn[left_atom] -= coordination_left*pair_factor[pair];
        dcn[right_atom] -= coordination_right*pair_factor[pair];
    }
    Py_END_ALLOW_THREADS
    cleanup();
    return Py_BuildValue("dNNNNNNd", energy, charge_derivative,
        coordination_derivative, c6_array, c8_array,
        coordination_left_array, coordination_right_array, maximum_self_c6);
}

PyObject* context_scf_state_fock_native(PyObject*, PyObject* args) {
    PyObject *capsule, *density_o;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &density_o)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context) return nullptr;
    if (!context_has_dense_model(context)
        || context->scf_arrays.size() != SCF_ARRAY_COUNT) {
        PyErr_SetString(PyExc_RuntimeError,
            "native GFN2 context has no complete SCF configuration");
        return nullptr;
    }
    auto* density = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        density_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const npy_intp n = PyArray_DIM(context->core, 0);
    if (!density || PyArray_SIZE(density) != n*n) {
        Py_XDECREF(density);
        PyErr_SetString(PyExc_ValueError, "incompatible native SCF density");
        return nullptr;
    }
    auto* observable_args = PyTuple_Pack(2, capsule, density);
    auto* observables = observable_args
        ? context_dense_observables_native(nullptr, observable_args) : nullptr;
    Py_XDECREF(observable_args);
    if (!observables) { Py_DECREF(density); return nullptr; }
    auto* shell_q = reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(observables, 0));
    auto* charges = PyTuple_GET_ITEM(observables, 1);
    auto* dipoles = PyTuple_GET_ITEM(observables, 2);
    auto* quadrupoles = PyTuple_GET_ITEM(observables, 3);
    const npy_intp nshell = PyArray_SIZE(shell_q);
    npy_intp shell_dims[1] = {nshell};
    auto* shell_potential = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, shell_dims, NPY_DOUBLE, 0));
    if (!shell_potential) { Py_DECREF(observables); Py_DECREF(density); return nullptr; }
    const auto* qshell = static_cast<const double*>(PyArray_DATA(shell_q));
    const auto* interaction = static_cast<const double*>(PyArray_DATA(
        context->scf_arrays[SCF_INTERACTION]));
    const auto* third = static_cast<const double*>(PyArray_DATA(
        context->scf_arrays[SCF_THIRD]));
    auto* potential = static_cast<double*>(PyArray_DATA(shell_potential));
    double monopole_energy_value = 0.0;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < nshell; ++i) {
        double quadratic = 0.0;
        for (npy_intp j = 0; j < nshell; ++j)
            quadratic += interaction[i*nshell+j]*qshell[j];
        potential[i] = quadratic + third[i]*qshell[i]*qshell[i];
        monopole_energy_value += 0.5*qshell[i]*quadratic
            + third[i]*qshell[i]*qshell[i]*qshell[i]/3.0;
    }
    Py_END_ALLOW_THREADS
    auto* anis_args = PyTuple_Pack(11, charges, dipoles, quadrupoles,
        context->scf_arrays[SCF_ANIS_DIPOLE_XC],
        context->scf_arrays[SCF_ANIS_QUADRUPOLE_XC],
        context->scf_arrays[SCF_ANIS_LEFT], context->scf_arrays[SCF_ANIS_RIGHT],
        context->scf_arrays[SCF_ANIS_DISPLACEMENT],
        context->scf_arrays[SCF_ANIS_F3], context->scf_arrays[SCF_ANIS_F5],
        context->scf_arrays[SCF_ANIS_DISTANCE2]);
    auto* anisotropic = anis_args
        ? anisotropic_energy_potentials_native(nullptr, anis_args) : nullptr;
    Py_XDECREF(anis_args);
    if (!anisotropic) {
        Py_DECREF(shell_potential); Py_DECREF(observables); Py_DECREF(density);
        return nullptr;
    }
    auto* d4_args = PyTuple_Pack(13, charges,
        context->scf_arrays[SCF_D4_FREQUENCY],
        context->scf_arrays[SCF_D4_WEIGHTS],
        context->scf_arrays[SCF_D4_WEIGHT_DERIVATIVES],
        context->scf_arrays[SCF_D4_REFERENCE_CHARGES],
        context->scf_arrays[SCF_D4_REFERENCES], context->scf_arrays[SCF_D4_MASK],
        context->scf_arrays[SCF_D4_EFFECTIVE], context->scf_arrays[SCF_D4_HARDNESS],
        context->scf_arrays[SCF_D4_ROWS], context->scf_arrays[SCF_D4_COLUMNS],
        context->scf_arrays[SCF_D4_RATIO], context->scf_arrays[SCF_D4_PAIR_FACTOR]);
    auto* d4 = d4_args ? d4_two_body_pairs_native(nullptr, d4_args) : nullptr;
    Py_XDECREF(d4_args);
    if (!d4) {
        Py_DECREF(anisotropic); Py_DECREF(shell_potential);
        Py_DECREF(observables); Py_DECREF(density); return nullptr;
    }
    auto* fock_args = PyTuple_Pack(6, capsule, shell_potential,
        PyTuple_GET_ITEM(anisotropic, 1), PyTuple_GET_ITEM(anisotropic, 2),
        PyTuple_GET_ITEM(anisotropic, 3), PyTuple_GET_ITEM(d4, 1));
    auto* fock = fock_args
        ? context_dense_fock_native(nullptr, fock_args) : nullptr;
    Py_XDECREF(fock_args);
    if (!fock) {
        Py_DECREF(d4); Py_DECREF(anisotropic); Py_DECREF(shell_potential);
        Py_DECREF(observables); Py_DECREF(density); return nullptr;
    }
    const auto* p = static_cast<const double*>(PyArray_DATA(density));
    const auto* h = static_cast<const double*>(PyArray_DATA(context->core));
    double electronic_energy = 0.0;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp index = 0; index < n*n; ++index)
        electronic_energy += p[index]*h[index];
    Py_END_ALLOW_THREADS
    const double anisotropic_energy = PyFloat_AsDouble(PyTuple_GET_ITEM(anisotropic, 0));
    const double d4_energy = PyFloat_AsDouble(PyTuple_GET_ITEM(d4, 0));
    const double total_energy = electronic_energy + monopole_energy_value
        + anisotropic_energy + d4_energy;
    Py_XDECREF(context->last_observables);
    Py_XDECREF(context->last_anisotropic);
    Py_XDECREF(context->last_d4);
    context->last_observables = observables;
    context->last_anisotropic = anisotropic;
    context->last_d4 = d4;
    context->last_scf_energy = total_energy;
    Py_DECREF(shell_potential); Py_DECREF(density);
    if (PyErr_Occurred()) { Py_DECREF(fock); return nullptr; }
    return Py_BuildValue("Nd", fock, total_energy);
}

PyObject* context_last_scf_state_native(PyObject*, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context) return nullptr;
    if (!context->last_observables || !context->last_anisotropic
        || !context->last_d4) {
        PyErr_SetString(PyExc_RuntimeError, "native SCF context has no final state");
        return nullptr;
    }
    return Py_BuildValue("OOOd", context->last_observables,
        context->last_anisotropic, context->last_d4,
        context->last_scf_energy);
}

std::vector<double> native_spin_occupations(
    const double* energies, npy_intp size, int count, double temperature) {
    std::vector<double> occupations(size, 0.0);
    if (count <= 0) return occupations;
    if (count >= size) {
        std::fill(occupations.begin(), occupations.end(), 1.0);
        return occupations;
    }
    constexpr double boltzmann_hartree_per_kelvin = 3.166808578545117e-6;
    const double thermal = boltzmann_hartree_per_kelvin*temperature;
    if (thermal <= 0.0) {
        std::fill(occupations.begin(), occupations.begin()+count, 1.0);
        return occupations;
    }
    double low = energies[0] - 64.0*thermal;
    double high = energies[size-1] + 64.0*thermal;
    for (int iteration = 0; iteration < 200; ++iteration) {
        const double chemical_potential = 0.5*(low+high);
        double population = 0.0;
        for (npy_intp orbital = 0; orbital < size; ++orbital) {
            const double argument = std::max(-700.0, std::min(700.0,
                (energies[orbital]-chemical_potential)/thermal));
            occupations[orbital] = 1.0/(std::exp(argument)+1.0);
            population += occupations[orbital];
        }
        if (std::abs(population-count) <= 2.0e-14) break;
        if (population > count) high = chemical_potential;
        else low = chemical_potential;
    }
    return occupations;
}

PyObject* context_resident_iscf_native(PyObject*, PyObject* args) {
    PyObject *capsule, *density_o;
    int electrons, multiplicity;
    double temperature, wavefunction_tolerance, energy_tolerance;
    if (!PyArg_ParseTuple(args, "OOiiddd", &capsule, &density_o, &electrons,
            &multiplicity, &temperature, &wavefunction_tolerance,
            &energy_tolerance)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context || !context->overlap) return nullptr;
    auto* density = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        density_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const npy_intp n = PyArray_DIM(context->overlap, 0);
    const int unpaired = multiplicity-1;
    if (!density || PyArray_SIZE(density) != n*n || electrons < 0
        || unpaired < 0 || (electrons-unpaired)%2 != 0) {
        Py_XDECREF(density);
        PyErr_SetString(PyExc_ValueError, "invalid native resident ISCF state");
        return nullptr;
    }
    auto* state_args = PyTuple_Pack(2, capsule, density);
    auto* initial_state = state_args
        ? context_scf_state_fock_native(nullptr, state_args) : nullptr;
    Py_XDECREF(state_args);
    if (!initial_state) { Py_DECREF(density); return nullptr; }
    auto* fock = PyTuple_GET_ITEM(initial_state, 0);
    const double initial_energy = PyFloat_AsDouble(PyTuple_GET_ITEM(initial_state, 1));
    auto* eigen_args = PyTuple_Pack(2, capsule, fock);
    auto* eigen = eigen_args
        ? context_generalized_eigh_native(nullptr, eigen_args) : nullptr;
    Py_XDECREF(eigen_args);
    if (!eigen) { Py_DECREF(initial_state); Py_DECREF(density); return nullptr; }
    auto* energies = reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(eigen, 0));
    auto* coefficients = reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(eigen, 1));
    const auto* eigenvalues = static_cast<const double*>(PyArray_DATA(energies));
    const int alpha_count = (electrons+unpaired)/2;
    const int beta_count = electrons-alpha_count;
    auto alpha = native_spin_occupations(eigenvalues, n, alpha_count, temperature);
    auto beta = native_spin_occupations(eigenvalues, n, beta_count, temperature);
    npy_intp vector_dims[1] = {n}, matrix_dims[2] = {n,n};
    auto* occupations = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, vector_dims, NPY_DOUBLE, 0));
    auto* updated = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(2, matrix_dims, NPY_DOUBLE, 0));
    if (!occupations || !updated) {
        Py_XDECREF(occupations); Py_XDECREF(updated); Py_DECREF(eigen);
        Py_DECREF(initial_state); Py_DECREF(density); return PyErr_NoMemory();
    }
    auto* occupation_values = static_cast<double*>(PyArray_DATA(occupations));
    const auto* coefficient_values = static_cast<const double*>(
        PyArray_DATA(coefficients));
    auto* density_values = static_cast<double*>(PyArray_DATA(updated));
    std::vector<double> weighted_coefficients(static_cast<std::size_t>(n)*n);
    for (npy_intp orbital = 0; orbital < n; ++orbital) {
        occupation_values[orbital] = alpha[orbital]+beta[orbital];
        const double scale = std::sqrt(std::max(0.0, occupation_values[orbital]));
        for (npy_intp row = 0; row < n; ++row)
            weighted_coefficients[row*n+orbital] = coefficient_values[row*n+orbital]*scale;
    }
#ifdef MATRIX_GFN2_NATIVE_LAPACK
    const int ni = static_cast<int>(n);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        ni, ni, ni, 1.0, weighted_coefficients.data(), ni,
        weighted_coefficients.data(), ni, 0.0, density_values, ni);
#else
    for (npy_intp row = 0; row < n; ++row) for (npy_intp column = 0; column < n; ++column) {
        double value = 0.0;
        for (npy_intp orbital = 0; orbital < n; ++orbital)
            value += weighted_coefficients[row*n+orbital]
                *weighted_coefficients[column*n+orbital];
        density_values[row*n+column] = value;
    }
#endif
    const auto* initial_density = static_cast<const double*>(PyArray_DATA(density));
    double density_residual = 0.0;
    for (npy_intp index = 0; index < n*n; ++index)
        density_residual = std::max(density_residual,
            std::abs(density_values[index]-initial_density[index]));
    auto* final_args = PyTuple_Pack(2, capsule, updated);
    auto* final_state = final_args
        ? context_scf_state_fock_native(nullptr, final_args) : nullptr;
    Py_XDECREF(final_args);
    if (!final_state) {
        Py_DECREF(occupations); Py_DECREF(updated); Py_DECREF(eigen);
        Py_DECREF(initial_state); Py_DECREF(density); return nullptr;
    }
    auto* final_fock = PyTuple_GET_ITEM(final_state, 0);
    const double final_energy = PyFloat_AsDouble(PyTuple_GET_ITEM(final_state, 1));
    const double energy_residual = std::abs(final_energy-initial_energy);
    const int converged = density_residual < wavefunction_tolerance
        && energy_residual < energy_tolerance;
    // Return the Fock matrix belonging to ``updated``.  Returning the input
    // Fock made a non-certified native step unusable by the Python safeguard:
    // it had to discard the entire step and repeat both the state build and
    // generalized diagonalization.  Keeping the final state makes this a
    // literal first ISCF cycle while preserving the independent convergence
    // tests performed by the caller.
    Py_INCREF(energies); Py_INCREF(coefficients); Py_INCREF(final_fock);
    Py_INCREF(fock);
    Py_DECREF(final_state); Py_DECREF(eigen); Py_DECREF(initial_state); Py_DECREF(density);
    return Py_BuildValue("iNNNNNNddd", converged, updated, energies,
        coefficients, occupations, final_fock, fock, density_residual,
        energy_residual, final_energy);
}

bool solve_small_dense_system(std::vector<double>& matrix,
                              std::vector<double>& rhs, int size) {
    for (int column = 0; column < size; ++column) {
        int pivot = column;
        for (int row = column + 1; row < size; ++row)
            if (std::abs(matrix[row*size + column])
                    > std::abs(matrix[pivot*size + column])) pivot = row;
        if (std::abs(matrix[pivot*size + column]) < 1.0e-24) return false;
        if (pivot != column) {
            for (int item = column; item < size; ++item)
                std::swap(matrix[column*size + item], matrix[pivot*size + item]);
            std::swap(rhs[column], rhs[pivot]);
        }
        const double diagonal = matrix[column*size + column];
        for (int row = column + 1; row < size; ++row) {
            const double factor = matrix[row*size + column] / diagonal;
            matrix[row*size + column] = 0.0;
            for (int item = column + 1; item < size; ++item)
                matrix[row*size + item] -= factor * matrix[column*size + item];
            rhs[row] -= factor * rhs[column];
        }
    }
    for (int row = size - 1; row >= 0; --row) {
        double value = rhs[row];
        for (int column = row + 1; column < size; ++column)
            value -= matrix[row*size + column] * rhs[column];
        rhs[row] = value / matrix[row*size + row];
    }
    return true;
}

void dense_commutator_residual(const double* fock, const double* density,
                               const double* overlap, int size,
                               std::vector<double>& residual) {
    const std::size_t count = static_cast<std::size_t>(size)*size;
    std::vector<double> fp(count), fps(count);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size,
        1.0, fock, size, density, size, 0.0, fp.data(), size);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size,
        1.0, fp.data(), size, overlap, size, 0.0, fps.data(), size);
    residual.resize(count);
    // F, P and S are symmetric, so (F P S)^T = S P F.  Form the
    // commutator with two matrix products instead of four.
    for (int row = 0; row < size; ++row)
        for (int column = 0; column < size; ++column)
            residual[static_cast<std::size_t>(row)*size+column]
                = fps[static_cast<std::size_t>(row)*size+column]
                - fps[static_cast<std::size_t>(column)*size+row];
}

std::vector<double> resident_diis_fock(
        const std::vector<std::vector<double>>& focks,
        const std::vector<std::vector<double>>& errors) {
    if (focks.size() < 2) return focks.back();
    const int count = static_cast<int>(focks.size());
    const int dimension = count + 1;
    std::vector<double> system(static_cast<std::size_t>(dimension)*dimension, 0.0);
    double scale = 1.0;
    for (int row = 0; row < count; ++row) for (int column = 0; column < count; ++column) {
        const double value = std::inner_product(
            errors[row].begin(), errors[row].end(), errors[column].begin(), 0.0);
        system[row*dimension + column] = value;
        scale = std::max(scale, std::abs(value));
    }
    for (int index = 0; index < count; ++index) {
        system[index*dimension + index] += scale*1.0e-12;
        system[index*dimension + count] = -1.0;
        system[count*dimension + index] = -1.0;
    }
    std::vector<double> rhs(dimension, 0.0);
    rhs[count] = -1.0;
    if (!solve_small_dense_system(system, rhs, dimension)) return focks.back();
    double coefficient_sum = 0.0, absolute_sum = 0.0, maximum = 0.0;
    for (int index = 0; index < count; ++index) {
        if (!std::isfinite(rhs[index])) return focks.back();
        coefficient_sum += rhs[index];
        absolute_sum += std::abs(rhs[index]);
        maximum = std::max(maximum, std::abs(rhs[index]));
    }
    if (std::abs(coefficient_sum - 1.0) > 1.0e-8
        || maximum > 8.0 || absolute_sum > 16.0) return focks.back();
    std::vector<double> result(focks.back().size(), 0.0);
    for (int item = 0; item < count; ++item)
        for (std::size_t index = 0; index < result.size(); ++index)
            result[index] += rhs[item]*focks[item][index];
    return result;
}

double native_tblite_mixer_rms(PyObject* before, PyObject* after) {
    if (!before || !after) return std::numeric_limits<double>::infinity();
    double squared = 0.0;
    npy_intp components = 0;
    for (int item : {0, 2}) {
        auto* left = reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(before, item));
        auto* right = reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(after, item));
        const auto* first = static_cast<const double*>(PyArray_DATA(left));
        const auto* second = static_cast<const double*>(PyArray_DATA(right));
        for (npy_intp index = 0; index < PyArray_SIZE(left); ++index) {
            const double difference = second[index]-first[index];
            squared += difference*difference;
            ++components;
        }
    }
    auto* left = reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(before, 3));
    auto* right = reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(after, 3));
    const auto* first = static_cast<const double*>(PyArray_DATA(left));
    const auto* second = static_cast<const double*>(PyArray_DATA(right));
    constexpr int upper[6] = {0, 1, 2, 4, 5, 8};
    for (npy_intp atom = 0; atom < PyArray_DIM(left, 0); ++atom)
        for (int index : upper) {
            const double difference = second[atom*9+index]-first[atom*9+index];
            squared += difference*difference;
            ++components;
        }
    return components ? std::sqrt(squared/components)
                      : std::numeric_limits<double>::infinity();
}

PyObject* context_resident_diis_native(PyObject*, PyObject* args) {
    PyObject *capsule, *density_o;
    int electrons, multiplicity, maximum_cycles;
    double temperature, wavefunction_tolerance, energy_tolerance;
    double mixer_tolerance = -1.0;
    if (!PyArg_ParseTuple(args, "OOiidddi|d", &capsule, &density_o, &electrons,
            &multiplicity, &temperature, &wavefunction_tolerance,
            &energy_tolerance, &maximum_cycles, &mixer_tolerance)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context || !context->overlap) return nullptr;
    auto* input_density = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        density_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const int n = static_cast<int>(PyArray_DIM(context->overlap, 0));
    const int unpaired = multiplicity - 1;
    if (!input_density || PyArray_SIZE(input_density) != n*n || electrons < 0
        || unpaired < 0 || (electrons-unpaired)%2 != 0 || maximum_cycles < 1) {
        Py_XDECREF(input_density);
        PyErr_SetString(PyExc_ValueError, "invalid native resident DIIS state");
        return nullptr;
    }
    maximum_cycles = std::min(maximum_cycles, 12);
    const std::size_t matrix_count = static_cast<std::size_t>(n)*n;
    const auto* overlap = static_cast<const double*>(PyArray_DATA(context->overlap));
    auto state_for = [&](PyArrayObject* candidate) -> PyObject* {
        auto* packed = PyTuple_Pack(2, capsule, candidate);
        auto* state = packed ? context_scf_state_fock_native(nullptr, packed) : nullptr;
        Py_XDECREF(packed);
        return state;
    };
    auto* initial_state = state_for(input_density);
    if (!initial_state) { Py_DECREF(input_density); return nullptr; }
    auto* initial_fock_object = reinterpret_cast<PyArrayObject*>(
        PyTuple_GET_ITEM(initial_state, 0));
    const double initial_energy = PyFloat_AsDouble(PyTuple_GET_ITEM(initial_state, 1));
    std::vector<std::vector<double>> focks(1), errors(1);
    const auto* initial_fock = static_cast<const double*>(PyArray_DATA(initial_fock_object));
    focks[0].assign(initial_fock, initial_fock + matrix_count);
    dense_commutator_residual(initial_fock,
        static_cast<const double*>(PyArray_DATA(input_density)), overlap, n, errors[0]);
    npy_intp matrix_dims[2] = {n,n};
    auto* initial_fock_copy = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(2, matrix_dims, NPY_DOUBLE));
    if (!initial_fock_copy) {
        Py_DECREF(initial_state); Py_DECREF(input_density); return nullptr;
    }
    std::copy(focks[0].begin(), focks[0].end(),
        static_cast<double*>(PyArray_DATA(initial_fock_copy)));
    Py_DECREF(initial_state);

    Py_INCREF(input_density);
    PyArrayObject* current_density = input_density;
    PyArrayObject *final_energies = nullptr, *final_coefficients = nullptr;
    PyArrayObject *final_occupations = nullptr, *final_fock = initial_fock_copy;
    PyArrayObject* restart_density = nullptr;
    const double* converged_density = nullptr;
    double* restart_values = nullptr;
    double contraction = 0.0, correction = 0.0;
    Py_INCREF(final_fock);
    double current_energy = initial_energy;
    double final_density_residual = std::numeric_limits<double>::infinity();
    double final_energy_residual = std::numeric_limits<double>::infinity();
    double final_mixer_residual = std::numeric_limits<double>::infinity();
    bool converged = false;
    int completed_cycles = 0;
    std::vector<double> previous_density_error, final_density_error;
    const int alpha_count = (electrons+unpaired)/2;
    const int beta_count = electrons-alpha_count;
    for (int cycle = 1; cycle <= maximum_cycles; ++cycle) {
        auto effective = resident_diis_fock(focks, errors);
        auto* effective_array = reinterpret_cast<PyArrayObject*>(
            PyArray_SimpleNew(2, matrix_dims, NPY_DOUBLE));
        if (!effective_array) goto resident_diis_failure;
        std::copy(effective.begin(), effective.end(),
            static_cast<double*>(PyArray_DATA(effective_array)));
        auto* eigen_args = PyTuple_Pack(2, capsule, effective_array);
        auto* eigen = eigen_args
            ? context_generalized_eigh_native(nullptr, eigen_args) : nullptr;
        Py_XDECREF(eigen_args); Py_DECREF(effective_array);
        if (!eigen) goto resident_diis_failure;
        auto* energies = reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(eigen, 0));
        auto* coefficients = reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(eigen, 1));
        const auto* eigenvalues = static_cast<const double*>(PyArray_DATA(energies));
        auto alpha = native_spin_occupations(eigenvalues, n, alpha_count, temperature);
        auto beta = native_spin_occupations(eigenvalues, n, beta_count, temperature);
        npy_intp vector_dims[1] = {n};
        auto* occupations = reinterpret_cast<PyArrayObject*>(
            PyArray_EMPTY(1, vector_dims, NPY_DOUBLE, 0));
        auto* updated = reinterpret_cast<PyArrayObject*>(
            PyArray_EMPTY(2, matrix_dims, NPY_DOUBLE, 0));
        if (!occupations || !updated) {
            Py_XDECREF(occupations); Py_XDECREF(updated); Py_DECREF(eigen);
            goto resident_diis_failure;
        }
        auto* occupation_values = static_cast<double*>(PyArray_DATA(occupations));
        const auto* coefficient_values = static_cast<const double*>(PyArray_DATA(coefficients));
        auto* density_values = static_cast<double*>(PyArray_DATA(updated));
        std::vector<double> weighted(matrix_count);
        for (int orbital = 0; orbital < n; ++orbital) {
            occupation_values[orbital] = alpha[orbital]+beta[orbital];
            const double scale = std::sqrt(std::max(0.0, occupation_values[orbital]));
            for (int row = 0; row < n; ++row)
                weighted[row*n+orbital] = coefficient_values[row*n+orbital]*scale;
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            n, n, n, 1.0, weighted.data(), n, weighted.data(), n,
            0.0, density_values, n);
        const auto* old_density = static_cast<const double*>(PyArray_DATA(current_density));
        final_density_residual = 0.0;
        previous_density_error = std::move(final_density_error);
        final_density_error.resize(matrix_count);
        for (std::size_t index = 0; index < matrix_count; ++index)
            final_density_residual = std::max(final_density_residual,
                std::abs(final_density_error[index] =
                    density_values[index]-old_density[index]));
        PyObject* previous_observables = mixer_tolerance > 0.0
            ? context->last_observables : nullptr;
        Py_XINCREF(previous_observables);
        auto* updated_state = state_for(updated);
        if (!updated_state) {
            Py_XDECREF(previous_observables);
            Py_DECREF(occupations); Py_DECREF(updated); Py_DECREF(eigen);
            goto resident_diis_failure;
        }
        if (mixer_tolerance > 0.0)
            final_mixer_residual = native_tblite_mixer_rms(
                previous_observables, context->last_observables);
        Py_XDECREF(previous_observables);
        auto* updated_fock = reinterpret_cast<PyArrayObject*>(
            PyTuple_GET_ITEM(updated_state, 0));
        const double updated_energy = PyFloat_AsDouble(PyTuple_GET_ITEM(updated_state, 1));
        final_energy_residual = std::abs(updated_energy-current_energy);
        const bool uphill = cycle >= 4 && updated_energy > current_energy + 1.0e-12;
        if (uphill) {
            Py_DECREF(updated_state); Py_DECREF(occupations);
            Py_DECREF(updated); Py_DECREF(eigen);
            break;
        }
        Py_XDECREF(final_energies); Py_XDECREF(final_coefficients);
        Py_XDECREF(final_occupations); Py_XDECREF(final_fock);
        final_energies = energies; Py_INCREF(final_energies);
        final_coefficients = coefficients; Py_INCREF(final_coefficients);
        final_occupations = occupations;
        final_fock = updated_fock; Py_INCREF(final_fock);
        Py_DECREF(current_density); current_density = updated;
        current_energy = updated_energy;
        ++completed_cycles;
        const auto* fock_values = static_cast<const double*>(PyArray_DATA(updated_fock));
        std::vector<double> next_fock(fock_values, fock_values+matrix_count);
        std::vector<double> next_error;
        dense_commutator_residual(fock_values, density_values, overlap, n, next_error);
        focks.push_back(std::move(next_fock));
        errors.push_back(std::move(next_error));
        if (focks.size() > 8) { focks.erase(focks.begin()); errors.erase(errors.begin()); }
        Py_DECREF(updated_state); Py_DECREF(eigen);
        converged = final_energy_residual < energy_tolerance
            && (mixer_tolerance > 0.0
                ? final_mixer_residual < mixer_tolerance
                : final_density_residual < wavefunction_tolerance);
        if (converged) break;
    }
    if (!final_energies || !final_coefficients || !final_occupations) {
        PyErr_SetString(PyExc_RuntimeError, "resident DIIS produced no orbital state");
        goto resident_diis_failure;
    }
    restart_density = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(2, matrix_dims, NPY_DOUBLE));
    if (!restart_density) goto resident_diis_failure;
    converged_density = static_cast<const double*>(
        PyArray_DATA(current_density));
    restart_values = static_cast<double*>(PyArray_DATA(restart_density));
    if (previous_density_error.size() == matrix_count
        && final_density_error.size() == matrix_count) {
        const double denominator = std::inner_product(
            previous_density_error.begin(), previous_density_error.end(),
            previous_density_error.begin(), 0.0);
        if (denominator > std::numeric_limits<double>::min())
            contraction = std::inner_product(
                previous_density_error.begin(), previous_density_error.end(),
                final_density_error.begin(), 0.0) / denominator;
        contraction = std::max(-0.15, std::min(0.15, contraction));
    }
    correction = contraction/(1.0-contraction);
    for (std::size_t index = 0; index < matrix_count; ++index)
        restart_values[index] = converged_density[index]
            + correction*(final_density_error.empty() ? 0.0
                                                       : final_density_error[index]);
    Py_DECREF(input_density);
    return Py_BuildValue("iNNNNNNNddddi", converged ? 1 : 0,
        current_density, final_energies, final_coefficients, final_occupations,
        final_fock, initial_fock_copy, restart_density, final_density_residual,
        final_energy_residual, current_energy, final_mixer_residual,
        completed_cycles);

resident_diis_failure:
    Py_DECREF(input_density); Py_DECREF(current_density);
    Py_DECREF(initial_fock_copy); Py_XDECREF(final_energies);
    Py_XDECREF(final_coefficients); Py_XDECREF(final_occupations);
    Py_XDECREF(final_fock); Py_XDECREF(restart_density);
    return nullptr;
}

PyObject* atomic_multipole_derivatives_native(PyObject*, PyObject* args) {
    PyObject *density_o, *overlap_o, *ds_o, *dipole_o, *dd_o;
    PyObject *dsecond_o, *coordinates_o, *atom_index_o;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &density_o, &overlap_o, &ds_o,
            &dipole_o, &dd_o, &dsecond_o, &coordinates_o, &atom_index_o)) {
        return nullptr;
    }
    auto* density = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(density_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* overlap = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(overlap_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* ds = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(ds_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* dipole = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(dipole_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* dd = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(dd_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* dsecond = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(dsecond_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* coordinates = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(coordinates_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* atom_index = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(atom_index_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    if (!density || !overlap || !ds || !dipole || !dd || !dsecond ||
        !coordinates || !atom_index) {
        Py_XDECREF(density); Py_XDECREF(overlap); Py_XDECREF(ds); Py_XDECREF(dipole);
        Py_XDECREF(dd); Py_XDECREF(dsecond); Py_XDECREF(coordinates); Py_XDECREF(atom_index);
        return nullptr;
    }
    const npy_intp n = PyArray_DIM(density, 0);
    const npy_intp nat = PyArray_DIM(coordinates, 0);
    const bool valid = PyArray_NDIM(density) == 2 && PyArray_DIM(density, 1) == n
        && PyArray_NDIM(overlap) == 2 && PyArray_DIM(overlap, 0) == n && PyArray_DIM(overlap, 1) == n
        && PyArray_NDIM(ds) == 4 && PyArray_DIM(ds, 0) == nat && PyArray_DIM(ds, 1) == 3
        && PyArray_DIM(ds, 2) == n && PyArray_DIM(ds, 3) == n
        && PyArray_NDIM(dipole) == 3 && PyArray_DIM(dipole, 0) == 3
        && PyArray_DIM(dipole, 1) == n && PyArray_DIM(dipole, 2) == n
        && PyArray_NDIM(dd) == 5 && PyArray_DIM(dd, 0) == 3 && PyArray_DIM(dd, 1) == nat
        && PyArray_DIM(dd, 2) == 3 && PyArray_DIM(dd, 3) == n && PyArray_DIM(dd, 4) == n
        && PyArray_NDIM(dsecond) == 6 && PyArray_DIM(dsecond, 0) == 3
        && PyArray_DIM(dsecond, 1) == 3 && PyArray_DIM(dsecond, 2) == nat
        && PyArray_DIM(dsecond, 3) == 3 && PyArray_DIM(dsecond, 4) == n
        && PyArray_DIM(dsecond, 5) == n
        && PyArray_NDIM(coordinates) == 2 && PyArray_DIM(coordinates, 1) == 3
        && PyArray_NDIM(atom_index) == 1 && PyArray_DIM(atom_index, 0) == n;
    if (!valid) {
        Py_DECREF(density); Py_DECREF(overlap); Py_DECREF(ds); Py_DECREF(dipole);
        Py_DECREF(dd); Py_DECREF(dsecond); Py_DECREF(coordinates); Py_DECREF(atom_index);
        PyErr_SetString(PyExc_ValueError, "incompatible atomic multipole derivative arrays");
        return nullptr;
    }
    npy_intp dq_dims[3] = {nat, nat, 3};
    npy_intp dmu_dims[4] = {nat, 3, nat, 3};
    npy_intp dquad_dims[5] = {nat, 3, 3, nat, 3};
    auto* dq = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(3, dq_dims, NPY_DOUBLE, 0));
    auto* dmu = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(4, dmu_dims, NPY_DOUBLE, 0));
    auto* draw = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(5, dquad_dims, NPY_DOUBLE, 0));
    auto* dquad = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(5, dquad_dims, NPY_DOUBLE, 0));
    if (!dq || !dmu || !draw || !dquad) {
        Py_XDECREF(dq); Py_XDECREF(dmu); Py_XDECREF(draw); Py_XDECREF(dquad);
        Py_DECREF(density); Py_DECREF(overlap); Py_DECREF(ds); Py_DECREF(dipole);
        Py_DECREF(dd); Py_DECREF(dsecond); Py_DECREF(coordinates); Py_DECREF(atom_index);
        return PyErr_NoMemory();
    }
    const auto* p = static_cast<const double*>(PyArray_DATA(density));
    const auto* s = static_cast<const double*>(PyArray_DATA(overlap));
    const auto* sd = static_cast<const double*>(PyArray_DATA(ds));
    const auto* mu = static_cast<const double*>(PyArray_DATA(dipole));
    const auto* mud = static_cast<const double*>(PyArray_DATA(dd));
    const auto* theta_d = static_cast<const double*>(PyArray_DATA(dsecond));
    const auto* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* atoms = static_cast<const std::int64_t*>(PyArray_DATA(atom_index));
    auto* qout = static_cast<double*>(PyArray_DATA(dq));
    auto* muout = static_cast<double*>(PyArray_DATA(dmu));
    auto* rawout = static_cast<double*>(PyArray_DATA(draw));
    auto* quadout = static_cast<double*>(PyArray_DATA(dquad));
    auto ds_at = [=](npy_intp t, int x, npy_intp k, npy_intp j) {
        return sd[((t*3+x)*n+k)*n+j];
    };
    auto dd_at = [=](int c, npy_intp t, int x, npy_intp k, npy_intp j) {
        return mud[((((c*nat+t)*3+x)*n+k)*n+j)];
    };
    auto d2_at = [=](int c, int o, npy_intp t, int x, npy_intp k, npy_intp j) {
        return theta_d[(((((c*3+o)*nat+t)*3+x)*n+k)*n+j)];
    };
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp j = 0; j < n; ++j) {
        const npy_intp source = atoms[j];
        double population = 0.0;
        double dipole_population[3] = {};
        for (npy_intp k = 0; k < n; ++k) {
            const double pjk = p[j*n+k];
            population += pjk * s[k*n+j];
            for (int c = 0; c < 3; ++c) dipole_population[c] += pjk * mu[(c*n+k)*n+j];
        }
        for (npy_intp t = 0; t < nat; ++t) for (int x = 0; x < 3; ++x) {
            double base = 0.0, dipbase[3] = {}, secondbase[3][3] = {};
            for (npy_intp k = 0; k < n; ++k) {
                const double pjk = p[j*n+k];
                base += pjk * ds_at(t, x, k, j);
                for (int c = 0; c < 3; ++c) {
                    dipbase[c] += pjk * dd_at(c, t, x, k, j);
                    for (int o = 0; o < 3; ++o) secondbase[c][o] += pjk * d2_at(c, o, t, x, k, j);
                }
            }
            qout[(source*nat+t)*3+x] -= base;
            for (int c = 0; c < 3; ++c) {
                const double shift_c = (t == source && x == c) ? 1.0 : 0.0;
                muout[((source*3+c)*nat+t)*3+x] +=
                    shift_c * population + xyz[source*3+c] * base - dipbase[c];
                for (int o = 0; o < 3; ++o) {
                    const double shift_o = (t == source && x == o) ? 1.0 : 0.0;
                    const double value =
                        shift_o * dipole_population[c] + xyz[source*3+o] * dipbase[c]
                        + shift_c * dipole_population[o] + xyz[source*3+c] * dipbase[o]
                        - (shift_c * xyz[source*3+o] + xyz[source*3+c] * shift_o) * population
                        - xyz[source*3+c] * xyz[source*3+o] * base - secondbase[c][o];
                    rawout[((((source*3+c)*3+o)*nat+t)*3+x)] += value;
                }
            }
        }
    }
    for (npy_intp aidx = 0; aidx < nat; ++aidx) for (npy_intp t = 0; t < nat; ++t) {
        for (int x = 0; x < 3; ++x) {
            double trace = 0.0;
            for (int c = 0; c < 3; ++c) trace += rawout[((((aidx*3+c)*3+c)*nat+t)*3+x)];
            for (int c = 0; c < 3; ++c) for (int o = 0; o < 3; ++o) {
                const npy_intp index = ((((aidx*3+c)*3+o)*nat+t)*3+x);
                quadout[index] = 1.5 * rawout[index] - (c == o ? 0.5 * trace : 0.0);
            }
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(draw);
    Py_DECREF(density); Py_DECREF(overlap); Py_DECREF(ds); Py_DECREF(dipole);
    Py_DECREF(dd); Py_DECREF(dsecond); Py_DECREF(coordinates); Py_DECREF(atom_index);
    return Py_BuildValue("NNN", dq, dmu, dquad);
}

PyObject* variational_gradient_native(PyObject*, PyObject* args) {
    PyObject *density_o, *weighted_o, *dh_o, *ds_o, *explicit_o;
    if (!PyArg_ParseTuple(args, "OOOOO", &density_o, &weighted_o, &dh_o, &ds_o, &explicit_o)) return nullptr;
    auto* density = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(density_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* weighted = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(weighted_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* dh = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(dh_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* ds = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(ds_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* explicit_values = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(explicit_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!density || !weighted || !dh || !ds || !explicit_values) {
        Py_XDECREF(density); Py_XDECREF(weighted); Py_XDECREF(dh); Py_XDECREF(ds); Py_XDECREF(explicit_values);
        return nullptr;
    }
    const npy_intp n = PyArray_DIM(density, 0), nat = PyArray_DIM(dh, 0);
    npy_intp dims[2] = {nat, 3};
    auto* result = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(2, dims, NPY_DOUBLE, 0));
    if (!result) return PyErr_NoMemory();
    const auto* p = static_cast<const double*>(PyArray_DATA(density));
    const auto* w = static_cast<const double*>(PyArray_DATA(weighted));
    const auto* h = static_cast<const double*>(PyArray_DATA(dh));
    const auto* s = static_cast<const double*>(PyArray_DATA(ds));
    const auto* e = static_cast<const double*>(PyArray_DATA(explicit_values));
    auto* out = static_cast<double*>(PyArray_DATA(result));
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp a = 0; a < nat; ++a) for (int x = 0; x < 3; ++x) {
        double value = e[a*3+x];
        const npy_intp offset = (a*3+x)*n*n;
        for (npy_intp i = 0; i < n*n; ++i) value += h[offset+i]*p[i] - s[offset+i]*w[i];
        out[a*3+x] = value;
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(density); Py_DECREF(weighted); Py_DECREF(dh); Py_DECREF(ds); Py_DECREF(explicit_values);
    return reinterpret_cast<PyObject*>(result);
}

PyObject* sparse_gradient_accumulate_native(PyObject*, PyObject* args) {
    PyObject *p_o,*w_o,*multiplicity_o,*left_o,*right_o,*dhl_o,*dhr_o,*dsl_o,*dsr_o,*gradient_o;
    int native_threads=1;
    if(!PyArg_ParseTuple(args,"OOOOOOOOOOi",&p_o,&w_o,&multiplicity_o,&left_o,&right_o,
            &dhl_o,&dhr_o,&dsl_o,&dsr_o,&gradient_o,&native_threads))return nullptr;
    auto f64=[](PyObject* o){return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(o,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));};
    auto i32=[](PyObject* o){return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(o,NPY_INT32,NPY_ARRAY_IN_ARRAY));};
    auto* p=f64(p_o);auto* w=f64(w_o);auto* multiplicity=f64(multiplicity_o);
    auto* left=i32(left_o);auto* right=i32(right_o);
    auto* dhl=f64(dhl_o);auto* dhr=f64(dhr_o);auto* dsl=f64(dsl_o);auto* dsr=f64(dsr_o);
    auto* gradient=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        gradient_o,NPY_DOUBLE,NPY_ARRAY_INOUT_ARRAY2));
    const npy_intp count=p?PyArray_SIZE(p):0;
    const bool valid=p&&w&&multiplicity&&left&&right&&dhl&&dhr&&dsl&&dsr&&gradient
        &&PyArray_SIZE(w)==count&&PyArray_SIZE(multiplicity)==count
        &&PyArray_SIZE(left)==count&&PyArray_SIZE(right)==count
        &&PyArray_SIZE(dhl)==3*count&&PyArray_SIZE(dhr)==3*count
        &&PyArray_SIZE(dsl)==3*count&&PyArray_SIZE(dsr)==3*count
        &&PyArray_NDIM(gradient)==2&&PyArray_DIM(gradient,1)==3;
    if(!valid){
        Py_XDECREF(p);Py_XDECREF(w);Py_XDECREF(multiplicity);Py_XDECREF(left);Py_XDECREF(right);
        Py_XDECREF(dhl);Py_XDECREF(dhr);Py_XDECREF(dsl);Py_XDECREF(dsr);
        if(gradient){PyArray_DiscardWritebackIfCopy(gradient);Py_DECREF(gradient);}
        PyErr_SetString(PyExc_ValueError,"invalid sparse gradient arrays");return nullptr;
    }
    const auto nat=PyArray_DIM(gradient,0);auto* g=static_cast<double*>(PyArray_DATA(gradient));
    const auto* pv=static_cast<const double*>(PyArray_DATA(p));
    const auto* wv=static_cast<const double*>(PyArray_DATA(w));
    const auto* mv=static_cast<const double*>(PyArray_DATA(multiplicity));
    const auto* la=static_cast<const std::int32_t*>(PyArray_DATA(left));
    const auto* ra=static_cast<const std::int32_t*>(PyArray_DATA(right));
    const auto* hl=static_cast<const double*>(PyArray_DATA(dhl));
    const auto* hr=static_cast<const double*>(PyArray_DATA(dhr));
    const auto* sl=static_cast<const double*>(PyArray_DATA(dsl));
    const auto* sr=static_cast<const double*>(PyArray_DATA(dsr));
    native_threads=std::max(1,std::min(native_threads,static_cast<int>(count)));
    std::vector<std::vector<double>> partial(native_threads,
        std::vector<double>(static_cast<std::size_t>(nat)*3,0.0));
    auto accumulate=[&](int thread,npy_intp begin,npy_intp end){
        auto& local=partial[thread];
        for(npy_intp pair=begin;pair<end;++pair){
            for(int axis=0;axis<3;++axis){
                local[la[pair]*3+axis]+=mv[pair]*(pv[pair]*hl[pair*3+axis]-wv[pair]*sl[pair*3+axis]);
                local[ra[pair]*3+axis]+=mv[pair]*(pv[pair]*hr[pair*3+axis]-wv[pair]*sr[pair*3+axis]);
            }
        }
    };
    Py_BEGIN_ALLOW_THREADS
    std::vector<std::thread> workers;workers.reserve(native_threads>1?native_threads-1:0);
    const npy_intp chunk=(count+native_threads-1)/native_threads;
    for(int thread=1;thread<native_threads;++thread){
        const auto begin=std::min<npy_intp>(count,thread*chunk);
        workers.emplace_back(accumulate,thread,begin,std::min<npy_intp>(count,begin+chunk));
    }
    accumulate(0,0,std::min<npy_intp>(count,chunk));
    for(auto& worker:workers)worker.join();
    for(const auto& local:partial)for(npy_intp index=0;index<nat*3;++index)g[index]+=local[index];
    Py_END_ALLOW_THREADS
    Py_DECREF(p);Py_DECREF(w);Py_DECREF(multiplicity);Py_DECREF(left);Py_DECREF(right);Py_DECREF(dhl);Py_DECREF(dhr);
    Py_DECREF(dsl);Py_DECREF(dsr);PyArray_ResolveWritebackIfCopy(gradient);Py_DECREF(gradient);
    Py_RETURN_NONE;
}

PyObject* pair_antisymmetric_gradient_native(PyObject*, PyObject* args) {
    PyObject *left_o, *right_o, *contribution_o;
    long long natoms;
    if (!PyArg_ParseTuple(args, "OOOL", &left_o, &right_o, &contribution_o,
            &natoms)) return nullptr;
    auto* left = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        left_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* right = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        right_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* contribution = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        contribution_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const npy_intp count = left ? PyArray_SIZE(left) : 0;
    const bool valid = left && right && contribution && natoms >= 0
        && PyArray_SIZE(right) == count
        && PyArray_NDIM(contribution) == 2
        && PyArray_DIM(contribution, 0) == count
        && PyArray_DIM(contribution, 1) == 3;
    if (!valid) {
        Py_XDECREF(left); Py_XDECREF(right); Py_XDECREF(contribution);
        if (!PyErr_Occurred()) PyErr_SetString(
            PyExc_ValueError, "invalid antisymmetric pair-gradient arrays");
        return nullptr;
    }
    npy_intp dims[2] = {static_cast<npy_intp>(natoms), 3};
    auto* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dims, NPY_DOUBLE, 0));
    if (!gradient) {
        Py_DECREF(left); Py_DECREF(right); Py_DECREF(contribution);
        return nullptr;
    }
    const auto* l = static_cast<const std::int64_t*>(PyArray_DATA(left));
    const auto* r = static_cast<const std::int64_t*>(PyArray_DATA(right));
    const auto* value = static_cast<const double*>(PyArray_DATA(contribution));
    auto* output = static_cast<double*>(PyArray_DATA(gradient));
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp pair = 0; pair < count; ++pair) {
        for (int axis = 0; axis < 3; ++axis) {
            output[l[pair]*3 + axis] += value[pair*3 + axis];
            output[r[pair]*3 + axis] -= value[pair*3 + axis];
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(left); Py_DECREF(right); Py_DECREF(contribution);
    return reinterpret_cast<PyObject*>(gradient);
}

PyObject* radial_pair_hessian_vector_native(PyObject*, PyObject* args) {
    PyObject *left_o, *right_o, *displacement_o, *radial_o, *second_o;
    PyObject *direction_o;
    long long natoms;
    if (!PyArg_ParseTuple(args, "OOOOOOL", &left_o, &right_o,
            &displacement_o, &radial_o, &second_o, &direction_o, &natoms))
        return nullptr;
    auto* left = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        left_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* right = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        right_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* displacement = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        displacement_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* radial = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        radial_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* second = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        second_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* direction = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        direction_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const npy_intp count = left ? PyArray_SIZE(left) : 0;
    const bool valid = left && right && displacement && radial && second
        && direction && natoms >= 0 && PyArray_SIZE(right) == count
        && PyArray_SIZE(radial) == count && PyArray_SIZE(second) == count
        && PyArray_NDIM(displacement) == 2
        && PyArray_DIM(displacement, 0) == count
        && PyArray_DIM(displacement, 1) == 3
        && PyArray_NDIM(direction) == 2
        && PyArray_DIM(direction, 0) == natoms
        && PyArray_DIM(direction, 1) == 3;
    if (!valid) {
        Py_XDECREF(left); Py_XDECREF(right); Py_XDECREF(displacement);
        Py_XDECREF(radial); Py_XDECREF(second); Py_XDECREF(direction);
        if (!PyErr_Occurred()) PyErr_SetString(
            PyExc_ValueError, "invalid radial pair-Hessian arrays");
        return nullptr;
    }
    npy_intp dims[2] = {static_cast<npy_intp>(natoms), 3};
    auto* result = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dims, NPY_DOUBLE, 0));
    if (!result) {
        Py_DECREF(left); Py_DECREF(right); Py_DECREF(displacement);
        Py_DECREF(radial); Py_DECREF(second); Py_DECREF(direction);
        return nullptr;
    }
    const auto* l = static_cast<const std::int64_t*>(PyArray_DATA(left));
    const auto* r = static_cast<const std::int64_t*>(PyArray_DATA(right));
    const auto* d = static_cast<const double*>(PyArray_DATA(displacement));
    const auto* first = static_cast<const double*>(PyArray_DATA(radial));
    const auto* curvature = static_cast<const double*>(PyArray_DATA(second));
    const auto* vector = static_cast<const double*>(PyArray_DATA(direction));
    auto* output = static_cast<double*>(PyArray_DATA(result));
    bool invalid = false;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp pair = 0; pair < count; ++pair) {
        if (l[pair] < 0 || r[pair] < 0 || l[pair] >= natoms || r[pair] >= natoms) {
            invalid = true;
            continue;
        }
        double distance2 = 0.0, projection = 0.0;
        double relative[3];
        for (int axis = 0; axis < 3; ++axis) {
            relative[axis] = vector[l[pair]*3+axis]-vector[r[pair]*3+axis];
            distance2 += d[pair*3+axis]*d[pair*3+axis];
            projection += d[pair*3+axis]*relative[axis];
        }
        if (!(distance2 > 0.0)) { invalid = true; continue; }
        const double distance = std::sqrt(distance2);
        const double longitudinal = projection/distance;
        for (int axis = 0; axis < 3; ++axis) {
            const double unit = d[pair*3+axis]/distance;
            const double value = curvature[pair]*longitudinal*unit
                + first[pair]/distance*(relative[axis]-longitudinal*unit);
            output[l[pair]*3+axis] += value;
            output[r[pair]*3+axis] -= value;
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(left); Py_DECREF(right); Py_DECREF(displacement);
    Py_DECREF(radial); Py_DECREF(second); Py_DECREF(direction);
    if (invalid) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError,
            "radial pair Hessian contains an invalid pair or distance");
        return nullptr;
    }
    return reinterpret_cast<PyObject*>(result);
}

PyObject* pair_symmetric_scalar_native(PyObject*, PyObject* args) {
    PyObject *left_o, *right_o, *value_o;
    long long natoms;
    if (!PyArg_ParseTuple(args, "OOOL", &left_o, &right_o, &value_o,
            &natoms)) return nullptr;
    auto* left = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        left_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* right = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        right_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* value = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        value_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const npy_intp count = left ? PyArray_SIZE(left) : 0;
    const bool valid = left && right && value && natoms >= 0
        && PyArray_SIZE(right) == count && PyArray_SIZE(value) == count;
    if (!valid) {
        Py_XDECREF(left); Py_XDECREF(right); Py_XDECREF(value);
        if (!PyErr_Occurred()) PyErr_SetString(
            PyExc_ValueError, "invalid symmetric pair-scalar arrays");
        return nullptr;
    }
    npy_intp dims[1] = {static_cast<npy_intp>(natoms)};
    auto* result = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, dims, NPY_DOUBLE, 0));
    if (!result) {
        Py_DECREF(left); Py_DECREF(right); Py_DECREF(value);
        return nullptr;
    }
    const auto* l = static_cast<const std::int64_t*>(PyArray_DATA(left));
    const auto* r = static_cast<const std::int64_t*>(PyArray_DATA(right));
    const auto* input = static_cast<const double*>(PyArray_DATA(value));
    auto* output = static_cast<double*>(PyArray_DATA(result));
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp pair = 0; pair < count; ++pair) {
        output[l[pair]] += input[pair];
        output[r[pair]] += input[pair];
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(left); Py_DECREF(right); Py_DECREF(value);
    return reinterpret_cast<PyObject*>(result);
}

PyObject* context_variational_gradient_sparse_native(PyObject*, PyObject* args) {
    if (PyTuple_Size(args) != 23) {
        PyErr_SetString(PyExc_TypeError,
            "context_variational_gradient_sparse expects 23 arguments");
        return nullptr;
    }
    auto* context = static_cast<IntegralContextNative*>(PyCapsule_GetPointer(
        PyTuple_GET_ITEM(args, 0), integral_context_capsule_name));
    if (!context) return nullptr;
    const auto size = PyLong_AsLongLong(PyTuple_GET_ITEM(args, 1));
    const auto natoms = PyLong_AsLongLong(PyTuple_GET_ITEM(args, 21));
    const auto native_threads_requested = PyLong_AsLong(PyTuple_GET_ITEM(args, 22));
    if (PyErr_Occurred()) return nullptr;
    auto f64 = [&](Py_ssize_t index) { return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(PyTuple_GET_ITEM(args, index), NPY_DOUBLE,
            NPY_ARRAY_IN_ARRAY)); };
    auto i32 = [&](Py_ssize_t index) { return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(PyTuple_GET_ITEM(args, index), NPY_INT32,
            NPY_ARRAY_IN_ARRAY)); };
    std::array<PyArrayObject*, 8> indices = {
        i32(2), i32(3), i32(5), i32(6), i32(8), i32(9), i32(10), i32(11)};
    std::array<PyArrayObject*, 12> floating = {
        f64(4), f64(7), f64(12), f64(13), f64(14), f64(15),
        f64(16), f64(17), f64(18), f64(19), f64(20), nullptr};
    auto release = [&]() {
        for (auto* item : indices) Py_XDECREF(item);
        for (auto* item : floating) Py_XDECREF(item);
    };
    const bool converted = std::all_of(indices.begin(), indices.end(),
            [](const auto* item) { return item != nullptr; })
        && std::all_of(floating.begin(), floating.end() - 1,
            [](const auto* item) { return item != nullptr; });
    const npy_intp p_count = converted ? PyArray_SIZE(indices[0]) : 0;
    const npy_intp w_count = converted ? PyArray_SIZE(indices[2]) : 0;
    const npy_intp pair_count = converted ? PyArray_SIZE(indices[4]) : 0;
    const bool valid = converted && size >= 0 && natoms >= 0
        && PyArray_SIZE(indices[1]) == p_count
        && PyArray_SIZE(floating[0]) == p_count
        && PyArray_SIZE(indices[3]) == w_count
        && PyArray_SIZE(floating[1]) == w_count
        && PyArray_SIZE(indices[5]) == pair_count
        && PyArray_SIZE(indices[6]) == pair_count
        && PyArray_SIZE(indices[7]) == pair_count
        && PyArray_SIZE(floating[2]) == pair_count*3
        && PyArray_SIZE(floating[3]) == pair_count*3
        && PyArray_SIZE(floating[4]) == pair_count*3
        && PyArray_SIZE(floating[5]) == pair_count*3
        && PyArray_SIZE(floating[6]) == pair_count
        && PyArray_SIZE(floating[7]) == pair_count*3
        && PyArray_SIZE(floating[8]) == pair_count*3
        && PyArray_SIZE(floating[9]) == pair_count
        && PyArray_SIZE(floating[10]) == pair_count;
    if (!valid) {
        release();
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError,
            "invalid persistent sparse variational-gradient arrays");
        return nullptr;
    }
    npy_intp gradient_dims[2] = {static_cast<npy_intp>(natoms), 3};
    npy_intp coefficient_dims[1] = {static_cast<npy_intp>(natoms)};
    auto* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, gradient_dims, NPY_DOUBLE, 0));
    auto* cn_coefficient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, coefficient_dims, NPY_DOUBLE, 0));
    if (!gradient || !cn_coefficient) {
        Py_XDECREF(gradient); Py_XDECREF(cn_coefficient); release();
        return PyErr_NoMemory();
    }
    const auto* p_rows = static_cast<const std::int32_t*>(PyArray_DATA(indices[0]));
    const auto* p_columns = static_cast<const std::int32_t*>(PyArray_DATA(indices[1]));
    const auto* p_values = static_cast<const double*>(PyArray_DATA(floating[0]));
    const auto* w_rows = static_cast<const std::int32_t*>(PyArray_DATA(indices[2]));
    const auto* w_columns = static_cast<const std::int32_t*>(PyArray_DATA(indices[3]));
    const auto* w_values = static_cast<const double*>(PyArray_DATA(floating[1]));
    const auto* rows = static_cast<const std::int32_t*>(PyArray_DATA(indices[4]));
    const auto* columns = static_cast<const std::int32_t*>(PyArray_DATA(indices[5]));
    const auto* left_atoms = static_cast<const std::int32_t*>(PyArray_DATA(indices[6]));
    const auto* right_atoms = static_cast<const std::int32_t*>(PyArray_DATA(indices[7]));
    const auto* h0_left = static_cast<const double*>(PyArray_DATA(floating[2]));
    const auto* h0_right = static_cast<const double*>(PyArray_DATA(floating[3]));
    const auto* camm_left = static_cast<const double*>(PyArray_DATA(floating[4]));
    const auto* camm_right = static_cast<const double*>(PyArray_DATA(floating[5]));
    const auto* pair_scalar = static_cast<const double*>(PyArray_DATA(floating[6]));
    const auto* ds_left = static_cast<const double*>(PyArray_DATA(floating[7]));
    const auto* ds_right = static_cast<const double*>(PyArray_DATA(floating[8]));
    const auto* coordination_left = static_cast<const double*>(PyArray_DATA(floating[9]));
    const auto* coordination_right = static_cast<const double*>(PyArray_DATA(floating[10]));
    auto* output = static_cast<double*>(PyArray_DATA(gradient));
    auto* cn_output = static_cast<double*>(PyArray_DATA(cn_coefficient));
    int native_threads = std::max(1, std::min(
        static_cast<int>(native_threads_requested), static_cast<int>(pair_count)));
    std::vector<std::vector<double>> partial_gradient;
    std::vector<std::vector<double>> partial_cn;
    if (native_threads > 1) {
        partial_gradient.assign(native_threads,
            std::vector<double>(static_cast<std::size_t>(natoms)*3, 0.0));
        partial_cn.assign(native_threads,
            std::vector<double>(static_cast<std::size_t>(natoms), 0.0));
    }
    auto accumulate = [&](int thread, npy_intp begin, npy_intp end) {
        double* local_gradient = native_threads == 1
            ? output : partial_gradient[thread].data();
        double* local_cn = native_threads == 1
            ? cn_output : partial_cn[thread].data();
        const auto first_key = begin < end
            ? static_cast<std::int64_t>(rows[begin])*size + columns[begin] : 0;
        auto lower_bound_key = [size](const std::int32_t* source_rows,
                const std::int32_t* source_columns, npy_intp source_count,
                std::int64_t key) {
            npy_intp low = 0, high = source_count;
            while (low < high) {
                const auto middle = low + (high-low)/2;
                const auto candidate = static_cast<std::int64_t>(source_rows[middle])*size
                    + source_columns[middle];
                if (candidate < key) low = middle + 1;
                else high = middle;
            }
            return low;
        };
        npy_intp p_position = lower_bound_key(
            p_rows, p_columns, p_count, first_key);
        npy_intp w_position = lower_bound_key(
            w_rows, w_columns, w_count, first_key);
        for (npy_intp pair = begin; pair < end; ++pair) {
            const auto key = static_cast<std::int64_t>(rows[pair])*size
                + columns[pair];
            while (p_position < p_count
                && static_cast<std::int64_t>(p_rows[p_position])*size
                    + p_columns[p_position] < key) ++p_position;
            while (w_position < w_count
                && static_cast<std::int64_t>(w_rows[w_position])*size
                    + w_columns[w_position] < key) ++w_position;
            const double p = p_position < p_count
                && p_rows[p_position] == rows[pair]
                && p_columns[p_position] == columns[pair]
                ? p_values[p_position] : 0.0;
            const double w = w_position < w_count
                && w_rows[w_position] == rows[pair]
                && w_columns[w_position] == columns[pair]
                ? w_values[w_position] : 0.0;
            const double multiplicity = rows[pair] == columns[pair] ? 1.0 : 2.0;
            local_cn[left_atoms[pair]] += multiplicity*p*coordination_left[pair];
            local_cn[right_atoms[pair]] += multiplicity*p*coordination_right[pair];
            for (int axis = 0; axis < 3; ++axis) {
                const auto offset = pair*3 + axis;
                const double left_h = h0_left[offset] + camm_left[offset]
                    + pair_scalar[pair]*ds_left[offset];
                const double right_h = h0_right[offset] + camm_right[offset]
                    + pair_scalar[pair]*ds_right[offset];
                local_gradient[left_atoms[pair]*3 + axis] += multiplicity*(
                    p*left_h - w*ds_left[offset]);
                local_gradient[right_atoms[pair]*3 + axis] += multiplicity*(
                    p*right_h - w*ds_right[offset]);
            }
        }
    };
    Py_BEGIN_ALLOW_THREADS
    std::vector<std::thread> workers;
    workers.reserve(native_threads > 1 ? native_threads - 1 : 0);
    const npy_intp chunk = pair_count > 0
        ? (pair_count + native_threads - 1)/native_threads : 0;
    for (int thread = 1; thread < native_threads; ++thread) {
        const auto begin = std::min<npy_intp>(pair_count, thread*chunk);
        workers.emplace_back(accumulate, thread, begin,
            std::min<npy_intp>(pair_count, begin + chunk));
    }
    accumulate(0, 0, std::min<npy_intp>(pair_count, chunk));
    for (auto& worker : workers) worker.join();
    if (native_threads > 1) {
        for (int thread = 0; thread < native_threads; ++thread) {
            for (npy_intp index = 0; index < natoms*3; ++index)
                output[index] += partial_gradient[thread][index];
            for (npy_intp atom = 0; atom < natoms; ++atom)
                cn_output[atom] += partial_cn[thread][atom];
        }
    }
    Py_END_ALLOW_THREADS
    release();
    return Py_BuildValue("NN", gradient, cn_coefficient);
}

PyObject* context_variational_gradient_dense_native(PyObject*, PyObject* args) {
    if (PyTuple_Size(args) != 18) {
        PyErr_SetString(PyExc_TypeError,
            "context_variational_gradient_dense expects 18 arguments");
        return nullptr;
    }
    auto* density = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        PyTuple_GET_ITEM(args, 1), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* weighted = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        PyTuple_GET_ITEM(args, 2), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* rows = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        PyTuple_GET_ITEM(args, 3), NPY_INT32, NPY_ARRAY_IN_ARRAY));
    auto* columns = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        PyTuple_GET_ITEM(args, 4), NPY_INT32, NPY_ARRAY_IN_ARRAY));
    const npy_intp size = density && PyArray_NDIM(density) == 2
        ? PyArray_DIM(density, 0) : 0;
    const npy_intp pair_count = rows ? PyArray_SIZE(rows) : 0;
    if (!density || !weighted || !rows || !columns
        || PyArray_DIM(density, 1) != size || PyArray_SIZE(weighted) != size*size
        || PyArray_SIZE(columns) != pair_count) {
        Py_XDECREF(density); Py_XDECREF(weighted);
        Py_XDECREF(rows); Py_XDECREF(columns);
        PyErr_SetString(PyExc_ValueError,
            "invalid dense variational-gradient matrices");
        return nullptr;
    }
    npy_intp dims[1] = {pair_count};
    auto* p_values = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, dims, NPY_DOUBLE, 0));
    auto* w_values = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, dims, NPY_DOUBLE, 0));
    if (!p_values || !w_values) {
        Py_XDECREF(p_values); Py_XDECREF(w_values); Py_DECREF(density);
        Py_DECREF(weighted); Py_DECREF(rows); Py_DECREF(columns);
        return PyErr_NoMemory();
    }
    const auto* p = static_cast<const double*>(PyArray_DATA(density));
    const auto* w = static_cast<const double*>(PyArray_DATA(weighted));
    const auto* r = static_cast<const std::int32_t*>(PyArray_DATA(rows));
    const auto* c = static_cast<const std::int32_t*>(PyArray_DATA(columns));
    auto* pv = static_cast<double*>(PyArray_DATA(p_values));
    auto* wv = static_cast<double*>(PyArray_DATA(w_values));
    for (npy_intp pair = 0; pair < pair_count; ++pair) {
        pv[pair] = p[r[pair]*size+c[pair]];
        wv[pair] = w[r[pair]*size+c[pair]];
    }
    auto* forwarded = PyTuple_New(23);
    auto* size_o = PyLong_FromLongLong(size);
    if (!forwarded || !size_o) {
        Py_XDECREF(forwarded); Py_XDECREF(size_o); Py_DECREF(p_values);
        Py_DECREF(w_values); Py_DECREF(density); Py_DECREF(weighted);
        Py_DECREF(rows); Py_DECREF(columns); return nullptr;
    }
    auto install = [&](Py_ssize_t target, PyObject* value) {
        Py_INCREF(value); PyTuple_SET_ITEM(forwarded, target, value);
    };
    install(0, PyTuple_GET_ITEM(args, 0));
    PyTuple_SET_ITEM(forwarded, 1, size_o);
    install(2, reinterpret_cast<PyObject*>(rows));
    install(3, reinterpret_cast<PyObject*>(columns));
    PyTuple_SET_ITEM(forwarded, 4, reinterpret_cast<PyObject*>(p_values));
    install(5, reinterpret_cast<PyObject*>(rows));
    install(6, reinterpret_cast<PyObject*>(columns));
    PyTuple_SET_ITEM(forwarded, 7, reinterpret_cast<PyObject*>(w_values));
    for (Py_ssize_t source = 3; source <= 17; ++source)
        install(source+5, PyTuple_GET_ITEM(args, source));
    auto* result = context_variational_gradient_sparse_native(nullptr, forwarded);
    Py_DECREF(forwarded); Py_DECREF(density); Py_DECREF(weighted);
    Py_DECREF(rows); Py_DECREF(columns);
    return result;
}

PyObject* coordination_response_gradient_native(PyObject*, PyObject* args) {
    PyObject *coefficients_o, *coordinates_o, *radii_o, *en_o;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOOd", &coefficients_o, &coordinates_o,
            &radii_o, &en_o, &cutoff)) return nullptr;
    auto* coefficients = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        coefficients_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* coordinates = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        coordinates_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* radii = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        radii_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* en = en_o == Py_None ? nullptr : reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(en_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const npy_intp natoms = coordinates && PyArray_NDIM(coordinates) == 2
        ? PyArray_DIM(coordinates, 0) : 0;
    const bool valid = coefficients && coordinates && radii
        && PyArray_NDIM(coordinates) == 2 && PyArray_DIM(coordinates, 1) == 3
        && PyArray_SIZE(coefficients) == natoms
        && PyArray_SIZE(radii) == natoms
        && (!en || PyArray_SIZE(en) == natoms)
        && std::isfinite(cutoff) && cutoff > 0.0;
    if (!valid) {
        Py_XDECREF(coefficients); Py_XDECREF(coordinates); Py_XDECREF(radii);
        Py_XDECREF(en);
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError,
            "invalid native coordination-response arrays");
        return nullptr;
    }
    npy_intp dims[2] = {natoms, 3};
    auto* result = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dims, NPY_DOUBLE, 0));
    if (!result) {
        Py_DECREF(coefficients); Py_DECREF(coordinates); Py_DECREF(radii);
        Py_XDECREF(en); return nullptr;
    }
    const auto* c = static_cast<const double*>(PyArray_DATA(coefficients));
    const auto* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* radius = static_cast<const double*>(PyArray_DATA(radii));
    const auto* electronegativity = en
        ? static_cast<const double*>(PyArray_DATA(en)) : nullptr;
    auto* gradient = static_cast<double*>(PyArray_DATA(result));
    const double cutoff2 = cutoff*cutoff;
    constexpr double inv_sqrt_pi = 0.56418958354775628695;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp left = 1; left < natoms; ++left) {
        for (npy_intp right = 0; right < left; ++right) {
            double displacement[3];
            double distance2 = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                displacement[axis] = xyz[left*3+axis]-xyz[right*3+axis];
                distance2 += displacement[axis]*displacement[axis];
            }
            if (distance2 > cutoff2 || distance2 <= 0.0) continue;
            const double distance = std::sqrt(distance2);
            double radial;
            if (!electronegativity) {
                const double scale1 = 4.0*(radius[left]+radius[right])/3.0;
                const double scale2 = scale1+2.0;
                const double first = 1.0/(1.0+std::exp(-10.0*(scale1/distance-1.0)));
                const double second = 1.0/(1.0+std::exp(-20.0*(scale2/distance-1.0)));
                radial = -10.0*scale1*first*(1.0-first)*second/distance2
                    -20.0*scale2*second*(1.0-second)*first/distance2;
            } else {
                const double critical = 4.0*(radius[left]+radius[right])/3.0;
                const double argument = -7.5*(distance-critical)/critical;
                const double en_delta = std::abs(
                    electronegativity[left]-electronegativity[right]);
                radial = -7.5*std::exp(-argument*argument)*inv_sqrt_pi/critical
                    *4.10451*std::exp(-(en_delta+19.08857)*(en_delta+19.08857)
                        /(2.0*11.28174*11.28174));
            }
            const double factor = (c[left]+c[right])*radial/distance;
            for (int axis = 0; axis < 3; ++axis) {
                const double value = factor*displacement[axis];
                gradient[left*3+axis] += value;
                gradient[right*3+axis] -= value;
            }
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(coefficients); Py_DECREF(coordinates); Py_DECREF(radii);
    Py_XDECREF(en);
    return reinterpret_cast<PyObject*>(result);
}

PyObject* repulsion_energy_gradient_native(PyObject*, PyObject* args) {
    PyObject *coordinates_o, *numbers_o, *alpha_o, *charge_o;
    double gradient_cutoff;
    if (!PyArg_ParseTuple(args, "OOOOd", &coordinates_o, &numbers_o,
            &alpha_o, &charge_o, &gradient_cutoff)) return nullptr;
    auto* coordinates = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        coordinates_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* numbers = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        numbers_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* alpha = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        alpha_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* charge = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        charge_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const npy_intp natoms = coordinates && PyArray_NDIM(coordinates) == 2
        ? PyArray_DIM(coordinates, 0) : 0;
    const bool valid = coordinates && numbers && alpha && charge
        && PyArray_NDIM(coordinates) == 2 && PyArray_DIM(coordinates, 1) == 3
        && PyArray_SIZE(numbers) == natoms && PyArray_SIZE(alpha) == natoms
        && PyArray_SIZE(charge) == natoms && std::isfinite(gradient_cutoff)
        && gradient_cutoff > 0.0;
    if (!valid) {
        Py_XDECREF(coordinates); Py_XDECREF(numbers); Py_XDECREF(alpha);
        Py_XDECREF(charge);
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError,
            "invalid native repulsion arrays");
        return nullptr;
    }
    npy_intp dims[2] = {natoms, 3};
    auto* gradient_array = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dims, NPY_DOUBLE, 0));
    if (!gradient_array) {
        Py_DECREF(coordinates); Py_DECREF(numbers); Py_DECREF(alpha);
        Py_DECREF(charge); return nullptr;
    }
    const auto* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* atomic_number = static_cast<const std::int64_t*>(
        PyArray_DATA(numbers));
    const auto* repulsion_alpha = static_cast<const double*>(PyArray_DATA(alpha));
    const auto* effective_charge = static_cast<const double*>(PyArray_DATA(charge));
    auto* gradient = static_cast<double*>(PyArray_DATA(gradient_array));
    const double cutoff2 = gradient_cutoff*gradient_cutoff;
    double energy_total = 0.0;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp left = 1; left < natoms; ++left) {
        for (npy_intp right = 0; right < left; ++right) {
            double displacement[3], distance2 = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                displacement[axis] = xyz[left*3+axis]-xyz[right*3+axis];
                distance2 += displacement[axis]*displacement[axis];
            }
            const double distance = std::sqrt(distance2);
            const double exponent = atomic_number[left] <= 2
                && atomic_number[right] <= 2 ? 1.0 : 1.5;
            const double alpha_pair = std::sqrt(
                repulsion_alpha[left]*repulsion_alpha[right]);
            const double energy = effective_charge[left]*effective_charge[right]
                /distance*std::exp(-alpha_pair*std::pow(distance, exponent));
            energy_total += energy;
            if (distance2 > cutoff2) continue;
            const double radial = energy*(-1.0/distance
                -alpha_pair*exponent*std::pow(distance, exponent-1.0));
            const double scale = radial/distance;
            for (int axis = 0; axis < 3; ++axis) {
                const double value = scale*displacement[axis];
                gradient[left*3+axis] += value;
                gradient[right*3+axis] -= value;
            }
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(coordinates); Py_DECREF(numbers); Py_DECREF(alpha); Py_DECREF(charge);
    return Py_BuildValue("dN", energy_total, gradient_array);
}

PyObject* d4_coordination_native(PyObject*, PyObject* args) {
    PyObject *coordinates_o, *radii_o, *electronegativity_o;
    if (!PyArg_ParseTuple(args, "OOO", &coordinates_o, &radii_o,
            &electronegativity_o)) return nullptr;
    auto* coordinates = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        coordinates_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* radii = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        radii_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* electronegativity = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        electronegativity_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const npy_intp n = coordinates && PyArray_NDIM(coordinates) == 2
        ? PyArray_DIM(coordinates, 0) : 0;
    const bool valid = coordinates && radii && electronegativity
        && PyArray_NDIM(coordinates) == 2 && PyArray_DIM(coordinates, 1) == 3
        && PyArray_SIZE(radii) == n && PyArray_SIZE(electronegativity) == n;
    if (!valid) {
        Py_XDECREF(coordinates); Py_XDECREF(radii); Py_XDECREF(electronegativity);
        if (!PyErr_Occurred()) PyErr_SetString(
            PyExc_ValueError, "invalid native D4 coordination arrays");
        return nullptr;
    }
    npy_intp dims[1] = {n};
    auto* result = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, dims, NPY_DOUBLE, 0));
    if (!result) {
        Py_DECREF(coordinates); Py_DECREF(radii); Py_DECREF(electronegativity);
        return nullptr;
    }
    const auto* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* radius = static_cast<const double*>(PyArray_DATA(radii));
    const auto* en = static_cast<const double*>(PyArray_DATA(electronegativity));
    auto* coordination = static_cast<double*>(PyArray_DATA(result));
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp left = 1; left < n; ++left) {
        for (npy_intp right = 0; right < left; ++right) {
            double distance2 = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                const double value = xyz[left*3+axis]-xyz[right*3+axis];
                distance2 += value*value;
            }
            const double distance = std::sqrt(distance2);
            const double critical = (4.0/3.0)*(radius[left]+radius[right]);
            const double argument = -7.5*(distance-critical)/critical;
            const double count = 0.5*(1.0+std::erf(argument));
            const double delta = std::abs(en[left]-en[right]);
            const double factor = 4.10451*std::exp(
                -(delta+19.08857)*(delta+19.08857)/(2.0*11.28174*11.28174));
            const double value = count*factor;
            coordination[left] += value;
            coordination[right] += value;
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(coordinates); Py_DECREF(radii); Py_DECREF(electronegativity);
    return reinterpret_cast<PyObject*>(result);
}

PyObject* gfn2_coordination_native(PyObject*, PyObject* args) {
    PyObject *coordinates_o, *radii_o;
    if (!PyArg_ParseTuple(args, "OO", &coordinates_o, &radii_o)) return nullptr;
    auto* coordinates = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        coordinates_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* radii = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        radii_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const npy_intp n = coordinates && PyArray_NDIM(coordinates) == 2
        ? PyArray_DIM(coordinates, 0) : 0;
    const bool valid = coordinates && radii
        && PyArray_NDIM(coordinates) == 2 && PyArray_DIM(coordinates, 1) == 3
        && PyArray_SIZE(radii) == n;
    if (!valid) {
        Py_XDECREF(coordinates); Py_XDECREF(radii);
        if (!PyErr_Occurred()) PyErr_SetString(
            PyExc_ValueError, "invalid native GFN2 coordination arrays");
        return nullptr;
    }
    npy_intp dims[1] = {n};
    auto* result = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, dims, NPY_DOUBLE, 0));
    if (!result) { Py_DECREF(coordinates); Py_DECREF(radii); return nullptr; }
    const auto* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* radius = static_cast<const double*>(PyArray_DATA(radii));
    auto* coordination = static_cast<double*>(PyArray_DATA(result));
    bool collision = false;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp left = 1; left < n; ++left) {
        for (npy_intp right = 0; right < left; ++right) {
            double distance2 = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                const double value = xyz[left*3+axis]-xyz[right*3+axis];
                distance2 += value*value;
            }
            if (distance2 <= 0.0) { collision = true; continue; }
            const double distance = std::sqrt(distance2);
            const double radius_sum = radius[left]+radius[right];
            const double first_argument =
                10.0*(4.0*radius_sum/(3.0*distance)-1.0);
            const double second_argument =
                20.0*((4.0*radius_sum/3.0+2.0)/distance-1.0);
            const double first = 1.0/(1.0+std::exp(-first_argument));
            const double second = 1.0/(1.0+std::exp(-second_argument));
            const double value = first*second;
            coordination[left] += value;
            coordination[right] += value;
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(coordinates); Py_DECREF(radii);
    if (collision) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError, "distinct atoms cannot occupy the same point");
        return nullptr;
    }
    return reinterpret_cast<PyObject*>(result);
}

PyObject* geometry_within_displacement_native(PyObject*, PyObject* args) {
    PyObject *current_o, *previous_o;
    double threshold;
    int native_threads = 1;
    if (!PyArg_ParseTuple(args, "OOd", &current_o, &previous_o, &threshold))
        return nullptr;
    auto* current = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        current_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* previous = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        previous_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const bool valid = current && previous && PyArray_NDIM(current) == 2
        && PyArray_DIM(current, 1) == 3
        && PyArray_NDIM(previous) == 2
        && PyArray_DIM(previous, 0) == PyArray_DIM(current, 0)
        && PyArray_DIM(previous, 1) == 3 && threshold >= 0.0;
    if (!valid) {
        Py_XDECREF(current); Py_XDECREF(previous);
        PyErr_SetString(PyExc_ValueError, "invalid displacement audit arrays");
        return nullptr;
    }
    const auto* now = static_cast<const double*>(PyArray_DATA(current));
    const auto* before = static_cast<const double*>(PyArray_DATA(previous));
    const npy_intp count = PyArray_DIM(current, 0);
    const double threshold2 = threshold*threshold;
    bool compatible = true;
    for (npy_intp atom = 0; atom < count && compatible; ++atom) {
        double distance2 = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double delta = now[atom*3+axis]-before[atom*3+axis];
            distance2 += delta*delta;
        }
        compatible = distance2 <= threshold2;
    }
    Py_DECREF(current); Py_DECREF(previous);
    if (compatible) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

PyObject* electronic_entropy_native(PyObject*, PyObject* args) {
    PyObject *alpha_o, *beta_o;
    double thermal;
    if (!PyArg_ParseTuple(args, "OOd", &alpha_o, &beta_o, &thermal))
        return nullptr;
    auto* alpha = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        alpha_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* beta = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        beta_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!alpha || !beta || PyArray_SIZE(alpha) != PyArray_SIZE(beta)) {
        Py_XDECREF(alpha); Py_XDECREF(beta);
        PyErr_SetString(PyExc_ValueError, "invalid electronic occupations");
        return nullptr;
    }
    double result = 0.0;
    for (auto* array : {alpha, beta}) {
        const auto* values = static_cast<const double*>(PyArray_DATA(array));
        for (npy_intp index = 0; index < PyArray_SIZE(array); ++index) {
            const double value = values[index];
            if (value > 0.0 && value < 1.0)
                result += thermal*(value*std::log(value)
                    +(1.0-value)*std::log(1.0-value));
        }
    }
    Py_DECREF(alpha); Py_DECREF(beta);
    return PyFloat_FromDouble(result);
}

PyObject* d4_reference_weights_native(PyObject*, PyObject* args) {
    PyObject *coordination_o, *reference_o, *gaussian_o, *mask_o;
    if (!PyArg_ParseTuple(args, "OOOO", &coordination_o, &reference_o,
            &gaussian_o, &mask_o)) return nullptr;
    auto* coordination = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        coordination_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* reference = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        reference_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* gaussian = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        gaussian_o, NPY_INT16, NPY_ARRAY_IN_ARRAY));
    auto* mask = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        mask_o, NPY_BOOL, NPY_ARRAY_IN_ARRAY));
    const npy_intp natoms = coordination ? PyArray_SIZE(coordination) : 0;
    const npy_intp references = reference && PyArray_NDIM(reference) == 2
        ? PyArray_DIM(reference, 1) : 0;
    const bool valid = coordination && reference && gaussian && mask
        && PyArray_NDIM(coordination) == 1
        && PyArray_NDIM(reference) == 2 && PyArray_DIM(reference, 0) == natoms
        && PyArray_NDIM(gaussian) == 2 && PyArray_DIM(gaussian, 0) == natoms
        && PyArray_DIM(gaussian, 1) == references
        && PyArray_NDIM(mask) == 2 && PyArray_DIM(mask, 0) == natoms
        && PyArray_DIM(mask, 1) == references;
    if (!valid) {
        Py_XDECREF(coordination); Py_XDECREF(reference);
        Py_XDECREF(gaussian); Py_XDECREF(mask);
        if (!PyErr_Occurred()) PyErr_SetString(
            PyExc_ValueError, "invalid native D4 reference arrays");
        return nullptr;
    }
    npy_intp dims[2] = {natoms, references};
    auto* weights = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dims, NPY_DOUBLE, 0));
    auto* derivatives = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dims, NPY_DOUBLE, 0));
    if (!weights || !derivatives) {
        Py_XDECREF(weights); Py_XDECREF(derivatives);
        Py_DECREF(coordination); Py_DECREF(reference);
        Py_DECREF(gaussian); Py_DECREF(mask); return nullptr;
    }
    const auto* cn = static_cast<const double*>(PyArray_DATA(coordination));
    const auto* ref = static_cast<const double*>(PyArray_DATA(reference));
    const auto* counts = static_cast<const std::int16_t*>(PyArray_DATA(gaussian));
    const auto* active = static_cast<const npy_bool*>(PyArray_DATA(mask));
    auto* out = static_cast<double*>(PyArray_DATA(weights));
    auto* dout = static_cast<double*>(PyArray_DATA(derivatives));
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp atom = 0; atom < natoms; ++atom) {
        double total = 0.0, total_derivative = 0.0;
        npy_intp nearest = 0;
        double nearest_distance = std::numeric_limits<double>::infinity();
        for (npy_intp state = 0; state < references; ++state) {
            const auto index = atom*references+state;
            if (!active[index]) continue;
            const double delta = cn[atom]-ref[index];
            if (std::abs(delta) < nearest_distance) {
                nearest_distance = std::abs(delta); nearest = state;
            }
            double raw = 0.0, derivative = 0.0;
            for (int order = 1; order <= counts[index]; ++order) {
                const double term = std::exp(-6.0*delta*delta*order);
                raw += term;
                derivative += -12.0*order*delta*term;
            }
            out[index] = raw; dout[index] = derivative;
            total += raw; total_derivative += derivative;
        }
        if (total <= std::numeric_limits<double>::min()) {
            out[atom*references+nearest] = 1.0;
            total = 1.0; total_derivative = 0.0;
        }
        const double inverse = 1.0/total;
        for (npy_intp state = 0; state < references; ++state) {
            const auto index = atom*references+state;
            const double raw = out[index], derivative = dout[index];
            out[index] = raw*inverse;
            dout[index] = (derivative*total-raw*total_derivative)*inverse*inverse;
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(coordination); Py_DECREF(reference);
    Py_DECREF(gaussian); Py_DECREF(mask);
    return Py_BuildValue("NN", weights, derivatives);
}

PyObject* anisotropic_geometry_native(PyObject*, PyObject* args) {
    PyObject *coordinates_o, *radii_o;
    double a3, a5;
    if (!PyArg_ParseTuple(args, "OOdd", &coordinates_o, &radii_o, &a3, &a5))
        return nullptr;
    auto* coordinates = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        coordinates_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* radii = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        radii_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const npy_intp n = coordinates && PyArray_NDIM(coordinates) == 2
        ? PyArray_DIM(coordinates, 0) : 0;
    if (!coordinates || !radii || PyArray_DIM(coordinates, 1) != 3
        || PyArray_SIZE(radii) != n || !std::isfinite(a3) || !std::isfinite(a5)) {
        Py_XDECREF(coordinates); Py_XDECREF(radii);
        if (!PyErr_Occurred()) PyErr_SetString(
            PyExc_ValueError, "invalid native anisotropic geometry arrays");
        return nullptr;
    }
    const npy_intp pairs = n*(n-1)/2;
    npy_intp vector_dims[1] = {pairs}, displacement_dims[2] = {pairs,3};
    npy_intp outer_dims[3] = {pairs,3,3};
    auto* left = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, vector_dims, NPY_INT64, 0));
    auto* right = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, vector_dims, NPY_INT64, 0));
    auto* displacement = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(2, displacement_dims, NPY_DOUBLE, 0));
    auto* distance2 = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, vector_dims, NPY_DOUBLE, 0));
    auto* f3 = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, vector_dims, NPY_DOUBLE, 0));
    auto* f5 = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, vector_dims, NPY_DOUBLE, 0));
    auto* outer = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(3, outer_dims, NPY_DOUBLE, 0));
    if (!left || !right || !displacement || !distance2 || !f3 || !f5 || !outer) {
        Py_XDECREF(left); Py_XDECREF(right); Py_XDECREF(displacement);
        Py_XDECREF(distance2); Py_XDECREF(f3); Py_XDECREF(f5); Py_XDECREF(outer);
        Py_DECREF(coordinates); Py_DECREF(radii); return nullptr;
    }
    const auto* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* radius = static_cast<const double*>(PyArray_DATA(radii));
    auto* lout = static_cast<std::int64_t*>(PyArray_DATA(left));
    auto* rout = static_cast<std::int64_t*>(PyArray_DATA(right));
    auto* delta = static_cast<double*>(PyArray_DATA(displacement));
    auto* r2out = static_cast<double*>(PyArray_DATA(distance2));
    auto* f3out = static_cast<double*>(PyArray_DATA(f3));
    auto* f5out = static_cast<double*>(PyArray_DATA(f5));
    auto* oout = static_cast<double*>(PyArray_DATA(outer));
    npy_intp pair = 0;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp l = 0; l < n; ++l) for (npy_intp r = l+1; r < n; ++r, ++pair) {
        lout[pair] = l; rout[pair] = r;
        double r2 = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            delta[pair*3+axis] = xyz[r*3+axis]-xyz[l*3+axis];
            r2 += delta[pair*3+axis]*delta[pair*3+axis];
        }
        const double distance = std::sqrt(r2);
        const double damping = 0.5*(radius[l]+radius[r]);
        r2out[pair] = r2;
        f3out[pair] = std::pow(distance,-3.0)
            /(1.0+6.0*std::pow(damping/distance,a3));
        f5out[pair] = std::pow(distance,-5.0)
            /(1.0+6.0*std::pow(damping/distance,a5));
        for (int first = 0; first < 3; ++first)
            for (int second = 0; second < 3; ++second)
                oout[(pair*3+first)*3+second] =
                    delta[pair*3+first]*delta[pair*3+second];
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(coordinates); Py_DECREF(radii);
    return Py_BuildValue("NNNNNNN", left, right, displacement,
        distance2, f3, f5, outer);
}

PyObject* shell_interaction_native(PyObject*, PyObject* args) {
    PyObject *coordinates_o, *shell_atom_o, *hardness_o;
    if (!PyArg_ParseTuple(args, "OOO", &coordinates_o, &shell_atom_o, &hardness_o))
        return nullptr;
    auto* coordinates = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        coordinates_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* shell_atom = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        shell_atom_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* hardness = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        hardness_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const npy_intp natoms = coordinates && PyArray_NDIM(coordinates) == 2
        ? PyArray_DIM(coordinates, 0) : 0;
    const npy_intp nshells = shell_atom ? PyArray_SIZE(shell_atom) : 0;
    if (!coordinates || !shell_atom || !hardness
        || PyArray_DIM(coordinates, 1) != 3
        || PyArray_SIZE(hardness) != nshells) {
        Py_XDECREF(coordinates); Py_XDECREF(shell_atom); Py_XDECREF(hardness);
        if (!PyErr_Occurred()) PyErr_SetString(
            PyExc_ValueError, "invalid native shell interaction arrays");
        return nullptr;
    }
    npy_intp dims[2] = {nshells,nshells};
    auto* result = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(2, dims, NPY_DOUBLE, 0));
    if (!result) {
        Py_DECREF(coordinates); Py_DECREF(shell_atom); Py_DECREF(hardness);
        return nullptr;
    }
    const auto* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* atoms = static_cast<const std::int64_t*>(PyArray_DATA(shell_atom));
    const auto* eta = static_cast<const double*>(PyArray_DATA(hardness));
    auto* out = static_cast<double*>(PyArray_DATA(result));
    bool invalid = false;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp left = 0; left < nshells; ++left) {
        if (atoms[left] < 0 || atoms[left] >= natoms) { invalid = true; break; }
        for (npy_intp right = 0; right <= left; ++right) {
            if (atoms[right] < 0 || atoms[right] >= natoms) { invalid = true; break; }
            double distance2 = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                const double delta = xyz[atoms[left]*3+axis]-xyz[atoms[right]*3+axis];
                distance2 += delta*delta;
            }
            const double average = 0.5*(eta[left]+eta[right]);
            const double value = 1.0/std::sqrt(distance2+1.0/(average*average));
            out[left*nshells+right] = value;
            out[right*nshells+left] = value;
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(coordinates); Py_DECREF(shell_atom); Py_DECREF(hardness);
    if (invalid) {
        Py_DECREF(result); PyErr_SetString(PyExc_ValueError,
            "shell atom index is outside the coordinate array"); return nullptr;
    }
    return reinterpret_cast<PyObject*>(result);
}

PyObject* context_prepare_small_scf_native(PyObject*, PyObject* args) {
    PyObject *capsule, *coordinates, *gfn_coordination, *configuration;
    if (!PyArg_ParseTuple(args, "OOOO", &capsule, &coordinates,
            &gfn_coordination, &configuration)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context || !PyDict_Check(configuration)) return nullptr;
    auto item = [&](const char* name) -> PyObject* {
        auto* value = PyDict_GetItemString(configuration, name);
        if (!value && !PyErr_Occurred())
            PyErr_Format(PyExc_KeyError, "missing small-SCF item: %s", name);
        return value;
    };
    auto scalar = [&](const char* name, double& value) -> bool {
        auto* object = item(name); if (!object) return false;
        value = PyFloat_AsDouble(object); return !PyErr_Occurred();
    };
    double anis_a3, anis_a5, s6, s8, d4_a1, d4_a2;
    if (!scalar("anis_a3", anis_a3) || !scalar("anis_a5", anis_a5)
        || !scalar("s6", s6) || !scalar("s8", s8)
        || !scalar("d4_a1", d4_a1) || !scalar("d4_a2", d4_a2)) return nullptr;
    auto* base_o = item("anis_base_radius");
    auto* valence_cn_o = item("anis_valence_coordination");
    if (!base_o || !valence_cn_o) return nullptr;
    auto* base = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        base_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* valence_cn = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        valence_cn_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* cn = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        gfn_coordination, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const npy_intp natom = cn ? PyArray_SIZE(cn) : 0;
    if (!base || !valence_cn || !cn || PyArray_SIZE(base) != natom
        || PyArray_SIZE(valence_cn) != natom) {
        Py_XDECREF(base); Py_XDECREF(valence_cn); Py_XDECREF(cn);
        PyErr_SetString(PyExc_ValueError, "invalid small-SCF anisotropic arrays");
        return nullptr;
    }
    npy_intp atom_dims[1] = {natom};
    auto* anis_radii = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, atom_dims, NPY_DOUBLE, 0));
    if (!anis_radii) { Py_DECREF(base); Py_DECREF(valence_cn); Py_DECREF(cn); return nullptr; }
    const auto* b = static_cast<const double*>(PyArray_DATA(base));
    const auto* vc = static_cast<const double*>(PyArray_DATA(valence_cn));
    const auto* c = static_cast<const double*>(PyArray_DATA(cn));
    auto* ar = static_cast<double*>(PyArray_DATA(anis_radii));
    for (npy_intp atom = 0; atom < natom; ++atom)
        ar[atom] = b[atom] + (5.0-b[atom])
            /(1.0+std::exp(-4.0*(c[atom]-vc[atom]-1.2)));
    Py_DECREF(base); Py_DECREF(valence_cn); Py_DECREF(cn);

    auto* shell_args = PyTuple_Pack(3, coordinates,
        item("shell_atom"), item("shell_hardness"));
    auto* interaction = shell_args
        ? shell_interaction_native(nullptr, shell_args) : nullptr;
    Py_XDECREF(shell_args);
    if (!interaction) { Py_DECREF(anis_radii); return nullptr; }
    auto* anis_args = Py_BuildValue("OOdd", coordinates, anis_radii,
        anis_a3, anis_a5);
    auto* anis = anis_args
        ? anisotropic_geometry_native(nullptr, anis_args) : nullptr;
    Py_XDECREF(anis_args);
    if (!anis) { Py_DECREF(interaction); Py_DECREF(anis_radii); return nullptr; }
    auto* d4_cn_args = PyTuple_Pack(3, coordinates,
        item("d4_radii"), item("electronegativity"));
    auto* d4_cn = d4_cn_args
        ? d4_coordination_native(nullptr, d4_cn_args) : nullptr;
    Py_XDECREF(d4_cn_args);
    if (!d4_cn) { Py_DECREF(anis); Py_DECREF(interaction); Py_DECREF(anis_radii); return nullptr; }
    auto* weight_args = PyTuple_Pack(4, d4_cn,
        item("d4_reference_coordination"), item("d4_gaussian_count"),
        item("d4_mask"));
    auto* weight_pair = weight_args
        ? d4_reference_weights_native(nullptr, weight_args) : nullptr;
    Py_XDECREF(weight_args);
    if (!weight_pair) {
        Py_DECREF(d4_cn); Py_DECREF(anis); Py_DECREF(interaction);
        Py_DECREF(anis_radii); return nullptr;
    }
    auto* weights = PyTuple_GET_ITEM(weight_pair, 0);
    auto* weight_derivatives = PyTuple_GET_ITEM(weight_pair, 1);
    const npy_intp npair = natom*(natom-1)/2;
    npy_intp pair_dims[1] = {npair};
    auto* rows = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, pair_dims, NPY_INT64, 0));
    auto* columns = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, pair_dims, NPY_INT64, 0));
    auto* ratio = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, pair_dims, NPY_DOUBLE, 0));
    auto* pair_factor = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, pair_dims, NPY_DOUBLE, 0));
    auto* r2r4 = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        item("d4_r2r4"), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* xyz = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        coordinates, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!rows || !columns || !ratio || !pair_factor || !r2r4 || !xyz
        || PyArray_SIZE(r2r4) != natom || PyArray_SIZE(xyz) != natom*3) {
        Py_XDECREF(rows); Py_XDECREF(columns); Py_XDECREF(ratio);
        Py_XDECREF(pair_factor); Py_XDECREF(r2r4); Py_XDECREF(xyz);
        Py_DECREF(weight_pair); Py_DECREF(d4_cn); Py_DECREF(anis);
        Py_DECREF(interaction); Py_DECREF(anis_radii); return nullptr;
    }
    const auto* coordinate_values = static_cast<const double*>(PyArray_DATA(xyz));
    const auto* radii_ratio = static_cast<const double*>(PyArray_DATA(r2r4));
    auto* row_values = static_cast<std::int64_t*>(PyArray_DATA(rows));
    auto* column_values = static_cast<std::int64_t*>(PyArray_DATA(columns));
    auto* ratios = static_cast<double*>(PyArray_DATA(ratio));
    auto* factors = static_cast<double*>(PyArray_DATA(pair_factor));
    npy_intp pair = 0;
    for (npy_intp left = 0; left < natom; ++left)
        for (npy_intp right = left+1; right < natom; ++right, ++pair) {
            row_values[pair] = left; column_values[pair] = right;
            double distance2 = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                const double delta = coordinate_values[left*3+axis]
                    -coordinate_values[right*3+axis];
                distance2 += delta*delta;
            }
            const double pair_ratio = 3.0*radii_ratio[left]*radii_ratio[right];
            const double cutoff = d4_a1*std::sqrt(pair_ratio)+d4_a2;
            ratios[pair] = pair_ratio;
            factors[pair] = s6/(std::pow(distance2,3.0)+std::pow(cutoff,6.0))
                +s8*pair_ratio/(std::pow(distance2,4.0)+std::pow(cutoff,8.0));
        }
    Py_DECREF(r2r4); Py_DECREF(xyz);
    auto* scf = PyDict_New();
    auto set = [&](const char* name, PyObject* value) {
        return value && PyDict_SetItemString(scf, name, value) == 0;
    };
    const bool configured = scf
        && set("interaction", interaction)
        && set("third_order", item("third_order"))
        && set("anis_dipole_xc", item("anis_dipole_xc"))
        && set("anis_quadrupole_xc", item("anis_quadrupole_xc"))
        && set("anis_left", PyTuple_GET_ITEM(anis, 0))
        && set("anis_right", PyTuple_GET_ITEM(anis, 1))
        && set("anis_displacement", PyTuple_GET_ITEM(anis, 2))
        && set("anis_distance2", PyTuple_GET_ITEM(anis, 3))
        && set("anis_f3", PyTuple_GET_ITEM(anis, 4))
        && set("anis_f5", PyTuple_GET_ITEM(anis, 5))
        && set("d4_frequency_weights", item("d4_frequency_weights"))
        && set("d4_weights", weights)
        && set("d4_weight_derivatives", weight_derivatives)
        && set("d4_reference_charges", item("d4_reference_charges"))
        && set("d4_references", item("d4_references"))
        && set("d4_mask", item("d4_mask"))
        && set("d4_effective", item("d4_effective"))
        && set("d4_hardness", item("d4_hardness"))
        && set("d4_rows", reinterpret_cast<PyObject*>(rows))
        && set("d4_columns", reinterpret_cast<PyObject*>(columns))
        && set("d4_ratio", reinterpret_cast<PyObject*>(ratio))
        && set("d4_pair_factor", reinterpret_cast<PyObject*>(pair_factor));
    auto* configure_args = configured ? PyTuple_Pack(2, capsule, scf) : nullptr;
    auto* configure_result = configure_args
        ? context_configure_scf_native(nullptr, configure_args) : nullptr;
    Py_XDECREF(configure_args); Py_XDECREF(scf);
    if (!configure_result) {
        Py_DECREF(rows); Py_DECREF(columns); Py_DECREF(ratio); Py_DECREF(pair_factor);
        Py_DECREF(weight_pair); Py_DECREF(d4_cn); Py_DECREF(anis);
        Py_DECREF(interaction); Py_DECREF(anis_radii); return nullptr;
    }
    Py_DECREF(configure_result);
    Py_INCREF(weights); Py_INCREF(weight_derivatives);
    Py_DECREF(weight_pair);
    return Py_BuildValue("NNNNNNNNNN", interaction, anis_radii, anis, d4_cn,
        weights, weight_derivatives, rows, columns, ratio, pair_factor);
}

PyObject* post_scf_explicit_gradient_native(PyObject*, PyObject* args) {
    PyObject *coordinates_o, *shell_charges_o, *shell_atoms_o, *hardness_o;
    PyObject *charges_o, *dipoles_o, *quadrupoles_o, *anis_radii_o;
    PyObject *gfn_radii_o, *radius_cn_o, *c6_o, *c8_o, *d4_cn_o;
    PyObject *d4_radii_o, *electronegativity_o;
    double anis_a3, anis_a5, s6, s8, d4_a1, d4_a2;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOdddddd",
            &coordinates_o, &shell_charges_o, &shell_atoms_o, &hardness_o,
            &charges_o, &dipoles_o, &quadrupoles_o, &anis_radii_o,
            &gfn_radii_o, &radius_cn_o, &c6_o, &c8_o, &d4_cn_o,
            &d4_radii_o, &electronegativity_o, &anis_a3, &anis_a5,
            &s6, &s8, &d4_a1, &d4_a2)) return nullptr;
    auto double_array=[](PyObject* object){return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));};
    auto* coordinates=double_array(coordinates_o);
    auto* shell_charges=double_array(shell_charges_o);
    auto* shell_atoms=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        shell_atoms_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* hardness=double_array(hardness_o);
    auto* charges=double_array(charges_o); auto* dipoles=double_array(dipoles_o);
    auto* quadrupoles=double_array(quadrupoles_o);
    auto* anis_radii=double_array(anis_radii_o); auto* gfn_radii=double_array(gfn_radii_o);
    auto* radius_cn=double_array(radius_cn_o); auto* c6=double_array(c6_o);
    auto* c8=double_array(c8_o); auto* d4_cn=double_array(d4_cn_o);
    auto* d4_radii=double_array(d4_radii_o);
    auto* electronegativity=double_array(electronegativity_o);
    const npy_intp n=coordinates&&PyArray_NDIM(coordinates)==2
        ?PyArray_DIM(coordinates,0):0;
    const npy_intp nshell=shell_charges?PyArray_SIZE(shell_charges):0;
    const bool valid=coordinates&&shell_charges&&shell_atoms&&hardness&&charges
        &&dipoles&&quadrupoles&&anis_radii&&gfn_radii&&radius_cn&&c6&&c8
        &&d4_cn&&d4_radii&&electronegativity
        &&PyArray_NDIM(coordinates)==2&&PyArray_DIM(coordinates,1)==3
        &&PyArray_SIZE(shell_atoms)==nshell&&PyArray_SIZE(hardness)==nshell
        &&PyArray_SIZE(charges)==n&&PyArray_NDIM(dipoles)==2
        &&PyArray_DIM(dipoles,0)==n&&PyArray_DIM(dipoles,1)==3
        &&PyArray_NDIM(quadrupoles)==3&&PyArray_DIM(quadrupoles,0)==n
        &&PyArray_DIM(quadrupoles,1)==3&&PyArray_DIM(quadrupoles,2)==3
        &&PyArray_SIZE(anis_radii)==n&&PyArray_SIZE(gfn_radii)==n
        &&PyArray_SIZE(radius_cn)==n&&PyArray_NDIM(c6)==2
        &&PyArray_DIM(c6,0)==n&&PyArray_DIM(c6,1)==n&&PyArray_NDIM(c8)==2
        &&PyArray_DIM(c8,0)==n&&PyArray_DIM(c8,1)==n
        &&PyArray_SIZE(d4_cn)==n&&PyArray_SIZE(d4_radii)==n
        &&PyArray_SIZE(electronegativity)==n;
    if(!valid){
        Py_XDECREF(coordinates);Py_XDECREF(shell_charges);Py_XDECREF(shell_atoms);
        Py_XDECREF(hardness);Py_XDECREF(charges);Py_XDECREF(dipoles);
        Py_XDECREF(quadrupoles);Py_XDECREF(anis_radii);Py_XDECREF(gfn_radii);
        Py_XDECREF(radius_cn);Py_XDECREF(c6);Py_XDECREF(c8);Py_XDECREF(d4_cn);
        Py_XDECREF(d4_radii);Py_XDECREF(electronegativity);
        PyErr_SetString(PyExc_ValueError,"invalid fused post-SCF arrays");return nullptr;
    }
    npy_intp dims[2]={n,3};auto* output=reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2,dims,NPY_DOUBLE,0));
    if(!output){
        Py_DECREF(coordinates);Py_DECREF(shell_charges);Py_DECREF(shell_atoms);
        Py_DECREF(hardness);Py_DECREF(charges);Py_DECREF(dipoles);
        Py_DECREF(quadrupoles);Py_DECREF(anis_radii);Py_DECREF(gfn_radii);
        Py_DECREF(radius_cn);Py_DECREF(c6);Py_DECREF(c8);Py_DECREF(d4_cn);
        Py_DECREF(d4_radii);Py_DECREF(electronegativity);return PyErr_NoMemory();
    }
    const auto* xyz=static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* sq=static_cast<const double*>(PyArray_DATA(shell_charges));
    const auto* satom=static_cast<const std::int64_t*>(PyArray_DATA(shell_atoms));
    const auto* eta=static_cast<const double*>(PyArray_DATA(hardness));
    const auto* q=static_cast<const double*>(PyArray_DATA(charges));
    const auto* mu=static_cast<const double*>(PyArray_DATA(dipoles));
    const auto* theta=static_cast<const double*>(PyArray_DATA(quadrupoles));
    const auto* ar=static_cast<const double*>(PyArray_DATA(anis_radii));
    const auto* gr=static_cast<const double*>(PyArray_DATA(gfn_radii));
    const auto* drdcn=static_cast<const double*>(PyArray_DATA(radius_cn));
    const auto* pc6=static_cast<const double*>(PyArray_DATA(c6));
    const auto* pc8=static_cast<const double*>(PyArray_DATA(c8));
    const auto* d4c=static_cast<const double*>(PyArray_DATA(d4_cn));
    const auto* d4r=static_cast<const double*>(PyArray_DATA(d4_radii));
    const auto* en=static_cast<const double*>(PyArray_DATA(electronegativity));
    auto* gradient=static_cast<double*>(PyArray_DATA(output));
    std::vector<double> anis_radius(static_cast<std::size_t>(n),0.0);
    constexpr double inv_sqrt_pi=0.56418958354775628695;
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp left=1;left<nshell;++left)for(npy_intp right=0;right<left;++right){
        const auto la=satom[left],ra=satom[right];if(la==ra)continue;
        double displacement[3],distance2=0.0;
        for(int axis=0;axis<3;++axis){displacement[axis]=xyz[la*3+axis]-xyz[ra*3+axis];
            distance2+=displacement[axis]*displacement[axis];}
        const double distance=std::sqrt(distance2),average=0.5*(eta[left]+eta[right]);
        const double radial=-distance/std::pow(distance2+1.0/(average*average),1.5);
        const double scale=sq[left]*sq[right]*radial/distance;
        for(int axis=0;axis<3;++axis){const double value=scale*displacement[axis];
            gradient[la*3+axis]+=value;gradient[ra*3+axis]-=value;}
    }
    for(npy_intp left=1;left<n;++left)for(npy_intp right=0;right<left;++right){
        double displacement[3],distance2=0.0;
        for(int axis=0;axis<3;++axis){displacement[axis]=xyz[right*3+axis]-xyz[left*3+axis];
            distance2+=displacement[axis]*displacement[axis];}
        const double distance=std::sqrt(distance2),radius=0.5*(ar[left]+ar[right]);
        const double ratio3=6.0*std::pow(radius/distance,anis_a3);
        const double ratio5=6.0*std::pow(radius/distance,anis_a5);
        const double f3=std::pow(distance,-3.0)/(1.0+ratio3);
        const double f5=std::pow(distance,-5.0)/(1.0+ratio5);
        const double df3=f3*(-3.0+anis_a3*ratio3/(1.0+ratio3))/distance;
        const double df5=f5*(-5.0+anis_a5*ratio5/(1.0+ratio5))/distance;
        const double da3=-f3*anis_a3*ratio3/((1.0+ratio3)*radius);
        const double da5=-f5*anis_a5*ratio5/((1.0+ratio5)*radius);
        double lp=0.0,rp=0.0,ldot=0.0,rdot=0.0,mudot=0.0;
        for(int a=0;a<3;++a){lp+=mu[left*3+a]*displacement[a];
            rp+=mu[right*3+a]*displacement[a];mudot+=mu[left*3+a]*mu[right*3+a];
            double lv=0.0,rv=0.0;for(int b=0;b<3;++b){
                lv+=theta[(left*3+a)*3+b]*displacement[b];
                rv+=theta[(right*3+a)*3+b]*displacement[b];}
            ldot+=displacement[a]*lv;rdot+=displacement[a]*rv;}
        const double angular3=-q[left]*rp+q[right]*lp;
        const double angular5=q[left]*rdot+q[right]*ldot-3.0*lp*rp+mudot*distance2;
        for(int axis=0;axis<3;++axis){
            double tl=0.0,tr=0.0;for(int b=0;b<3;++b){
                tl+=theta[(left*3+axis)*3+b]*displacement[b];
                tr+=theta[(right*3+axis)*3+b]*displacement[b];}
            const double dag3=-q[left]*mu[right*3+axis]+q[right]*mu[left*3+axis];
            const double dag5=2.0*q[left]*tr+2.0*q[right]*tl
                -3.0*(mu[left*3+axis]*rp+mu[right*3+axis]*lp)
                +2.0*mudot*displacement[axis];
            const double value=(df3*angular3/distance)*displacement[axis]+f3*dag3
                +(df5*angular5/distance)*displacement[axis]+f5*dag5;
            gradient[right*3+axis]+=value;gradient[left*3+axis]-=value;
        }
        const double pair_radius=0.5*(da3*angular3+da5*angular5);
        anis_radius[left]+=pair_radius;anis_radius[right]+=pair_radius;
    }
    for(npy_intp left=1;left<n;++left)for(npy_intp right=0;right<left;++right){
        double displacement[3],distance2=0.0;
        for(int axis=0;axis<3;++axis){displacement[axis]=xyz[left*3+axis]-xyz[right*3+axis];
            distance2+=displacement[axis]*displacement[axis];}
        const double distance=std::sqrt(distance2);
        const double scale1=4.0*(gr[left]+gr[right])/3.0,scale2=scale1+2.0;
        const double first=1.0/(1.0+std::exp(-10.0*(scale1/distance-1.0)));
        const double second=1.0/(1.0+std::exp(-20.0*(scale2/distance-1.0)));
        const double gfn_radial=-10.0*scale1*first*(1.0-first)*second/distance2
            -20.0*scale2*second*(1.0-second)*first/distance2;
        const double c6pair=pc6[left*n+right],c8pair=pc8[left*n+right];
        const double cutoff=d4_a1*std::sqrt(c8pair/c6pair)+d4_a2;
        const double d4_radial=s6*c6pair*6.0*std::pow(distance,5.0)
                /std::pow(std::pow(distance,6.0)+std::pow(cutoff,6.0),2.0)
            +s8*c8pair*8.0*std::pow(distance,7.0)
                /std::pow(std::pow(distance,8.0)+std::pow(cutoff,8.0),2.0);
        const double d4_critical=4.0*(d4r[left]+d4r[right])/3.0;
        const double argument=-7.5*(distance-d4_critical)/d4_critical;
        const double delta=std::abs(en[left]-en[right]);
        const double d4_cn_radial=-7.5*std::exp(-argument*argument)*inv_sqrt_pi/d4_critical
            *4.10451*std::exp(-(delta+19.08857)*(delta+19.08857)
                /(2.0*11.28174*11.28174));
        const double scale=((anis_radius[left]*drdcn[left]+anis_radius[right]*drdcn[right])
                *gfn_radial+(d4c[left]+d4c[right])*d4_cn_radial+d4_radial)/distance;
        for(int axis=0;axis<3;++axis){const double value=scale*displacement[axis];
            gradient[left*3+axis]+=value;gradient[right*3+axis]-=value;}
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(coordinates);Py_DECREF(shell_charges);Py_DECREF(shell_atoms);
    Py_DECREF(hardness);Py_DECREF(charges);Py_DECREF(dipoles);Py_DECREF(quadrupoles);
    Py_DECREF(anis_radii);Py_DECREF(gfn_radii);Py_DECREF(radius_cn);Py_DECREF(c6);
    Py_DECREF(c8);Py_DECREF(d4_cn);Py_DECREF(d4_radii);Py_DECREF(electronegativity);
    return reinterpret_cast<PyObject*>(output);
}

PyObject* localized_domains_native(PyObject*, PyObject* args) {
    PyObject *coordinates_o, *atom_index_o;
    double cutoff;
    int hops;
    if (!PyArg_ParseTuple(args, "OOdi", &coordinates_o, &atom_index_o, &cutoff, &hops)) return nullptr;
    auto* coordinates = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(coordinates_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* atom_index = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(atom_index_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    if (!coordinates || !atom_index) { Py_XDECREF(coordinates); Py_XDECREF(atom_index); return nullptr; }
    const npy_intp nat = PyArray_DIM(coordinates, 0), norb = PyArray_SIZE(atom_index);
    const auto* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* atoms = static_cast<const std::int64_t*>(PyArray_DATA(atom_index));
    std::unordered_map<CellKey, std::vector<std::int64_t>, CellHash> cells;
    std::vector<CellKey> atom_cells(nat);
    for (npy_intp atom = 0; atom < nat; ++atom) {
        CellKey key{
            static_cast<std::int64_t>(std::floor(xyz[atom*3] / cutoff)),
            static_cast<std::int64_t>(std::floor(xyz[atom*3+1] / cutoff)),
            static_cast<std::int64_t>(std::floor(xyz[atom*3+2] / cutoff))};
        atom_cells[atom] = key; cells[key].push_back(atom);
    }
    std::vector<std::unordered_set<std::int64_t>> base(nat);
    const double cutoff2 = cutoff * cutoff;
    for (npy_intp atom = 0; atom < nat; ++atom) {
        base[atom].insert(atom);
        const auto key = atom_cells[atom];
        for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
            auto found = cells.find(CellKey{key.x+dx, key.y+dy, key.z+dz});
            if (found == cells.end()) continue;
            for (auto other : found->second) {
                if (other <= atom) continue;
                double distance2 = 0.0;
                for (int axis = 0; axis < 3; ++axis) {
                    const double d = xyz[atom*3+axis] - xyz[other*3+axis]; distance2 += d*d;
                }
                if (distance2 <= cutoff2) { base[atom].insert(other); base[other].insert(atom); }
            }
        }
    }
    auto expanded = base;
    for (int hop = 0; hop < hops; ++hop) {
        std::vector<std::unordered_set<std::int64_t>> next(nat);
        for (npy_intp atom = 0; atom < nat; ++atom) {
            for (auto member : expanded[atom]) next[atom].insert(base[member].begin(), base[member].end());
        }
        expanded.swap(next);
    }
    std::vector<std::vector<std::int64_t>> orbitals_by_atom(nat);
    for (npy_intp orbital = 0; orbital < norb; ++orbital) orbitals_by_atom[atoms[orbital]].push_back(orbital);
    std::vector<std::int64_t> offsets(nat + 1, 0), neighbor_indices;
    std::vector<std::int64_t> pair_left, pair_right;
    for (npy_intp atom = 0; atom < nat; ++atom) {
        std::vector<std::int64_t> sorted(expanded[atom].begin(), expanded[atom].end());
        std::sort(sorted.begin(), sorted.end());
        neighbor_indices.insert(neighbor_indices.end(), sorted.begin(), sorted.end());
        offsets[atom+1] = neighbor_indices.size();
    }
    for (npy_intp left = 0; left < norb; ++left) {
        std::vector<std::int64_t> rights;
        for (auto atom : expanded[atoms[left]]) {
            for (auto right : orbitals_by_atom[atom]) if (right <= left) rights.push_back(right);
        }
        std::sort(rights.begin(), rights.end());
        for (auto right : rights) { pair_left.push_back(left); pair_right.push_back(right); }
    }
    auto make_i64 = [](const std::vector<std::int64_t>& source) -> PyArrayObject* {
        npy_intp dims[1] = {static_cast<npy_intp>(source.size())};
        auto* out = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, dims, NPY_INT64, 0));
        if (out) std::copy(source.begin(), source.end(), static_cast<std::int64_t*>(PyArray_DATA(out)));
        return out;
    };
    auto make_i32 = [](const std::vector<std::int64_t>& source) -> PyArrayObject* {
        npy_intp dims[1] = {static_cast<npy_intp>(source.size())};
        auto* out = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, dims, NPY_INT32, 0));
        if (out) {
            auto* values = static_cast<std::int32_t*>(PyArray_DATA(out));
            std::transform(source.begin(), source.end(), values, [](std::int64_t value) {
                return static_cast<std::int32_t>(value);
            });
        }
        return out;
    };
    auto* offsets_o = make_i64(offsets); auto* neighbors_o = make_i32(neighbor_indices);
    auto* left_o = make_i32(pair_left); auto* right_o = make_i32(pair_right);
    Py_DECREF(coordinates); Py_DECREF(atom_index);
    if (!offsets_o || !neighbors_o || !left_o || !right_o) {
        Py_XDECREF(offsets_o); Py_XDECREF(neighbors_o); Py_XDECREF(left_o); Py_XDECREF(right_o);
        return PyErr_NoMemory();
    }
    return Py_BuildValue("NNNN", offsets_o, neighbors_o, left_o, right_o);
}

PyObject* sparse_from_dense_native(PyObject*, PyObject* args) {
    PyObject *dense_o, *left_o, *right_o;
    double threshold;
    if (!PyArg_ParseTuple(args, "OOOd", &dense_o, &left_o, &right_o, &threshold)) return nullptr;
    auto* dense = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(dense_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* left = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(left_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    auto* right = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(right_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    if (!dense || !left || !right) { Py_XDECREF(dense); Py_XDECREF(left); Py_XDECREF(right); return nullptr; }
    const npy_intp size = PyArray_DIM(dense, 0), count = PyArray_SIZE(left);
    const auto* matrix = static_cast<const double*>(PyArray_DATA(dense));
    const auto* rows = static_cast<const std::int32_t*>(PyArray_DATA(left));
    const auto* columns = static_cast<const std::int32_t*>(PyArray_DATA(right));
    SparseMatrixNative result; result.size = size;
    result.rows.reserve(count); result.columns.reserve(count); result.values.reserve(count);
    for (npy_intp i = 0; i < count; ++i) {
        const double value = matrix[rows[i]*size+columns[i]];
        if (std::abs(value - matrix[columns[i]*size+rows[i]]) > 1.0e-13) {
            Py_DECREF(dense); Py_DECREF(left); Py_DECREF(right);
            PyErr_SetString(PyExc_ValueError, "sparse symmetric matrix requires symmetric input");
            return nullptr;
        }
        if (rows[i] == columns[i] || std::abs(value) >= threshold) {
            result.rows.push_back(rows[i]); result.columns.push_back(columns[i]); result.values.push_back(value);
        }
    }
    Py_DECREF(dense); Py_DECREF(left); Py_DECREF(right);
    return sparse_arrays(result);
}

PyObject* sparse_topology_native(PyObject*, PyObject* args) {
    long long size;
    PyObject *left_o, *right_o;
    int include_intersections = 0;
    if (!PyArg_ParseTuple(args, "LOO|p", &size, &left_o, &right_o,
            &include_intersections)) return nullptr;
    auto* left = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        left_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    auto* right = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        right_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    if (!left || !right || PyArray_SIZE(left) != PyArray_SIZE(right)) {
        Py_XDECREF(left); Py_XDECREF(right);
        PyErr_SetString(PyExc_ValueError, "invalid sparse topology pair arrays");
        return nullptr;
    }
    using Entry = std::tuple<std::int64_t, std::int64_t, std::int64_t>;
    const auto count = PyArray_SIZE(left);
    const auto* l = static_cast<const std::int32_t*>(PyArray_DATA(left));
    const auto* r = static_cast<const std::int32_t*>(PyArray_DATA(right));
    std::vector<std::vector<Entry>> adjacency(size);
    for (npy_intp pair = 0; pair < count; ++pair) {
        adjacency[l[pair]].push_back({r[pair], pair, 0});
        if (l[pair] != r[pair]) adjacency[r[pair]].push_back({l[pair], pair, 0});
    }
    std::vector<std::int64_t> offsets(size + 1, 0), neighbors, pair_indices;
    for (long long row = 0; row < size; ++row) {
        auto& entries = adjacency[row];
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            return std::get<0>(a) < std::get<0>(b);
        });
        offsets[row + 1] = offsets[row] + static_cast<std::int64_t>(entries.size());
        for (const auto& entry : entries) {
            neighbors.push_back(std::get<0>(entry));
            pair_indices.push_back(std::get<1>(entry));
        }
    }
    auto make_array = [](const std::vector<std::int64_t>& values) -> PyObject* {
        npy_intp dims[1] = {static_cast<npy_intp>(values.size())};
        auto* result = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, dims, NPY_INT64, 0));
        if (!result) return nullptr;
        std::copy(values.begin(), values.end(), static_cast<std::int64_t*>(PyArray_DATA(result)));
        return reinterpret_cast<PyObject*>(result);
    };
    auto make_index_array = [](const std::vector<std::int64_t>& values) -> PyObject* {
        npy_intp dims[1] = {static_cast<npy_intp>(values.size())};
        auto* result = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, dims, NPY_INT32, 0));
        if (!result) return nullptr;
        auto* output = static_cast<std::int32_t*>(PyArray_DATA(result));
        std::transform(values.begin(), values.end(), output, [](std::int64_t value) {
            return static_cast<std::int32_t>(value);
        });
        return reinterpret_cast<PyObject*>(result);
    };
    PyObject* offsets_o = make_array(offsets);
    PyObject* neighbors_o = make_index_array(neighbors);
    PyObject* pairs_o = make_index_array(pair_indices);
    if (!offsets_o || !neighbors_o || !pairs_o) {
        Py_DECREF(left); Py_DECREF(right);
        Py_XDECREF(offsets_o); Py_XDECREF(neighbors_o); Py_XDECREF(pairs_o);
        return PyErr_NoMemory();
    }
    if (include_intersections) {
        std::vector<std::int64_t> common_offsets(count + 1, 0);
        std::vector<std::int64_t> common_left, common_right;
        for (npy_intp pair = 0; pair < count; ++pair) {
            const auto& left_row = adjacency[l[pair]];
            const auto& right_row = adjacency[r[pair]];
            std::size_t li = 0, ri = 0;
            while (li < left_row.size() && ri < right_row.size()) {
                if (std::get<0>(left_row[li]) < std::get<0>(right_row[ri])) ++li;
                else if (std::get<0>(right_row[ri]) < std::get<0>(left_row[li])) ++ri;
                else {
                    common_left.push_back(std::get<1>(left_row[li]));
                    common_right.push_back(std::get<1>(right_row[ri]));
                    ++li; ++ri;
                }
            }
            common_offsets[pair + 1] = static_cast<std::int64_t>(common_left.size());
        }
        PyObject* common_offsets_o = make_array(common_offsets);
        PyObject* common_left_o = make_index_array(common_left);
        PyObject* common_right_o = make_index_array(common_right);
        Py_DECREF(left); Py_DECREF(right);
        if (!common_offsets_o || !common_left_o || !common_right_o) {
            Py_DECREF(offsets_o); Py_DECREF(neighbors_o); Py_DECREF(pairs_o);
            Py_XDECREF(common_offsets_o); Py_XDECREF(common_left_o);
            Py_XDECREF(common_right_o);
            return PyErr_NoMemory();
        }
        return Py_BuildValue("NNNNNN", offsets_o, neighbors_o, pairs_o,
            common_offsets_o, common_left_o, common_right_o);
    }
    Py_DECREF(left); Py_DECREF(right);
    return Py_BuildValue("NNN", offsets_o, neighbors_o, pairs_o);
}

PyObject* sparse_sp2_native(PyObject*, PyObject* args) {
    PyObject *rows_o, *columns_o, *values_o, *allowed_left_o, *allowed_right_o;
    PyObject *topology_offsets_o = nullptr, *topology_neighbors_o = nullptr,
             *topology_pairs_o = nullptr, *common_offsets_o = nullptr,
             *common_left_o = nullptr, *common_right_o = nullptr;
    long long size, occupied;
    double lower, upper, tolerance, threshold;
    int max_iterations;
    int native_threads = 1;
    if (!PyArg_ParseTuple(args, "LOOOLOOdddid|OOOOOOi", &size, &rows_o, &columns_o, &values_o,
            &occupied, &allowed_left_o, &allowed_right_o, &lower, &upper,
            &tolerance, &max_iterations, &threshold, &topology_offsets_o,
            &topology_neighbors_o, &topology_pairs_o, &common_offsets_o,
            &common_left_o, &common_right_o, &native_threads)) return nullptr;
    auto* rows = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(rows_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    auto* columns = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(columns_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    auto* values = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(values_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* allowed_left = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(allowed_left_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    auto* allowed_right = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(allowed_right_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    if (!rows || !columns || !values || !allowed_left || !allowed_right) {
        Py_XDECREF(rows); Py_XDECREF(columns); Py_XDECREF(values); Py_XDECREF(allowed_left); Py_XDECREF(allowed_right); return nullptr;
    }
    const npy_intp nnz = PyArray_SIZE(values), allowed_count = PyArray_SIZE(allowed_left);
    if (static_cast<unsigned long long>(allowed_count)
        > static_cast<unsigned long long>(std::numeric_limits<std::uint32_t>::max())) {
        Py_DECREF(rows); Py_DECREF(columns); Py_DECREF(values);
        Py_DECREF(allowed_left); Py_DECREF(allowed_right);
        PyErr_SetString(PyExc_OverflowError, "sparse SP2 pair pattern exceeds 32-bit indexing");
        return nullptr;
    }
    native_threads = std::max(1, std::min(
        native_threads, static_cast<int>(std::max<npy_intp>(1, allowed_count))));
    SparseMatrixNative fock; fock.size = size;
    const auto* r = static_cast<const std::int32_t*>(PyArray_DATA(rows));
    const auto* c = static_cast<const std::int32_t*>(PyArray_DATA(columns));
    const auto* v = static_cast<const double*>(PyArray_DATA(values));
    fock.rows.assign(r, r+nnz); fock.columns.assign(c, c+nnz); fock.values.assign(v, v+nnz);
    const auto* al = static_cast<const std::int32_t*>(PyArray_DATA(allowed_left));
    const auto* ar = static_cast<const std::int32_t*>(PyArray_DATA(allowed_right));
    const double width = upper - lower;
    std::unordered_map<std::int64_t,double> fock_values;
    fock_values.reserve(fock.values.size()*2+1);
    for(std::size_t i=0;i<fock.values.size();++i)
        fock_values[fock.rows[i]*size+fock.columns[i]]=fock.values[i];
    std::vector<double> projector(allowed_count,0.0),square(allowed_count,0.0);
    std::vector<std::size_t> diagonal_indices;diagonal_indices.reserve(size);
    double discarded_squared=0.0;
    for(npy_intp pair=0;pair<allowed_count;++pair){
        const auto row=al[pair],column=ar[pair];auto found=fock_values.find(row*size+column);
        double value=(row==column?upper/width:0.0)-(found==fock_values.end()?0.0:found->second/width);
        if(row!=column&&std::abs(value)<threshold){discarded_squared+=2.0*value*value;value=0.0;}
        projector[pair]=value;
        if(row==column) diagonal_indices.push_back(static_cast<std::size_t>(pair));
    }
    // Compact symmetric CSR topology.  The row pattern is invariant during
    // an SCF cycle, so the Python domain normally supplies these arrays from
    // its cache.  The fallback builds the same representation without nested
    // vectors or per-row heap allocations.
    std::vector<std::int64_t> row_offsets(size + 1, 0);
    std::vector<std::int32_t> neighbors;
    std::vector<std::uint32_t> adjacency_pairs;
    const bool supplied_topology = topology_offsets_o && topology_neighbors_o
        && topology_pairs_o && topology_offsets_o != Py_None
        && topology_neighbors_o != Py_None && topology_pairs_o != Py_None;
    if (supplied_topology) {
        auto* offsets = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            topology_offsets_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
        auto* neighbors_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            topology_neighbors_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
        auto* pair_indices = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            topology_pairs_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
        if (!offsets || !neighbors_array || !pair_indices
            || PyArray_SIZE(offsets) != size + 1
            || PyArray_SIZE(neighbors_array) != PyArray_SIZE(pair_indices)) {
            Py_XDECREF(offsets); Py_XDECREF(neighbors_array); Py_XDECREF(pair_indices);
            Py_DECREF(rows); Py_DECREF(columns); Py_DECREF(values);
            Py_DECREF(allowed_left); Py_DECREF(allowed_right);
            PyErr_SetString(PyExc_ValueError, "invalid sparse SP2 topology");
            return nullptr;
        }
        const auto* offsets_data = static_cast<const std::int64_t*>(PyArray_DATA(offsets));
        const auto* neighbors_data = static_cast<const std::int32_t*>(PyArray_DATA(neighbors_array));
        const auto* pair_data = static_cast<const std::int32_t*>(PyArray_DATA(pair_indices));
        row_offsets.assign(offsets_data, offsets_data + size + 1);
        const auto topology_count = PyArray_SIZE(neighbors_array);
        neighbors.assign(neighbors_data, neighbors_data + topology_count);
        adjacency_pairs.resize(topology_count);
        for (npy_intp index = 0; index < topology_count; ++index) {
            if (pair_data[index] < 0 || pair_data[index] >= allowed_count) {
                Py_DECREF(offsets); Py_DECREF(neighbors_array); Py_DECREF(pair_indices);
                Py_DECREF(rows); Py_DECREF(columns); Py_DECREF(values);
                Py_DECREF(allowed_left); Py_DECREF(allowed_right);
                PyErr_SetString(PyExc_ValueError, "invalid sparse SP2 topology pair index");
                return nullptr;
            }
            adjacency_pairs[index] = static_cast<std::uint32_t>(pair_data[index]);
        }
        Py_DECREF(offsets); Py_DECREF(neighbors_array); Py_DECREF(pair_indices);
    } else {
        for (npy_intp pair = 0; pair < allowed_count; ++pair) {
            ++row_offsets[al[pair] + 1];
            if (al[pair] != ar[pair]) ++row_offsets[ar[pair] + 1];
        }
        for (long long row = 0; row < size; ++row)
            row_offsets[row + 1] += row_offsets[row];
        neighbors.resize(row_offsets.back());
        adjacency_pairs.resize(row_offsets.back());
        std::vector<std::int64_t> cursor = row_offsets;
        for (npy_intp pair = 0; pair < allowed_count; ++pair) {
            auto index = cursor[al[pair]]++;
            neighbors[index] = ar[pair]; adjacency_pairs[index] = static_cast<std::uint32_t>(pair);
            if (al[pair] != ar[pair]) {
                index = cursor[ar[pair]]++;
                neighbors[index] = al[pair]; adjacency_pairs[index] = static_cast<std::uint32_t>(pair);
            }
        }
        for (long long row = 0; row < size; ++row) {
            std::vector<std::pair<std::int64_t,std::uint32_t>> entries;
            entries.reserve(row_offsets[row + 1] - row_offsets[row]);
            for (auto index = row_offsets[row]; index < row_offsets[row + 1]; ++index)
                entries.push_back({neighbors[index], adjacency_pairs[index]});
            std::sort(entries.begin(), entries.end());
            for (std::size_t local = 0; local < entries.size(); ++local) {
                const auto index = row_offsets[row] + static_cast<std::int64_t>(local);
                neighbors[index] = entries[local].first;
                adjacency_pairs[index] = entries[local].second;
            }
        }
    }
    // Optionally materialize the row intersections when the caller's memory
    // estimate allows it; otherwise intersect the compact CSR rows per step.
    std::vector<std::int64_t> common_offsets(allowed_count + 1, 0);
    std::vector<std::uint32_t> common_left, common_right;
    const bool supplied_intersections = common_offsets_o && common_left_o
        && common_right_o && common_offsets_o != Py_None
        && common_left_o != Py_None && common_right_o != Py_None;
    if (supplied_intersections) {
        auto* offsets_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            common_offsets_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
        auto* left_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            common_left_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
        auto* right_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            common_right_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
        if (!offsets_array || !left_array || !right_array
            || PyArray_SIZE(offsets_array) != allowed_count + 1
            || PyArray_SIZE(left_array) != PyArray_SIZE(right_array)) {
            Py_XDECREF(offsets_array); Py_XDECREF(left_array); Py_XDECREF(right_array);
            Py_DECREF(rows); Py_DECREF(columns); Py_DECREF(values);
            Py_DECREF(allowed_left); Py_DECREF(allowed_right);
            PyErr_SetString(PyExc_ValueError, "invalid sparse SP2 intersections");
            return nullptr;
        }
        const auto* offsets_data = static_cast<const std::int64_t*>(PyArray_DATA(offsets_array));
        const auto* left_data = static_cast<const std::int32_t*>(PyArray_DATA(left_array));
        const auto* right_data = static_cast<const std::int32_t*>(PyArray_DATA(right_array));
        common_offsets.assign(offsets_data, offsets_data + allowed_count + 1);
        const auto common_count = PyArray_SIZE(left_array);
        common_left.resize(common_count); common_right.resize(common_count);
        for (npy_intp index = 0; index < common_count; ++index) {
            if (left_data[index] < 0 || left_data[index] >= allowed_count
                || right_data[index] < 0 || right_data[index] >= allowed_count) {
                Py_DECREF(offsets_array); Py_DECREF(left_array); Py_DECREF(right_array);
                Py_DECREF(rows); Py_DECREF(columns); Py_DECREF(values);
                Py_DECREF(allowed_left); Py_DECREF(allowed_right);
                PyErr_SetString(PyExc_ValueError, "invalid sparse SP2 intersection index");
                return nullptr;
            }
            common_left[index] = static_cast<std::uint32_t>(left_data[index]);
            common_right[index] = static_cast<std::uint32_t>(right_data[index]);
        }
        Py_DECREF(offsets_array); Py_DECREF(left_array); Py_DECREF(right_array);
    }
    double discarded=std::sqrt(discarded_squared),bound=discarded,error=INFINITY;
    const double scale=size<128?1.0:std::sqrt(static_cast<double>(size));
    const double extensive_tolerance=tolerance*scale;
    const double trace_tolerance=tolerance*(size<128?1.0:static_cast<double>(size));
    int updates = 0;
    bool converged = false;
    std::vector<std::vector<double>> history;
    std::vector<std::int8_t> branches;
    history.push_back(projector);
    const npy_intp pair_chunk =
        (allowed_count + native_threads - 1) / native_threads;
    std::vector<double> discarded_by_thread(native_threads, 0.0);
    auto run_pair_ranges = [&](auto&& operation) {
        std::vector<std::thread> workers;
        workers.reserve(native_threads > 1 ? native_threads - 1 : 0);
        for (int thread = 1; thread < native_threads; ++thread)
            workers.emplace_back(operation, thread);
        operation(0);
        for (auto& worker : workers) worker.join();
    };
    Py_BEGIN_ALLOW_THREADS
    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        auto square_range = [&](int thread) {
            const npy_intp begin = static_cast<npy_intp>(thread) * pair_chunk;
            const npy_intp end = std::min(allowed_count, begin + pair_chunk);
            double local_discarded = 0.0;
            for (npy_intp pair = begin; pair < end; ++pair) {
                double value = 0.0;
                if (supplied_intersections) {
                    for (std::int64_t common = common_offsets[pair];
                            common < common_offsets[pair + 1]; ++common)
                        value += projector[common_left[common]]
                            *projector[common_right[common]];
                } else {
                    auto li = row_offsets[al[pair]], ri = row_offsets[ar[pair]];
                    const auto li_end = row_offsets[al[pair] + 1];
                    const auto ri_end = row_offsets[ar[pair] + 1];
                    while (li < li_end && ri < ri_end) {
                        if (neighbors[li] < neighbors[ri]) ++li;
                        else if (neighbors[ri] < neighbors[li]) ++ri;
                        else {
                            value += projector[adjacency_pairs[li]]
                                *projector[adjacency_pairs[ri]];
                            ++li; ++ri;
                        }
                    }
                }
                if (al[pair] != ar[pair] && std::abs(value) < threshold) {
                    local_discarded += 2.0 * value * value;
                    value = 0.0;
                }
                square[pair] = value;
            }
            discarded_by_thread[thread] = local_discarded;
        };
        run_pair_ranges(square_range);
        discarded_squared = 0.0;
        for (double local : discarded_by_thread) discarded_squared += local;
        discarded=std::sqrt(discarded_squared);
        bound += discarded;
        double norm2=0.0;for(npy_intp pair=0;pair<allowed_count;++pair){const double delta=square[pair]-projector[pair];norm2+=(al[pair]==ar[pair]?1.0:2.0)*delta*delta;}
        error = std::sqrt(norm2) + discarded;
        double trace=0.0;for(auto pair:diagonal_indices)trace+=projector[pair];
        if (error <= extensive_tolerance && std::abs(trace-occupied) <= trace_tolerance) {
            converged = true; break;
        }
        if (trace > occupied) {
            branches.push_back(1);
            projector.swap(square);
        }
        else {
            branches.push_back(-1);
            auto update_range = [&](int thread) {
                const npy_intp begin = static_cast<npy_intp>(thread) * pair_chunk;
                const npy_intp end = std::min(allowed_count, begin + pair_chunk);
                double local_discarded = 0.0;
                for (npy_intp pair = begin; pair < end; ++pair) {
                    double value = 2.0 * projector[pair] - square[pair];
                    if (al[pair] != ar[pair] && std::abs(value) < threshold) {
                        local_discarded += 2.0 * value * value;
                        value = 0.0;
                    }
                    projector[pair] = value;
                }
                discarded_by_thread[thread] = local_discarded;
            };
            run_pair_ranges(update_range);
            discarded_squared = 0.0;
            for (double local : discarded_by_thread) discarded_squared += local;
            discarded=std::sqrt(discarded_squared);
            bound += discarded;
        }
        history.push_back(projector);
        updates = iteration;
        if (!std::all_of(projector.begin(), projector.end(), [](double value) {
                return std::isfinite(value);
            })) {
            break;
        }
    }
    if (!converged) {
        double trace=0.0;for(auto pair:diagonal_indices)trace+=projector[pair];
        converged=error<=extensive_tolerance&&std::abs(trace-occupied)<=trace_tolerance;
    }
    Py_END_ALLOW_THREADS
    SparseMatrixNative projector_sparse;projector_sparse.size=size;
    for(npy_intp pair=0;pair<allowed_count;++pair)if(al[pair]==ar[pair]||projector[pair]!=0.0){
        projector_sparse.rows.push_back(al[pair]);projector_sparse.columns.push_back(ar[pair]);projector_sparse.values.push_back(projector[pair]);
    }
    PyObject* arrays = sparse_arrays(projector_sparse);
    Py_DECREF(rows); Py_DECREF(columns); Py_DECREF(values); Py_DECREF(allowed_left); Py_DECREF(allowed_right);
    if (!arrays) return nullptr;
    PyObject *out_rows, *out_columns, *out_values;
    if (!PyArg_ParseTuple(arrays, "OOO", &out_rows, &out_columns, &out_values)) { Py_DECREF(arrays); return nullptr; }
    Py_INCREF(out_rows); Py_INCREF(out_columns); Py_INCREF(out_values); Py_DECREF(arrays);
    npy_intp history_dims[2] = {
        static_cast<npy_intp>(history.size()), allowed_count
    };
    auto* history_array = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(2, history_dims, NPY_DOUBLE, 0));
    npy_intp branch_dims[1] = {static_cast<npy_intp>(branches.size())};
    auto* branch_array = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, branch_dims, NPY_INT8, 0));
    if (!history_array || !branch_array) {
        Py_DECREF(out_rows); Py_DECREF(out_columns); Py_DECREF(out_values);
        Py_XDECREF(history_array); Py_XDECREF(branch_array);
        return PyErr_NoMemory();
    }
    auto* history_data = static_cast<double*>(PyArray_DATA(history_array));
    for (std::size_t step = 0; step < history.size(); ++step)
        std::copy(history[step].begin(), history[step].end(),
                  history_data + step * allowed_count);
    auto* branch_data = static_cast<std::int8_t*>(PyArray_DATA(branch_array));
    std::copy(branches.begin(), branches.end(), branch_data);
    return Py_BuildValue("NNNidddNN", out_rows, out_columns, out_values, updates,
        converged ? 1.0 : 0.0, bound, error, history_array, branch_array);
}

PyObject* sparse_sp2_incremental_native(PyObject*, PyObject* args) {
    PyObject *rows_o, *columns_o, *values_o, *allowed_left_o, *allowed_right_o;
    PyObject *history_o, *branches_o;
    long long size, occupied;
    double lower, upper, tolerance, threshold;
    if (!PyArg_ParseTuple(args, "LOOOLOOddddOO", &size, &rows_o, &columns_o,
            &values_o, &occupied, &allowed_left_o, &allowed_right_o, &lower,
            &upper, &tolerance, &threshold, &history_o, &branches_o)) return nullptr;
    auto* rows = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        rows_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* columns = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        columns_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* values = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        values_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* allowed_left = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        allowed_left_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* allowed_right = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        allowed_right_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* old_history = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        history_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* branches = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        branches_o, NPY_INT8, NPY_ARRAY_IN_ARRAY));
    auto cleanup = [&]() {
        Py_XDECREF(rows); Py_XDECREF(columns); Py_XDECREF(values);
        Py_XDECREF(allowed_left); Py_XDECREF(allowed_right);
        Py_XDECREF(old_history); Py_XDECREF(branches);
    };
    if (!rows || !columns || !values || !allowed_left || !allowed_right
        || !old_history || !branches) {
        cleanup(); return nullptr;
    }
    const npy_intp allowed_count = PyArray_SIZE(allowed_left);
    const npy_intp branch_count = PyArray_SIZE(branches);
    if (PyArray_NDIM(old_history) != 2
        || PyArray_DIM(old_history, 0) != branch_count + 1
        || PyArray_DIM(old_history, 1) != allowed_count
        || PyArray_SIZE(allowed_right) != allowed_count) {
        cleanup();
        PyErr_SetString(PyExc_ValueError, "invalid sparse SP2 history dimensions");
        return nullptr;
    }
    const double width = upper - lower;
    if (!(width > std::numeric_limits<double>::epsilon())) {
        cleanup();
        PyErr_SetString(PyExc_ValueError, "invalid incremental SP2 spectral interval");
        return nullptr;
    }
    const auto* al = static_cast<const std::int64_t*>(PyArray_DATA(allowed_left));
    const auto* ar = static_cast<const std::int64_t*>(PyArray_DATA(allowed_right));
    const auto* old = static_cast<const double*>(PyArray_DATA(old_history));
    const auto* branch = static_cast<const std::int8_t*>(PyArray_DATA(branches));
    std::unordered_map<std::int64_t, double> fock_values;
    const auto* fr = static_cast<const std::int64_t*>(PyArray_DATA(rows));
    const auto* fc = static_cast<const std::int64_t*>(PyArray_DATA(columns));
    const auto* fv = static_cast<const double*>(PyArray_DATA(values));
    for (npy_intp index = 0; index < PyArray_SIZE(values); ++index)
        fock_values[fr[index] * size + fc[index]] = fv[index];
    std::unordered_map<std::int64_t, std::size_t> allowed_index;
    allowed_index.reserve(static_cast<std::size_t>(allowed_count) * 2 + 1);
    std::vector<std::size_t> diagonal_indices;
    for (npy_intp pair = 0; pair < allowed_count; ++pair) {
        allowed_index[al[pair] * size + ar[pair]] = static_cast<std::size_t>(pair);
        if (al[pair] == ar[pair]) diagonal_indices.push_back(pair);
    }
    npy_intp history_dims[2] = {branch_count + 1, allowed_count};
    auto* new_history = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(2, history_dims, NPY_DOUBLE, 0));
    if (!new_history) { cleanup(); return PyErr_NoMemory(); }
    auto* current = static_cast<double*>(PyArray_DATA(new_history));
    std::vector<double> delta(allowed_count, 0.0), next_delta(allowed_count, 0.0);
    double discarded_squared = 0.0;
    for (npy_intp pair = 0; pair < allowed_count; ++pair) {
        const auto found = fock_values.find(al[pair] * size + ar[pair]);
        const double fock_value = found == fock_values.end() ? 0.0 : found->second;
        double value = (al[pair] == ar[pair] ? upper / width : 0.0)
            - fock_value / width;
        delta[pair] = value - old[pair];
        if (al[pair] != ar[pair] && std::abs(delta[pair]) < threshold) {
            discarded_squared += 2.0 * delta[pair] * delta[pair];
            delta[pair] = 0.0;
        }
        current[pair] = old[pair] + delta[pair];
    }
    double bound = std::sqrt(discarded_squared);
    using Entry = std::pair<std::int64_t, double>;
    std::vector<double> cross(allowed_count, 0.0);
    std::vector<double> delta_square(allowed_count, 0.0);
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp step = 0; step < branch_count; ++step) {
        std::vector<std::vector<Entry>> x_rows(size), d_rows(size);
        const double* x = old + step * allowed_count;
        for (npy_intp pair = 0; pair < allowed_count; ++pair) {
            if (x[pair] != 0.0) {
                x_rows[al[pair]].push_back({ar[pair], x[pair]});
                if (al[pair] != ar[pair]) x_rows[ar[pair]].push_back({al[pair], x[pair]});
            }
            if (delta[pair] != 0.0) {
                d_rows[al[pair]].push_back({ar[pair], delta[pair]});
                if (al[pair] != ar[pair]) d_rows[ar[pair]].push_back({al[pair], delta[pair]});
            }
        }
        std::fill(cross.begin(), cross.end(), 0.0);
        std::fill(delta_square.begin(), delta_square.end(), 0.0);
        for (std::int64_t k = 0; k < size; ++k) {
            for (const auto& xe : x_rows[k]) for (const auto& de : d_rows[k]) {
                const auto high = std::max(xe.first, de.first);
                const auto low = std::min(xe.first, de.first);
                const auto key = high * size + low;
                const auto target = allowed_index.find(key);
                if (target != allowed_index.end())
                    cross[target->second] += xe.second * de.second;
            }
            for (const auto& left : d_rows[k]) for (const auto& right : d_rows[k]) {
                const auto high = std::max(left.first, right.first);
                const auto low = std::min(left.first, right.first);
                const auto key = high * size + low;
                const auto target = allowed_index.find(key);
                if (target != allowed_index.end())
                    delta_square[target->second] += left.second * right.second;
            }
        }
        discarded_squared = 0.0;
        double* next = current + (step + 1) * allowed_count;
        const double* old_next = old + (step + 1) * allowed_count;
        for (npy_intp pair = 0; pair < allowed_count; ++pair) {
            double anti = cross[pair];
            double square = delta_square[pair];
            if (al[pair] == ar[pair]) anti *= 2.0;
            else square *= 0.5;
            double value = branch[step] > 0
                ? anti + square : 2.0 * delta[pair] - anti - square;
            if (al[pair] != ar[pair] && std::abs(value) < threshold) {
                discarded_squared += 2.0 * value * value;
                value = 0.0;
            }
            next_delta[pair] = value;
            next[pair] = old_next[pair] + value;
        }
        bound += std::sqrt(discarded_squared);
        delta.swap(next_delta);
        std::fill(next_delta.begin(), next_delta.end(), 0.0);
    }
    Py_END_ALLOW_THREADS
    const double* projector = current + branch_count * allowed_count;
    SparseMatrixNative projector_sparse; projector_sparse.size = size;
    for (npy_intp pair = 0; pair < allowed_count; ++pair) {
        if (al[pair] == ar[pair] || std::abs(projector[pair]) >= threshold) {
            projector_sparse.rows.push_back(al[pair]);
            projector_sparse.columns.push_back(ar[pair]);
            projector_sparse.values.push_back(projector[pair]);
        }
    }
    double square_discarded = 0.0;
    auto square = sparse_square_native(projector_sparse, al, ar, allowed_count,
                                       0.0, square_discarded);
    std::unordered_map<std::int64_t, double> square_values;
    for (std::size_t index = 0; index < square.values.size(); ++index)
        square_values[square.rows[index] * size + square.columns[index]] = square.values[index];
    double error_squared = 0.0, trace = 0.0;
    for (npy_intp pair = 0; pair < allowed_count; ++pair) {
        const double difference = square_values[al[pair] * size + ar[pair]] - projector[pair];
        error_squared += (al[pair] == ar[pair] ? 1.0 : 2.0) * difference * difference;
        if (al[pair] == ar[pair]) trace += projector[pair];
    }
    const double error = std::sqrt(error_squared) + square_discarded;
    const double scale = size < 128 ? 1.0 : std::sqrt(static_cast<double>(size));
    const double trace_scale = size < 128 ? 1.0 : static_cast<double>(size);
    std::vector<std::vector<Entry>> projector_rows(size), fock_rows(size);
    for (std::size_t index = 0; index < projector_sparse.values.size(); ++index) {
        const auto row = projector_sparse.rows[index], column = projector_sparse.columns[index];
        const double value = projector_sparse.values[index];
        projector_rows[row].push_back({column, value});
        if (row != column) projector_rows[column].push_back({row, value});
    }
    for (npy_intp index = 0; index < PyArray_SIZE(values); ++index) {
        fock_rows[fr[index]].push_back({fc[index], fv[index]});
        if (fr[index] != fc[index]) fock_rows[fc[index]].push_back({fr[index], fv[index]});
    }
    for (auto* collection : {&projector_rows, &fock_rows})
        for (auto& row : *collection)
            std::sort(row.begin(), row.end(), [](const Entry& a, const Entry& b) {
                return a.first < b.first;
            });
    auto dot_rows = [](const std::vector<Entry>& left, const std::vector<Entry>& right) {
        std::size_t i = 0, j = 0; double value = 0.0;
        while (i < left.size() && j < right.size()) {
            if (left[i].first < right[j].first) ++i;
            else if (right[j].first < left[i].first) ++j;
            else { value += left[i].second * right[j].second; ++i; ++j; }
        }
        return value;
    };
    double commutator_squared = 0.0;
    for (npy_intp pair = 0; pair < allowed_count; ++pair) {
        if (al[pair] == ar[pair]) continue;
        const double pf_ij = dot_rows(projector_rows[al[pair]], fock_rows[ar[pair]]);
        const double pf_ji = dot_rows(projector_rows[ar[pair]], fock_rows[al[pair]]);
        const double residual = pf_ij - pf_ji;
        commutator_squared += 2.0 * residual * residual;
    }
    const double stationarity = std::sqrt(commutator_squared);
    const bool converged = error <= tolerance * scale
        && std::abs(trace - occupied) <= tolerance * trace_scale
        && stationarity <= 20.0 * tolerance * scale;
    PyObject* arrays = sparse_arrays(projector_sparse);
    cleanup();
    if (!arrays) { Py_DECREF(new_history); return nullptr; }
    PyObject *out_rows, *out_columns, *out_values;
    if (!PyArg_ParseTuple(arrays, "OOO", &out_rows, &out_columns, &out_values)) {
        Py_DECREF(arrays); Py_DECREF(new_history); return nullptr;
    }
    Py_INCREF(out_rows); Py_INCREF(out_columns); Py_INCREF(out_values); Py_DECREF(arrays);
    return Py_BuildValue("NNNddddN", out_rows, out_columns, out_values,
                         converged ? 1.0 : 0.0, bound, error, stationarity,
                         new_history);
}

bool parse_sparse_matrix(
    long long size, PyObject* rows_o, PyObject* columns_o, PyObject* values_o,
    SparseMatrixNative& matrix, PyArrayObject*& rows, PyArrayObject*& columns,
    PyArrayObject*& values) {
    rows = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(rows_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    columns = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(columns_o, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    values = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(values_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!rows || !columns || !values || PyArray_SIZE(rows) != PyArray_SIZE(values)
        || PyArray_SIZE(columns) != PyArray_SIZE(values)) return false;
    const npy_intp count = PyArray_SIZE(values);
    const auto* r = static_cast<const std::int32_t*>(PyArray_DATA(rows));
    const auto* c = static_cast<const std::int32_t*>(PyArray_DATA(columns));
    const auto* v = static_cast<const double*>(PyArray_DATA(values));
    matrix.size = size; matrix.rows.assign(r, r+count);
    matrix.columns.assign(c, c+count); matrix.values.assign(v, v+count);
    return true;
}

PyObject* sparse_scaled_add_py(PyObject*, PyObject* args) {
    long long size; PyObject *lr_o, *lc_o, *lv_o, *rr_o, *rc_o, *rv_o;
    double ls, rs, threshold;
    if (!PyArg_ParseTuple(args, "LOOOOOOddd", &size, &lr_o, &lc_o, &lv_o,
            &rr_o, &rc_o, &rv_o, &ls, &rs, &threshold)) return nullptr;
    SparseMatrixNative left, right; PyArrayObject *lr=nullptr,*lc=nullptr,*lv=nullptr,*rr=nullptr,*rc=nullptr,*rv=nullptr;
    if (!parse_sparse_matrix(size,lr_o,lc_o,lv_o,left,lr,lc,lv)
        || !parse_sparse_matrix(size,rr_o,rc_o,rv_o,right,rr,rc,rv)) {
        Py_XDECREF(lr);Py_XDECREF(lc);Py_XDECREF(lv);Py_XDECREF(rr);Py_XDECREF(rc);Py_XDECREF(rv);
        PyErr_SetString(PyExc_ValueError,"invalid sparse matrix arrays"); return nullptr;
    }
    double discarded=0.0; auto result=sparse_scaled_add_native(left,right,ls,rs,threshold,discarded);
    Py_DECREF(lr);Py_DECREF(lc);Py_DECREF(lv);Py_DECREF(rr);Py_DECREF(rc);Py_DECREF(rv);
    PyObject* arrays=sparse_arrays(result); if(!arrays)return nullptr;
    PyObject *ro,*co,*vo; PyArg_ParseTuple(arrays,"OOO",&ro,&co,&vo); Py_INCREF(ro);Py_INCREF(co);Py_INCREF(vo);Py_DECREF(arrays);
    return Py_BuildValue("NNNd",ro,co,vo,discarded);
}

PyObject* sparse_square_py(PyObject*, PyObject* args) {
    long long size; PyObject *rows_o,*columns_o,*values_o,*left_o,*right_o; double threshold;
    if(!PyArg_ParseTuple(args,"LOOOOOd",&size,&rows_o,&columns_o,&values_o,&left_o,&right_o,&threshold))return nullptr;
    SparseMatrixNative matrix; PyArrayObject *rows=nullptr,*columns=nullptr,*values=nullptr;
    auto* left=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(left_o,NPY_INT32,NPY_ARRAY_IN_ARRAY));
    auto* right=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(right_o,NPY_INT32,NPY_ARRAY_IN_ARRAY));
    if(!left||!right||!parse_sparse_matrix(size,rows_o,columns_o,values_o,matrix,rows,columns,values)){
        Py_XDECREF(left);Py_XDECREF(right);Py_XDECREF(rows);Py_XDECREF(columns);Py_XDECREF(values);
        PyErr_SetString(PyExc_ValueError,"invalid sparse square arrays");return nullptr;
    }
    double discarded=0.0; auto result=sparse_square_native(matrix,
        static_cast<const std::int32_t*>(PyArray_DATA(left)),static_cast<const std::int32_t*>(PyArray_DATA(right)),
        PyArray_SIZE(left),threshold,discarded);
    Py_DECREF(left);Py_DECREF(right);Py_DECREF(rows);Py_DECREF(columns);Py_DECREF(values);
    PyObject* arrays=sparse_arrays(result);if(!arrays)return nullptr;PyObject *ro,*co,*vo;
    PyArg_ParseTuple(arrays,"OOO",&ro,&co,&vo);Py_INCREF(ro);Py_INCREF(co);Py_INCREF(vo);Py_DECREF(arrays);
    return Py_BuildValue("NNNd",ro,co,vo,discarded);
}

PyObject* sparse_product_py(PyObject*, PyObject* args) {
    long long size;
    PyObject *lr_o,*lc_o,*lv_o,*rr_o,*rc_o,*rv_o,*left_o,*right_o;
    double threshold;
    int native_threads = 1;
    if(!PyArg_ParseTuple(args,"LOOOOOOOOd|i",&size,&lr_o,&lc_o,&lv_o,
            &rr_o,&rc_o,&rv_o,&left_o,&right_o,&threshold,&native_threads))return nullptr;
    SparseMatrixNative left_matrix,right_matrix;
    PyArrayObject *lr=nullptr,*lc=nullptr,*lv=nullptr,*rr=nullptr,*rc=nullptr,*rv=nullptr;
    auto* allowed_left=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(left_o,NPY_INT32,NPY_ARRAY_IN_ARRAY));
    auto* allowed_right=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(right_o,NPY_INT32,NPY_ARRAY_IN_ARRAY));
    if(!allowed_left||!allowed_right
        ||!parse_sparse_matrix(size,lr_o,lc_o,lv_o,left_matrix,lr,lc,lv)
        ||!parse_sparse_matrix(size,rr_o,rc_o,rv_o,right_matrix,rr,rc,rv)){
        Py_XDECREF(allowed_left);Py_XDECREF(allowed_right);
        Py_XDECREF(lr);Py_XDECREF(lc);Py_XDECREF(lv);
        Py_XDECREF(rr);Py_XDECREF(rc);Py_XDECREF(rv);
        PyErr_SetString(PyExc_ValueError,"invalid sparse product arrays");return nullptr;
    }
    double discarded=0.0;
    SparseMatrixNative result;
    Py_BEGIN_ALLOW_THREADS
    result=sparse_product_native(left_matrix,right_matrix,
        static_cast<const std::int32_t*>(PyArray_DATA(allowed_left)),
        static_cast<const std::int32_t*>(PyArray_DATA(allowed_right)),
        PyArray_SIZE(allowed_left),threshold,discarded,native_threads);
    Py_END_ALLOW_THREADS
    Py_DECREF(allowed_left);Py_DECREF(allowed_right);
    Py_DECREF(lr);Py_DECREF(lc);Py_DECREF(lv);Py_DECREF(rr);Py_DECREF(rc);Py_DECREF(rv);
    PyObject* arrays=sparse_arrays(result);if(!arrays)return nullptr;PyObject *ro,*co,*vo;
    PyArg_ParseTuple(arrays,"OOO",&ro,&co,&vo);Py_INCREF(ro);Py_INCREF(co);Py_INCREF(vo);Py_DECREF(arrays);
    return Py_BuildValue("NNNd",ro,co,vo,discarded);
}

PyObject* sparse_congruence_py(PyObject*, PyObject* args) {
    long long size;PyObject *xr_o,*xc_o,*xv_o,*ar_o,*ac_o,*av_o,*left_o,*right_o;double threshold;
    PyObject *topology_offsets_o=nullptr,*topology_neighbors_o=nullptr,*topology_pairs_o=nullptr;
    int native_threads=1;
    if(!PyArg_ParseTuple(args,"LOOOOOOOOd|OOOi",&size,&xr_o,&xc_o,&xv_o,&ar_o,&ac_o,&av_o,
            &left_o,&right_o,&threshold,&topology_offsets_o,&topology_neighbors_o,
            &topology_pairs_o,&native_threads))return nullptr;
    SparseMatrixNative x,a;PyArrayObject *xr=nullptr,*xc=nullptr,*xv=nullptr,*ar=nullptr,*ac=nullptr,*av=nullptr;
    auto* left=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(left_o,NPY_INT32,NPY_ARRAY_IN_ARRAY));
    auto* right=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(right_o,NPY_INT32,NPY_ARRAY_IN_ARRAY));
    if(!left||!right||PyArray_SIZE(left)!=PyArray_SIZE(right)
        ||!parse_sparse_matrix(size,xr_o,xc_o,xv_o,x,xr,xc,xv)
        ||!parse_sparse_matrix(size,ar_o,ac_o,av_o,a,ar,ac,av)){
        Py_XDECREF(left);Py_XDECREF(right);Py_XDECREF(xr);Py_XDECREF(xc);Py_XDECREF(xv);
        Py_XDECREF(ar);Py_XDECREF(ac);Py_XDECREF(av);
        PyErr_SetString(PyExc_ValueError,"invalid sparse congruence arrays");return nullptr;
    }
    auto* topology_offsets = topology_offsets_o
        ? reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            topology_offsets_o, NPY_INT64, NPY_ARRAY_IN_ARRAY)) : nullptr;
    auto* topology_neighbors = topology_neighbors_o
        ? reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            topology_neighbors_o, NPY_INT32, NPY_ARRAY_IN_ARRAY)) : nullptr;
    auto* topology_pairs = topology_pairs_o
        ? reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            topology_pairs_o, NPY_INT32, NPY_ARRAY_IN_ARRAY)) : nullptr;
    double discarded=0.0;SparseMatrixNative result;
    Py_BEGIN_ALLOW_THREADS
    result=sparse_congruence_native(x,a,
        static_cast<const std::int32_t*>(PyArray_DATA(left)),
        static_cast<const std::int32_t*>(PyArray_DATA(right)),PyArray_SIZE(left),threshold,discarded,
        topology_offsets ? static_cast<const std::int64_t*>(PyArray_DATA(topology_offsets)) : nullptr,
        topology_neighbors ? static_cast<const std::int32_t*>(PyArray_DATA(topology_neighbors)) : nullptr,
        topology_pairs ? static_cast<const std::int32_t*>(PyArray_DATA(topology_pairs)) : nullptr,
        native_threads);
    Py_END_ALLOW_THREADS
    Py_DECREF(left);Py_DECREF(right);Py_DECREF(xr);Py_DECREF(xc);Py_DECREF(xv);Py_DECREF(ar);Py_DECREF(ac);Py_DECREF(av);
    Py_XDECREF(topology_offsets);Py_XDECREF(topology_neighbors);Py_XDECREF(topology_pairs);
    PyObject* arrays=sparse_arrays(result);if(!arrays)return nullptr;PyObject *ro,*co,*vo;
    PyArg_ParseTuple(arrays,"OOO",&ro,&co,&vo);Py_INCREF(ro);Py_INCREF(co);Py_INCREF(vo);Py_DECREF(arrays);
    return Py_BuildValue("NNNd",ro,co,vo,discarded);
}

PyObject* sparse_orthogonal_observables_native(PyObject*, PyObject* args) {
    long long size, nshells, natoms;
    PyObject *xr_o,*xc_o,*xv_o,*pr_o,*pc_o,*pv_o,*left_o,*right_o;
    PyObject *orbital_shell_o,*reference_o,*valence_o,*op_rows_o,*op_cols_o,*atom_left_o,*atom_right_o;
    PyObject *overlap_o,*dmu_left_o,*dmu_right_o,*theta_left_o,*theta_right_o;
    int native_threads = 1;
    if (!PyArg_ParseTuple(args, "LLLOOOOOOOOOOOOOOOOOOOO|i", &size, &nshells, &natoms,
            &xr_o,&xc_o,&xv_o,&pr_o,&pc_o,&pv_o,&left_o,&right_o,
            &orbital_shell_o,&reference_o,&valence_o,&op_rows_o,&op_cols_o,&atom_left_o,
            &atom_right_o,&overlap_o,&dmu_left_o,&dmu_right_o,
            &theta_left_o,&theta_right_o,&native_threads)) return nullptr;
    SparseMatrixNative x, projector;
    PyArrayObject *xr=nullptr,*xc=nullptr,*xv=nullptr,*pr=nullptr,*pc=nullptr,*pv=nullptr;
    if (!parse_sparse_matrix(size, xr_o, xc_o, xv_o, x, xr, xc, xv)
        || !parse_sparse_matrix(size, pr_o, pc_o, pv_o, projector, pr, pc, pv)) {
        Py_XDECREF(xr);Py_XDECREF(xc);Py_XDECREF(xv);Py_XDECREF(pr);Py_XDECREF(pc);Py_XDECREF(pv);
        PyErr_SetString(PyExc_ValueError, "invalid sparse orthogonal observable matrices");
        return nullptr;
    }
    auto as_i64 = [](PyObject* object) { return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object, NPY_INT64, NPY_ARRAY_IN_ARRAY)); };
    auto as_i32 = [](PyObject* object) { return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object, NPY_INT32, NPY_ARRAY_IN_ARRAY)); };
    auto as_f64 = [](PyObject* object) { return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)); };
    auto* allowed_left=as_i32(left_o); auto* allowed_right=as_i32(right_o);
    auto* orbital_shell=as_i64(orbital_shell_o); auto* reference=as_f64(reference_o);
    auto* valence=as_f64(valence_o); auto* op_rows=as_i64(op_rows_o); auto* op_cols=as_i64(op_cols_o);
    auto* atom_left=as_i64(atom_left_o); auto* atom_right=as_i64(atom_right_o);
    auto* overlap=as_f64(overlap_o); auto* dmu_left=as_f64(dmu_left_o);
    auto* dmu_right=as_f64(dmu_right_o); auto* theta_left=as_f64(theta_left_o);
    auto* theta_right=as_f64(theta_right_o);
    const bool compact = dmu_left && dmu_right && theta_left && theta_right
        && PyArray_NDIM(dmu_left)==2 && PyArray_DIM(dmu_left,1)==3
        && PyArray_NDIM(dmu_right)==3 && PyArray_DIM(dmu_right,1)==3
        && PyArray_DIM(dmu_right,2)==3
        && PyArray_NDIM(theta_left)==2 && PyArray_DIM(theta_left,1)==3
        && PyArray_NDIM(theta_right)==2 && PyArray_DIM(theta_right,1)==3;
    const bool expanded = dmu_left && dmu_right && theta_left && theta_right
        && PyArray_SIZE(dmu_left)==3*PyArray_SIZE(op_rows)
        && PyArray_SIZE(dmu_right)==3*PyArray_SIZE(op_rows)
        && PyArray_SIZE(theta_left)==9*PyArray_SIZE(op_rows)
        && PyArray_SIZE(theta_right)==9*PyArray_SIZE(op_rows);
    const bool valid = allowed_left && allowed_right && orbital_shell && reference && valence
        && op_rows && op_cols && atom_left && atom_right && overlap
        && dmu_left && dmu_right && theta_left && theta_right
        && PyArray_SIZE(allowed_left)==PyArray_SIZE(allowed_right)
        && PyArray_SIZE(op_rows)==PyArray_SIZE(op_cols)
        && PyArray_SIZE(op_rows)==PyArray_SIZE(atom_left)
        && PyArray_SIZE(op_rows)==PyArray_SIZE(atom_right)
        && PyArray_SIZE(op_rows)==PyArray_SIZE(overlap)
        && (compact || expanded)
        && PyArray_SIZE(orbital_shell)==size && PyArray_SIZE(reference)==nshells
        && PyArray_SIZE(valence)==natoms;
    if (!valid) {
        Py_XDECREF(xr);Py_XDECREF(xc);Py_XDECREF(xv);Py_XDECREF(pr);Py_XDECREF(pc);Py_XDECREF(pv);
        Py_XDECREF(allowed_left);Py_XDECREF(allowed_right);Py_XDECREF(orbital_shell);Py_XDECREF(reference);Py_XDECREF(valence);
        Py_XDECREF(op_rows);Py_XDECREF(op_cols);Py_XDECREF(atom_left);Py_XDECREF(atom_right);
        Py_XDECREF(overlap);Py_XDECREF(dmu_left);Py_XDECREF(dmu_right);Py_XDECREF(theta_left);Py_XDECREF(theta_right);
        PyErr_SetString(PyExc_ValueError, "invalid sparse orthogonal observable arrays");
        return nullptr;
    }
    double discarded = 0.0;
    SparseMatrixNative density;
    density = sparse_congruence_native(
        x, projector,
        static_cast<const std::int32_t*>(PyArray_DATA(allowed_left)),
        static_cast<const std::int32_t*>(PyArray_DATA(allowed_right)),
        PyArray_SIZE(allowed_left), 0.0, discarded,
        nullptr, nullptr, nullptr, native_threads);
    std::unordered_map<std::int64_t, double> values;
    values.reserve(density.values.size()*2+1);
    for (std::size_t index=0; index<density.values.size(); ++index)
        values[density.rows[index]*size+density.columns[index]] =
            2.0*density.values[index];
    npy_intp shell_dim[1]={nshells}, atom_charge_dim[1]={natoms}, atom_dim[2]={natoms,3}, quad_dim[3]={natoms,3,3};
    auto* shell_q=reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1,shell_dim,NPY_DOUBLE,0));
    auto* charges=reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1,atom_charge_dim,NPY_DOUBLE,0));
    auto* dipoles=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2,atom_dim,NPY_DOUBLE,0));
    auto* quadrupoles=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(3,quad_dim,NPY_DOUBLE,0));
    if(!shell_q||!charges||!dipoles||!quadrupoles){Py_XDECREF(shell_q);Py_XDECREF(charges);Py_XDECREF(dipoles);Py_XDECREF(quadrupoles);return PyErr_NoMemory();}
    auto* shell_values=static_cast<double*>(PyArray_DATA(shell_q));
    const auto* ref=static_cast<const double*>(PyArray_DATA(reference));
    const auto* nuc=static_cast<const double*>(PyArray_DATA(valence));
    auto* charge_values=static_cast<double*>(PyArray_DATA(charges));
    std::copy(nuc,nuc+natoms,charge_values);
    std::copy(ref,ref+nshells,shell_values);
    auto* dip=static_cast<double*>(PyArray_DATA(dipoles));
    auto* quad=static_cast<double*>(PyArray_DATA(quadrupoles));
    const auto* os=static_cast<const std::int64_t*>(PyArray_DATA(orbital_shell));
    const auto* ol=static_cast<const std::int64_t*>(PyArray_DATA(op_rows));
    const auto* oc=static_cast<const std::int64_t*>(PyArray_DATA(op_cols));
    const auto* al=static_cast<const std::int64_t*>(PyArray_DATA(atom_left));
    const auto* ar=static_cast<const std::int64_t*>(PyArray_DATA(atom_right));
    const auto* ov=static_cast<const double*>(PyArray_DATA(overlap));
    const auto* dl=static_cast<const double*>(PyArray_DATA(dmu_left));
    const auto* dr=static_cast<const double*>(PyArray_DATA(dmu_right));
    const auto* tl=static_cast<const double*>(PyArray_DATA(theta_left));
    const auto* tr=static_cast<const double*>(PyArray_DATA(theta_right));
    auto raw_endpoint=[&](const npy_intp pair,const double* center,const int i,const int j){
        return dl[pair*3+i]*center[j]+center[i]*dl[pair*3+j]
            -ov[pair]*center[i]*center[j]-dr[(pair*3+i)*3+j];
    };
    for(npy_intp pair=0; pair<PyArray_SIZE(op_rows); ++pair){
        const auto found=values.find(ol[pair]*size+oc[pair]);
        const double value=found==values.end()?0.0:found->second;
        shell_values[os[ol[pair]]] -= value*ov[pair];
        charge_values[al[pair]] -= value*ov[pair];
        for(int component=0;component<3;++component){
            const double dmul=compact?tl[pair*3+component]*ov[pair]-dl[pair*3+component]
                :dl[pair*3+component];
            const double thetal=compact?raw_endpoint(pair,tl+pair*3,component,component)
                :tl[pair*9+component*3+component];
            dip[al[pair]*3+component]+=value*dmul;
            quad[(al[pair]*3+component)*3+component]+=value*thetal;
        }
        if(ol[pair]!=oc[pair]){
            shell_values[os[oc[pair]]] -= value*ov[pair];
            charge_values[ar[pair]] -= value*ov[pair];
            for(int component=0;component<3;++component){
                const double dmur=compact?tr[pair*3+component]*ov[pair]-dl[pair*3+component]
                    :dr[pair*3+component];
                const double thetar=compact?raw_endpoint(pair,tr+pair*3,component,component)
                    :tr[pair*9+component*3+component];
                dip[ar[pair]*3+component]+=value*dmur;
                quad[(ar[pair]*3+component)*3+component]+=value*thetar;
            }
        }
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){
            if(i==j) continue;
            const double thetal=compact?raw_endpoint(pair,tl+pair*3,i,j):tl[pair*9+i*3+j];
            quad[(al[pair]*3+i)*3+j]+=value*thetal;
            if(ol[pair]!=oc[pair]){
                const double thetar=compact?raw_endpoint(pair,tr+pair*3,i,j):tr[pair*9+i*3+j];
                quad[(ar[pair]*3+i)*3+j]+=value*thetar;
            }
        }
    }
    for (long long atom=0; atom<natoms; ++atom) {
        const double trace = quad[(atom*3)*3 + 0]
            + quad[(atom*3+1)*3 + 1] + quad[(atom*3+2)*3 + 2];
        for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) {
            const double raw = quad[(atom*3+i)*3+j];
            quad[(atom*3+i)*3+j] = 1.5*raw
                - 0.5*trace*(i==j ? 1.0 : 0.0);
        }
    }
    Py_DECREF(xr);Py_DECREF(xc);Py_DECREF(xv);Py_DECREF(pr);Py_DECREF(pc);Py_DECREF(pv);
    Py_DECREF(allowed_left);Py_DECREF(allowed_right);Py_DECREF(orbital_shell);Py_DECREF(reference);Py_DECREF(valence);
    Py_DECREF(op_rows);Py_DECREF(op_cols);Py_DECREF(atom_left);Py_DECREF(atom_right);
    Py_DECREF(overlap);Py_DECREF(dmu_left);Py_DECREF(dmu_right);Py_DECREF(theta_left);Py_DECREF(theta_right);
    return Py_BuildValue("NNNNd",shell_q,charges,dipoles,quadrupoles,discarded);
}

PyObject* sparse_multipole_congruence_native(PyObject*, PyObject* args) {
    PyObject *rows_o,*columns_o,*values_o,*dl_o,*dr_o;
    PyObject *offsets_o,*cartesian_o,*coefficients_o,*target_left_o,*target_right_o;
    long long cartesian_size;
    if(!PyArg_ParseTuple(args,"LOOOOOOOOOO",&cartesian_size,&rows_o,&columns_o,
            &values_o,&dl_o,&dr_o,&offsets_o,&cartesian_o,&coefficients_o,
            &target_left_o,&target_right_o))return nullptr;
    auto* rows=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(rows_o,NPY_INT64,NPY_ARRAY_IN_ARRAY));
    auto* columns=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(columns_o,NPY_INT64,NPY_ARRAY_IN_ARRAY));
    auto* values=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(values_o,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
    auto* dl=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(dl_o,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
    auto* dr=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(dr_o,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
    auto* offsets=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(offsets_o,NPY_INT64,NPY_ARRAY_IN_ARRAY));
    auto* cartesian=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(cartesian_o,NPY_INT64,NPY_ARRAY_IN_ARRAY));
    auto* coefficients=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(coefficients_o,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
    auto* target_left=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(target_left_o,NPY_INT64,NPY_ARRAY_IN_ARRAY));
    auto* target_right=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(target_right_o,NPY_INT64,NPY_ARRAY_IN_ARRAY));
    const bool valid=rows&&columns&&values&&dl&&dr&&offsets&&cartesian&&coefficients
        &&target_left&&target_right&&PyArray_NDIM(values)==2&&PyArray_DIM(values,1)==13
        &&PyArray_NDIM(dl)==3&&PyArray_DIM(dl,1)==13&&PyArray_DIM(dl,2)==3
        &&PyArray_NDIM(dr)==3&&PyArray_DIM(dr,1)==13&&PyArray_DIM(dr,2)==3
        &&PyArray_SIZE(rows)==PyArray_DIM(values,0)&&PyArray_SIZE(columns)==PyArray_DIM(values,0)
        &&PyArray_DIM(dl,0)==PyArray_DIM(values,0)&&PyArray_DIM(dr,0)==PyArray_DIM(values,0)
        &&PyArray_SIZE(cartesian)==PyArray_SIZE(coefficients)
        &&PyArray_SIZE(target_left)==PyArray_SIZE(target_right);
    if(!valid){
        Py_XDECREF(rows);Py_XDECREF(columns);Py_XDECREF(values);Py_XDECREF(dl);Py_XDECREF(dr);
        Py_XDECREF(offsets);Py_XDECREF(cartesian);Py_XDECREF(coefficients);
        Py_XDECREF(target_left);Py_XDECREF(target_right);
        PyErr_SetString(PyExc_ValueError,"invalid sparse multipole congruence arrays");return nullptr;
    }
    const npy_intp pair_count=PyArray_DIM(values,0), target_count=PyArray_SIZE(target_left);
    npy_intp value_dims[2]={target_count,13};
    npy_intp derivative_dims[3]={target_count,13,3};
    auto* output_values=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2,value_dims,NPY_DOUBLE,0));
    auto* output_dl=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(3,derivative_dims,NPY_DOUBLE,0));
    auto* output_dr=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(3,derivative_dims,NPY_DOUBLE,0));
    if(!output_values||!output_dl||!output_dr){
        Py_XDECREF(output_values);Py_XDECREF(output_dl);Py_XDECREF(output_dr);
        Py_DECREF(rows);Py_DECREF(columns);Py_DECREF(values);Py_DECREF(dl);Py_DECREF(dr);
        Py_DECREF(offsets);Py_DECREF(cartesian);Py_DECREF(coefficients);Py_DECREF(target_left);Py_DECREF(target_right);
        return PyErr_NoMemory();
    }
    const auto* row_data=static_cast<const std::int64_t*>(PyArray_DATA(rows));
    const auto* column_data=static_cast<const std::int64_t*>(PyArray_DATA(columns));
    const auto* value_data=static_cast<const double*>(PyArray_DATA(values));
    const auto* dl_data=static_cast<const double*>(PyArray_DATA(dl));
    const auto* dr_data=static_cast<const double*>(PyArray_DATA(dr));
    const auto* offset_data=static_cast<const std::int64_t*>(PyArray_DATA(offsets));
    const auto* cartesian_data=static_cast<const std::int64_t*>(PyArray_DATA(cartesian));
    const auto* coefficient_data=static_cast<const double*>(PyArray_DATA(coefficients));
    const auto* left_data=static_cast<const std::int64_t*>(PyArray_DATA(target_left));
    const auto* right_data=static_cast<const std::int64_t*>(PyArray_DATA(target_right));
    auto* output_value_data=static_cast<double*>(PyArray_DATA(output_values));
    auto* output_dl_data=static_cast<double*>(PyArray_DATA(output_dl));
    auto* output_dr_data=static_cast<double*>(PyArray_DATA(output_dr));
    std::unordered_map<std::int64_t,npy_intp> pair_lookup;
    pair_lookup.reserve(static_cast<std::size_t>(pair_count*1.4)+1);
    for(npy_intp pair=0;pair<pair_count;++pair){
        pair_lookup[row_data[pair]*cartesian_size+column_data[pair]]=pair;
    }
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp target=0;target<target_count;++target){
        const auto left=left_data[target],right=right_data[target];
        for(auto li=offset_data[left];li<offset_data[left+1];++li){
            const auto a=cartesian_data[li];
            for(auto ri=offset_data[right];ri<offset_data[right+1];++ri){
                const auto b=cartesian_data[ri];
                const auto high=std::max(a,b),low=std::min(a,b);
                const auto found=pair_lookup.find(high*cartesian_size+low);
                if(found==pair_lookup.end())continue;
                const npy_intp source=found->second;
                const double weight=coefficient_data[li]*coefficient_data[ri];
                for(int component=0;component<13;++component){
                    output_value_data[target*13+component]+=weight*value_data[source*13+component];
                    for(int axis=0;axis<3;++axis){
                        const auto destination=(target*13+component)*3+axis;
                        const auto source_index=(source*13+component)*3+axis;
                        if(a>=b){
                            output_dl_data[destination]+=weight*dl_data[source_index];
                            output_dr_data[destination]+=weight*dr_data[source_index];
                        }else{
                            output_dl_data[destination]+=weight*dr_data[source_index];
                            output_dr_data[destination]+=weight*dl_data[source_index];
                        }
                    }
                }
            }
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(rows);Py_DECREF(columns);Py_DECREF(values);Py_DECREF(dl);Py_DECREF(dr);
    Py_DECREF(offsets);Py_DECREF(cartesian);Py_DECREF(coefficients);Py_DECREF(target_left);Py_DECREF(target_right);
    return Py_BuildValue("NNN",output_values,output_dl,output_dr);
}

PyObject* anisotropic_fock_compact_native(PyObject*, PyObject* args) {
    PyObject *rows_o,*columns_o,*left_atom_o,*right_atom_o,*overlap_o;
    PyObject *dipole_o,*second_o,*left_center_o,*right_center_o;
    PyObject *charge_o,*dipole_potential_o,*quadrupole_o;
    if(!PyArg_ParseTuple(args,"OOOOOOOOOOOO",&rows_o,&columns_o,&left_atom_o,
            &right_atom_o,&overlap_o,&dipole_o,&second_o,&left_center_o,
            &right_center_o,&charge_o,&dipole_potential_o,&quadrupole_o))return nullptr;
    auto array=[](PyObject* object,int type){return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object,type,NPY_ARRAY_IN_ARRAY));};
    auto* rows=array(rows_o,NPY_INT64);auto* columns=array(columns_o,NPY_INT64);
    auto* left_atom=array(left_atom_o,NPY_INT64);auto* right_atom=array(right_atom_o,NPY_INT64);
    auto* overlap=array(overlap_o,NPY_DOUBLE);auto* dipole=array(dipole_o,NPY_DOUBLE);
    auto* second=array(second_o,NPY_DOUBLE);auto* left_center=array(left_center_o,NPY_DOUBLE);
    auto* right_center=array(right_center_o,NPY_DOUBLE);auto* charge=array(charge_o,NPY_DOUBLE);
    auto* dipole_potential=array(dipole_potential_o,NPY_DOUBLE);
    auto* quadrupole=array(quadrupole_o,NPY_DOUBLE);
    const npy_intp pairs=overlap?PyArray_SIZE(overlap):0;
    const npy_intp atoms=charge?PyArray_SIZE(charge):0;
    const bool valid=rows&&columns&&left_atom&&right_atom&&overlap&&dipole&&second
        &&left_center&&right_center&&charge&&dipole_potential&&quadrupole
        &&PyArray_SIZE(rows)==pairs&&PyArray_SIZE(columns)==pairs
        &&PyArray_SIZE(left_atom)==pairs&&PyArray_SIZE(right_atom)==pairs
        &&PyArray_NDIM(dipole)==2&&PyArray_DIM(dipole,0)==pairs&&PyArray_DIM(dipole,1)==3
        &&PyArray_NDIM(second)==3&&PyArray_DIM(second,0)==pairs
        &&PyArray_DIM(second,1)==3&&PyArray_DIM(second,2)==3
        &&PyArray_NDIM(left_center)==2&&PyArray_DIM(left_center,0)==pairs&&PyArray_DIM(left_center,1)==3
        &&PyArray_NDIM(right_center)==2&&PyArray_DIM(right_center,0)==pairs&&PyArray_DIM(right_center,1)==3
        &&PyArray_NDIM(dipole_potential)==2&&PyArray_DIM(dipole_potential,0)==atoms
        &&PyArray_DIM(dipole_potential,1)==3
        &&PyArray_NDIM(quadrupole)==3&&PyArray_DIM(quadrupole,0)==atoms
        &&PyArray_DIM(quadrupole,1)==3&&PyArray_DIM(quadrupole,2)==3;
    if(!valid){
        Py_XDECREF(rows);Py_XDECREF(columns);Py_XDECREF(left_atom);Py_XDECREF(right_atom);
        Py_XDECREF(overlap);Py_XDECREF(dipole);Py_XDECREF(second);
        Py_XDECREF(left_center);Py_XDECREF(right_center);Py_XDECREF(charge);
        Py_XDECREF(dipole_potential);Py_XDECREF(quadrupole);
        PyErr_SetString(PyExc_ValueError,"invalid compact anisotropic Fock arrays");return nullptr;
    }
    npy_intp dims[1]={pairs};
    auto* output=reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1,dims,NPY_DOUBLE,0));
    if(!output)return PyErr_NoMemory();
    const auto* row=static_cast<const std::int64_t*>(PyArray_DATA(rows));
    const auto* column=static_cast<const std::int64_t*>(PyArray_DATA(columns));
    const auto* al=static_cast<const std::int64_t*>(PyArray_DATA(left_atom));
    const auto* ar=static_cast<const std::int64_t*>(PyArray_DATA(right_atom));
    const auto* s=static_cast<const double*>(PyArray_DATA(overlap));
    const auto* mu=static_cast<const double*>(PyArray_DATA(dipole));
    const auto* m2=static_cast<const double*>(PyArray_DATA(second));
    const auto* cl=static_cast<const double*>(PyArray_DATA(left_center));
    const auto* cr=static_cast<const double*>(PyArray_DATA(right_center));
    const auto* q=static_cast<const double*>(PyArray_DATA(charge));
    const auto* mup=static_cast<const double*>(PyArray_DATA(dipole_potential));
    const auto* qp=static_cast<const double*>(PyArray_DATA(quadrupole));
    auto* out=static_cast<double*>(PyArray_DATA(output));
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp pair=0;pair<pairs;++pair){
        auto endpoint=[&](const std::int64_t atom,const double* center){
            double value=-q[atom]*s[pair];
            double raw[9],trace=0.0;
            for(int i=0;i<3;++i){
                const double dmu=center[i]*s[pair]-mu[pair*3+i];
                value+=mup[atom*3+i]*dmu;
                for(int j=0;j<3;++j){
                    const double theta=mu[pair*3+i]*center[j]
                        +center[i]*mu[pair*3+j]-s[pair]*center[i]*center[j]
                        -m2[(pair*3+i)*3+j];
                    raw[i*3+j]=theta;if(i==j)trace+=theta;
                }
            }
            for(int i=0;i<3;++i)for(int j=0;j<3;++j){
                const double traceless=1.5*raw[i*3+j]-0.5*trace*(i==j?1.0:0.0);
                value+=qp[(atom*3+i)*3+j]*traceless;
            }
            return value;
        };
        const double left=endpoint(al[pair],cl+pair*3);
        const double right=endpoint(ar[pair],cr+pair*3);
        out[pair]=row[pair]==column[pair]?left:0.5*(left+right);
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(rows);Py_DECREF(columns);Py_DECREF(left_atom);Py_DECREF(right_atom);
    Py_DECREF(overlap);Py_DECREF(dipole);Py_DECREF(second);Py_DECREF(left_center);
    Py_DECREF(right_center);Py_DECREF(charge);Py_DECREF(dipole_potential);Py_DECREF(quadrupole);
    return reinterpret_cast<PyObject*>(output);
}

PyObject* anisotropic_fock_derivatives_native(PyObject*, PyObject* args) {
    PyObject *overlap_o,*dipole_o,*dl_o,*dr_o,*left_atom_o,*right_atom_o;
    PyObject *coordinates_o,*charge_o,*dipole_potential_o,*quadrupole_o;
    if(!PyArg_ParseTuple(args,"OOOOOOOOOO",&overlap_o,&dipole_o,&dl_o,&dr_o,
            &left_atom_o,&right_atom_o,&coordinates_o,&charge_o,
            &dipole_potential_o,&quadrupole_o))return nullptr;
    auto array=[](PyObject* object,int type){
        return reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(object,type,NPY_ARRAY_IN_ARRAY));
    };
    auto* overlap=array(overlap_o,NPY_DOUBLE);auto* dipole=array(dipole_o,NPY_DOUBLE);
    auto* dl=array(dl_o,NPY_DOUBLE);auto* dr=array(dr_o,NPY_DOUBLE);
    auto* left_atom=array(left_atom_o,NPY_INT64);auto* right_atom=array(right_atom_o,NPY_INT64);
    auto* coordinates=array(coordinates_o,NPY_DOUBLE);auto* charge=array(charge_o,NPY_DOUBLE);
    auto* dipole_potential=array(dipole_potential_o,NPY_DOUBLE);
    auto* quadrupole=array(quadrupole_o,NPY_DOUBLE);
    const npy_intp pairs=overlap?PyArray_SIZE(overlap):0;
    const npy_intp atoms=coordinates&&PyArray_NDIM(coordinates)==2?PyArray_DIM(coordinates,0):0;
    const bool valid=overlap&&dipole&&dl&&dr&&left_atom&&right_atom&&coordinates
        &&charge&&dipole_potential&&quadrupole
        &&PyArray_NDIM(dipole)==2&&PyArray_DIM(dipole,0)==pairs&&PyArray_DIM(dipole,1)==3
        &&PyArray_NDIM(dl)==3&&PyArray_DIM(dl,0)==pairs&&PyArray_DIM(dl,1)==13&&PyArray_DIM(dl,2)==3
        &&PyArray_NDIM(dr)==3&&PyArray_DIM(dr,0)==pairs&&PyArray_DIM(dr,1)==13&&PyArray_DIM(dr,2)==3
        &&PyArray_SIZE(left_atom)==pairs&&PyArray_SIZE(right_atom)==pairs
        &&PyArray_NDIM(coordinates)==2&&PyArray_DIM(coordinates,1)==3
        &&PyArray_SIZE(charge)==atoms
        &&PyArray_NDIM(dipole_potential)==2&&PyArray_DIM(dipole_potential,0)==atoms&&PyArray_DIM(dipole_potential,1)==3
        &&PyArray_NDIM(quadrupole)==3&&PyArray_DIM(quadrupole,0)==atoms&&PyArray_DIM(quadrupole,1)==3&&PyArray_DIM(quadrupole,2)==3;
    if(!valid){
        Py_XDECREF(overlap);Py_XDECREF(dipole);Py_XDECREF(dl);Py_XDECREF(dr);
        Py_XDECREF(left_atom);Py_XDECREF(right_atom);Py_XDECREF(coordinates);
        Py_XDECREF(charge);Py_XDECREF(dipole_potential);Py_XDECREF(quadrupole);
        PyErr_SetString(PyExc_ValueError,"invalid anisotropic derivative arrays");return nullptr;
    }
    npy_intp dims[2]={pairs,3};
    auto* output_left=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2,dims,NPY_DOUBLE,0));
    auto* output_right=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2,dims,NPY_DOUBLE,0));
    if(!output_left||!output_right){Py_XDECREF(output_left);Py_XDECREF(output_right);return PyErr_NoMemory();}
    const auto* s=static_cast<const double*>(PyArray_DATA(overlap));
    const auto* mu=static_cast<const double*>(PyArray_DATA(dipole));
    const double* deriv[2]={static_cast<const double*>(PyArray_DATA(dl)),static_cast<const double*>(PyArray_DATA(dr))};
    const std::int64_t* atom_index[2]={static_cast<const std::int64_t*>(PyArray_DATA(left_atom)),static_cast<const std::int64_t*>(PyArray_DATA(right_atom))};
    const auto* xyz=static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* qpot=static_cast<const double*>(PyArray_DATA(charge));
    const auto* mupot=static_cast<const double*>(PyArray_DATA(dipole_potential));
    const auto* thetapot=static_cast<const double*>(PyArray_DATA(quadrupole));
    double* outputs[2]={static_cast<double*>(PyArray_DATA(output_left)),static_cast<double*>(PyArray_DATA(output_right))};
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp pair=0;pair<pairs;++pair){
        const auto endpoint_left=atom_index[0][pair],endpoint_right=atom_index[1][pair];
        for(int derivative_endpoint=0;derivative_endpoint<2;++derivative_endpoint){
            for(int allocation_endpoint=0;allocation_endpoint<2;++allocation_endpoint){
                const auto atom=atom_index[allocation_endpoint][pair];
                const int occurrences=(atom==endpoint_left)+(atom==endpoint_right);
                const double shift_weight=(atom==atom_index[derivative_endpoint][pair])
                    ?1.0/static_cast<double>(occurrences):0.0;
                const auto* d=deriv[derivative_endpoint]+pair*39;
                for(int axis=0;axis<3;++axis){
                    const double ds=d[axis];
                    double value=-qpot[atom]*ds;
                    double trace=0.0,draw[9];
                    for(int i=0;i<3;++i){
                        const double delta_i=(i==axis)?shift_weight:0.0;
                        const double dd_i=d[(1+i)*3+axis];
                        const double dmu=delta_i*s[pair]+xyz[atom*3+i]*ds-dd_i;
                        value+=mupot[atom*3+i]*dmu;
                        for(int j=0;j<3;++j){
                            const double delta_j=(j==axis)?shift_weight:0.0;
                            const double dd_j=d[(1+j)*3+axis];
                            const double item=dd_i*xyz[atom*3+j]+mu[pair*3+i]*delta_j
                                +delta_i*mu[pair*3+j]+xyz[atom*3+i]*dd_j
                                -ds*xyz[atom*3+i]*xyz[atom*3+j]
                                -s[pair]*(delta_i*xyz[atom*3+j]+xyz[atom*3+i]*delta_j)
                                -d[(4+3*i+j)*3+axis];
                            draw[3*i+j]=item;if(i==j)trace+=item;
                        }
                    }
                    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
                        const double dq=1.5*draw[3*i+j]-(i==j?0.5*trace:0.0);
                        value+=thetapot[(atom*3+i)*3+j]*dq;
                    }
                    outputs[derivative_endpoint][pair*3+axis]+=0.5*value;
                }
            }
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(overlap);Py_DECREF(dipole);Py_DECREF(dl);Py_DECREF(dr);
    Py_DECREF(left_atom);Py_DECREF(right_atom);Py_DECREF(coordinates);
    Py_DECREF(charge);Py_DECREF(dipole_potential);Py_DECREF(quadrupole);
    return Py_BuildValue("NN",output_left,output_right);
}

struct DirectionalDual {
    double value;
    double direction;
    DirectionalDual(double v = 0.0, double d = 0.0) : value(v), direction(d) {}
};
inline DirectionalDual operator+(DirectionalDual a, DirectionalDual b) {
    return {a.value+b.value, a.direction+b.direction};
}
inline DirectionalDual operator-(DirectionalDual a, DirectionalDual b) {
    return {a.value-b.value, a.direction-b.direction};
}
inline DirectionalDual operator-(DirectionalDual a) {
    return {-a.value, -a.direction};
}
inline DirectionalDual operator*(DirectionalDual a, DirectionalDual b) {
    return {a.value*b.value, a.direction*b.value+a.value*b.direction};
}
inline DirectionalDual operator/(DirectionalDual a, DirectionalDual b) {
    return {a.value/b.value,
        (a.direction*b.value-a.value*b.direction)/(b.value*b.value)};
}
inline DirectionalDual directional_sqrt(DirectionalDual a) {
    const double root = std::sqrt(a.value);
    return {root, 0.5*a.direction/root};
}
inline DirectionalDual directional_pow(DirectionalDual a, double exponent) {
    const double value = std::pow(a.value, exponent);
    return {value, exponent*std::pow(a.value, exponent-1.0)*a.direction};
}

PyObject* anisotropic_fock_derivatives_directional_native(PyObject*, PyObject* args) {
    // Differentiate the native first-derivative contraction in one forward
    // analytic pass.  Inputs 8-9 are exact directional AO integral seconds.
    constexpr int count = 16;
    if (PyTuple_GET_SIZE(args) != count) {
        PyErr_SetString(PyExc_TypeError,
            "directional anisotropic Fock contraction needs 16 arrays");
        return nullptr;
    }
    std::array<PyArrayObject*, count> arrays{};
    for (int index = 0; index < count; ++index) {
        arrays[index] = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            PyTuple_GET_ITEM(args, index),
            index == 4 || index == 5 ? NPY_INT64 : NPY_DOUBLE,
            NPY_ARRAY_IN_ARRAY));
        if (!arrays[index]) {
            for (auto* item : arrays) Py_XDECREF(item);
            return nullptr;
        }
    }
    const npy_intp pairs = PyArray_SIZE(arrays[0]);
    const npy_intp atoms = PyArray_NDIM(arrays[6]) == 2
        ? PyArray_DIM(arrays[6], 0) : 0;
    auto shape2 = [&](int index, npy_intp first, npy_intp second) {
        return PyArray_NDIM(arrays[index]) == 2
            && PyArray_DIM(arrays[index], 0) == first
            && PyArray_DIM(arrays[index], 1) == second;
    };
    auto shape3 = [&](int index, npy_intp first, npy_intp second,
            npy_intp third) {
        return PyArray_NDIM(arrays[index]) == 3
            && PyArray_DIM(arrays[index], 0) == first
            && PyArray_DIM(arrays[index], 1) == second
            && PyArray_DIM(arrays[index], 2) == third;
    };
    const bool valid = PyArray_NDIM(arrays[0]) == 1
        && shape2(1, pairs, 3)
        && shape3(2, pairs, 13, 3) && shape3(3, pairs, 13, 3)
        && PyArray_SIZE(arrays[4]) == pairs && PyArray_SIZE(arrays[5]) == pairs
        && shape2(6, atoms, 3) && shape2(7, atoms, 3)
        && shape3(8, pairs, 13, 3) && shape3(9, pairs, 13, 3)
        && PyArray_SIZE(arrays[10]) == atoms && shape2(11, atoms, 3)
        && shape3(12, atoms, 3, 3)
        && PyArray_SIZE(arrays[13]) == atoms && shape2(14, atoms, 3)
        && shape3(15, atoms, 3, 3);
    if (!valid) {
        for (auto* item : arrays) Py_DECREF(item);
        PyErr_SetString(PyExc_ValueError,
            "invalid directional anisotropic Fock arrays");
        return nullptr;
    }
    const auto* left_indices = static_cast<const std::int64_t*>(
        PyArray_DATA(arrays[4]));
    const auto* right_indices = static_cast<const std::int64_t*>(
        PyArray_DATA(arrays[5]));
    for (npy_intp pair = 0; pair < pairs; ++pair)
        if (left_indices[pair] < 0 || left_indices[pair] >= atoms
            || right_indices[pair] < 0 || right_indices[pair] >= atoms) {
            for (auto* item : arrays) Py_DECREF(item);
            PyErr_SetString(PyExc_ValueError,
                "directional anisotropic Fock atom index is out of bounds");
            return nullptr;
        }
    npy_intp dims[2] = {pairs, 3};
    auto* output_left = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dims, NPY_DOUBLE, 0));
    auto* output_right = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dims, NPY_DOUBLE, 0));
    if (!output_left || !output_right) {
        Py_XDECREF(output_left); Py_XDECREF(output_right);
        for (auto* item : arrays) Py_DECREF(item);
        return PyErr_NoMemory();
    }
    auto doubles = [&](int index) {
        return static_cast<const double*>(PyArray_DATA(arrays[index]));
    };
    const auto* overlap = doubles(0);
    const auto* dipole = doubles(1);
    const double* derivatives[2] = {doubles(2), doubles(3)};
    const std::int64_t* atom_index[2] = {
        static_cast<const std::int64_t*>(PyArray_DATA(arrays[4])),
        static_cast<const std::int64_t*>(PyArray_DATA(arrays[5]))
    };
    const auto* coordinates = doubles(6);
    const auto* direction = doubles(7);
    const double* seconds[2] = {doubles(8), doubles(9)};
    const auto* qpot = doubles(10);
    const auto* mupot = doubles(11);
    const auto* thetapot = doubles(12);
    const auto* qpot_dot = doubles(13);
    const auto* mupot_dot = doubles(14);
    const auto* thetapot_dot = doubles(15);
    double* output[2] = {
        static_cast<double*>(PyArray_DATA(output_left)),
        static_cast<double*>(PyArray_DATA(output_right))
    };
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp pair = 0; pair < pairs; ++pair) {
        const auto left_atom = atom_index[0][pair];
        const auto right_atom = atom_index[1][pair];
        DirectionalDual s(overlap[pair]);
        DirectionalDual mu[3];
        for (int component = 0; component < 3; ++component)
            mu[component] = {dipole[pair*3+component], 0.0};
        for (int endpoint = 0; endpoint < 2; ++endpoint)
            for (int component = 0; component < 3; ++component) {
                const auto atom = atom_index[endpoint][pair];
                const double velocity = direction[atom*3+component];
                s.direction += derivatives[endpoint][pair*39+component]
                    * velocity;
                for (int moment = 0; moment < 3; ++moment)
                    mu[moment].direction += derivatives[endpoint][
                        pair*39+(1+moment)*3+component] * velocity;
            }
        for (int derivative_endpoint = 0; derivative_endpoint < 2;
                ++derivative_endpoint)
            for (int allocation_endpoint = 0; allocation_endpoint < 2;
                    ++allocation_endpoint) {
                const auto atom = atom_index[allocation_endpoint][pair];
                const int occurrences = (atom == left_atom)+(atom == right_atom);
                const double shift_weight =
                    atom == atom_index[derivative_endpoint][pair]
                    ? 1.0/static_cast<double>(occurrences) : 0.0;
                const auto* first = derivatives[derivative_endpoint]+pair*39;
                const auto* second = seconds[derivative_endpoint]+pair*39;
                for (int axis = 0; axis < 3; ++axis) {
                    DirectionalDual ds(first[axis], second[axis]);
                    DirectionalDual value = -DirectionalDual(
                        qpot[atom], qpot_dot[atom])*ds;
                    DirectionalDual trace, draw[9];
                    for (int i = 0; i < 3; ++i) {
                        DirectionalDual xi(
                            coordinates[atom*3+i], direction[atom*3+i]);
                        const double delta_i = i == axis ? shift_weight : 0.0;
                        DirectionalDual ddi(first[(1+i)*3+axis],
                            second[(1+i)*3+axis]);
                        DirectionalDual dmu = delta_i*s+xi*ds-ddi;
                        value = value+DirectionalDual(
                            mupot[atom*3+i], mupot_dot[atom*3+i])*dmu;
                        for (int j = 0; j < 3; ++j) {
                            DirectionalDual xj(
                                coordinates[atom*3+j], direction[atom*3+j]);
                            const double delta_j = j == axis ? shift_weight : 0.0;
                            DirectionalDual ddj(first[(1+j)*3+axis],
                                second[(1+j)*3+axis]);
                            DirectionalDual item = ddi*xj+mu[i]*delta_j
                                +delta_i*mu[j]+xi*ddj-ds*xi*xj
                                -s*(delta_i*xj+xi*delta_j)
                                -DirectionalDual(first[(4+3*i+j)*3+axis],
                                    second[(4+3*i+j)*3+axis]);
                            draw[3*i+j] = item;
                            if (i == j) trace = trace+item;
                        }
                    }
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j) {
                            DirectionalDual dq = 1.5*draw[3*i+j]
                                -(i == j ? 0.5*trace : DirectionalDual());
                            value = value+DirectionalDual(
                                thetapot[(atom*3+i)*3+j],
                                thetapot_dot[(atom*3+i)*3+j])*dq;
                        }
                    output[derivative_endpoint][pair*3+axis]
                        += 0.5*value.direction;
                }
            }
    }
    Py_END_ALLOW_THREADS
    for (auto* item : arrays) Py_DECREF(item);
    return Py_BuildValue("NN", output_left, output_right);
}

PyObject* atm_three_body_hvp_core_native(PyObject*, PyObject* args) {
    // Return the geometric ATM Hessian action and the two CN coefficients.
    // The CN-gradient and CN-Hessian contractions remain in their existing
    // native-backed sparse routines on the Python side.
    constexpr int count = 11;
    if (PyTuple_GET_SIZE(args) != count+4) {
        PyErr_SetString(PyExc_TypeError, "ATM HVP core needs 11 arrays and 4 scalars");
        return nullptr;
    }
    std::array<PyArrayObject*, count> arrays{};
    for (int index = 0; index < count; ++index) {
        arrays[index] = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            PyTuple_GET_ITEM(args, index),
            index == 2 || index == 3 ? NPY_INT64 : NPY_DOUBLE,
            NPY_ARRAY_IN_ARRAY));
        if (!arrays[index]) {
            for (auto* item : arrays) Py_XDECREF(item);
            return nullptr;
        }
    }
    const double s9 = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 11));
    const double a1 = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 12));
    const double a2 = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 13));
    const double alpha = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 14));
    if (PyErr_Occurred()) {
        for (auto* item : arrays) Py_DECREF(item);
        return nullptr;
    }
    const npy_intp atoms = PyArray_NDIM(arrays[0]) == 2
        ? PyArray_DIM(arrays[0], 0) : 0;
    const npy_intp triples = PyArray_NDIM(arrays[2]) == 2
        ? PyArray_DIM(arrays[2], 0) : 0;
    const npy_intp pairs = PyArray_SIZE(arrays[4]);
    const bool valid = PyArray_NDIM(arrays[0]) == 2
        && PyArray_DIM(arrays[0], 1) == 3
        && PyArray_NDIM(arrays[1]) == 2
        && PyArray_DIM(arrays[1], 0) == atoms
        && PyArray_DIM(arrays[1], 1) == 3
        && PyArray_NDIM(arrays[2]) == 2
        && PyArray_DIM(arrays[2], 1) == 3
        && PyArray_NDIM(arrays[3]) == 2
        && PyArray_DIM(arrays[3], 0) == triples
        && PyArray_DIM(arrays[3], 1) == 3;
    bool pairs_valid = valid;
    for (int index = 5; index < count; ++index)
        pairs_valid &= PyArray_SIZE(arrays[index]) == pairs;
    if (!pairs_valid) {
        for (auto* item : arrays) Py_DECREF(item);
        PyErr_SetString(PyExc_ValueError, "invalid sparse ATM HVP arrays");
        return nullptr;
    }
    const auto* triangle = static_cast<const std::int64_t*>(PyArray_DATA(arrays[2]));
    const auto* position = static_cast<const std::int64_t*>(PyArray_DATA(arrays[3]));
    for (npy_intp item = 0; item < triples*3; ++item)
        if (triangle[item] < 0 || triangle[item] >= atoms
            || position[item] < 0 || position[item] >= pairs) {
            for (auto* array : arrays) Py_DECREF(array);
            PyErr_SetString(PyExc_ValueError,
                "sparse ATM triangle or pair index is out of bounds");
            return nullptr;
        }
    npy_intp geometry_dims[2] = {atoms, 3};
    npy_intp cn_dims[1] = {atoms};
    auto* geometric = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, geometry_dims, NPY_DOUBLE, 0));
    auto* cn = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, cn_dims, NPY_DOUBLE, 0));
    auto* cn_dot = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, cn_dims, NPY_DOUBLE, 0));
    if (!geometric || !cn || !cn_dot) {
        Py_XDECREF(geometric); Py_XDECREF(cn); Py_XDECREF(cn_dot);
        for (auto* item : arrays) Py_DECREF(item);
        return PyErr_NoMemory();
    }
    auto doubles = [&](int index) {
        return static_cast<const double*>(PyArray_DATA(arrays[index]));
    };
    const auto* xyz = doubles(0);
    const auto* velocity = doubles(1);
    const auto* c6 = doubles(4);
    const auto* c8 = doubles(5);
    const auto* dc_left = doubles(6);
    const auto* dc_right = doubles(7);
    const auto* c6_dot = doubles(8);
    const auto* dc_left_dot = doubles(9);
    const auto* dc_right_dot = doubles(10);
    auto* geom_out = static_cast<double*>(PyArray_DATA(geometric));
    auto* cn_out = static_cast<double*>(PyArray_DATA(cn));
    auto* cn_dot_out = static_cast<double*>(PyArray_DATA(cn_dot));
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp item = 0; item < triples; ++item) {
        const auto kat = triangle[item*3];
        const auto jat = triangle[item*3+1];
        const auto iat = triangle[item*3+2];
        const auto jk = position[item*3];
        const auto ik = position[item*3+1];
        const auto ij = position[item*3+2];
        DirectionalDual vij[3], vik[3], vjk[3];
        DirectionalDual a, b, c;
        for (int axis = 0; axis < 3; ++axis) {
            vij[axis] = {xyz[jat*3+axis]-xyz[iat*3+axis],
                velocity[jat*3+axis]-velocity[iat*3+axis]};
            vik[axis] = {xyz[kat*3+axis]-xyz[iat*3+axis],
                velocity[kat*3+axis]-velocity[iat*3+axis]};
            vjk[axis] = {xyz[kat*3+axis]-xyz[jat*3+axis],
                velocity[kat*3+axis]-velocity[jat*3+axis]};
            a = a+vij[axis]*vij[axis];
            b = b+vik[axis]*vik[axis];
            c = c+vjk[axis]*vjk[axis];
        }
        DirectionalDual c6ij(c6[ij], c6_dot[ij]);
        DirectionalDual c6ik(c6[ik], c6_dot[ik]);
        DirectionalDual c6jk(c6[jk], c6_dot[jk]);
        const double r0ij = a1*std::sqrt(c8[ij]/c6[ij])+a2;
        const double r0ik = a1*std::sqrt(c8[ik]/c6[ik])+a2;
        const double r0jk = a1*std::sqrt(c8[jk]/c6[jk])+a2;
        DirectionalDual product = a*b*c;
        DirectionalDual r1 = directional_sqrt(product);
        DirectionalDual r3 = product*r1;
        DirectionalDual r5 = r3*product;
        DirectionalDual ratio = directional_pow(
            (r0ij*r0ik*r0jk)/r1, alpha/3.0);
        DirectionalDual fdmp = 1.0/(1.0+6.0*ratio);
        DirectionalDual angular = 0.375*(a+c-b)*(a-c+b)*(-a+c+b)/r5
            +1.0/r3;
        DirectionalDual c9 = -s9*directional_sqrt(c6ij*c6ik*c6jk);
        DirectionalDual energy = angular*fdmp*c9;
        DirectionalDual dfdmp = -2.0*alpha*ratio*fdmp*fdmp;
        auto coefficient = [&](DirectionalDual ra, DirectionalDual rb,
                DirectionalDual rc) {
            DirectionalDual polynomial = ra*ra*ra+ra*ra*(rb+rc)
                +ra*(3.0*rb*rb+2.0*rb*rc+3.0*rc*rc)
                -5.0*(rb-rc)*(rb-rc)*(rb+rc);
            DirectionalDual dang = -0.375*polynomial/r5;
            return c9*(-dang*fdmp+angular*dfdmp)/ra;
        };
        DirectionalDual cij = coefficient(a,c,b);
        DirectionalDual cik = coefficient(b,c,a);
        DirectionalDual cjk = coefficient(c,b,a);
        for (int axis = 0; axis < 3; ++axis) {
            const double dgij = (cij*vij[axis]).direction;
            const double dgik = (cik*vik[axis]).direction;
            const double dgjk = (cjk*vjk[axis]).direction;
            geom_out[iat*3+axis] -= dgij+dgik;
            geom_out[jat*3+axis] += dgij-dgjk;
            geom_out[kat*3+axis] += dgik+dgjk;
        }
        auto add_cn = [&](npy_intp atom, DirectionalDual derivative,
                DirectionalDual coefficient_value) {
            DirectionalDual contribution = -0.5*energy
                *derivative/coefficient_value;
            cn_out[atom] += contribution.value;
            cn_dot_out[atom] += contribution.direction;
        };
        add_cn(iat, {dc_left[ij], dc_left_dot[ij]}, c6ij);
        add_cn(jat, {dc_right[ij], dc_right_dot[ij]}, c6ij);
        add_cn(iat, {dc_left[ik], dc_left_dot[ik]}, c6ik);
        add_cn(kat, {dc_right[ik], dc_right_dot[ik]}, c6ik);
        add_cn(jat, {dc_left[jk], dc_left_dot[jk]}, c6jk);
        add_cn(kat, {dc_right[jk], dc_right_dot[jk]}, c6jk);
    }
    Py_END_ALLOW_THREADS
    for (auto* item : arrays) Py_DECREF(item);
    return Py_BuildValue("NNN", geometric, cn, cn_dot);
}

PyObject* anisotropic_energy_potentials_native(PyObject*, PyObject* args) {
    PyObject *charges_o,*dipoles_o,*quadrupoles_o,*dipole_xc_o,*quadrupole_xc_o;
    PyObject *left_o,*right_o,*displacement_o,*f3_o,*f5_o,*distance2_o;
    if(!PyArg_ParseTuple(args,"OOOOOOOOOOO",&charges_o,&dipoles_o,&quadrupoles_o,
            &dipole_xc_o,&quadrupole_xc_o,&left_o,&right_o,&displacement_o,
            &f3_o,&f5_o,&distance2_o))return nullptr;
    auto array=[](PyObject* object,int type){
        return reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(object,type,NPY_ARRAY_IN_ARRAY));
    };
    auto* charges=array(charges_o,NPY_DOUBLE);auto* dipoles=array(dipoles_o,NPY_DOUBLE);
    auto* quadrupoles=array(quadrupoles_o,NPY_DOUBLE);auto* dipole_xc=array(dipole_xc_o,NPY_DOUBLE);
    auto* quadrupole_xc=array(quadrupole_xc_o,NPY_DOUBLE);auto* left=array(left_o,NPY_INT64);
    auto* right=array(right_o,NPY_INT64);auto* displacement=array(displacement_o,NPY_DOUBLE);
    auto* f3=array(f3_o,NPY_DOUBLE);auto* f5=array(f5_o,NPY_DOUBLE);
    auto* distance2=array(distance2_o,NPY_DOUBLE);
    const npy_intp atoms=charges?PyArray_SIZE(charges):0,pairs=left?PyArray_SIZE(left):0;
    const bool valid=charges&&dipoles&&quadrupoles&&dipole_xc&&quadrupole_xc
        &&left&&right&&displacement&&f3&&f5&&distance2
        &&PyArray_NDIM(dipoles)==2&&PyArray_DIM(dipoles,0)==atoms&&PyArray_DIM(dipoles,1)==3
        &&PyArray_NDIM(quadrupoles)==3&&PyArray_DIM(quadrupoles,0)==atoms
        &&PyArray_DIM(quadrupoles,1)==3&&PyArray_DIM(quadrupoles,2)==3
        &&PyArray_SIZE(dipole_xc)==atoms&&PyArray_SIZE(quadrupole_xc)==atoms
        &&PyArray_SIZE(right)==pairs&&PyArray_SIZE(f3)==pairs&&PyArray_SIZE(f5)==pairs
        &&PyArray_SIZE(distance2)==pairs&&PyArray_NDIM(displacement)==2
        &&PyArray_DIM(displacement,0)==pairs&&PyArray_DIM(displacement,1)==3;
    if(!valid){
        Py_XDECREF(charges);Py_XDECREF(dipoles);Py_XDECREF(quadrupoles);Py_XDECREF(dipole_xc);
        Py_XDECREF(quadrupole_xc);Py_XDECREF(left);Py_XDECREF(right);Py_XDECREF(displacement);
        Py_XDECREF(f3);Py_XDECREF(f5);Py_XDECREF(distance2);
        PyErr_SetString(PyExc_ValueError,"invalid anisotropic energy arrays");return nullptr;
    }
    npy_intp qdims[1]={atoms},mudims[2]={atoms,3},tdims[3]={atoms,3,3};
    auto* qout=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(1,qdims,NPY_DOUBLE,0));
    auto* muout=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2,mudims,NPY_DOUBLE,0));
    auto* tout=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(3,tdims,NPY_DOUBLE,0));
    if(!qout||!muout||!tout){Py_XDECREF(qout);Py_XDECREF(muout);Py_XDECREF(tout);return PyErr_NoMemory();}
    const auto* q=static_cast<const double*>(PyArray_DATA(charges));
    const auto* mu=static_cast<const double*>(PyArray_DATA(dipoles));
    const auto* theta=static_cast<const double*>(PyArray_DATA(quadrupoles));
    const auto* muxc=static_cast<const double*>(PyArray_DATA(dipole_xc));
    const auto* txc=static_cast<const double*>(PyArray_DATA(quadrupole_xc));
    const auto* li=static_cast<const std::int64_t*>(PyArray_DATA(left));
    const auto* ri=static_cast<const std::int64_t*>(PyArray_DATA(right));
    const auto* dr=static_cast<const double*>(PyArray_DATA(displacement));
    const auto* pf3=static_cast<const double*>(PyArray_DATA(f3));
    const auto* pf5=static_cast<const double*>(PyArray_DATA(f5));
    const auto* r2=static_cast<const double*>(PyArray_DATA(distance2));
    auto* qp=static_cast<double*>(PyArray_DATA(qout));
    auto* mup=static_cast<double*>(PyArray_DATA(muout));
    auto* tp=static_cast<double*>(PyArray_DATA(tout));
    double energy=0.0;
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp atom=0;atom<atoms;++atom){
        for(int i=0;i<3;++i){
            energy+=muxc[atom]*mu[atom*3+i]*mu[atom*3+i];
            mup[atom*3+i]=2.0*muxc[atom]*mu[atom*3+i];
            for(int j=0;j<3;++j){
                const auto index=(atom*3+i)*3+j;
                energy+=txc[atom]*theta[index]*theta[index];
                tp[index]=2.0*txc[atom]*theta[index];
            }
        }
    }
    for(npy_intp pair=0;pair<pairs;++pair){
        const auto l=li[pair],r=ri[pair];double lp=0.0,rp=0.0,lq=0.0,rq=0.0,mudot=0.0;
        for(int i=0;i<3;++i){
            lp+=mu[l*3+i]*dr[pair*3+i];rp+=mu[r*3+i]*dr[pair*3+i];
            mudot+=mu[l*3+i]*mu[r*3+i];
            for(int j=0;j<3;++j){
                const double outer=dr[pair*3+i]*dr[pair*3+j];
                lq+=outer*theta[(l*3+i)*3+j];rq+=outer*theta[(r*3+i)*3+j];
            }
        }
        energy+=pf3[pair]*(-q[l]*rp+q[r]*lp)
            +pf5[pair]*(q[l]*rq+q[r]*lq-3.0*lp*rp+mudot*r2[pair]);
        qp[l]+=-pf3[pair]*rp+pf5[pair]*rq;qp[r]+=pf3[pair]*lp+pf5[pair]*lq;
        for(int i=0;i<3;++i){
            const double d=dr[pair*3+i];
            mup[l*3+i]+=pf3[pair]*q[r]*d+pf5[pair]*(-3.0*d*rp+mu[r*3+i]*r2[pair]);
            mup[r*3+i]+=-pf3[pair]*q[l]*d+pf5[pair]*(-3.0*d*lp+mu[l*3+i]*r2[pair]);
            for(int j=0;j<3;++j){
                const double outer=d*dr[pair*3+j];
                tp[(l*3+i)*3+j]+=pf5[pair]*q[r]*outer;
                tp[(r*3+i)*3+j]+=pf5[pair]*q[l]*outer;
            }
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(charges);Py_DECREF(dipoles);Py_DECREF(quadrupoles);Py_DECREF(dipole_xc);
    Py_DECREF(quadrupole_xc);Py_DECREF(left);Py_DECREF(right);Py_DECREF(displacement);
    Py_DECREF(f3);Py_DECREF(f5);Py_DECREF(distance2);
    return Py_BuildValue("dNNN",energy,qout,muout,tout);
}

PyObject* atm_three_body_energy_native(PyObject*, PyObject* args) {
    PyObject *coordinates_o,*c6_o,*c8_o;double s9,a1,a2;
    if(!PyArg_ParseTuple(args,"OOOddd",&coordinates_o,&c6_o,&c8_o,&s9,&a1,&a2))return nullptr;
    auto array=[](PyObject* object){return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));};
    auto* coordinates=array(coordinates_o);auto* c6=array(c6_o);auto* c8=array(c8_o);
    const npy_intp n=coordinates&&PyArray_NDIM(coordinates)==2?PyArray_DIM(coordinates,0):0;
    const bool valid=coordinates&&c6&&c8&&PyArray_DIM(coordinates,1)==3
        &&PyArray_NDIM(c6)==2&&PyArray_DIM(c6,0)==n&&PyArray_DIM(c6,1)==n
        &&PyArray_NDIM(c8)==2&&PyArray_DIM(c8,0)==n&&PyArray_DIM(c8,1)==n;
    if(!valid){Py_XDECREF(coordinates);Py_XDECREF(c6);Py_XDECREF(c8);
        PyErr_SetString(PyExc_ValueError,"invalid ATM energy arrays");return nullptr;}
    const auto* xyz=static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* pc6=static_cast<const double*>(PyArray_DATA(c6));
    const auto* pc8=static_cast<const double*>(PyArray_DATA(c8));
    double energy=0.0;
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp left=0;left<n;++left)for(npy_intp middle=left+1;middle<n;++middle)
        for(npy_intp right=middle+1;right<n;++right){
            double rab[3],rac[3],rbc[3];double dab2=0.0,dac2=0.0,dbc2=0.0;
            for(int axis=0;axis<3;++axis){
                rab[axis]=xyz[left*3+axis]-xyz[middle*3+axis];
                rac[axis]=xyz[left*3+axis]-xyz[right*3+axis];
                rbc[axis]=xyz[middle*3+axis]-xyz[right*3+axis];
                dab2+=rab[axis]*rab[axis];dac2+=rac[axis]*rac[axis];dbc2+=rbc[axis]*rbc[axis];
            }
            const double dab=std::sqrt(dab2),dac=std::sqrt(dac2),dbc=std::sqrt(dbc2);
            double dot_l=0.0,dot_m=0.0,dot_r=0.0;
            for(int axis=0;axis<3;++axis){
                dot_l+=rab[axis]*rac[axis];dot_m-=rab[axis]*rbc[axis];dot_r+=rac[axis]*rbc[axis];
            }
            const double angular=1.0+3.0*dot_l*dot_m*dot_r/(dab2*dac2*dbc2);
            const double c6ab=pc6[left*n+middle],c6ac=pc6[left*n+right],c6bc=pc6[middle*n+right];
            const double c9=std::sqrt(c6ab*c6ac*c6bc);
            const double r0ab=a1*std::sqrt(pc8[left*n+middle]/c6ab)+a2;
            const double r0ac=a1*std::sqrt(pc8[left*n+right]/c6ac)+a2;
            const double r0bc=a1*std::sqrt(pc8[middle*n+right]/c6bc)+a2;
            const double product=dab*dac*dbc;
            const double ratio=(r0ab*r0ac*r0bc)/product;
            const double damping=1.0/(1.0+6.0*std::pow(ratio,16.0/3.0));
            energy+=s9*angular*c9*damping/(product*product*product);
        }
    Py_END_ALLOW_THREADS
    Py_DECREF(coordinates);Py_DECREF(c6);Py_DECREF(c8);return PyFloat_FromDouble(energy);
}

PyObject* atm_three_body_gradient_native(PyObject*, PyObject* args) {
    PyObject *coordinates_o,*c6_o,*c8_o,*dc6_o,*jacobian_o;
    double s9,a1,a2,alpha;
    if(!PyArg_ParseTuple(args,"OOOOOdddd",&coordinates_o,&c6_o,&c8_o,&dc6_o,
            &jacobian_o,&s9,&a1,&a2,&alpha))return nullptr;
    auto array=[](PyObject* object){return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));};
    auto* coordinates=array(coordinates_o);auto* c6=array(c6_o);auto* c8=array(c8_o);
    auto* dc6=array(dc6_o);auto* jacobian=array(jacobian_o);
    const npy_intp n=coordinates&&PyArray_NDIM(coordinates)==2?PyArray_DIM(coordinates,0):0;
    const bool valid=coordinates&&c6&&c8&&dc6&&jacobian&&PyArray_DIM(coordinates,1)==3
        &&PyArray_NDIM(c6)==2&&PyArray_DIM(c6,0)==n&&PyArray_DIM(c6,1)==n
        &&PyArray_NDIM(c8)==2&&PyArray_DIM(c8,0)==n&&PyArray_DIM(c8,1)==n
        &&PyArray_NDIM(dc6)==3&&PyArray_DIM(dc6,0)==n&&PyArray_DIM(dc6,1)==n&&PyArray_DIM(dc6,2)==n
        &&PyArray_NDIM(jacobian)==3&&PyArray_DIM(jacobian,0)==n
        &&PyArray_DIM(jacobian,1)==n&&PyArray_DIM(jacobian,2)==3;
    if(!valid){Py_XDECREF(coordinates);Py_XDECREF(c6);Py_XDECREF(c8);Py_XDECREF(dc6);
        Py_XDECREF(jacobian);PyErr_SetString(PyExc_ValueError,"invalid ATM gradient arrays");return nullptr;}
    npy_intp dims[2]={n,3};auto* output=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2,dims,NPY_DOUBLE,0));
    if(!output)return PyErr_NoMemory();
    const auto* xyz=static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* pc6=static_cast<const double*>(PyArray_DATA(c6));
    const auto* pc8=static_cast<const double*>(PyArray_DATA(c8));
    const auto* pdc6=static_cast<const double*>(PyArray_DATA(dc6));
    const auto* jac=static_cast<const double*>(PyArray_DATA(jacobian));
    auto* gradient=static_cast<double*>(PyArray_DATA(output));
    std::vector<double> dedcn(static_cast<std::size_t>(n),0.0);
    const double alpha3=alpha/3.0;
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp kat=0;kat<n;++kat)for(npy_intp jat=kat+1;jat<n;++jat)
        for(npy_intp iat=jat+1;iat<n;++iat){
            double vij[3],vik[3],vjk[3];double r2ij=0.0,r2ik=0.0,r2jk=0.0;
            for(int axis=0;axis<3;++axis){
                vij[axis]=xyz[jat*3+axis]-xyz[iat*3+axis];
                vik[axis]=xyz[kat*3+axis]-xyz[iat*3+axis];
                vjk[axis]=xyz[kat*3+axis]-xyz[jat*3+axis];
                r2ij+=vij[axis]*vij[axis];r2ik+=vik[axis]*vik[axis];r2jk+=vjk[axis]*vjk[axis];
            }
            const double c6ij=pc6[iat*n+jat],c6ik=pc6[iat*n+kat],c6jk=pc6[jat*n+kat];
            const double r0ij=a1*std::sqrt(pc8[iat*n+jat]/c6ij)+a2;
            const double r0ik=a1*std::sqrt(pc8[iat*n+kat]/c6ik)+a2;
            const double r0jk=a1*std::sqrt(pc8[jat*n+kat]/c6jk)+a2;
            const double r2=r2ij*r2ik*r2jk,r1=std::sqrt(r2),r3=r2*r1,r5=r3*r2;
            const double ratio=std::pow((r0ij*r0ik*r0jk)/r1,alpha3);
            const double fdmp=1.0/(1.0+6.0*ratio);
            const double angular=0.375*(r2ij+r2jk-r2ik)
                *(r2ij-r2jk+r2ik)*(-r2ij+r2jk+r2ik)/r5+1.0/r3;
            const double c9=-s9*std::sqrt(c6ij*c6ik*c6jk),de=angular*fdmp*c9;
            const double dfdmp=-2.0*alpha*ratio*fdmp*fdmp;
            auto coefficient=[&](double ra,double rb,double rc){
                const double dang=-0.375*(ra*ra*ra+ra*ra*(rb+rc)
                    +ra*(3.0*rb*rb+2.0*rb*rc+3.0*rc*rc)
                    -5.0*(rb-rc)*(rb-rc)*(rb+rc))/r5;
                return c9*(-dang*fdmp+angular*dfdmp)/ra;
            };
            const double cij=coefficient(r2ij,r2jk,r2ik);
            const double cik=coefficient(r2ik,r2jk,r2ij);
            const double cjk=coefficient(r2jk,r2ik,r2ij);
            for(int axis=0;axis<3;++axis){
                const double dgij=cij*vij[axis],dgik=cik*vik[axis],dgjk=cjk*vjk[axis];
                gradient[iat*3+axis]-=dgij+dgik;
                gradient[jat*3+axis]+=dgij-dgjk;
                gradient[kat*3+axis]+=dgik+dgjk;
            }
            // D4 atomic polarizabilities are local in CN: C6_AB depends
            // only on CN_A and CN_B.  Contract the six nonzero endpoint
            // derivatives directly instead of scanning an n-wide zero slab
            // for each triple (the former O(n^4) implementation).
            const double scale=-0.5*de;
            dedcn[iat]+=scale*(
                pdc6[(iat*n+jat)*n+iat]/c6ij
                +pdc6[(iat*n+kat)*n+iat]/c6ik);
            dedcn[jat]+=scale*(
                pdc6[(iat*n+jat)*n+jat]/c6ij
                +pdc6[(jat*n+kat)*n+jat]/c6jk);
            dedcn[kat]+=scale*(
                pdc6[(iat*n+kat)*n+kat]/c6ik
                +pdc6[(jat*n+kat)*n+kat]/c6jk);
        }
    for(npy_intp atom=0;atom<n;++atom)for(npy_intp target=0;target<n;++target)
        for(int axis=0;axis<3;++axis)
            gradient[target*3+axis]+=dedcn[atom]*jac[(atom*n+target)*3+axis];
    Py_END_ALLOW_THREADS
    Py_DECREF(coordinates);Py_DECREF(c6);Py_DECREF(c8);Py_DECREF(dc6);Py_DECREF(jacobian);
    return reinterpret_cast<PyObject*>(output);
}

PyObject* atm_three_body_energy_gradient_native(PyObject*, PyObject* args) {
    PyObject *coordinates_o,*c6_o,*c8_o,*dc6_o,*radii_o,*en_o;
    double s9,a1,a2,alpha;
    if(!PyArg_ParseTuple(args,"OOOOOOdddd",&coordinates_o,&c6_o,&c8_o,
            &dc6_o,&radii_o,&en_o,&s9,&a1,&a2,&alpha))return nullptr;
    auto array=[](PyObject* object){return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));};
    auto* coordinates=array(coordinates_o);auto* c6=array(c6_o);auto* c8=array(c8_o);
    auto* dc6=array(dc6_o);auto* radii=array(radii_o);auto* en=array(en_o);
    const npy_intp n=coordinates&&PyArray_NDIM(coordinates)==2?PyArray_DIM(coordinates,0):0;
    const bool valid=coordinates&&c6&&c8&&dc6&&radii&&en
        &&PyArray_DIM(coordinates,1)==3
        &&PyArray_NDIM(c6)==2&&PyArray_DIM(c6,0)==n&&PyArray_DIM(c6,1)==n
        &&PyArray_NDIM(c8)==2&&PyArray_DIM(c8,0)==n&&PyArray_DIM(c8,1)==n
        &&PyArray_NDIM(dc6)==3&&PyArray_DIM(dc6,0)==n
        &&PyArray_DIM(dc6,1)==n&&PyArray_DIM(dc6,2)==n
        &&PyArray_SIZE(radii)==n&&PyArray_SIZE(en)==n;
    if(!valid){Py_XDECREF(coordinates);Py_XDECREF(c6);Py_XDECREF(c8);
        Py_XDECREF(dc6);Py_XDECREF(radii);Py_XDECREF(en);
        PyErr_SetString(PyExc_ValueError,"invalid fused ATM arrays");return nullptr;}
    npy_intp dims[2]={n,3};auto* output=reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2,dims,NPY_DOUBLE,0));
    if(!output){Py_DECREF(coordinates);Py_DECREF(c6);Py_DECREF(c8);
        Py_DECREF(dc6);Py_DECREF(radii);Py_DECREF(en);return PyErr_NoMemory();}
    const auto* xyz=static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* pc6=static_cast<const double*>(PyArray_DATA(c6));
    const auto* pc8=static_cast<const double*>(PyArray_DATA(c8));
    const auto* pdc6=static_cast<const double*>(PyArray_DATA(dc6));
    const auto* radius=static_cast<const double*>(PyArray_DATA(radii));
    const auto* electronegativity=static_cast<const double*>(PyArray_DATA(en));
    auto* gradient=static_cast<double*>(PyArray_DATA(output));
    std::vector<double> dedcn(static_cast<std::size_t>(n),0.0);
    std::vector<double> pair_r2(static_cast<std::size_t>(n*n),0.0);
    std::vector<double> pair_r0(static_cast<std::size_t>(n*n),0.0);
    std::vector<double> pair_vector(static_cast<std::size_t>(n*n*3),0.0);
    const double alpha3=alpha/3.0;
    constexpr double inv_sqrt_pi=0.56418958354775628695;
    double energy_total=0.0;
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp left=1;left<n;++left)for(npy_intp right=0;right<left;++right){
        double distance2=0.0;
        for(int axis=0;axis<3;++axis){
            const double delta=xyz[left*3+axis]-xyz[right*3+axis];
            distance2+=delta*delta;
            pair_vector[(left*n+right)*3+axis]=-delta;
            pair_vector[(right*n+left)*3+axis]=delta;
        }
        const double pair_c6=pc6[left*n+right];
        const double damping_radius=a1*std::sqrt(pc8[left*n+right]/pair_c6)+a2;
        pair_r2[left*n+right]=pair_r2[right*n+left]=distance2;
        pair_r0[left*n+right]=pair_r0[right*n+left]=damping_radius;
    }
    for(npy_intp kat=0;kat<n;++kat)for(npy_intp jat=kat+1;jat<n;++jat)
        for(npy_intp iat=jat+1;iat<n;++iat){
            double vij[3],vik[3],vjk[3];
            for(int axis=0;axis<3;++axis){
                vij[axis]=pair_vector[(iat*n+jat)*3+axis];
                vik[axis]=pair_vector[(iat*n+kat)*3+axis];
                vjk[axis]=pair_vector[(jat*n+kat)*3+axis];
            }
            const double r2ij=pair_r2[iat*n+jat];
            const double r2ik=pair_r2[iat*n+kat];
            const double r2jk=pair_r2[jat*n+kat];
            const double c6ij=pc6[iat*n+jat],c6ik=pc6[iat*n+kat],c6jk=pc6[jat*n+kat];
            const double r0ij=pair_r0[iat*n+jat];
            const double r0ik=pair_r0[iat*n+kat];
            const double r0jk=pair_r0[jat*n+kat];
            const double r2=r2ij*r2ik*r2jk,r1=std::sqrt(r2),r3=r2*r1,r5=r3*r2;
            const double ratio=std::pow((r0ij*r0ik*r0jk)/r1,alpha3);
            const double fdmp=1.0/(1.0+6.0*ratio);
            const double angular=0.375*(r2ij+r2jk-r2ik)
                *(r2ij-r2jk+r2ik)*(-r2ij+r2jk+r2ik)/r5+1.0/r3;
            const double c9=-s9*std::sqrt(c6ij*c6ik*c6jk);
            const double de=angular*fdmp*c9;
            energy_total-=de;
            const double dfdmp=-2.0*alpha*ratio*fdmp*fdmp;
            auto coefficient=[&](double ra,double rb,double rc){
                const double dang=-0.375*(ra*ra*ra+ra*ra*(rb+rc)
                    +ra*(3.0*rb*rb+2.0*rb*rc+3.0*rc*rc)
                    -5.0*(rb-rc)*(rb-rc)*(rb+rc))/r5;
                return c9*(-dang*fdmp+angular*dfdmp)/ra;
            };
            const double cij=coefficient(r2ij,r2jk,r2ik);
            const double cik=coefficient(r2ik,r2jk,r2ij);
            const double cjk=coefficient(r2jk,r2ik,r2ij);
            for(int axis=0;axis<3;++axis){
                const double dgij=cij*vij[axis],dgik=cik*vik[axis],dgjk=cjk*vjk[axis];
                gradient[iat*3+axis]-=dgij+dgik;
                gradient[jat*3+axis]+=dgij-dgjk;
                gradient[kat*3+axis]+=dgik+dgjk;
            }
            const double scale=-0.5*de;
            dedcn[iat]+=scale*(pdc6[(iat*n+jat)*n+iat]/c6ij
                +pdc6[(iat*n+kat)*n+iat]/c6ik);
            dedcn[jat]+=scale*(pdc6[(iat*n+jat)*n+jat]/c6ij
                +pdc6[(jat*n+kat)*n+jat]/c6jk);
            dedcn[kat]+=scale*(pdc6[(iat*n+kat)*n+kat]/c6ik
                +pdc6[(jat*n+kat)*n+kat]/c6jk);
        }
    // Contract dE/dCN directly with the D4 pair derivative.  This removes
    // the dense n x n x 3 Jacobian formerly materialized in Python.
    for(npy_intp left=1;left<n;++left)for(npy_intp right=0;right<left;++right){
        double displacement[3],distance2=0.0;
        for(int axis=0;axis<3;++axis){
            displacement[axis]=xyz[left*3+axis]-xyz[right*3+axis];
            distance2+=displacement[axis]*displacement[axis];
        }
        const double distance=std::sqrt(distance2);
        const double critical=4.0*(radius[left]+radius[right])/3.0;
        const double argument=-7.5*(distance-critical)/critical;
        const double delta=std::abs(electronegativity[left]-electronegativity[right]);
        const double radial=-7.5*std::exp(-argument*argument)*inv_sqrt_pi/critical
            *4.10451*std::exp(-(delta+19.08857)*(delta+19.08857)
                /(2.0*11.28174*11.28174));
        const double factor=(dedcn[left]+dedcn[right])*radial/distance;
        for(int axis=0;axis<3;++axis){
            const double value=factor*displacement[axis];
            gradient[left*3+axis]+=value;gradient[right*3+axis]-=value;
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(coordinates);Py_DECREF(c6);Py_DECREF(c8);Py_DECREF(dc6);
    Py_DECREF(radii);Py_DECREF(en);
    return Py_BuildValue("dN",energy_total,output);
}

PyObject* sparse_matvec_py(PyObject*, PyObject* args) {
    long long size; PyObject *rows_o,*columns_o,*values_o,*vector_o;
    if(!PyArg_ParseTuple(args,"LOOOO",&size,&rows_o,&columns_o,&values_o,&vector_o))return nullptr;
    SparseMatrixNative matrix;PyArrayObject *rows=nullptr,*columns=nullptr,*values=nullptr;
    auto* vector=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(vector_o,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
    if(!vector||PyArray_SIZE(vector)!=size||!parse_sparse_matrix(size,rows_o,columns_o,values_o,matrix,rows,columns,values)){
        Py_XDECREF(vector);Py_XDECREF(rows);Py_XDECREF(columns);Py_XDECREF(values);
        PyErr_SetString(PyExc_ValueError,"invalid sparse matvec arrays");return nullptr;
    }
    npy_intp dims[1]={static_cast<npy_intp>(size)};auto* result=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(1,dims,NPY_DOUBLE,0));
    const auto* x=static_cast<const double*>(PyArray_DATA(vector));auto* y=static_cast<double*>(PyArray_DATA(result));
    for(std::size_t i=0;i<matrix.values.size();++i){auto r=matrix.rows[i],c=matrix.columns[i];double v=matrix.values[i];y[r]+=v*x[c];if(r!=c)y[c]+=v*x[r];}
    Py_DECREF(vector);Py_DECREF(rows);Py_DECREF(columns);Py_DECREF(values);return reinterpret_cast<PyObject*>(result);
}

PyObject* sparse_gather_py(PyObject*, PyObject* args) {
    long long size;
    PyObject *rows_o,*columns_o,*values_o,*target_rows_o,*target_columns_o;
    if(!PyArg_ParseTuple(args,"LOOOOO",&size,&rows_o,&columns_o,&values_o,
            &target_rows_o,&target_columns_o))return nullptr;
    SparseMatrixNative matrix;PyArrayObject *rows=nullptr,*columns=nullptr,*values=nullptr;
    auto* target_rows=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(target_rows_o,NPY_INT64,NPY_ARRAY_IN_ARRAY));
    auto* target_columns=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(target_columns_o,NPY_INT64,NPY_ARRAY_IN_ARRAY));
    if(!target_rows||!target_columns||PyArray_SIZE(target_rows)!=PyArray_SIZE(target_columns)
        ||!parse_sparse_matrix(size,rows_o,columns_o,values_o,matrix,rows,columns,values)){
        Py_XDECREF(target_rows);Py_XDECREF(target_columns);Py_XDECREF(rows);Py_XDECREF(columns);Py_XDECREF(values);
        PyErr_SetString(PyExc_ValueError,"invalid sparse gather arrays");return nullptr;
    }
    const npy_intp count=PyArray_SIZE(target_rows);npy_intp dims[1]={count};
    auto* result=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(1,dims,NPY_DOUBLE,0));
    const auto* tr=static_cast<const std::int64_t*>(PyArray_DATA(target_rows));
    const auto* tc=static_cast<const std::int64_t*>(PyArray_DATA(target_columns));
    auto* output=static_cast<double*>(PyArray_DATA(result));
    std::size_t source=0;
    for(npy_intp target=0;target<count;++target){
        while(source<matrix.values.size()&&(matrix.rows[source]<tr[target]
            ||(matrix.rows[source]==tr[target]&&matrix.columns[source]<tc[target])))++source;
        if(source<matrix.values.size()&&matrix.rows[source]==tr[target]
            &&matrix.columns[source]==tc[target])output[target]=matrix.values[source];
    }
    Py_DECREF(target_rows);Py_DECREF(target_columns);Py_DECREF(rows);Py_DECREF(columns);Py_DECREF(values);
    return reinterpret_cast<PyObject*>(result);
}

PyObject* dense_scc_observables_native(PyObject*, PyObject* args) {
    PyObject *density_o, *overlap_o, *dipole_o, *second_o, *centers_o,
             *atom_o, *shell_o, *reference_o, *valence_o;
    if (!PyArg_ParseTuple(args, "OOOOOOOOO", &density_o, &overlap_o,
            &dipole_o, &second_o, &centers_o, &atom_o, &shell_o,
            &reference_o, &valence_o)) return nullptr;
    auto* density = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        density_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* overlap = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        overlap_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* dipole = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        dipole_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* second = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        second_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* centers = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        centers_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* atom = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        atom_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* shell = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        shell_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* reference = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        reference_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* valence = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        valence_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto cleanup = [&]() {
        Py_XDECREF(density); Py_XDECREF(overlap); Py_XDECREF(dipole);
        Py_XDECREF(second); Py_XDECREF(centers); Py_XDECREF(atom); Py_XDECREF(shell);
        Py_XDECREF(reference); Py_XDECREF(valence);
    };
    if (!density || !overlap || !dipole || !second || !centers || !atom || !shell
        || !reference || !valence || PyArray_NDIM(density) != 2
        || PyArray_DIM(density, 0) != PyArray_DIM(density, 1)) {
        cleanup(); PyErr_SetString(PyExc_ValueError, "invalid dense SCC observable arrays");
        return nullptr;
    }
    const npy_intp n = PyArray_DIM(density, 0);
    const npy_intp npair = n*(n+1)/2;
    const npy_intp nshell = PyArray_SIZE(reference), natom = PyArray_SIZE(valence);
    if (PyArray_SIZE(overlap) != npair || PyArray_SIZE(dipole) != npair*3
        || PyArray_SIZE(second) != npair*9 || PyArray_SIZE(centers) != n*3
        || PyArray_SIZE(atom) != n
        || PyArray_SIZE(shell) != n) {
        cleanup(); PyErr_SetString(PyExc_ValueError, "incompatible dense SCC observable dimensions");
        return nullptr;
    }
    npy_intp shell_dims[1] = {nshell}, charge_dims[1] = {natom};
    npy_intp dipole_dims[2] = {natom, 3}, theta_dims[3] = {natom, 3, 3};
    auto* shell_q = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, shell_dims, NPY_DOUBLE, 0));
    auto* charges = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, charge_dims, NPY_DOUBLE, 0));
    auto* dipoles = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2, dipole_dims, NPY_DOUBLE, 0));
    auto* quadrupoles = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(3, theta_dims, NPY_DOUBLE, 0));
    if (!shell_q || !charges || !dipoles || !quadrupoles) {
        cleanup(); Py_XDECREF(shell_q); Py_XDECREF(charges);
        Py_XDECREF(dipoles); Py_XDECREF(quadrupoles); return PyErr_NoMemory();
    }
    const auto* p = static_cast<const double*>(PyArray_DATA(density));
    const auto* s = static_cast<const double*>(PyArray_DATA(overlap));
    const auto* dip = static_cast<const double*>(PyArray_DATA(dipole));
    const auto* sec = static_cast<const double*>(PyArray_DATA(second));
    const auto* center = static_cast<const double*>(PyArray_DATA(centers));
    const auto* atoms = static_cast<const std::int64_t*>(PyArray_DATA(atom));
    const auto* shells = static_cast<const std::int64_t*>(PyArray_DATA(shell));
    const auto* reference_data = static_cast<const double*>(PyArray_DATA(reference));
    const auto* valence_data = static_cast<const double*>(PyArray_DATA(valence));
    auto* qshell = static_cast<double*>(PyArray_DATA(shell_q));
    auto* qatom = static_cast<double*>(PyArray_DATA(charges));
    auto* atomic_mu = static_cast<double*>(PyArray_DATA(dipoles));
    auto* atomic_theta = static_cast<double*>(PyArray_DATA(quadrupoles));
    std::copy(reference_data, reference_data + nshell, qshell);
    std::copy(valence_data, valence_data + natom, qatom);
    Py_BEGIN_ALLOW_THREADS
    npy_intp packed = 0;
    for (npy_intp j = 0; j < n; ++j) {
        for (npy_intp i = 0; i <= j; ++i, ++packed) {
            const double density_j = p[j*n+i];
            const double density_i = p[i*n+j];
            const double overlap_value = s[packed];
            const auto accumulate = [&](npy_intp orbital, double density_value) {
                const auto a = atoms[orbital], sh = shells[orbital];
                qshell[sh] -= density_value * overlap_value;
                qatom[a] -= density_value * overlap_value;
                for (int axis = 0; axis < 3; ++axis) {
                    const double dmu = center[orbital*3+axis]*overlap_value
                        - dip[packed*3+axis];
                    atomic_mu[a*3+axis] += density_value*dmu;
                }
                for (int x = 0; x < 3; ++x) for (int y = 0; y < 3; ++y) {
                    const int component = x*3+y;
                    const double raw = dip[packed*3+x]*center[orbital*3+y]
                        + center[orbital*3+x]*dip[packed*3+y]
                        - overlap_value*center[orbital*3+x]*center[orbital*3+y]
                        - sec[packed*9+component];
                    atomic_theta[a*9+component] += density_value*raw;
                }
            };
            accumulate(j, density_j);
            if (i != j) accumulate(i, density_i);
        }
    }
    for (npy_intp a = 0; a < natom; ++a) {
        const double trace = atomic_theta[a*9] + atomic_theta[a*9+4]
            + atomic_theta[a*9+8];
        for (int x = 0; x < 3; ++x)
            for (int y = 0; y < 3; ++y) {
                const auto index = a*9 + x*3 + y;
                atomic_theta[index] = 1.5*atomic_theta[index]
                    - (x == y ? 0.5*trace : 0.0);
            }
    }
    Py_END_ALLOW_THREADS
    cleanup();
    return Py_BuildValue("NNNN", shell_q, charges, dipoles, quadrupoles);
}

PyObject* dense_scc_fock_native(PyObject*, PyObject* args) {
    PyObject *core_o, *overlap_o, *pair_overlap_o, *dipole_o, *second_o,
             *centers_o, *atom_o, *shell_o, *shell_potential_o, *qpot_o,
             *mupot_o, *thetapot_o, *d4_o;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOO", &core_o, &overlap_o,
            &pair_overlap_o, &dipole_o, &second_o, &centers_o, &atom_o,
            &shell_o, &shell_potential_o, &qpot_o, &mupot_o, &thetapot_o,
            &d4_o)) return nullptr;
    std::array<PyArrayObject*, 13> arrays = {
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(core_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(overlap_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(pair_overlap_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(dipole_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(second_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(centers_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(atom_o, NPY_INT64, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(shell_o, NPY_INT64, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(shell_potential_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(qpot_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(mupot_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(thetapot_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(d4_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY))};
    for (auto* array : arrays) if (!array) {
        for (auto* item : arrays) Py_XDECREF(item); return nullptr;
    }
    const npy_intp n = PyArray_DIM(arrays[0], 0);
    const npy_intp npair = n*(n+1)/2;
    npy_intp dims[2] = {n, n};
    auto* result = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(2, dims, NPY_DOUBLE, 0));
    if (!result || PyArray_SIZE(arrays[0]) != n*n || PyArray_SIZE(arrays[1]) != n*n
        || PyArray_SIZE(arrays[2]) != npair || PyArray_SIZE(arrays[3]) != npair*3
        || PyArray_SIZE(arrays[4]) != npair*9 || PyArray_SIZE(arrays[5]) != n*3
        || PyArray_SIZE(arrays[6]) != n || PyArray_SIZE(arrays[7]) != n) {
        for (auto* item : arrays) Py_XDECREF(item); Py_XDECREF(result);
        PyErr_SetString(PyExc_ValueError, "incompatible dense SCC Fock dimensions");
        return nullptr;
    }
    const auto* core = static_cast<const double*>(PyArray_DATA(arrays[0]));
    const auto* overlap = static_cast<const double*>(PyArray_DATA(arrays[1]));
    const auto* pair_overlap = static_cast<const double*>(PyArray_DATA(arrays[2]));
    const auto* dipole = static_cast<const double*>(PyArray_DATA(arrays[3]));
    const auto* second = static_cast<const double*>(PyArray_DATA(arrays[4]));
    const auto* centers = static_cast<const double*>(PyArray_DATA(arrays[5]));
    const auto* atom = static_cast<const std::int64_t*>(PyArray_DATA(arrays[6]));
    const auto* shell = static_cast<const std::int64_t*>(PyArray_DATA(arrays[7]));
    const auto* shell_potential = static_cast<const double*>(PyArray_DATA(arrays[8]));
    const auto* qpot = static_cast<const double*>(PyArray_DATA(arrays[9]));
    const auto* mupot = static_cast<const double*>(PyArray_DATA(arrays[10]));
    const auto* thetapot = static_cast<const double*>(PyArray_DATA(arrays[11]));
    const auto* d4 = static_cast<const double*>(PyArray_DATA(arrays[12]));
    auto* output = static_cast<double*>(PyArray_DATA(result));
    Py_BEGIN_ALLOW_THREADS
    npy_intp packed = 0;
    for (npy_intp j = 0; j < n; ++j) for (npy_intp i = 0; i <= j; ++i, ++packed) {
        const auto pair = j*n+i, transpose = i*n+j;
        const auto aj = atom[j], ai = atom[i];
        const double s = pair_overlap[packed];
        const auto endpoint = [&](npy_intp orbital, npy_intp a) {
            double raw = -qpot[a]*s;
            for (int axis = 0; axis < 3; ++axis) {
                const double dmu = centers[orbital*3+axis]*s
                    - dipole[packed*3+axis];
                raw += mupot[a*3+axis]*dmu;
            }
            double trace = 0.0, quadrupole_raw = 0.0;
            for (int x = 0; x < 3; ++x) for (int y = 0; y < 3; ++y) {
                const int component = x*3+y;
                const double moment = dipole[packed*3+x]*centers[orbital*3+y]
                    + centers[orbital*3+x]*dipole[packed*3+y]
                    - s*centers[orbital*3+x]*centers[orbital*3+y]
                    - second[packed*9+component];
                quadrupole_raw += thetapot[a*9+component]*moment;
                if (x == y) trace += moment;
            }
            double potential_trace = thetapot[a*9] + thetapot[a*9+4]
                + thetapot[a*9+8];
            return raw + 1.5*quadrupole_raw - 0.5*potential_trace*trace;
        };
        const double raw_j = endpoint(j, aj), raw_i = endpoint(i, ai);
        const double vj = shell_potential[shell[j]] + d4[aj];
        const double vi = shell_potential[shell[i]] + d4[ai];
        const double value = core[pair] - 0.5*overlap[pair]*(vj+vi)
            + 0.5*(raw_j+raw_i);
        output[pair] = value;
        output[transpose] = value;
    }
    Py_END_ALLOW_THREADS
    for (auto* item : arrays) Py_DECREF(item);
    return reinterpret_cast<PyObject*>(result);
}

PyObject* sparse_mayer_bond_orders_native(PyObject*,PyObject* args){
    long long size;PyObject *dr_o,*dc_o,*dv_o,*sr_o,*sc_o,*sv_o,*assignment_o;
    if(!PyArg_ParseTuple(args,"LOOOOOOO",&size,&dr_o,&dc_o,&dv_o,&sr_o,&sc_o,&sv_o,&assignment_o))return nullptr;
    SparseMatrixNative density,overlap;PyArrayObject *dr=nullptr,*dc=nullptr,*dv=nullptr,*sr=nullptr,*sc=nullptr,*sv=nullptr;
    auto* assignment=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(assignment_o,NPY_INT64,NPY_ARRAY_IN_ARRAY));
    if(!assignment||PyArray_SIZE(assignment)!=size
        ||!parse_sparse_matrix(size,dr_o,dc_o,dv_o,density,dr,dc,dv)
        ||!parse_sparse_matrix(size,sr_o,sc_o,sv_o,overlap,sr,sc,sv)){
        Py_XDECREF(assignment);Py_XDECREF(dr);Py_XDECREF(dc);Py_XDECREF(dv);
        Py_XDECREF(sr);Py_XDECREF(sc);Py_XDECREF(sv);
        PyErr_SetString(PyExc_ValueError,"invalid sparse Mayer arrays");return nullptr;
    }
    using Row=std::unordered_map<std::int64_t,double>;
    std::vector<Row> prows(size),srows(size);
    auto expand=[](const SparseMatrixNative& matrix,std::vector<Row>& rows){
        for(std::size_t index=0;index<matrix.values.size();++index){
            const auto i=matrix.rows[index],j=matrix.columns[index];const double value=matrix.values[index];
            rows[i][j]=value;rows[j][i]=value;
        }
    };
    expand(density,prows);expand(overlap,srows);
    auto dot=[](const Row& left,const Row& right){
        const Row* small=&left;const Row* large=&right;if(small->size()>large->size())std::swap(small,large);
        double value=0.0;for(const auto& [key,scalar]:*small){auto found=large->find(key);if(found!=large->end())value+=scalar*found->second;}return value;
    };
    const auto* atom=static_cast<const std::int64_t*>(PyArray_DATA(assignment));
    std::int64_t natoms=0;for(npy_intp i=0;i<size;++i)natoms=std::max(natoms,atom[i]+1);
    std::unordered_map<std::int64_t,double> atomic;atomic.reserve(overlap.values.size()/2+1);
    Py_BEGIN_ALLOW_THREADS
    for(std::size_t pair=0;pair<overlap.values.size();++pair){
        const auto left=overlap.rows[pair],right=overlap.columns[pair];
        if(atom[left]==atom[right])continue;
        const auto high=std::max(atom[left],atom[right]),low=std::min(atom[left],atom[right]);
        atomic[high*natoms+low]+=dot(prows[left],srows[right])*dot(prows[right],srows[left]);
    }
    Py_END_ALLOW_THREADS
    std::vector<std::tuple<std::int64_t,std::int64_t,double>> items;items.reserve(atomic.size());
    for(const auto& [key,value]:atomic)items.push_back({key/natoms,key%natoms,value});
    std::sort(items.begin(),items.end());npy_intp dims[1]={static_cast<npy_intp>(items.size())};
    auto* rows_out=reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1,dims,NPY_INT64,0));
    auto* cols_out=reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1,dims,NPY_INT64,0));
    auto* vals_out=reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1,dims,NPY_DOUBLE,0));
    if(!rows_out||!cols_out||!vals_out){Py_XDECREF(rows_out);Py_XDECREF(cols_out);Py_XDECREF(vals_out);return PyErr_NoMemory();}
    auto* ro=static_cast<std::int64_t*>(PyArray_DATA(rows_out));auto* co=static_cast<std::int64_t*>(PyArray_DATA(cols_out));auto* vo=static_cast<double*>(PyArray_DATA(vals_out));
    for(std::size_t i=0;i<items.size();++i){ro[i]=std::get<0>(items[i]);co[i]=std::get<1>(items[i]);vo[i]=std::get<2>(items[i]);}
    Py_DECREF(assignment);Py_DECREF(dr);Py_DECREF(dc);Py_DECREF(dv);Py_DECREF(sr);Py_DECREF(sc);Py_DECREF(sv);
    return Py_BuildValue("NNN",rows_out,cols_out,vals_out);
}

PyObject* sparse_triangles_native(PyObject*, PyObject* args) {
    long long size;PyObject *rows_o,*columns_o;
    if(!PyArg_ParseTuple(args,"LOO",&size,&rows_o,&columns_o))return nullptr;
    auto* rows=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(rows_o,NPY_INT64,NPY_ARRAY_IN_ARRAY));
    auto* columns=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(columns_o,NPY_INT64,NPY_ARRAY_IN_ARRAY));
    if(!rows||!columns||PyArray_SIZE(rows)!=PyArray_SIZE(columns)){
        Py_XDECREF(rows);Py_XDECREF(columns);PyErr_SetString(PyExc_ValueError,"invalid sparse edge arrays");return nullptr;
    }
    std::vector<std::unordered_set<std::int64_t>> lower(static_cast<std::size_t>(size));
    const auto* r=static_cast<const std::int64_t*>(PyArray_DATA(rows));
    const auto* c=static_cast<const std::int64_t*>(PyArray_DATA(columns));
    for(npy_intp edge=0;edge<PyArray_SIZE(rows);++edge)if(r[edge]!=c[edge])lower[r[edge]].insert(c[edge]);
    std::vector<std::array<std::int64_t,3>> triangles;
    for(std::int64_t high=0;high<size;++high){
        for(auto middle:lower[high]){
            for(auto low:lower[middle])if(lower[high].find(low)!=lower[high].end()){
                triangles.push_back({low,middle,high});
            }
        }
    }
    std::sort(triangles.begin(),triangles.end());
    npy_intp dims[2]={static_cast<npy_intp>(triangles.size()),3};
    auto* output=reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(2,dims,NPY_INT64,0));
    auto* out=static_cast<std::int64_t*>(PyArray_DATA(output));
    for(std::size_t index=0;index<triangles.size();++index)
        for(int component=0;component<3;++component)out[index*3+component]=triangles[index][component];
    Py_DECREF(rows);Py_DECREF(columns);return reinterpret_cast<PyObject*>(output);
}

PyObject* sparse_atm_energy_gradient_native(PyObject*, PyObject* args) {
    PyObject *coordinates_o,*rows_o,*columns_o,*c6_o,*c8_o,*dc6_left_o,*dc6_right_o;
    double s9,a1,a2,alpha;
    if(!PyArg_ParseTuple(args,"OOOOOOOdddd",&coordinates_o,&rows_o,&columns_o,
            &c6_o,&c8_o,&dc6_left_o,&dc6_right_o,&s9,&a1,&a2,&alpha))return nullptr;
    auto real_array=[](PyObject* object){return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));};
    auto integer_array=[](PyObject* object){return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object,NPY_INT64,NPY_ARRAY_IN_ARRAY));};
    auto* coordinates=real_array(coordinates_o);
    auto* rows=integer_array(rows_o);
    auto* columns=integer_array(columns_o);
    auto* c6=real_array(c6_o);
    auto* c8=real_array(c8_o);
    auto* dc6_left=real_array(dc6_left_o);
    auto* dc6_right=real_array(dc6_right_o);
    const npy_intp n=coordinates&&PyArray_NDIM(coordinates)==2
        ?PyArray_DIM(coordinates,0):0;
    const npy_intp edges=rows?PyArray_SIZE(rows):0;
    const bool valid=coordinates&&rows&&columns&&c6&&c8&&dc6_left&&dc6_right
        &&PyArray_DIM(coordinates,1)==3&&PyArray_NDIM(rows)==1
        &&PyArray_NDIM(columns)==1&&PyArray_NDIM(c6)==1&&PyArray_NDIM(c8)==1
        &&PyArray_NDIM(dc6_left)==1&&PyArray_NDIM(dc6_right)==1
        &&PyArray_SIZE(columns)==edges&&PyArray_SIZE(c6)==edges
        &&PyArray_SIZE(c8)==edges&&PyArray_SIZE(dc6_left)==edges
        &&PyArray_SIZE(dc6_right)==edges&&std::isfinite(s9)
        &&std::isfinite(a1)&&std::isfinite(a2)&&std::isfinite(alpha)&&alpha>0.0;
    if(!valid){
        Py_XDECREF(coordinates);Py_XDECREF(rows);Py_XDECREF(columns);Py_XDECREF(c6);
        Py_XDECREF(c8);Py_XDECREF(dc6_left);Py_XDECREF(dc6_right);
        PyErr_SetString(PyExc_ValueError,"invalid sparse ATM energy-gradient arrays");
        return nullptr;
    }
    const auto* row=static_cast<const std::int64_t*>(PyArray_DATA(rows));
    const auto* column=static_cast<const std::int64_t*>(PyArray_DATA(columns));
    const auto* xyz=static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* pc6=static_cast<const double*>(PyArray_DATA(c6));
    const auto* pc8=static_cast<const double*>(PyArray_DATA(c8));
    const auto* pdc6_left=static_cast<const double*>(PyArray_DATA(dc6_left));
    const auto* pdc6_right=static_cast<const double*>(PyArray_DATA(dc6_right));
    using Edge=std::pair<std::int64_t,std::int64_t>;
    std::vector<std::vector<Edge>> upper(static_cast<std::size_t>(n));
    for(npy_intp edge=0;edge<edges;++edge){
        if(row[edge]<0||row[edge]>=n||column[edge]<0||column[edge]>=row[edge]){
            Py_DECREF(coordinates);Py_DECREF(rows);Py_DECREF(columns);Py_DECREF(c6);
            Py_DECREF(c8);Py_DECREF(dc6_left);Py_DECREF(dc6_right);
            PyErr_SetString(PyExc_ValueError,"sparse ATM edges must be canonical lower-triangle pairs");
            return nullptr;
        }
        upper[static_cast<std::size_t>(column[edge])].push_back(
            {row[edge],static_cast<std::int64_t>(edge)});
    }
    for(auto& neighbors:upper)std::sort(neighbors.begin(),neighbors.end());
    std::vector<double> edge_distance2(static_cast<std::size_t>(edges));
    std::vector<double> edge_damping_ratio(static_cast<std::size_t>(edges));
    for(npy_intp edge=0;edge<edges;++edge){
        double distance2=0.0;
        for(int axis=0;axis<3;++axis){
            const double delta=xyz[row[edge]*3+axis]-xyz[column[edge]*3+axis];
            distance2+=delta*delta;
        }
        edge_distance2[static_cast<std::size_t>(edge)]=distance2;
        const double critical=a1*std::sqrt(pc8[edge]/pc6[edge])+a2;
        edge_damping_ratio[static_cast<std::size_t>(edge)]
            =std::pow(critical/std::sqrt(distance2),alpha/3.0);
    }

    npy_intp gradient_dims[2]={n,3};
    npy_intp atom_dims[1]={n};
    auto* gradient=reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2,gradient_dims,NPY_DOUBLE,0));
    auto* dedcn_array=reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1,atom_dims,NPY_DOUBLE,0));
    if(!gradient||!dedcn_array){
        Py_XDECREF(gradient);Py_XDECREF(dedcn_array);Py_DECREF(coordinates);
        Py_DECREF(rows);Py_DECREF(columns);Py_DECREF(c6);Py_DECREF(c8);
        Py_DECREF(dc6_left);Py_DECREF(dc6_right);return PyErr_NoMemory();
    }
    auto* grad=static_cast<double*>(PyArray_DATA(gradient));
    auto* dedcn=static_cast<double*>(PyArray_DATA(dedcn_array));
    std::int64_t triangle_count=0;
    double energy=0.0,minimum_edge2=std::numeric_limits<double>::infinity();
    double maximum_c9=0.0;
    Py_BEGIN_ALLOW_THREADS
    for(std::int64_t low=0;low<n;++low){
        const auto& low_neighbors=upper[static_cast<std::size_t>(low)];
        for(auto low_middle=low_neighbors.begin();low_middle!=low_neighbors.end();++low_middle){
            const std::int64_t middle=low_middle->first;
            const std::int64_t edge_lm=low_middle->second;
            const auto& middle_neighbors=upper[static_cast<std::size_t>(middle)];
            auto left=std::next(low_middle);
            auto right=middle_neighbors.begin();
            while(left!=low_neighbors.end()&&right!=middle_neighbors.end()){
                if(left->first<right->first){++left;continue;}
                if(right->first<left->first){++right;continue;}
                const std::int64_t high=left->first;
                const std::int64_t edge_lh=left->second;
                const std::int64_t edge_mh=right->second;
                const double c6ij=pc6[edge_mh],c6ik=pc6[edge_lh],c6jk=pc6[edge_lm];
                double vij[3],vik[3],vjk[3];
                const double r2ij=edge_distance2[static_cast<std::size_t>(edge_mh)];
                const double r2ik=edge_distance2[static_cast<std::size_t>(edge_lh)];
                const double r2jk=edge_distance2[static_cast<std::size_t>(edge_lm)];
                for(int axis=0;axis<3;++axis){
                    vij[axis]=xyz[middle*3+axis]-xyz[high*3+axis];
                    vik[axis]=xyz[low*3+axis]-xyz[high*3+axis];
                    vjk[axis]=xyz[low*3+axis]-xyz[middle*3+axis];
                }
                const double r2=r2ij*r2ik*r2jk,r1=std::sqrt(r2),r3=r2*r1,r5=r3*r2;
                const double ratio=edge_damping_ratio[static_cast<std::size_t>(edge_mh)]
                    *edge_damping_ratio[static_cast<std::size_t>(edge_lh)]
                    *edge_damping_ratio[static_cast<std::size_t>(edge_lm)];
                const double damping=1.0/(1.0+6.0*ratio);
                const double angular=0.375*(r2ij+r2jk-r2ik)
                    *(r2ij-r2jk+r2ik)*(-r2ij+r2jk+r2ik)/r5+1.0/r3;
                const double positive_c9=std::sqrt(c6ij*c6ik*c6jk);
                const double c9=-s9*positive_c9;
                const double de=angular*damping*c9;
                energy+=s9*angular*positive_c9*damping;
                maximum_c9=std::max(maximum_c9,positive_c9);
                minimum_edge2=std::min(minimum_edge2,
                    std::min(r2ij,std::min(r2ik,r2jk)));
                auto coefficient=[&](double ra,double rb,double rc){
                    const double dang=-0.375*(ra*ra*ra+ra*ra*(rb+rc)
                        +ra*(3.0*rb*rb+2.0*rb*rc+3.0*rc*rc)
                        -5.0*(rb-rc)*(rb-rc)*(rb+rc))/r5;
                    const double df=-2.0*alpha*ratio*damping*damping;
                    return c9*(-dang*damping+angular*df)/ra;
                };
                const double cij=coefficient(r2ij,r2jk,r2ik);
                const double cik=coefficient(r2ik,r2jk,r2ij);
                const double cjk=coefficient(r2jk,r2ik,r2ij);
                for(int axis=0;axis<3;++axis){
                    const double dgij=cij*vij[axis],dgik=cik*vik[axis],dgjk=cjk*vjk[axis];
                    grad[high*3+axis]-=dgij+dgik;
                    grad[middle*3+axis]+=dgij-dgjk;
                    grad[low*3+axis]+=dgik+dgjk;
                }
                const double scale=-0.5*de;
                dedcn[high]+=scale*(pdc6_left[edge_mh]/c6ij+pdc6_left[edge_lh]/c6ik);
                dedcn[middle]+=scale*(pdc6_right[edge_mh]/c6ij+pdc6_left[edge_lm]/c6jk);
                dedcn[low]+=scale*(pdc6_right[edge_lh]/c6ik+pdc6_right[edge_lm]/c6jk);
                ++triangle_count;++left;++right;
            }
        }
    }
    Py_END_ALLOW_THREADS
    const std::int64_t total_triangles=static_cast<std::int64_t>(n)
        *(n-1)*(n-2)/6;
    const std::int64_t omitted=total_triangles-triangle_count;
    const double tail_bound=triangle_count==0
        ?(total_triangles?std::numeric_limits<double>::infinity():0.0)
        :omitted*std::abs(s9)*4.0*maximum_c9
            /std::pow(std::sqrt(minimum_edge2),9.0);
    PyObject* output=PyTuple_New(6);
    if(!output){Py_DECREF(gradient);Py_DECREF(dedcn_array);Py_DECREF(coordinates);
        Py_DECREF(rows);Py_DECREF(columns);Py_DECREF(c6);Py_DECREF(c8);
        Py_DECREF(dc6_left);Py_DECREF(dc6_right);return nullptr;}
    PyTuple_SET_ITEM(output,0,PyFloat_FromDouble(energy));
    PyTuple_SET_ITEM(output,1,reinterpret_cast<PyObject*>(gradient));
    PyTuple_SET_ITEM(output,2,reinterpret_cast<PyObject*>(dedcn_array));
    PyTuple_SET_ITEM(output,3,PyLong_FromLongLong(omitted));
    PyTuple_SET_ITEM(output,4,PyFloat_FromDouble(tail_bound));
    PyTuple_SET_ITEM(output,5,PyLong_FromLongLong(triangle_count));
    Py_DECREF(coordinates);Py_DECREF(rows);Py_DECREF(columns);Py_DECREF(c6);Py_DECREF(c8);
    Py_DECREF(dc6_left);Py_DECREF(dc6_right);
    return output;
}

PyObject* closed_shell_stda_native(PyObject*, PyObject* args) {
    PyObject *qia_o, *qij_o, *qab_o, *gamma_j_o, *gamma_k_o, *gaps_o;
    double threshold, maximum, perturbative_threshold;
    int roots;
    if (!PyArg_ParseTuple(args, "OOOOOOdddi", &qia_o, &qij_o, &qab_o,
            &gamma_j_o, &gamma_k_o, &gaps_o, &threshold, &maximum,
            &perturbative_threshold, &roots)) return nullptr;
    std::array<PyArrayObject*, 6> arrays = {
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(qia_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(qij_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(qab_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(gamma_j_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(gamma_k_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(gaps_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY))};
    auto cleanup = [&]() { for (auto* array : arrays) Py_XDECREF(array); };
    if (std::any_of(arrays.begin(), arrays.end(), [](auto* value) { return value == nullptr; })) {
        cleanup(); return nullptr;
    }
    auto* qia = arrays[0]; auto* qij = arrays[1]; auto* qab = arrays[2];
    auto* gamma_j = arrays[3]; auto* gamma_k = arrays[4]; auto* gaps = arrays[5];
    const npy_intp nconfig = PyArray_DIM(qia, 0);
    const npy_intp natom = PyArray_DIM(qia, 1);
    const npy_intp nocc = PyArray_DIM(qij, 0);
    const npy_intp nvirt = PyArray_DIM(qab, 0);
    const bool valid = PyArray_NDIM(qia) == 2 && PyArray_NDIM(qij) == 3
        && PyArray_NDIM(qab) == 3 && PyArray_NDIM(gamma_j) == 2
        && PyArray_NDIM(gamma_k) == 2 && PyArray_NDIM(gaps) == 1
        && PyArray_DIM(qij, 1) == nocc && PyArray_DIM(qij, 2) == natom
        && PyArray_DIM(qab, 1) == nvirt && PyArray_DIM(qab, 2) == natom
        && PyArray_DIM(gamma_j, 0) == natom && PyArray_DIM(gamma_j, 1) == natom
        && PyArray_DIM(gamma_k, 0) == natom && PyArray_DIM(gamma_k, 1) == natom
        && PyArray_SIZE(gaps) == nconfig && nconfig == nocc*nvirt
        && roots > 0 && roots <= nconfig;
    if (!valid || nconfig > std::numeric_limits<int>::max()) {
        cleanup();
        PyErr_SetString(PyExc_ValueError, "incompatible native closed-shell sTDA arrays");
        return nullptr;
    }
#ifndef MATRIX_GFN2_NATIVE_LAPACK
    cleanup();
    PyErr_SetString(PyExc_RuntimeError, "native sTDA was built without LAPACK");
    return nullptr;
#else
    const auto* q = static_cast<const double*>(PyArray_DATA(qia));
    const auto* oo = static_cast<const double*>(PyArray_DATA(qij));
    const auto* vv = static_cast<const double*>(PyArray_DATA(qab));
    const auto* gj = static_cast<const double*>(PyArray_DATA(gamma_j));
    const auto* gk = static_cast<const double*>(PyArray_DATA(gamma_k));
    const auto* gap = static_cast<const double*>(PyArray_DATA(gaps));
    std::vector<double> response(static_cast<std::size_t>(nconfig)*nconfig, 0.0);
    std::vector<double> exchange_work(static_cast<std::size_t>(nconfig)*natom);
    std::vector<double> contracted_oo(static_cast<std::size_t>(nocc)*nocc*natom);
    std::vector<double> diagonal(nconfig), exchange_diagonal(nconfig);
    std::vector<npy_intp> primary, secondary, included;
    std::vector<double> lowering;
    int info = 0, found = 0;
    int operator_batches = 0;
    bool used_davidson = false;
    std::vector<double> values, vectors;
    Py_BEGIN_ALLOW_THREADS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        static_cast<int>(nconfig), static_cast<int>(natom), static_cast<int>(natom),
        1.0, q, static_cast<int>(natom), gk, static_cast<int>(natom),
        0.0, exchange_work.data(), static_cast<int>(natom));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        static_cast<int>(nconfig), static_cast<int>(nconfig), static_cast<int>(natom),
        2.0, exchange_work.data(), static_cast<int>(natom), q, static_cast<int>(natom),
        0.0, response.data(), static_cast<int>(nconfig));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        static_cast<int>(nocc*nocc), static_cast<int>(natom), static_cast<int>(natom),
        1.0, oo, static_cast<int>(natom), gj, static_cast<int>(natom),
        0.0, contracted_oo.data(), static_cast<int>(natom));
    for (npy_intp x = 0; x < nconfig; ++x) {
        const npy_intp i = x/nvirt, a = x%nvirt;
        for (npy_intp y = 0; y <= x; ++y) {
            const npy_intp j = y/nvirt, b = y%nvirt;
            double coulomb = 0.0;
            const auto* occupied_pair = contracted_oo.data() + (i*nocc + j)*natom;
            const auto* virtual_pair = vv + (a*nvirt + b)*natom;
            for (npy_intp atom = 0; atom < natom; ++atom)
                coulomb += occupied_pair[atom]*virtual_pair[atom];
            response[x*nconfig + y] -= coulomb;
            response[y*nconfig + x] = response[x*nconfig + y];
            if (x == y) exchange_diagonal[x] = 0.5*(response[x*nconfig + x] + coulomb);
        }
    }
    constexpr double hartree_to_ev = 27.211386245988;
    const double shift_max = 0.500/hartree_to_ev;
    const double shift_width = 0.10/hartree_to_ev;
    for (npy_intp x = 0; x < nconfig; ++x) {
        const double ratio = exchange_diagonal[x]/shift_width;
        diagonal[x] = gap[x] + response[x + x*nconfig]
            + shift_max/(1.0 + ratio*ratio*ratio*ratio);
        if (diagonal[x] <= threshold) primary.push_back(x);
        else if (diagonal[x] < maximum) secondary.push_back(x);
    }
    lowering.assign(primary.size(), 0.0);
    for (const auto candidate : secondary) {
        double perturbation = 0.0;
        for (std::size_t p = 0; p < primary.size(); ++p) {
            const double coupling = response[candidate + primary[p]*nconfig];
            perturbation += coupling*coupling
                /(diagonal[candidate] - diagonal[primary[p]] + 1.0e-10);
        }
        if (perturbation > perturbative_threshold) included.push_back(candidate);
        else for (std::size_t p = 0; p < primary.size(); ++p) {
            const double coupling = response[candidate + primary[p]*nconfig];
            lowering[p] += coupling*coupling
                /(diagonal[candidate] - diagonal[primary[p]] + 1.0e-10);
        }
    }
    std::vector<npy_intp> selected = primary;
    selected.insert(selected.end(), included.begin(), included.end());
    int dimension = static_cast<int>(selected.size());
    if (dimension > 0) {
        std::vector<double> matrix(static_cast<std::size_t>(dimension)*dimension);
        std::vector<double> selected_diagonal(dimension);
        for (int column = 0; column < dimension; ++column)
            for (int row = 0; row < dimension; ++row)
                matrix[row + column*dimension] = response[
                    selected[row] + selected[column]*nconfig];
        for (int index = 0; index < dimension; ++index) {
            matrix[index + index*dimension] = diagonal[selected[index]];
            if (index < static_cast<int>(primary.size()))
                matrix[index + index*dimension] -= lowering[index];
            selected_diagonal[index] = matrix[index + index*dimension];
        }
        const int requested = std::min(roots, dimension);
        if (dimension >= 150) {
            used_davidson = native_lowest_davidson(
                matrix, selected_diagonal, dimension, requested, 1.0e-8,
                100, std::max(32, 6*requested + 8), values, vectors,
                operator_batches);
            if (used_davidson) found = requested;
        }
        if (!used_davidson) {
            values.resize(requested);
            vectors.resize(static_cast<std::size_t>(dimension)*requested);
            std::vector<int> support(2*requested);
            char jobz = 'V', range = 'I', uplo = 'L';
            int lda = dimension, ldz = dimension, il = 1, iu = requested;
            double vl = 0.0, vu = 0.0, abstol = 0.0, work_query = 0.0;
            int lwork = -1, liwork = -1, iwork_query = 0;
            dsyevr_(&jobz, &range, &uplo, &dimension, matrix.data(), &lda,
                &vl, &vu, &il, &iu, &abstol, &found, values.data(), vectors.data(),
                &ldz, support.data(), &work_query, &lwork, &iwork_query, &liwork, &info);
            if (info == 0) {
                lwork = std::max(1, static_cast<int>(work_query));
                liwork = std::max(1, iwork_query);
                std::vector<double> work(lwork); std::vector<int> iwork(liwork);
                dsyevr_(&jobz, &range, &uplo, &dimension, matrix.data(), &lda,
                    &vl, &vu, &il, &iu, &abstol, &found, values.data(), vectors.data(),
                    &ldz, support.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
            }
        }
    }
    Py_END_ALLOW_THREADS
    if (primary.empty() || info != 0 || found <= 0) {
        cleanup();
        if (primary.empty()) PyErr_SetString(PyExc_ValueError,
            "no primary configurations survive the native sTDA threshold");
        else PyErr_Format(PyExc_RuntimeError, "native sTDA LAPACK failure %d", info);
        return nullptr;
    }
    npy_intp value_dim[1] = {found};
    npy_intp vector_dim[2] = {found, static_cast<npy_intp>(primary.size() + included.size())};
    npy_intp index_dim[1] = {static_cast<npy_intp>(primary.size() + included.size())};
    auto* values_out = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, value_dim, NPY_DOUBLE, 0));
    auto* vectors_out = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(2, vector_dim, NPY_DOUBLE, 0));
    auto* indices_out = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, index_dim, NPY_INT64, 0));
    if (!values_out || !vectors_out || !indices_out) {
        Py_XDECREF(values_out); Py_XDECREF(vectors_out); Py_XDECREF(indices_out); cleanup();
        return PyErr_NoMemory();
    }
    std::copy_n(values.data(), found, static_cast<double*>(PyArray_DATA(values_out)));
    auto* vector_out = static_cast<double*>(PyArray_DATA(vectors_out));
    const int dimension = static_cast<int>(primary.size() + included.size());
    for (int root = 0; root < found; ++root)
        for (int index = 0; index < dimension; ++index)
            vector_out[root*dimension + index] = vectors[index + root*dimension];
    auto* index_out = static_cast<std::int64_t*>(PyArray_DATA(indices_out));
    for (std::size_t index = 0; index < primary.size(); ++index) index_out[index] = primary[index];
    for (std::size_t index = 0; index < included.size(); ++index)
        index_out[primary.size() + index] = included[index];
    cleanup();
    return Py_BuildValue("NNNnnsi", values_out, vectors_out, indices_out,
        static_cast<Py_ssize_t>(primary.size()), static_cast<Py_ssize_t>(included.size()),
        used_davidson ? "native-davidson" : "native-lapack", operator_batches);
#endif
}

PyObject* localized_projected_orbitals_native(PyObject*, PyObject* args) {
    int size, rank, complement_flag; double threshold;
    PyObject *rows_o, *columns_o, *values_o, *domain_rows_o, *domain_columns_o;
    if (!PyArg_ParseTuple(args, "iOOOOOidp", &size, &rows_o, &columns_o, &values_o,
            &domain_rows_o, &domain_columns_o, &rank, &threshold,
            &complement_flag)) return nullptr;
    auto* rows = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        rows_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* columns = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        columns_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* values = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        values_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* domain_rows = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        domain_rows_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto* domain_columns = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        domain_columns_o, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    auto cleanup = [&]() {
        Py_XDECREF(rows); Py_XDECREF(columns); Py_XDECREF(values);
        Py_XDECREF(domain_rows); Py_XDECREF(domain_columns);
    };
    if (!rows || !columns || !values || !domain_rows || !domain_columns
        || PyArray_SIZE(rows) != PyArray_SIZE(columns)
        || PyArray_SIZE(rows) != PyArray_SIZE(values)
        || PyArray_SIZE(domain_rows) != PyArray_SIZE(domain_columns)
        || size < 1 || rank < 1 || rank >= size || threshold < 0.0) {
        cleanup(); PyErr_SetString(PyExc_ValueError,
            "invalid sparse projected-orbital inputs"); return nullptr;
    }
    const auto* pr = static_cast<const std::int64_t*>(PyArray_DATA(rows));
    const auto* pc = static_cast<const std::int64_t*>(PyArray_DATA(columns));
    const auto* pv = static_cast<const double*>(PyArray_DATA(values));
    const auto* dr = static_cast<const std::int64_t*>(PyArray_DATA(domain_rows));
    const auto* dc = static_cast<const std::int64_t*>(PyArray_DATA(domain_columns));
    std::vector<std::unordered_map<int, double>> projector(size);
    std::vector<std::vector<int>> neighbors(size);
    std::vector<double> residual(size, 0.0);
    for (npy_intp index = 0; index < PyArray_SIZE(rows); ++index) {
        const int left = static_cast<int>(pr[index]), right = static_cast<int>(pc[index]);
        if (left < 0 || left >= size || right < 0 || right >= size) {
            cleanup(); PyErr_SetString(PyExc_ValueError,
                "projector pair index is outside the AO space"); return nullptr;
        }
        projector[left][right] = pv[index]; projector[right][left] = pv[index];
        if (left == right) residual[left] = pv[index];
    }
    for (npy_intp index = 0; index < PyArray_SIZE(domain_rows); ++index) {
        const int left = static_cast<int>(dr[index]), right = static_cast<int>(dc[index]);
        if (left < 0 || left >= size || right < 0 || right >= size) continue;
        neighbors[left].push_back(right);
        if (left != right) neighbors[right].push_back(left);
    }
    for (int row = 0; row < size; ++row) {
        if (neighbors[row].empty()) neighbors[row].push_back(row);
        std::sort(neighbors[row].begin(), neighbors[row].end());
        neighbors[row].erase(std::unique(neighbors[row].begin(), neighbors[row].end()),
            neighbors[row].end());
    }
    const bool complement = complement_flag != 0;
    const int target_rank = rank;
    if (complement) for (int row = 0; row < size; ++row)
        residual[row] = 1.0 - residual[row];
    using HeapItem = std::pair<double, int>;
    std::priority_queue<HeapItem> heap;
    for (int row = 0; row < size; ++row) heap.push({residual[row], row});
    std::vector<std::vector<std::pair<int, double>>> row_members(size);
    std::vector<std::int64_t> offsets(1, 0), output_rows;
    std::vector<double> output_values;
    bool failed = false;
    Py_BEGIN_ALLOW_THREADS
    for (int orbital = 0; orbital < target_rank; ++orbital) {
        while (!heap.empty() && std::abs(heap.top().first - residual[heap.top().second]) > 1.0e-14)
            heap.pop();
        if (heap.empty() || heap.top().first <= 1.0e-12) { failed = true; break; }
        const int pivot = heap.top().second; heap.pop();
        const double denominator = std::sqrt(residual[pivot]);
        std::vector<std::pair<int, double>> column;
        for (const int row : neighbors[pivot]) {
            double base = 0.0;
            const auto found = projector[pivot].find(row);
            if (found != projector[pivot].end()) base = found->second;
            if (complement) base = (row == pivot ? 1.0 : 0.0) - base;
            const auto& left = row_members[row]; const auto& right = row_members[pivot];
            std::size_t il = 0, ir = 0; double correction = 0.0;
            while (il < left.size() && ir < right.size()) {
                if (left[il].first < right[ir].first) ++il;
                else if (right[ir].first < left[il].first) ++ir;
                else { correction += left[il].second*right[ir].second; ++il; ++ir; }
            }
            const double value = (base - correction)/denominator;
            if (row == pivot || std::abs(value) >= threshold) column.push_back({row, value});
        }
        for (const auto& [row, value] : column) {
            output_rows.push_back(row); output_values.push_back(value);
            row_members[row].push_back({orbital, value});
            residual[row] = std::max(0.0, residual[row] - value*value);
            heap.push({residual[row], row});
        }
        offsets.push_back(static_cast<std::int64_t>(output_rows.size()));
    }
    Py_END_ALLOW_THREADS
    if (failed || static_cast<int>(offsets.size()) != target_rank + 1) {
        cleanup(); PyErr_SetString(PyExc_RuntimeError,
            "localized projected-orbital factorization lost rank"); return nullptr;
    }
    double discarded = 0.0;
    for (const auto value : residual) discarded += std::max(0.0, value);
    npy_intp offset_dim[1] = {static_cast<npy_intp>(offsets.size())};
    npy_intp value_dim[1] = {static_cast<npy_intp>(output_rows.size())};
    auto* offsets_out = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, offset_dim, NPY_INT64, 0));
    auto* rows_out = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, value_dim, NPY_INT64, 0));
    auto* values_out = reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, value_dim, NPY_DOUBLE, 0));
    if (!offsets_out || !rows_out || !values_out) {
        Py_XDECREF(offsets_out); Py_XDECREF(rows_out); Py_XDECREF(values_out);
        cleanup(); return PyErr_NoMemory();
    }
    std::copy(offsets.begin(), offsets.end(),
        static_cast<std::int64_t*>(PyArray_DATA(offsets_out)));
    std::copy(output_rows.begin(), output_rows.end(),
        static_cast<std::int64_t*>(PyArray_DATA(rows_out)));
    std::copy(output_values.begin(), output_values.end(),
        static_cast<double*>(PyArray_DATA(values_out)));
    cleanup();
    return Py_BuildValue("NNNd", offsets_out, rows_out, values_out, std::sqrt(discarded));
}

PyObject* sparse_orbital_quadratic_forms_native(PyObject*, PyObject* args) {
    int size; PyObject *frows_o, *fcolumns_o, *fvalues_o, *offsets_o, *rows_o, *values_o;
    if (!PyArg_ParseTuple(args, "iOOOOOO", &size, &frows_o, &fcolumns_o,
            &fvalues_o, &offsets_o, &rows_o, &values_o)) return nullptr;
    std::array<PyArrayObject*, 6> arrays = {
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(frows_o, NPY_INT64, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(fcolumns_o, NPY_INT64, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(fvalues_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(offsets_o, NPY_INT64, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(rows_o, NPY_INT64, NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(values_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY))};
    auto cleanup = [&]() { for (auto* array : arrays) Py_XDECREF(array); };
    if (std::any_of(arrays.begin(), arrays.end(), [](auto* item){return item == nullptr;})
        || PyArray_SIZE(arrays[0]) != PyArray_SIZE(arrays[1])
        || PyArray_SIZE(arrays[0]) != PyArray_SIZE(arrays[2])
        || PyArray_SIZE(arrays[4]) != PyArray_SIZE(arrays[5])
        || PyArray_SIZE(arrays[3]) < 2) {
        cleanup(); PyErr_SetString(PyExc_ValueError,
            "invalid sparse orbital quadratic-form arrays"); return nullptr;
    }
    std::vector<std::unordered_map<int,double>> fock(size);
    const auto* fr = static_cast<const std::int64_t*>(PyArray_DATA(arrays[0]));
    const auto* fc = static_cast<const std::int64_t*>(PyArray_DATA(arrays[1]));
    const auto* fv = static_cast<const double*>(PyArray_DATA(arrays[2]));
    for (npy_intp index=0; index<PyArray_SIZE(arrays[0]); ++index) {
        fock[fr[index]][fc[index]]=fv[index]; fock[fc[index]][fr[index]]=fv[index];
    }
    const auto* offsets = static_cast<const std::int64_t*>(PyArray_DATA(arrays[3]));
    const auto* rows = static_cast<const std::int64_t*>(PyArray_DATA(arrays[4]));
    const auto* values = static_cast<const double*>(PyArray_DATA(arrays[5]));
    const npy_intp count = PyArray_SIZE(arrays[3])-1;
    npy_intp dims[1]={count};
    auto* output=reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1,dims,NPY_DOUBLE,0));
    if(!output){cleanup();return PyErr_NoMemory();}
    auto* out=static_cast<double*>(PyArray_DATA(output));
    Py_BEGIN_ALLOW_THREADS
    for(npy_intp orbital=0;orbital<count;++orbital){
        double value=0.0;
        for(auto left=offsets[orbital];left<offsets[orbital+1];++left)
            for(auto right=offsets[orbital];right<offsets[orbital+1];++right){
                const auto found=fock[rows[left]].find(static_cast<int>(rows[right]));
                if(found!=fock[rows[left]].end())value+=values[left]*found->second*values[right];
            }
        out[orbital]=value;
    }
    Py_END_ALLOW_THREADS
    cleanup(); return reinterpret_cast<PyObject*>(output);
}

PyObject* localized_stda_pairs_native(PyObject*, PyObject* args) {
    PyObject* configuration;
    if (!PyArg_ParseTuple(args, "O", &configuration) || !PyDict_Check(configuration)) {
        PyErr_SetString(PyExc_TypeError, "localized sTDA input must be a dictionary");
        return nullptr;
    }
    constexpr std::array<const char*, 19> names = {
        "ao_atom", "atomic_numbers", "coordinates", "hardness",
        "fock_rows", "fock_columns", "fock_values",
        "occupied_offsets", "occupied_rows", "occupied_values",
        "virtual_offsets", "virtual_rows", "virtual_values",
        "configuration_occupied", "configuration_virtual",
        "edge_rows", "edge_columns", "gaps", "value_threshold"};
    constexpr std::array<int, 19> types = {
        NPY_INT64, NPY_INT64, NPY_DOUBLE, NPY_DOUBLE,
        NPY_INT64, NPY_INT64, NPY_DOUBLE,
        NPY_INT64, NPY_INT64, NPY_DOUBLE,
        NPY_INT64, NPY_INT64, NPY_DOUBLE,
        NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64, NPY_DOUBLE, NPY_DOUBLE};
    std::array<PyArrayObject*, 19> values{};
    for (std::size_t index=0;index<names.size();++index) {
        auto* item=PyDict_GetItemString(configuration,names[index]);
        if(!item){for(auto* array:values)Py_XDECREF(array);PyErr_Format(PyExc_KeyError,
            "missing localized sTDA array: %s",names[index]);return nullptr;}
        values[index]=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            item,types[index],NPY_ARRAY_IN_ARRAY));
        if(!values[index]){for(auto* array:values)Py_XDECREF(array);return nullptr;}
    }
    auto cleanup=[&](){for(auto* array:values)Py_XDECREF(array);};
    const auto* ao=static_cast<const std::int64_t*>(PyArray_DATA(values[0]));
    const auto* numbers=static_cast<const std::int64_t*>(PyArray_DATA(values[1]));
    const auto* coordinates=static_cast<const double*>(PyArray_DATA(values[2]));
    const auto* hardness=static_cast<const double*>(PyArray_DATA(values[3]));
    const auto* fr=static_cast<const std::int64_t*>(PyArray_DATA(values[4]));
    const auto* fc=static_cast<const std::int64_t*>(PyArray_DATA(values[5]));
    const auto* fv=static_cast<const double*>(PyArray_DATA(values[6]));
    const auto* oo=static_cast<const std::int64_t*>(PyArray_DATA(values[7]));
    const auto* ore=static_cast<const std::int64_t*>(PyArray_DATA(values[8]));
    const auto* ove=static_cast<const double*>(PyArray_DATA(values[9]));
    const auto* vo=static_cast<const std::int64_t*>(PyArray_DATA(values[10]));
    const auto* vre=static_cast<const std::int64_t*>(PyArray_DATA(values[11]));
    const auto* vve=static_cast<const double*>(PyArray_DATA(values[12]));
    const auto* ci=static_cast<const std::int64_t*>(PyArray_DATA(values[13]));
    const auto* ca=static_cast<const std::int64_t*>(PyArray_DATA(values[14]));
    const auto* er=static_cast<const std::int64_t*>(PyArray_DATA(values[15]));
    const auto* ec=static_cast<const std::int64_t*>(PyArray_DATA(values[16]));
    const auto* gaps=static_cast<const double*>(PyArray_DATA(values[17]));
    const double value_threshold=*static_cast<const double*>(PyArray_DATA(values[18]));
    const int nao=static_cast<int>(PyArray_SIZE(values[0]));
    const int natom=static_cast<int>(PyArray_SIZE(values[1]));
    const int nconfig=static_cast<int>(PyArray_SIZE(values[13]));
    const int nedge=static_cast<int>(PyArray_SIZE(values[15]));
    if(PyArray_SIZE(values[14])!=nconfig||PyArray_SIZE(values[17])!=nconfig
        ||PyArray_SIZE(values[16])!=nedge||PyArray_SIZE(values[2])!=3*natom
        ||PyArray_SIZE(values[3])!=natom||PyArray_SIZE(values[4])!=PyArray_SIZE(values[5])
        ||PyArray_SIZE(values[4])!=PyArray_SIZE(values[6])){
        cleanup();PyErr_SetString(PyExc_ValueError,"inconsistent localized sTDA arrays");return nullptr;
    }
    std::vector<std::unordered_map<int,double>> fock(nao);
    for(npy_intp index=0;index<PyArray_SIZE(values[4]);++index){
        fock[fr[index]][fc[index]]=fv[index];fock[fc[index]][fr[index]]=fv[index];
    }
    using AtomVector=std::vector<std::pair<int,double>>;
    std::unordered_map<std::uint64_t,AtomVector> occupied_cache,virtual_cache,transition_cache;
    std::unordered_map<std::uint64_t,double> occupied_fock_cache,virtual_fock_cache;
    auto key=[](int left,int right){if(left>right)std::swap(left,right);
        return(static_cast<std::uint64_t>(left)<<32)|static_cast<std::uint32_t>(right);};
    auto atom_pair=[&](bool occupied,int left,int right)->const AtomVector&{
        auto& cache=occupied?occupied_cache:virtual_cache;const auto cache_key=key(left,right);
        auto found=cache.find(cache_key);if(found!=cache.end())return found->second;
        const auto* offsets=occupied?oo:vo;const auto* rows=occupied?ore:vre;
        const auto* coefficients=occupied?ove:vve;auto il=offsets[left],ir=offsets[right];
        std::unordered_map<int,double> atoms;
        while(il<offsets[left+1]&&ir<offsets[right+1]){
            if(rows[il]<rows[ir])++il;else if(rows[ir]<rows[il])++ir;else{
                atoms[ao[rows[il]]]+=coefficients[il]*coefficients[ir];++il;++ir;}}
        AtomVector result(atoms.begin(),atoms.end());std::sort(result.begin(),result.end());
        return cache.emplace(cache_key,std::move(result)).first->second;};
    auto transition=[&](int occupied,int virtual_orbital)->const AtomVector&{
        const auto cache_key=(static_cast<std::uint64_t>(occupied)<<32)
            |static_cast<std::uint32_t>(virtual_orbital);auto found=transition_cache.find(cache_key);
        if(found!=transition_cache.end())return found->second;auto il=oo[occupied],ir=vo[virtual_orbital];
        std::unordered_map<int,double> atoms;
        while(il<oo[occupied+1]&&ir<vo[virtual_orbital+1]){
            if(ore[il]<vre[ir])++il;else if(vre[ir]<ore[il])++ir;else{
                atoms[ao[ore[il]]]+=ove[il]*vve[ir];++il;++ir;}}
        AtomVector result(atoms.begin(),atoms.end());std::sort(result.begin(),result.end());
        return transition_cache.emplace(cache_key,std::move(result)).first->second;};
    auto fock_pair=[&](bool occupied,int left,int right){auto& cache=occupied?occupied_fock_cache:virtual_fock_cache;
        const auto cache_key=key(left,right);auto found=cache.find(cache_key);if(found!=cache.end())return found->second;
        const auto* offsets=occupied?oo:vo;const auto* rows=occupied?ore:vre;const auto* coefficients=occupied?ove:vve;
        double result=0.0;for(auto l=offsets[left];l<offsets[left+1];++l)for(auto r=offsets[right];r<offsets[right+1];++r){
            const auto item=fock[rows[l]].find(static_cast<int>(rows[r]));if(item!=fock[rows[l]].end())result+=coefficients[l]*item->second*coefficients[r];}
        cache.emplace(cache_key,result);return result;};
    auto contract=[&](const AtomVector& left,const AtomVector& right,bool exchange){double result=0.0;
        const double exponent=exchange?2.0:4.0;for(const auto& [la,lv]:left)for(const auto& [ra,rv]:right){
            const double dx=coordinates[3*la]-coordinates[3*ra],dy=coordinates[3*la+1]-coordinates[3*ra+1],dz=coordinates[3*la+2]-coordinates[3*ra+2];
            const double distance=std::sqrt(dx*dx+dy*dy+dz*dz);double screened=0.5*(hardness[la]+hardness[ra]);if(!exchange)screened*=0.5;
            const double gamma=1.0/std::pow(std::pow(distance,exponent)+std::pow(screened,-exponent),1.0/exponent);result+=lv*gamma*rv;}return result;};
    auto coupling=[&](int left,int right){const int oi=ci[left],oj=ci[right],va=ca[left],vb=ca[right];
        double result=2.0*contract(transition(oi,va),transition(oj,vb),true)
            -contract(atom_pair(true,oi,oj),atom_pair(false,va,vb),false);
        if(oi==oj)result+=fock_pair(false,va,vb);if(va==vb)result-=fock_pair(true,oi,oj);return result;};
    npy_intp diagonal_dim[1]={nconfig},edge_dim[1]={nedge};
    auto* diagonal_out=reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1,diagonal_dim,NPY_DOUBLE,0));
    auto* edge_out=reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1,edge_dim,NPY_DOUBLE,0));
    if(!diagonal_out||!edge_out){Py_XDECREF(diagonal_out);Py_XDECREF(edge_out);cleanup();return PyErr_NoMemory();}
    auto* diagonal=static_cast<double*>(PyArray_DATA(diagonal_out));auto* edges=static_cast<double*>(PyArray_DATA(edge_out));
    constexpr double hartree_to_ev=27.211386245988;const double shift_max=.5/hartree_to_ev,shift_width=.1/hartree_to_ev;
    Py_BEGIN_ALLOW_THREADS
    for(int index=0;index<nconfig;++index){const auto& q=transition(ci[index],ca[index]);const double exchange=contract(q,q,true);
        diagonal[index]=coupling(index,index)+shift_max/(1.0+std::pow(exchange/shift_width,4));}
    for(int index=0;index<nedge;++index){const double item=coupling(er[index],ec[index]);edges[index]=std::abs(item)>=value_threshold?item:0.0;}
    Py_END_ALLOW_THREADS
    cleanup();return Py_BuildValue("NN",diagonal_out,edge_out);
}

PyObject* sparse_symmetric_davidson_native(PyObject*, PyObject* args) {
    PyObject *diagonal_o, *rows_o, *columns_o, *values_o;
    int roots; double tolerance;
    if(!PyArg_ParseTuple(args,"OOOOid",&diagonal_o,&rows_o,&columns_o,&values_o,
            &roots,&tolerance))return nullptr;
    std::array<PyArrayObject*,4> arrays={
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(diagonal_o,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(rows_o,NPY_INT64,NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(columns_o,NPY_INT64,NPY_ARRAY_IN_ARRAY)),
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(values_o,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY))};
    auto cleanup=[&](){for(auto* array:arrays)Py_XDECREF(array);};
    if(std::any_of(arrays.begin(),arrays.end(),[](auto* item){return item==nullptr;})
        ||PyArray_SIZE(arrays[1])!=PyArray_SIZE(arrays[2])
        ||PyArray_SIZE(arrays[1])!=PyArray_SIZE(arrays[3])){
        cleanup();PyErr_SetString(PyExc_ValueError,"invalid sparse Davidson arrays");return nullptr;}
    const int dimension=static_cast<int>(PyArray_SIZE(arrays[0]));
    if(roots<1||roots>=dimension||tolerance<=0.0){cleanup();PyErr_SetString(
        PyExc_ValueError,"invalid sparse Davidson dimensions");return nullptr;}
    const auto* diagonal_data=static_cast<const double*>(PyArray_DATA(arrays[0]));
    const auto* rows=static_cast<const std::int64_t*>(PyArray_DATA(arrays[1]));
    const auto* columns=static_cast<const std::int64_t*>(PyArray_DATA(arrays[2]));
    const auto* edge_values=static_cast<const double*>(PyArray_DATA(arrays[3]));
    const auto edges=PyArray_SIZE(arrays[1]);
    std::vector<double> diagonal(diagonal_data,diagonal_data+dimension),eigenvalues,eigenvectors;
    int batches=0;bool converged=false;
    auto apply=[&](const double* input,int block,double* output){
        for(int vector=0;vector<block;++vector){
            const auto* x=input+static_cast<std::size_t>(vector)*dimension;
            auto* y=output+static_cast<std::size_t>(vector)*dimension;
            for(int row=0;row<dimension;++row)y[row]=diagonal[row]*x[row];
            for(npy_intp edge=0;edge<edges;++edge){const int left=rows[edge],right=columns[edge];
                const double value=edge_values[edge];y[left]+=value*x[right];y[right]+=value*x[left];}
        }};
    Py_BEGIN_ALLOW_THREADS
    converged=native_lowest_davidson_action(apply,diagonal,dimension,roots,tolerance,
        120,std::max(32,6*roots+8),eigenvalues,eigenvectors,batches);
    Py_END_ALLOW_THREADS
    npy_intp value_dim[1]={roots},vector_dim[2]={roots,dimension};
    auto* values_out=reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1,value_dim,NPY_DOUBLE,0));
    auto* vectors_out=reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(2,vector_dim,NPY_DOUBLE,0));
    if(!values_out||!vectors_out){Py_XDECREF(values_out);Py_XDECREF(vectors_out);cleanup();return PyErr_NoMemory();}
    std::copy_n(eigenvalues.data(),roots,static_cast<double*>(PyArray_DATA(values_out)));
    auto* vector_out=static_cast<double*>(PyArray_DATA(vectors_out));
    for(int root=0;root<roots;++root)for(int row=0;row<dimension;++row)
        vector_out[root*dimension+row]=eigenvectors[row+root*dimension];
    cleanup();return Py_BuildValue("NNii",values_out,vectors_out,converged?1:0,batches);
}

PyObject* sparse_rotation_to_ao_native(PyObject*, PyObject* args) {
    int nao;
    PyObject *offsets_o, *orbital_rows_o, *orbital_values_o;
    PyObject *left_o, *right_o, *rotation_values_o;
    PyObject *ao_rows_o, *ao_columns_o, *diagonal_o;
    if (!PyArg_ParseTuple(args, "iOOOOOOOOO", &nao, &offsets_o,
            &orbital_rows_o, &orbital_values_o, &left_o, &right_o,
            &rotation_values_o, &ao_rows_o, &ao_columns_o, &diagonal_o))
        return nullptr;
    constexpr std::array<int, 9> types = {
        NPY_INT64, NPY_INT64, NPY_DOUBLE, NPY_INT64, NPY_INT64,
        NPY_DOUBLE, NPY_INT64, NPY_INT64, NPY_DOUBLE};
    const std::array<PyObject*, 9> inputs = {
        offsets_o, orbital_rows_o, orbital_values_o, left_o, right_o,
        rotation_values_o, ao_rows_o, ao_columns_o, diagonal_o};
    std::array<PyArrayObject*, 9> arrays{};
    for (std::size_t index = 0; index < arrays.size(); ++index) {
        arrays[index] = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            inputs[index], types[index], NPY_ARRAY_IN_ARRAY));
        if (!arrays[index]) {
            for (auto* array : arrays) Py_XDECREF(array);
            return nullptr;
        }
    }
    auto cleanup = [&]() { for (auto* array : arrays) Py_XDECREF(array); };
    const int nmo = static_cast<int>(PyArray_SIZE(arrays[0])) - 1;
    const npy_intp orbital_nnz = PyArray_SIZE(arrays[1]);
    const npy_intp rotations = PyArray_SIZE(arrays[3]);
    const npy_intp ao_pairs = PyArray_SIZE(arrays[6]);
    if (nao < 1 || nmo < 1 || PyArray_SIZE(arrays[2]) != orbital_nnz
        || PyArray_SIZE(arrays[4]) != rotations
        || PyArray_SIZE(arrays[5]) != rotations
        || PyArray_SIZE(arrays[7]) != ao_pairs
        || PyArray_SIZE(arrays[8]) != nmo) {
        cleanup(); PyErr_SetString(PyExc_ValueError,
            "inconsistent sparse orbital-to-AO arrays"); return nullptr;
    }
    const auto* offsets = static_cast<const std::int64_t*>(PyArray_DATA(arrays[0]));
    const auto* orbital_rows = static_cast<const std::int64_t*>(PyArray_DATA(arrays[1]));
    const auto* orbital_values = static_cast<const double*>(PyArray_DATA(arrays[2]));
    const auto* left = static_cast<const std::int64_t*>(PyArray_DATA(arrays[3]));
    const auto* right = static_cast<const std::int64_t*>(PyArray_DATA(arrays[4]));
    const auto* rotation_values = static_cast<const double*>(PyArray_DATA(arrays[5]));
    const auto* ao_rows = static_cast<const std::int64_t*>(PyArray_DATA(arrays[6]));
    const auto* ao_columns = static_cast<const std::int64_t*>(PyArray_DATA(arrays[7]));
    const auto* diagonal = static_cast<const double*>(PyArray_DATA(arrays[8]));
    if (offsets[0] != 0 || offsets[nmo] != orbital_nnz) {
        cleanup(); PyErr_SetString(PyExc_ValueError,
            "sparse orbital offsets do not span their values"); return nullptr;
    }
    auto key = [](int first, int second) {
        if (first < second) std::swap(first, second);
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(first)) << 32)
            | static_cast<std::uint32_t>(second);
    };
    std::unordered_map<std::uint64_t, int> pair_index;
    pair_index.reserve(static_cast<std::size_t>(ao_pairs) * 2);
    for (npy_intp pair = 0; pair < ao_pairs; ++pair) {
        if (ao_rows[pair] < ao_columns[pair] || ao_rows[pair] >= nao
            || ao_columns[pair] < 0) {
            cleanup(); PyErr_SetString(PyExc_ValueError,
                "AO pair pattern is not a valid lower triangle"); return nullptr;
        }
        pair_index[key(static_cast<int>(ao_rows[pair]),
            static_cast<int>(ao_columns[pair]))] = static_cast<int>(pair);
    }
    npy_intp output_dim[1] = {ao_pairs};
    auto* output = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(
        1, output_dim, NPY_DOUBLE, 0));
    if (!output) { cleanup(); return PyErr_NoMemory(); }
    auto* result = static_cast<double*>(PyArray_DATA(output));
    bool invalid = false;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp edge = 0; edge < rotations && !invalid; ++edge) {
        const int i = static_cast<int>(left[edge]);
        const int j = static_cast<int>(right[edge]);
        if (i < 0 || j < 0 || i >= nmo || j >= nmo || i == j) {
            invalid = true; break;
        }
        for (auto a = offsets[i]; a < offsets[i + 1]; ++a)
            for (auto b = offsets[j]; b < offsets[j + 1]; ++b) {
                const int mu = static_cast<int>(orbital_rows[a]);
                const int nu = static_cast<int>(orbital_rows[b]);
                const auto found = pair_index.find(key(mu, nu));
                if (found == pair_index.end()) continue;
                result[found->second] += (mu == nu ? 2.0 : 1.0)
                    * rotation_values[edge] * orbital_values[a] * orbital_values[b];
            }
    }
    for (int orbital = 0; orbital < nmo && !invalid; ++orbital)
        for (auto a = offsets[orbital]; a < offsets[orbital + 1]; ++a)
            for (auto b = offsets[orbital]; b <= a; ++b) {
                const int mu = static_cast<int>(orbital_rows[a]);
                const int nu = static_cast<int>(orbital_rows[b]);
                const auto found = pair_index.find(key(mu, nu));
                if (found != pair_index.end())
                    result[found->second] += diagonal[orbital]
                        * orbital_values[a] * orbital_values[b];
            }
    Py_END_ALLOW_THREADS
    cleanup();
    if (invalid) {
        Py_DECREF(output); PyErr_SetString(PyExc_ValueError,
            "orbital rotation index is invalid"); return nullptr;
    }
    return reinterpret_cast<PyObject*>(output);
}

PyObject* sparse_ao_to_rotation_native(PyObject*, PyObject* args) {
    int nao;
    PyObject *offsets_o, *orbital_rows_o, *orbital_values_o;
    PyObject *ao_rows_o, *ao_columns_o, *ao_values_o, *left_o, *right_o;
    if (!PyArg_ParseTuple(args, "iOOOOOOOO", &nao, &offsets_o,
            &orbital_rows_o, &orbital_values_o, &ao_rows_o, &ao_columns_o,
            &ao_values_o, &left_o, &right_o)) return nullptr;
    constexpr std::array<int, 8> types = {
        NPY_INT64, NPY_INT64, NPY_DOUBLE, NPY_INT64,
        NPY_INT64, NPY_DOUBLE, NPY_INT64, NPY_INT64};
    const std::array<PyObject*, 8> inputs = {
        offsets_o, orbital_rows_o, orbital_values_o, ao_rows_o,
        ao_columns_o, ao_values_o, left_o, right_o};
    std::array<PyArrayObject*, 8> arrays{};
    for (std::size_t index = 0; index < arrays.size(); ++index) {
        arrays[index] = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            inputs[index], types[index], NPY_ARRAY_IN_ARRAY));
        if (!arrays[index]) {
            for (auto* array : arrays) Py_XDECREF(array);
            return nullptr;
        }
    }
    auto cleanup = [&]() { for (auto* array : arrays) Py_XDECREF(array); };
    const int nmo = static_cast<int>(PyArray_SIZE(arrays[0])) - 1;
    const npy_intp orbital_nnz = PyArray_SIZE(arrays[1]);
    const npy_intp ao_pairs = PyArray_SIZE(arrays[3]);
    const npy_intp rotations = PyArray_SIZE(arrays[6]);
    if (nao < 1 || nmo < 1 || PyArray_SIZE(arrays[2]) != orbital_nnz
        || PyArray_SIZE(arrays[4]) != ao_pairs
        || PyArray_SIZE(arrays[5]) != ao_pairs
        || PyArray_SIZE(arrays[7]) != rotations) {
        cleanup(); PyErr_SetString(PyExc_ValueError,
            "inconsistent sparse AO-to-orbital arrays"); return nullptr;
    }
    const auto* offsets = static_cast<const std::int64_t*>(PyArray_DATA(arrays[0]));
    const auto* orbital_rows = static_cast<const std::int64_t*>(PyArray_DATA(arrays[1]));
    const auto* orbital_values = static_cast<const double*>(PyArray_DATA(arrays[2]));
    const auto* ao_rows = static_cast<const std::int64_t*>(PyArray_DATA(arrays[3]));
    const auto* ao_columns = static_cast<const std::int64_t*>(PyArray_DATA(arrays[4]));
    const auto* ao_values = static_cast<const double*>(PyArray_DATA(arrays[5]));
    const auto* left = static_cast<const std::int64_t*>(PyArray_DATA(arrays[6]));
    const auto* right = static_cast<const std::int64_t*>(PyArray_DATA(arrays[7]));
    auto key = [](int first, int second) {
        if (first < second) std::swap(first, second);
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(first)) << 32)
            | static_cast<std::uint32_t>(second);
    };
    std::unordered_map<std::uint64_t, double> ao_matrix;
    ao_matrix.reserve(static_cast<std::size_t>(ao_pairs) * 2);
    for (npy_intp pair = 0; pair < ao_pairs; ++pair)
        ao_matrix[key(static_cast<int>(ao_rows[pair]),
            static_cast<int>(ao_columns[pair]))] = ao_values[pair];
    npy_intp rotation_dim[1] = {rotations}, diagonal_dim[1] = {nmo};
    auto* selected_out = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(
        1, rotation_dim, NPY_DOUBLE, 0));
    auto* diagonal_out = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(
        1, diagonal_dim, NPY_DOUBLE, 0));
    if (!selected_out || !diagonal_out) {
        Py_XDECREF(selected_out); Py_XDECREF(diagonal_out); cleanup();
        return PyErr_NoMemory();
    }
    auto bilinear = [&](int i, int j) {
        double value = 0.0;
        for (auto a = offsets[i]; a < offsets[i + 1]; ++a)
            for (auto b = offsets[j]; b < offsets[j + 1]; ++b) {
                const auto found = ao_matrix.find(key(
                    static_cast<int>(orbital_rows[a]),
                    static_cast<int>(orbital_rows[b])));
                if (found != ao_matrix.end())
                    value += orbital_values[a] * found->second * orbital_values[b];
            }
        return value;
    };
    auto* selected = static_cast<double*>(PyArray_DATA(selected_out));
    auto* diagonal = static_cast<double*>(PyArray_DATA(diagonal_out));
    bool invalid = false;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp edge = 0; edge < rotations; ++edge) {
        const int i = static_cast<int>(left[edge]), j = static_cast<int>(right[edge]);
        if (i < 0 || j < 0 || i >= nmo || j >= nmo || i == j) {
            invalid = true; break;
        }
        selected[edge] = bilinear(i, j);
    }
    for (int orbital = 0; orbital < nmo && !invalid; ++orbital)
        diagonal[orbital] = bilinear(orbital, orbital);
    Py_END_ALLOW_THREADS
    cleanup();
    if (invalid) {
        Py_DECREF(selected_out); Py_DECREF(diagonal_out);
        PyErr_SetString(PyExc_ValueError, "orbital rotation index is invalid");
        return nullptr;
    }
    return Py_BuildValue("NN", selected_out, diagonal_out);
}

PyObject* context_extreme_singlepoint_native(PyObject*, PyObject* args) {
    PyObject *capsule, *cartesian_centers, *coordinates, *configuration;
    PyObject* density_o;
    int electrons, multiplicity, maximum_cycles;
    double temperature, wavefunction_tolerance, energy_tolerance;
    double mixer_tolerance = -1.0;
    if (!PyArg_ParseTuple(args, "OOOOOiidddi|d", &capsule,
            &cartesian_centers, &coordinates, &configuration, &density_o,
            &electrons, &multiplicity, &temperature, &wavefunction_tolerance,
            &energy_tolerance, &maximum_cycles, &mixer_tolerance)) return nullptr;
    auto* context = static_cast<IntegralContextNative*>(
        PyCapsule_GetPointer(capsule, integral_context_capsule_name));
    if (!context || !PyDict_Check(configuration)) return nullptr;
    auto item = [&](const char* name) -> PyObject* {
        auto* value = PyDict_GetItemString(configuration, name);
        if (!value && !PyErr_Occurred())
            PyErr_Format(PyExc_KeyError, "missing extreme-kernel item: %s", name);
        return value;
    };
    auto scalar = [&](const char* name, double& value) -> bool {
        auto* object = item(name);
        if (!object) return false;
        value = PyFloat_AsDouble(object);
        return !PyErr_Occurred();
    };
    double anis_a3, anis_a5, s6, s8, d4_a1, d4_a2, s9;
    if (!scalar("anis_a3", anis_a3) || !scalar("anis_a5", anis_a5)
        || !scalar("s6", s6) || !scalar("s8", s8)
        || !scalar("d4_a1", d4_a1) || !scalar("d4_a2", d4_a2)
        || !scalar("s9", s9)) return nullptr;

    PyObject *geometry = nullptr, *small = nullptr, *diis = nullptr;
    PyObject *neutral_d4 = nullptr, *repulsion = nullptr, *atm = nullptr;
    PyObject *post = nullptr, *h0 = nullptr, *camm = nullptr;
    PyObject *variational = nullptr, *cn_gradient = nullptr;
    PyArrayObject *density = nullptr, *weighted = nullptr;
    PyArrayObject *charged_c6 = nullptr, *charged_c8 = nullptr;
    PyArrayObject *neutral_c6 = nullptr, *neutral_c8 = nullptr;
    PyArrayObject *neutral_dc6 = nullptr, *radius_derivative = nullptr;
    PyArrayObject *left_atoms = nullptr, *right_atoms = nullptr;
    PyArrayObject *orbital_rows = nullptr, *orbital_columns = nullptr;
    PyArrayObject *pair_scalar = nullptr, *gradient = nullptr;
    PyArrayObject *zero_charges = nullptr;
    auto fail = [&]() -> PyObject* {
        Py_XDECREF(geometry); Py_XDECREF(small); Py_XDECREF(diis);
        Py_XDECREF(neutral_d4); Py_XDECREF(repulsion); Py_XDECREF(atm);
        Py_XDECREF(post); Py_XDECREF(h0); Py_XDECREF(camm);
        Py_XDECREF(variational); Py_XDECREF(cn_gradient);
        Py_XDECREF(density); Py_XDECREF(weighted);
        Py_XDECREF(charged_c6); Py_XDECREF(charged_c8);
        Py_XDECREF(neutral_c6); Py_XDECREF(neutral_c8);
        Py_XDECREF(neutral_dc6); Py_XDECREF(radius_derivative);
        Py_XDECREF(left_atoms); Py_XDECREF(right_atoms);
        Py_XDECREF(orbital_rows); Py_XDECREF(orbital_columns);
        Py_XDECREF(pair_scalar); Py_XDECREF(gradient); Py_XDECREF(zero_charges);
        return nullptr;
    };

    auto* one = PyLong_FromLong(1);
    npy_intp origin_dims[1] = {3};
    auto* origin = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(
        1, origin_dims, NPY_DOUBLE, 0));
    auto* geometry_args = one && origin ? PyTuple_Pack(10, capsule,
        cartesian_centers, reinterpret_cast<PyObject*>(origin), coordinates,
        item("coordination_radii"), item("orbital_atom"), item("orbital_shell"),
        item("reference_population"), item("valence"), one) : nullptr;
    Py_XDECREF(one); Py_XDECREF(origin);
    geometry = geometry_args
        ? context_prepare_dense_geometry_native(nullptr, geometry_args) : nullptr;
    Py_XDECREF(geometry_args);
    if (!geometry) return fail();
    auto* overlap = reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(geometry, 5));
    auto* coordination = PyTuple_GET_ITEM(geometry, 7);
    const npy_intp norb = PyArray_DIM(overlap, 0);
    auto* xyz = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        coordinates, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    const npy_intp natom = xyz && PyArray_NDIM(xyz) == 2
        ? PyArray_DIM(xyz, 0) : 0;
    if (!xyz || PyArray_DIM(xyz, 1) != 3) { Py_XDECREF(xyz); return fail(); }
    Py_DECREF(xyz);

    density = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        density_o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSURECOPY));
    if (!density || PyArray_SIZE(density) != norb*norb) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError,
            "incompatible extreme-kernel restart density");
        return fail();
    }
    auto* p0 = static_cast<double*>(PyArray_DATA(density));
    const auto* s = static_cast<const double*>(PyArray_DATA(overlap));
    double population = 0.0;
    for (npy_intp index = 0; index < norb*norb; ++index)
        population += p0[index]*s[index];
    if (!(population > std::numeric_limits<double>::min())) {
        PyErr_SetString(PyExc_ValueError, "empty extreme-kernel restart density");
        return fail();
    }
    const double population_scale = electrons/population;
    for (npy_intp index = 0; index < norb*norb; ++index)
        p0[index] *= population_scale;

    auto* small_args = PyTuple_Pack(4, capsule, coordinates, coordination,
        configuration);
    small = small_args
        ? context_prepare_small_scf_native(nullptr, small_args) : nullptr;
    Py_XDECREF(small_args);
    if (!small) return fail();
    auto* diis_args = Py_BuildValue("OOiidddid", capsule, density, electrons,
        multiplicity, temperature, wavefunction_tolerance, energy_tolerance,
        maximum_cycles, mixer_tolerance);
    diis = diis_args ? context_resident_diis_native(nullptr, diis_args) : nullptr;
    Py_XDECREF(diis_args);
    if (!diis) return fail();
    if (!PyObject_IsTrue(PyTuple_GET_ITEM(diis, 0))) {
        Py_DECREF(geometry); geometry = nullptr;
        Py_DECREF(small); small = nullptr;
        Py_DECREF(diis); diis = nullptr;
        Py_DECREF(density); density = nullptr;
        Py_RETURN_NONE;
    }
    auto* final_density = reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(diis, 1));
    auto* energies = reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(diis, 2));
    auto* coefficients = reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(diis, 3));
    auto* occupations = reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(diis, 4));
    auto* restart_density = PyTuple_GET_ITEM(diis, 7);
    auto* observables = context->last_observables;
    auto* anisotropic = context->last_anisotropic;
    auto* charged_d4 = context->last_d4;
    if (!observables || !anisotropic || !charged_d4) {
        PyErr_SetString(PyExc_RuntimeError, "extreme kernel lost final SCF state");
        return fail();
    }
    auto* shell_q = reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(observables, 0));
    auto* charges = PyTuple_GET_ITEM(observables, 1);
    auto* dipoles = PyTuple_GET_ITEM(observables, 2);
    auto* quadrupoles = PyTuple_GET_ITEM(observables, 3);

    npy_intp atom_dims[1] = {natom};
    zero_charges = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, atom_dims, NPY_DOUBLE, 0));
    auto* neutral_args = zero_charges ? PyTuple_Pack(13, zero_charges,
        context->scf_arrays[SCF_D4_FREQUENCY],
        context->scf_arrays[SCF_D4_WEIGHTS],
        context->scf_arrays[SCF_D4_WEIGHT_DERIVATIVES],
        context->scf_arrays[SCF_D4_REFERENCE_CHARGES],
        context->scf_arrays[SCF_D4_REFERENCES], context->scf_arrays[SCF_D4_MASK],
        context->scf_arrays[SCF_D4_EFFECTIVE], context->scf_arrays[SCF_D4_HARDNESS],
        context->scf_arrays[SCF_D4_ROWS], context->scf_arrays[SCF_D4_COLUMNS],
        context->scf_arrays[SCF_D4_RATIO], context->scf_arrays[SCF_D4_PAIR_FACTOR])
        : nullptr;
    neutral_d4 = neutral_args
        ? d4_two_body_pairs_native(nullptr, neutral_args) : nullptr;
    Py_XDECREF(neutral_args);
    if (!neutral_d4) return fail();

    auto dense_d4 = [&](PyObject* packed, PyArrayObject** c6_out,
                        PyArrayObject** c8_out, PyArrayObject** dc6_out) -> bool {
        npy_intp matrix_dims[2] = {natom,natom};
        *c6_out = reinterpret_cast<PyArrayObject*>(
            PyArray_ZEROS(2, matrix_dims, NPY_DOUBLE, 0));
        *c8_out = reinterpret_cast<PyArrayObject*>(
            PyArray_ZEROS(2, matrix_dims, NPY_DOUBLE, 0));
        if (dc6_out) {
            npy_intp tensor_dims[3] = {natom,natom,natom};
            *dc6_out = reinterpret_cast<PyArrayObject*>(
                PyArray_ZEROS(3, tensor_dims, NPY_DOUBLE, 0));
        }
        if (!*c6_out || !*c8_out || (dc6_out && !*dc6_out)) return false;
        const auto* rows = static_cast<const std::int64_t*>(PyArray_DATA(
            context->scf_arrays[SCF_D4_ROWS]));
        const auto* columns = static_cast<const std::int64_t*>(PyArray_DATA(
            context->scf_arrays[SCF_D4_COLUMNS]));
        const npy_intp npair = PyArray_SIZE(context->scf_arrays[SCF_D4_ROWS]);
        const auto* pair_c6 = static_cast<const double*>(PyArray_DATA(
            reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(packed, 3))));
        const auto* pair_c8 = static_cast<const double*>(PyArray_DATA(
            reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(packed, 4))));
        const auto* cn_left = static_cast<const double*>(PyArray_DATA(
            reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(packed, 5))));
        const auto* cn_right = static_cast<const double*>(PyArray_DATA(
            reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(packed, 6))));
        auto* c6 = static_cast<double*>(PyArray_DATA(*c6_out));
        auto* c8 = static_cast<double*>(PyArray_DATA(*c8_out));
        auto* dc6 = dc6_out ? static_cast<double*>(PyArray_DATA(*dc6_out)) : nullptr;
        for (npy_intp pair = 0; pair < npair; ++pair) {
            const auto left = rows[pair], right = columns[pair];
            c6[left*natom+right] = c6[right*natom+left] = pair_c6[pair];
            c8[left*natom+right] = c8[right*natom+left] = pair_c8[pair];
            if (dc6) {
                dc6[(left*natom+right)*natom+left] = cn_left[pair];
                dc6[(right*natom+left)*natom+left] = cn_left[pair];
                dc6[(left*natom+right)*natom+right] = cn_right[pair];
                dc6[(right*natom+left)*natom+right] = cn_right[pair];
            }
        }
        return true;
    };
    if (!dense_d4(charged_d4, &charged_c6, &charged_c8, nullptr)
        || !dense_d4(neutral_d4, &neutral_c6, &neutral_c8, &neutral_dc6))
        return fail();

    auto* rep_args = Py_BuildValue("OOOOd", coordinates, item("atomic_numbers"),
        item("repulsion_alpha"), item("effective_nuclear_charges"), 24.0);
    repulsion = rep_args
        ? repulsion_energy_gradient_native(nullptr, rep_args) : nullptr;
    Py_XDECREF(rep_args);
    if (!repulsion) return fail();
    double energy = context->last_scf_energy
        + PyFloat_AsDouble(PyTuple_GET_ITEM(repulsion, 0));

    npy_intp grad_dims[2] = {natom,3};
    PyObject* atm_gradient_object = nullptr;
    if (natom >= 3) {
        auto* atm_args = Py_BuildValue("OOOOOOdddd", coordinates, neutral_c6,
            neutral_c8, neutral_dc6, item("d4_coordination_radii"),
            item("electronegativity"), s9, d4_a1, d4_a2, 16.0);
        atm = atm_args
            ? atm_three_body_energy_gradient_native(nullptr, atm_args) : nullptr;
        Py_XDECREF(atm_args);
        if (!atm) return fail();
        energy += PyFloat_AsDouble(PyTuple_GET_ITEM(atm, 0));
        atm_gradient_object = PyTuple_GET_ITEM(atm, 1);
    } else {
        atm_gradient_object = reinterpret_cast<PyObject*>(PyArray_ZEROS(
            2, grad_dims, NPY_DOUBLE, 0));
        if (!atm_gradient_object) return fail();
    }

    radius_derivative = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, atom_dims, NPY_DOUBLE, 0));
    auto* base = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        item("anis_base_radius"), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* valence_cn = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        item("anis_valence_coordination"), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* cn = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        coordination, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!radius_derivative || !base || !valence_cn || !cn) {
        Py_XDECREF(base); Py_XDECREF(valence_cn); Py_XDECREF(cn);
        if (natom < 3) Py_DECREF(atm_gradient_object);
        return fail();
    }
    auto* rd = static_cast<double*>(PyArray_DATA(radius_derivative));
    const auto* bv = static_cast<const double*>(PyArray_DATA(base));
    const auto* vv = static_cast<const double*>(PyArray_DATA(valence_cn));
    const auto* cv = static_cast<const double*>(PyArray_DATA(cn));
    for (npy_intp atom = 0; atom < natom; ++atom) {
        const double exponential = std::exp(-4.0*(cv[atom]-vv[atom]-1.2));
        rd[atom] = 4.0*(5.0-bv[atom])*exponential
            /((1.0+exponential)*(1.0+exponential));
    }
    Py_DECREF(base); Py_DECREF(valence_cn); Py_DECREF(cn);
    auto* post_args = Py_BuildValue("OOOOOOOOOOOOOOOdddddd", coordinates,
        shell_q, item("shell_atom"), item("shell_hardness"), charges, dipoles,
        quadrupoles, PyTuple_GET_ITEM(small, 1), item("coordination_radii"),
        radius_derivative, charged_c6, charged_c8, PyTuple_GET_ITEM(charged_d4, 2),
        item("d4_coordination_radii"), item("electronegativity"), anis_a3,
        anis_a5, s6, s8, d4_a1, d4_a2);
    post = post_args
        ? post_scf_explicit_gradient_native(nullptr, post_args) : nullptr;
    Py_XDECREF(post_args);
    if (!post) {
        if (natom < 3) Py_DECREF(atm_gradient_object);
        return fail();
    }

    auto* left_derivatives = reinterpret_cast<PyArrayObject*>(
        PyTuple_GET_ITEM(geometry, 3));
    auto* right_derivatives = reinterpret_cast<PyArrayObject*>(
        PyTuple_GET_ITEM(geometry, 4));
    const npy_intp npair = PyArray_SIZE(context->spherical_left);
    npy_intp derivative_dims[2] = {npair,3};
    auto* left_ds = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(2, derivative_dims, NPY_DOUBLE, 0));
    auto* right_ds = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(2, derivative_dims, NPY_DOUBLE, 0));
    if (!left_ds || !right_ds) {
        Py_XDECREF(left_ds); Py_XDECREF(right_ds);
        if (natom < 3) Py_DECREF(atm_gradient_object);
        return fail();
    }
    const auto* ld_all = static_cast<const double*>(PyArray_DATA(left_derivatives));
    const auto* rd_all = static_cast<const double*>(PyArray_DATA(right_derivatives));
    auto* lds = static_cast<double*>(PyArray_DATA(left_ds));
    auto* rds = static_cast<double*>(PyArray_DATA(right_ds));
    for (npy_intp pair = 0; pair < npair; ++pair)
        for (int axis = 0; axis < 3; ++axis) {
            lds[pair*3+axis] = ld_all[pair*39+axis];
            rds[pair*3+axis] = rd_all[pair*39+axis];
        }
    npy_intp pair_dims[1] = {npair};
    orbital_rows = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, pair_dims, NPY_INT32, 0));
    orbital_columns = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, pair_dims, NPY_INT32, 0));
    left_atoms = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, pair_dims, NPY_INT32, 0));
    right_atoms = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, pair_dims, NPY_INT32, 0));
    const auto* rows = static_cast<const std::int64_t*>(
        PyArray_DATA(context->spherical_left));
    const auto* columns = static_cast<const std::int64_t*>(
        PyArray_DATA(context->spherical_right));
    const auto* orbital_atom = static_cast<const std::int64_t*>(
        PyArray_DATA(context->atom));
    if (!orbital_rows || !orbital_columns || !left_atoms || !right_atoms) {
        Py_DECREF(left_ds); Py_DECREF(right_ds);
        if (natom < 3) Py_DECREF(atm_gradient_object);
        return fail();
    }
    auto* la = static_cast<std::int32_t*>(PyArray_DATA(left_atoms));
    auto* ra = static_cast<std::int32_t*>(PyArray_DATA(right_atoms));
    auto* orows = static_cast<std::int32_t*>(PyArray_DATA(orbital_rows));
    auto* ocolumns = static_cast<std::int32_t*>(PyArray_DATA(orbital_columns));
    for (npy_intp pair = 0; pair < npair; ++pair) {
        orows[pair] = rows[pair];
        ocolumns[pair] = columns[pair];
        la[pair] = static_cast<std::int32_t>(orbital_atom[rows[pair]]);
        ra[pair] = static_cast<std::int32_t>(orbital_atom[columns[pair]]);
    }
    auto* h0_args = PyTuple_Pack(8, capsule, orbital_rows,
        orbital_columns, context->pair_overlap, left_ds, right_ds,
        coordinates, coordination);
    h0 = h0_args ? context_h0_derivatives_native(nullptr, h0_args) : nullptr;
    Py_XDECREF(h0_args);
    if (!h0) {
        Py_DECREF(left_ds); Py_DECREF(right_ds);
        if (natom < 3) Py_DECREF(atm_gradient_object);
        return fail();
    }
    auto* camm_args = PyTuple_Pack(10, context->pair_overlap,
        context->pair_dipole, left_derivatives, right_derivatives,
        left_atoms, right_atoms, coordinates, PyTuple_GET_ITEM(anisotropic, 1),
        PyTuple_GET_ITEM(anisotropic, 2), PyTuple_GET_ITEM(anisotropic, 3));
    camm = camm_args
        ? anisotropic_fock_derivatives_native(nullptr, camm_args) : nullptr;
    Py_XDECREF(camm_args);
    if (!camm) {
        Py_DECREF(left_ds); Py_DECREF(right_ds);
        if (natom < 3) Py_DECREF(atm_gradient_object);
        return fail();
    }

    pair_scalar = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(1, pair_dims, NPY_DOUBLE, 0));
    if (!pair_scalar) {
        Py_DECREF(left_ds); Py_DECREF(right_ds);
        if (natom < 3) Py_DECREF(atm_gradient_object);
        return fail();
    }
    const npy_intp nshell = PyArray_SIZE(shell_q);
    const auto* qshell = static_cast<const double*>(PyArray_DATA(shell_q));
    const auto* interaction = static_cast<const double*>(PyArray_DATA(
        context->scf_arrays[SCF_INTERACTION]));
    const auto* third = static_cast<const double*>(PyArray_DATA(
        context->scf_arrays[SCF_THIRD]));
    const auto* shell_index = static_cast<const std::int64_t*>(
        PyArray_DATA(context->shell));
    const auto* d4_charge = static_cast<const double*>(PyArray_DATA(
        reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(charged_d4, 1))));
    std::vector<double> shell_potential(nshell);
    for (npy_intp shell = 0; shell < nshell; ++shell) {
        double value = third[shell]*qshell[shell]*qshell[shell];
        for (npy_intp other = 0; other < nshell; ++other)
            value += interaction[shell*nshell+other]*qshell[other];
        shell_potential[shell] = value;
    }
    auto* ps = static_cast<double*>(PyArray_DATA(pair_scalar));
    for (npy_intp pair = 0; pair < npair; ++pair) {
        const auto left = rows[pair], right = columns[pair];
        const double lv = shell_potential[shell_index[left]]
            + d4_charge[orbital_atom[left]];
        const double rv = shell_potential[shell_index[right]]
            + d4_charge[orbital_atom[right]];
        ps[pair] = -0.5*(lv+rv);
    }

    npy_intp matrix_dims[2] = {norb,norb};
    weighted = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(2, matrix_dims, NPY_DOUBLE, 0));
    if (!weighted) {
        Py_DECREF(left_ds); Py_DECREF(right_ds);
        if (natom < 3) Py_DECREF(atm_gradient_object);
        return fail();
    }
    const auto* c = static_cast<const double*>(PyArray_DATA(coefficients));
    const auto* eps = static_cast<const double*>(PyArray_DATA(energies));
    const auto* occ = static_cast<const double*>(PyArray_DATA(occupations));
    auto* w = static_cast<double*>(PyArray_DATA(weighted));
    std::fill(w, w+norb*norb, 0.0);
    for (npy_intp row = 0; row < norb; ++row)
        for (npy_intp column = 0; column < norb; ++column) {
            double value = 0.0;
            for (npy_intp orbital = 0; orbital < norb; ++orbital)
                value += c[row*norb+orbital]*c[column*norb+orbital]
                    *occ[orbital]*eps[orbital];
            w[row*norb+column] = value;
        }
    auto* natom_o = PyLong_FromLongLong(natom);
    auto* thread_o = PyLong_FromLong(1);
    auto* variational_args = natom_o && thread_o ? PyTuple_Pack(18, capsule,
        final_density, weighted, orbital_rows, orbital_columns,
        left_atoms, right_atoms, PyTuple_GET_ITEM(h0, 0), PyTuple_GET_ITEM(h0, 1),
        PyTuple_GET_ITEM(camm, 0), PyTuple_GET_ITEM(camm, 1), pair_scalar,
        left_ds, right_ds, PyTuple_GET_ITEM(h0, 2), PyTuple_GET_ITEM(h0, 3),
        natom_o, thread_o) : nullptr;
    Py_XDECREF(natom_o); Py_XDECREF(thread_o);
    variational = variational_args
        ? context_variational_gradient_dense_native(nullptr, variational_args)
        : nullptr;
    Py_XDECREF(variational_args); Py_DECREF(left_ds); Py_DECREF(right_ds);
    if (!variational) {
        if (natom < 3) Py_DECREF(atm_gradient_object);
        return fail();
    }
    auto* cn_args = Py_BuildValue("OOOOd", PyTuple_GET_ITEM(variational, 1),
        coordinates, item("coordination_radii"), Py_None, 24.0);
    cn_gradient = cn_args
        ? coordination_response_gradient_native(nullptr, cn_args) : nullptr;
    Py_XDECREF(cn_args);
    if (!cn_gradient) {
        if (natom < 3) Py_DECREF(atm_gradient_object);
        return fail();
    }

    gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_EMPTY(2, grad_dims, NPY_DOUBLE, 0));
    if (!gradient) {
        if (natom < 3) Py_DECREF(atm_gradient_object);
        return fail();
    }
    auto* g = static_cast<double*>(PyArray_DATA(gradient));
    const auto* electronic_g = static_cast<const double*>(PyArray_DATA(
        reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(variational, 0))));
    const auto* cn_g = static_cast<const double*>(PyArray_DATA(
        reinterpret_cast<PyArrayObject*>(cn_gradient)));
    const auto* post_g = static_cast<const double*>(PyArray_DATA(
        reinterpret_cast<PyArrayObject*>(post)));
    const auto* rep_g = static_cast<const double*>(PyArray_DATA(
        reinterpret_cast<PyArrayObject*>(PyTuple_GET_ITEM(repulsion, 1))));
    const auto* atm_g = static_cast<const double*>(PyArray_DATA(
        reinterpret_cast<PyArrayObject*>(atm_gradient_object)));
    for (npy_intp index = 0; index < natom*3; ++index)
        g[index] = electronic_g[index]+cn_g[index]+post_g[index]
            +rep_g[index]+atm_g[index];
    if (natom < 3) Py_DECREF(atm_gradient_object);

    constexpr double boltzmann_hartree_per_kelvin = 3.166808578545117e-6;
    const double thermal = boltzmann_hartree_per_kelvin*temperature;
    if (multiplicity == 1 && thermal > 0.0) {
        for (npy_intp orbital = 0; orbital < norb; ++orbital) {
            const double value = 0.5*occ[orbital];
            if (value > 0.0 && value < 1.0)
                energy += 2.0*thermal*(value*std::log(value)
                    +(1.0-value)*std::log(1.0-value));
        }
    }
    if (PyErr_Occurred()) return fail();

    Py_INCREF(charges); Py_INCREF(final_density); Py_INCREF(restart_density);
    Py_INCREF(energies); Py_INCREF(occupations);
    auto* result = Py_BuildValue("dNNNNNNiddd", energy, gradient, charges,
        final_density, restart_density, energies, occupations,
        static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(diis, 12))),
        PyFloat_AsDouble(PyTuple_GET_ITEM(diis, 8)),
        PyFloat_AsDouble(PyTuple_GET_ITEM(diis, 9)),
        PyFloat_AsDouble(PyTuple_GET_ITEM(diis, 11)));
    gradient = nullptr;
    Py_DECREF(geometry); geometry = nullptr; Py_DECREF(small); small = nullptr;
    Py_DECREF(diis); diis = nullptr; Py_DECREF(neutral_d4); neutral_d4 = nullptr;
    Py_DECREF(repulsion); repulsion = nullptr; Py_XDECREF(atm); atm = nullptr;
    Py_DECREF(post); post = nullptr; Py_DECREF(h0); h0 = nullptr;
    Py_DECREF(camm); camm = nullptr; Py_DECREF(variational); variational = nullptr;
    Py_DECREF(cn_gradient); cn_gradient = nullptr; Py_DECREF(density); density = nullptr;
    Py_DECREF(weighted); weighted = nullptr; Py_DECREF(charged_c6); charged_c6 = nullptr;
    Py_DECREF(charged_c8); charged_c8 = nullptr; Py_DECREF(neutral_c6); neutral_c6 = nullptr;
    Py_DECREF(neutral_c8); neutral_c8 = nullptr; Py_DECREF(neutral_dc6); neutral_dc6 = nullptr;
    Py_DECREF(radius_derivative); radius_derivative = nullptr;
    Py_DECREF(left_atoms); left_atoms = nullptr; Py_DECREF(right_atoms); right_atoms = nullptr;
    Py_DECREF(orbital_rows); orbital_rows = nullptr;
    Py_DECREF(orbital_columns); orbital_columns = nullptr;
    Py_DECREF(pair_scalar); pair_scalar = nullptr; Py_DECREF(zero_charges); zero_charges = nullptr;
    return result;
}

PyMethodDef methods[] = {
    {"multipole_pair_batch", multipole_pair_batch, METH_VARARGS,
     "Fused AO multipole values and centre derivatives."},
    {"create_integral_context", create_integral_context_native, METH_VARARGS,
     "Create one persistent normalized AO integral context."},
    {"integral_context_evaluate", integral_context_evaluate_native, METH_VARARGS,
     "Evaluate geometry-dependent AO integrals in a persistent context."},
    {"context_configure_spherical", context_configure_spherical_native, METH_VARARGS,
     "Install a persistent sparse Cartesian-to-spherical transformation."},
    {"context_evaluate_spherical", context_evaluate_spherical_native, METH_VARARGS,
     "Evaluate pair-packed spherical multipoles directly in the native context."},
    {"context_spherical_directional_seconds",
     context_spherical_directional_seconds_native, METH_VARARGS,
     "Differentiate pair-packed spherical AO centre gradients along one direction."},
    {"context_update_dense_model", context_update_dense_model_native, METH_VARARGS,
     "Update geometry-dependent dense model arrays in a persistent context."},
    {"context_configure_core", context_configure_core_native, METH_VARARGS,
     "Install geometry-invariant extended-Hueckel parameters."},
    {"context_build_core", context_build_core_native, METH_VARARGS,
     "Build the dense extended-Hueckel core in a persistent context."},
    {"context_prepare_dense_geometry", context_prepare_dense_geometry_native,
     METH_VARARGS,
     "Fuse dense integral, overlap, coordination, core, and model setup."},
    {"context_h0_derivatives", context_h0_derivatives_native, METH_VARARGS,
     "Build pair-packed extended-Hueckel derivatives in the persistent context."},
    {"context_dense_observables", context_dense_observables_native, METH_VARARGS,
     "Contract dense SCC observables using a persistent context."},
    {"context_dense_fock", context_dense_fock_native, METH_VARARGS,
     "Assemble a dense Fock matrix using a persistent context."},
    {"context_configure_scf", context_configure_scf_native, METH_VARARGS,
     "Install persistent SCC, anisotropic, and D4 geometry arrays."},
    {"context_scf_state_fock", context_scf_state_fock_native, METH_VARARGS,
     "Evaluate SCC, anisotropic D4, energy, and Fock in one native call."},
    {"context_last_scf_state", context_last_scf_state_native, METH_VARARGS,
     "Return cached observables from the most recent native SCC state."},
    {"context_resident_iscf", context_resident_iscf_native, METH_VARARGS,
     "Execute and certify a resident SCC iteration entirely in native code."},
    {"context_resident_diis", context_resident_diis_native, METH_VARARGS,
     "Execute resident SCC, Pulay DIIS, density builds and certification."},
    {"context_generalized_eigh", context_generalized_eigh_native, METH_VARARGS,
     "Solve FC=SCe with the vendor LAPACK driver owned by the context."},
    {"d4_two_body_pairs", d4_two_body_pairs_native, METH_VARARGS,
     "Evaluate charge-dependent pair-packed D4 and SCC derivatives."},
    {"atomic_multipole_derivatives", atomic_multipole_derivatives_native, METH_VARARGS,
     "Contract AO integral derivatives into atomic CAMM derivatives."},
    {"variational_gradient", variational_gradient_native, METH_VARARGS,
     "Contract stationary density matrices with AO derivative tensors."},
    {"sparse_gradient_accumulate", sparse_gradient_accumulate_native, METH_VARARGS,
     "Accumulate pair-packed variational-gradient chunks in parallel."},
    {"pair_antisymmetric_gradient", pair_antisymmetric_gradient_native, METH_VARARGS,
     "Scatter antisymmetric pair vectors directly to atomic gradients."},
    {"radial_pair_hessian_vector", radial_pair_hessian_vector_native, METH_VARARGS,
     "Apply central pair Hessian blocks without materializing them."},
    {"pair_symmetric_scalar", pair_symmetric_scalar_native, METH_VARARGS,
     "Scatter symmetric pair scalars directly to atoms."},
    {"context_variational_gradient_sparse", context_variational_gradient_sparse_native,
     METH_VARARGS,
     "Gather sparse densities and contract the complete AO gradient in one context call."},
    {"context_variational_gradient_dense", context_variational_gradient_dense_native,
     METH_VARARGS,
     "Contract dense resident densities without Python sparse containers."},
    {"coordination_response_gradient", coordination_response_gradient_native,
     METH_VARARGS,
     "Contract small-system GFN/D4 coordination derivatives in native code."},
    {"repulsion_energy_gradient", repulsion_energy_gradient_native,
     METH_VARARGS,
     "Evaluate repulsion energy and its small-system gradient in one native pass."},
    {"d4_coordination", d4_coordination_native, METH_VARARGS,
     "Evaluate D4 electronegativity-weighted coordination in one native pass."},
    {"gfn2_coordination", gfn2_coordination_native, METH_VARARGS,
     "Evaluate dense GFN2 coordination in one native pass."},
    {"geometry_within_displacement", geometry_within_displacement_native,
     METH_VARARGS,
     "Certify a resident geometry displacement bound without temporaries."},
    {"electronic_entropy", electronic_entropy_native, METH_VARARGS,
     "Evaluate the finite-temperature electronic entropy in native code."},
    {"d4_reference_weights", d4_reference_weights_native, METH_VARARGS,
     "Normalize D4 coordination-reference weights and derivatives."},
    {"anisotropic_geometry", anisotropic_geometry_native, METH_VARARGS,
     "Build dense anisotropic pair geometry in one native pass."},
    {"shell_interaction", shell_interaction_native, METH_VARARGS,
     "Build the dense SCC shell interaction in one native pass."},
    {"context_prepare_small_scf", context_prepare_small_scf_native,
     METH_VARARGS,
     "Prepare and install all geometry-dependent small-system SCF arrays."},
    {"context_extreme_singlepoint", context_extreme_singlepoint_native,
     METH_VARARGS,
     "Run a complete certified energy/gradient kernel for extremely small systems."},
    {"post_scf_explicit_gradient", post_scf_explicit_gradient_native,
     METH_VARARGS,
     "Fuse frozen SCC, anisotropic and D4 explicit gradient contributions."},
    {"localized_domains", localized_domains_native, METH_VARARGS,
     "Build cell-list atom domains and localized AO pairs."},
    {"sparse_from_dense", sparse_from_dense_native, METH_VARARGS,
     "Pack a domain-restricted dense symmetric matrix."},
    {"sparse_topology", sparse_topology_native, METH_VARARGS,
     "Build a reusable contiguous sparse adjacency topology."},
    {"sparse_sp2", sparse_sp2_native, METH_VARARGS,
     "Run domain-restricted sparse SP2 entirely in native code."},
    {"sparse_sp2_incremental", sparse_sp2_incremental_native, METH_VARARGS,
     "Update a sparse SP2 projector from a certified purification history."},
    {"sparse_scaled_add", sparse_scaled_add_py, METH_VARARGS,
     "Merge and truncate two native sparse symmetric matrices."},
    {"sparse_square", sparse_square_py, METH_VARARGS,
     "Square a sparse symmetric matrix inside localized domains."},
    {"sparse_product", sparse_product_py, METH_VARARGS,
     "Form a symmetric sparse Jordan product inside localized domains."},
    {"sparse_congruence", sparse_congruence_py, METH_VARARGS,
     "Form X A X directly inside localized domains."},
    {"sparse_orthogonal_observables", sparse_orthogonal_observables_native, METH_VARARGS,
     "Contract XPX directly into shell and multipole observables."},
    {"sparse_multipole_congruence", sparse_multipole_congruence_native, METH_VARARGS,
     "Transform pair-packed Cartesian multipoles to a spherical basis."},
    {"anisotropic_fock_compact", anisotropic_fock_compact_native, METH_VARARGS,
     "Assemble sparse anisotropic Fock values without endpoint buffers."},
    {"anisotropic_fock_derivatives", anisotropic_fock_derivatives_native, METH_VARARGS,
     "Contract pair-packed anisotropic Fock derivatives."},
    {"anisotropic_fock_derivatives_directional",
     anisotropic_fock_derivatives_directional_native, METH_VARARGS,
     "Differentiate pair-packed anisotropic Fock derivatives analytically."},
    {"atm_three_body_hvp_core", atm_three_body_hvp_core_native, METH_VARARGS,
     "Differentiate sparse ATM geometric and coordination contractions."},
    {"anisotropic_energy_potentials", anisotropic_energy_potentials_native, METH_VARARGS,
     "Evaluate anisotropic energy and moment potentials."},
    {"atm_three_body_energy", atm_three_body_energy_native, METH_VARARGS,
     "Evaluate the dense ATM three-body energy."},
    {"atm_three_body_gradient", atm_three_body_gradient_native, METH_VARARGS,
     "Evaluate the dense ATM geometry and coordination gradient."},
    {"atm_three_body_energy_gradient", atm_three_body_energy_gradient_native,
     METH_VARARGS,
     "Evaluate dense ATM energy, geometry gradient and CN response in one pass."},
    {"sparse_matvec", sparse_matvec_py, METH_VARARGS,
     "Multiply a native sparse symmetric matrix by a vector."},
    {"sparse_gather", sparse_gather_py, METH_VARARGS,
     "Gather a sparse symmetric matrix on a sorted target pair list."},
    {"dense_scc_observables", dense_scc_observables_native, METH_VARARGS,
     "Contract dense density into shell charges and atomic multipoles."},
    {"dense_scc_fock", dense_scc_fock_native, METH_VARARGS,
     "Assemble monopole, anisotropic, and D4 dense Fock contributions."},
    {"sparse_mayer_bond_orders", sparse_mayer_bond_orders_native, METH_VARARGS,
     "Evaluate lazy Mayer bond orders from sparse AO matrices."},
    {"sparse_triangles", sparse_triangles_native, METH_VARARGS,
     "Enumerate triangles in a sparse lower-triangular edge graph."},
    {"sparse_atm_energy_gradient", sparse_atm_energy_gradient_native, METH_VARARGS,
     "Evaluate sparse ATM energy and analytic gradient without materializing triangles."},
    {"closed_shell_stda", closed_shell_stda_native, METH_VARARGS,
     "Build, select, and solve a closed-shell sTDA problem in native code."},
    {"localized_projected_orbitals", localized_projected_orbitals_native, METH_VARARGS,
     "Factor a sparse occupied or virtual projector into local orbitals."},
    {"sparse_orbital_quadratic_forms", sparse_orbital_quadratic_forms_native, METH_VARARGS,
     "Contract sparse local orbitals with a pair-packed Fock matrix."},
    {"localized_stda_pairs", localized_stda_pairs_native, METH_VARARGS,
     "Assemble local closed-shell sTDA diagonal and pair couplings."},
    {"sparse_symmetric_davidson", sparse_symmetric_davidson_native, METH_VARARGS,
     "Solve a pair-packed symmetric problem with resident block Davidson."},
    {"sparse_rotation_to_ao", sparse_rotation_to_ao_native, METH_VARARGS,
     "Transform local orbital rotations to a pair-packed AO pattern."},
    {"sparse_ao_to_rotation", sparse_ao_to_rotation_native, METH_VARARGS,
     "Contract a pair-packed AO matrix with local orbital rotations."},
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_gfn2_native", nullptr, -1, methods};
}  // namespace

PyMODINIT_FUNC PyInit__gfn2_native() {
    import_array();
    return PyModule_Create(&module);
}
