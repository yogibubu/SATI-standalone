#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_24_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <limits>
#include <numeric>
#include <set>
#include <utility>
#include <vector>
#include <thread>
#include <stdexcept>

#define MATRIX_STRINGIFY_DETAIL(value) #value
#define MATRIX_STRINGIFY(value) MATRIX_STRINGIFY_DETAIL(value)

namespace {

using Edge = std::pair<npy_intp, npy_intp>;
using Cycle = std::vector<npy_intp>;
using BitVector = std::vector<std::uint64_t>;

constexpr std::size_t LOCAL_DERIVATIVE_DIMENSION = 12;
constexpr npy_intp PARALLEL_PRIMITIVE_ROW_THRESHOLD = 4096;
constexpr double MATRIX_PI = 3.141592653589793238462643383279502884;

struct LocalDual {
    double value = 0.0;
    std::array<double, LOCAL_DERIVATIVE_DIMENSION> derivative{};
};

using LocalVector = std::array<LocalDual, 3>;

LocalDual dual_constant(double value) {
    LocalDual result;
    result.value = value;
    return result;
}

LocalDual dual_variable(double value, std::size_t index) {
    LocalDual result = dual_constant(value);
    result.derivative[index] = 1.0;
    return result;
}

LocalDual dual_add(const LocalDual& left, const LocalDual& right) {
    LocalDual result;
    result.value = left.value + right.value;
    for (std::size_t i = 0; i < LOCAL_DERIVATIVE_DIMENSION; ++i) {
        result.derivative[i] = left.derivative[i] + right.derivative[i];
    }
    return result;
}

LocalDual dual_subtract(const LocalDual& left, const LocalDual& right) {
    LocalDual result;
    result.value = left.value - right.value;
    for (std::size_t i = 0; i < LOCAL_DERIVATIVE_DIMENSION; ++i) {
        result.derivative[i] = left.derivative[i] - right.derivative[i];
    }
    return result;
}

LocalDual dual_multiply(const LocalDual& left, const LocalDual& right) {
    LocalDual result;
    result.value = left.value * right.value;
    for (std::size_t i = 0; i < LOCAL_DERIVATIVE_DIMENSION; ++i) {
        result.derivative[i] =
            left.value * right.derivative[i] + right.value * left.derivative[i];
    }
    return result;
}

LocalDual dual_scale(const LocalDual& value, double factor) {
    LocalDual result;
    result.value = factor * value.value;
    for (std::size_t i = 0; i < LOCAL_DERIVATIVE_DIMENSION; ++i) {
        result.derivative[i] = factor * value.derivative[i];
    }
    return result;
}

LocalDual dual_divide(const LocalDual& numerator, const LocalDual& denominator) {
    LocalDual result;
    const double inverse = 1.0 / denominator.value;
    result.value = numerator.value * inverse;
    for (std::size_t i = 0; i < LOCAL_DERIVATIVE_DIMENSION; ++i) {
        result.derivative[i] =
            (numerator.derivative[i] - result.value * denominator.derivative[i]) * inverse;
    }
    return result;
}

LocalDual dual_sqrt(const LocalDual& value) {
    if (value.value <= 1.0e-24) {
        return dual_constant(0.0);
    }
    LocalDual result;
    result.value = std::sqrt(value.value);
    const double factor = 0.5 / result.value;
    for (std::size_t i = 0; i < LOCAL_DERIVATIVE_DIMENSION; ++i) {
        result.derivative[i] = factor * value.derivative[i];
    }
    return result;
}

LocalDual dual_exp(const LocalDual& value) {
    LocalDual result;
    result.value = std::exp(value.value);
    for (std::size_t index = 0; index < result.derivative.size(); ++index) {
        result.derivative[index] = result.value * value.derivative[index];
    }
    return result;
}

LocalDual dual_acos(const LocalDual& value) {
    LocalDual result;
    const double clipped = std::max(-1.0, std::min(1.0, value.value));
    result.value = std::acos(clipped);
    const double denominator = std::sqrt(std::max(1.0 - clipped * clipped, 1.0e-16));
    for (std::size_t i = 0; i < LOCAL_DERIVATIVE_DIMENSION; ++i) {
        result.derivative[i] = -value.derivative[i] / denominator;
    }
    return result;
}

LocalDual dual_asin(const LocalDual& value) {
    LocalDual result;
    const double clipped = std::max(-0.999999, std::min(0.999999, value.value));
    result.value = std::asin(std::max(-1.0, std::min(1.0, value.value)));
    const double denominator = std::sqrt(std::max(1.0 - clipped * clipped, 1.0e-24));
    for (std::size_t i = 0; i < LOCAL_DERIVATIVE_DIMENSION; ++i) {
        result.derivative[i] = value.derivative[i] / denominator;
    }
    return result;
}

LocalDual dual_atan2(const LocalDual& y, const LocalDual& x) {
    LocalDual result;
    result.value = std::atan2(y.value, x.value);
    const double denominator = x.value * x.value + y.value * y.value;
    if (denominator <= 1.0e-24) {
        return result;
    }
    for (std::size_t i = 0; i < LOCAL_DERIVATIVE_DIMENSION; ++i) {
        result.derivative[i] =
            (x.value * y.derivative[i] - y.value * x.derivative[i]) / denominator;
    }
    return result;
}

LocalVector vector_add(const LocalVector& left, const LocalVector& right) {
    return {dual_add(left[0], right[0]), dual_add(left[1], right[1]),
            dual_add(left[2], right[2])};
}

LocalVector vector_subtract(const LocalVector& left, const LocalVector& right) {
    return {dual_subtract(left[0], right[0]), dual_subtract(left[1], right[1]),
            dual_subtract(left[2], right[2])};
}

LocalDual vector_dot(const LocalVector& left, const LocalVector& right) {
    return dual_add(
        dual_add(dual_multiply(left[0], right[0]), dual_multiply(left[1], right[1])),
        dual_multiply(left[2], right[2]));
}

LocalVector vector_cross(const LocalVector& left, const LocalVector& right) {
    return {
        dual_subtract(dual_multiply(left[1], right[2]), dual_multiply(left[2], right[1])),
        dual_subtract(dual_multiply(left[2], right[0]), dual_multiply(left[0], right[2])),
        dual_subtract(dual_multiply(left[0], right[1]), dual_multiply(left[1], right[0])),
    };
}

LocalVector vector_unit(const LocalVector& vector) {
    const LocalDual length = dual_sqrt(vector_dot(vector, vector));
    if (length.value <= 1.0e-12) {
        return {dual_constant(0.0), dual_constant(0.0), dual_constant(0.0)};
    }
    return {dual_divide(vector[0], length), dual_divide(vector[1], length),
            dual_divide(vector[2], length)};
}

LocalDual local_angle(
    const LocalVector& first, const LocalVector& center, const LocalVector& last) {
    return dual_acos(vector_dot(
        vector_unit(vector_subtract(first, center)),
        vector_unit(vector_subtract(last, center))));
}

LocalDual local_dihedral(
    const LocalVector& first,
    const LocalVector& second,
    const LocalVector& third,
    const LocalVector& fourth) {
    const LocalVector b1 = vector_subtract(first, second);
    const LocalVector b2 = vector_subtract(third, second);
    const LocalVector b3 = vector_subtract(fourth, third);
    const LocalVector n1 = vector_cross(b1, b2);
    const LocalVector n2 = vector_cross(b2, b3);
    const LocalDual x = vector_dot(n1, n2);
    const LocalDual y = vector_dot(vector_cross(n1, n2), vector_unit(b2));
    return dual_atan2(y, x);
}

LocalDual evaluate_local_primitive(
    npy_intp kind,
    npy_intp atom_count,
    npy_intp mode,
    const std::array<LocalVector, 4>& points) {
    if (kind == 1) {
        return dual_sqrt(vector_dot(
            vector_subtract(points[0], points[1]),
            vector_subtract(points[0], points[1])));
    }
    if (kind == 7 || kind == 8 || kind == 9) {
        const LocalDual radius = dual_sqrt(vector_dot(
            vector_subtract(points[0], points[1]),
            vector_subtract(points[0], points[1])));
        if (kind == 7) return dual_divide(dual_constant(1.0), radius);
        if (kind == 8) {
            return dual_divide(
                dual_constant(1.0),
                dual_multiply(dual_multiply(radius, radius), radius));
        }
        return dual_exp(dual_scale(radius, -1.0));
    }
    if (kind == 2) {
        return local_angle(points[0], points[1], points[2]);
    }
    if (kind == 3) {
        return local_dihedral(points[0], points[1], points[2], points[3]);
    }
    if (kind == 4 || kind == 5) {
        const LocalVector vector = vector_subtract(points[3], points[0]);
        const LocalVector left = vector_subtract(points[1], points[0]);
        const LocalVector right = vector_subtract(points[2], points[0]);
        const LocalVector normal = vector_cross(left, right);
        if (kind == 4) {
            return dual_scale(
                dual_asin(vector_dot(vector_unit(vector), vector_unit(normal))), -1.0);
        }
        return vector_dot(vector, vector_unit(normal));
    }
    if (kind == 6 && atom_count == 3) {
        const LocalVector u = vector_unit(vector_subtract(points[0], points[1]));
        const LocalVector v = vector_unit(vector_subtract(points[2], points[1]));
        LocalVector axis = {dual_constant(1.0), dual_constant(0.0), dual_constant(0.0)};
        if (std::abs(u[0].value) > 0.9) {
            axis = {dual_constant(0.0), dual_constant(1.0), dual_constant(0.0)};
        }
        const LocalVector first = vector_unit(vector_cross(u, axis));
        const LocalVector second = vector_cross(u, first);
        return vector_dot(vector_add(v, u), mode == -1 ? first : second);
    }
    if (kind == 6 && atom_count == 4) {
        if (mode == -1) {
            return dual_add(
                local_angle(points[0], points[1], points[3]),
                local_angle(points[3], points[1], points[2]));
        }
        const LocalDual first = local_dihedral(
            points[0], points[1], points[3], points[2]);
        const LocalDual second = local_dihedral(
            points[0], points[3], points[1], points[2]);
        LocalDual result = dual_scale(dual_subtract(first, second), 0.5);
        double wrapped = std::fmod(
            first.value - second.value + MATRIX_PI, 2.0 * MATRIX_PI);
        if (wrapped < 0.0) {
            wrapped += 2.0 * MATRIX_PI;
        }
        result.value = MATRIX_PI + 0.5 * (wrapped - MATRIX_PI);
        return result;
    }
    return dual_constant(std::numeric_limits<double>::quiet_NaN());
}

using ScalarVector = std::array<double, 3>;

ScalarVector scalar_subtract(const ScalarVector& a, const ScalarVector& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
ScalarVector scalar_add(const ScalarVector& a, const ScalarVector& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
double scalar_dot(const ScalarVector& a, const ScalarVector& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
ScalarVector scalar_cross(const ScalarVector& a, const ScalarVector& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
ScalarVector scalar_unit(const ScalarVector& value) {
    const double length = std::sqrt(scalar_dot(value, value));
    if (length <= 1.0e-12) return {0.0, 0.0, 0.0};
    return {value[0] / length, value[1] / length, value[2] / length};
}
double scalar_angle(const ScalarVector& a, const ScalarVector& b, const ScalarVector& c) {
    const double cosine = scalar_dot(scalar_unit(scalar_subtract(a, b)),
                                     scalar_unit(scalar_subtract(c, b)));
    return std::acos(std::max(-1.0, std::min(1.0, cosine)));
}
double scalar_dihedral(const ScalarVector& a, const ScalarVector& b,
                       const ScalarVector& c, const ScalarVector& d) {
    const ScalarVector b1 = scalar_subtract(a, b);
    const ScalarVector b2 = scalar_subtract(c, b);
    const ScalarVector b3 = scalar_subtract(d, c);
    const ScalarVector n1 = scalar_cross(b1, b2);
    const ScalarVector n2 = scalar_cross(b2, b3);
    return std::atan2(scalar_dot(scalar_cross(n1, n2), scalar_unit(b2)),
                      scalar_dot(n1, n2));
}
double evaluate_local_primitive_value(npy_intp kind, npy_intp atom_count,
                                      npy_intp mode,
                                      const std::array<ScalarVector, 4>& p) {
    if (kind == 1 || kind == 7 || kind == 8 || kind == 9) {
        const ScalarVector delta = scalar_subtract(p[0], p[1]);
        const double radius = std::sqrt(scalar_dot(delta, delta));
        if (kind == 1) return radius;
        if (kind == 7) return 1.0 / radius;
        if (kind == 8) return 1.0 / (radius * radius * radius);
        return std::exp(-radius);
    }
    if (kind == 2) return scalar_angle(p[0], p[1], p[2]);
    if (kind == 3) return scalar_dihedral(p[0], p[1], p[2], p[3]);
    if (kind == 4 || kind == 5) {
        const ScalarVector vector = scalar_subtract(p[3], p[0]);
        const ScalarVector normal = scalar_cross(
            scalar_subtract(p[1], p[0]), scalar_subtract(p[2], p[0]));
        if (kind == 4) {
            const double sine = scalar_dot(scalar_unit(vector), scalar_unit(normal));
            return -std::asin(std::max(-1.0, std::min(1.0, sine)));
        }
        return scalar_dot(vector, scalar_unit(normal));
    }
    if (kind == 6 && atom_count == 3) {
        const ScalarVector u = scalar_unit(scalar_subtract(p[0], p[1]));
        const ScalarVector v = scalar_unit(scalar_subtract(p[2], p[1]));
        const ScalarVector axis = std::abs(u[0]) > 0.9
            ? ScalarVector{0.0, 1.0, 0.0} : ScalarVector{1.0, 0.0, 0.0};
        const ScalarVector first = scalar_unit(scalar_cross(u, axis));
        const ScalarVector second = scalar_cross(u, first);
        return scalar_dot(scalar_add(v, u), mode == -1 ? first : second);
    }
    if (kind == 6 && atom_count == 4) {
        if (mode == -1)
            return scalar_angle(p[0], p[1], p[3]) + scalar_angle(p[3], p[1], p[2]);
        const double first = scalar_dihedral(p[0], p[1], p[3], p[2]);
        const double second = scalar_dihedral(p[0], p[3], p[1], p[2]);
        double wrapped = std::fmod(first - second + MATRIX_PI, 2.0 * MATRIX_PI);
        if (wrapped < 0.0) wrapped += 2.0 * MATRIX_PI;
        return MATRIX_PI + 0.5 * (wrapped - MATRIX_PI);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

struct CycleOrder {
    bool operator()(const Cycle& left, const Cycle& right) const {
        if (left.size() != right.size()) {
            return left.size() < right.size();
        }
        return left < right;
    }
};

npy_intp cycle_rank(
    npy_intp natoms,
    const std::vector<unsigned char>& allowed,
    const std::vector<Edge>& edges);
std::set<Cycle, CycleOrder> horton_candidates(
    npy_intp natoms,
    const std::vector<unsigned char>& allowed,
    const std::vector<Edge>& all_edges,
    npy_intp ring_max);
std::vector<Cycle> minimum_cycle_basis(
    const std::set<Cycle, CycleOrder>& candidates,
    const std::vector<Edge>& edges,
    npy_intp target_rank);

template <typename T>
PyObject* numpy_vector(const std::vector<T>& values, int type) {
    npy_intp dimension = static_cast<npy_intp>(values.size());
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, &dimension, type));
    if (array == nullptr) {
        return nullptr;
    }
    std::copy(values.begin(), values.end(), static_cast<T*>(PyArray_DATA(array)));
    return reinterpret_cast<PyObject*>(array);
}

#include "_smith_primitive_native.hpp"

double effective_radius(
    npy_intp atomic_number,
    double coordination,
    const double* standard_radii,
    const double* pyykko,
    npy_intp pyykko_columns) {
    if (atomic_number < 0 || atomic_number > 118) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<npy_intp> keys;
    for (npy_intp key = 0; key < pyykko_columns; ++key) {
        if (std::isfinite(pyykko[atomic_number * pyykko_columns + key])) {
            keys.push_back(key);
        }
    }
    if (keys.empty()) {
        return standard_radii[atomic_number];
    }
    if (keys.size() == 1 || coordination <= keys.front()) {
        return pyykko[atomic_number * pyykko_columns + keys.front()];
    }
    if (coordination >= keys.back()) {
        return pyykko[atomic_number * pyykko_columns + keys.back()];
    }
    auto upper = std::upper_bound(keys.begin(), keys.end(), coordination);
    const npy_intp upper_key = *upper;
    const npy_intp lower_key = *(upper - 1);
    const double lower_radius =
        pyykko[atomic_number * pyykko_columns + lower_key];
    const double upper_radius =
        pyykko[atomic_number * pyykko_columns + upper_key];
    const double t =
        (coordination - lower_key) / static_cast<double>(upper_key - lower_key);
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double interval_slope = upper_radius - lower_radius;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * lower_radius +
        (t3 - 2.0 * t2 + t) * interval_slope +
        (-2.0 * t3 + 3.0 * t2) * upper_radius +
        (t3 - t2) * interval_slope;
}

PyObject* perceive_continuous_graph(PyObject*, PyObject* args) {
    PyObject *coordinates_object, *numbers_object, *standard_object, *pyykko_object;
    double cutoff, cna_alpha, distance_scale, switch_alpha, lambda_strong, lambda_weak;
    if (!PyArg_ParseTuple(
            args, "OOOOdddddd",
            &coordinates_object, &numbers_object, &standard_object, &pyykko_object,
            &cutoff, &cna_alpha, &distance_scale, &switch_alpha,
            &lambda_strong, &lambda_weak)) {
        return nullptr;
    }
    PyArrayObject* coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* numbers = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(numbers_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* standard = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(standard_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* pyykko = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(pyykko_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (coordinates == nullptr || numbers == nullptr || standard == nullptr ||
        pyykko == nullptr) {
        Py_XDECREF(coordinates);
        Py_XDECREF(numbers);
        Py_XDECREF(standard);
        Py_XDECREF(pyykko);
        return nullptr;
    }
    const npy_intp natoms = PyArray_DIM(coordinates, 0);
    if (PyArray_NDIM(coordinates) != 2 || PyArray_DIM(coordinates, 1) != 3 ||
        PyArray_NDIM(numbers) != 1 || PyArray_DIM(numbers, 0) != natoms ||
        PyArray_NDIM(standard) != 1 || PyArray_DIM(standard, 0) < 119 ||
        PyArray_NDIM(pyykko) != 2 || PyArray_DIM(pyykko, 0) < 119) {
        Py_DECREF(coordinates);
        Py_DECREF(numbers);
        Py_DECREF(standard);
        Py_DECREF(pyykko);
        PyErr_SetString(PyExc_ValueError, "native continuous-graph arrays are invalid");
        return nullptr;
    }
    const auto* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* z = static_cast<const npy_intp*>(PyArray_DATA(numbers));
    const auto* radii = static_cast<const double*>(PyArray_DATA(standard));
    const auto* pyykko_data = static_cast<const double*>(PyArray_DATA(pyykko));
    const npy_intp pyykko_columns = PyArray_DIM(pyykko, 1);
    std::vector<npy_intp> pair_left;
    std::vector<npy_intp> pair_right;
    std::vector<double> pair_distance;
    std::vector<double> coordination(natoms, 0.0);
    std::vector<double> effective(natoms, 0.0);
    std::vector<npy_intp> discrete_left;
    std::vector<npy_intp> discrete_right;
    std::vector<double> connectivity;
    std::vector<npy_intp> accepted_left;
    std::vector<npy_intp> accepted_right;
    npy_intp accepted_cycle_rank = 0;
    std::set<Cycle, CycleOrder> accepted_candidates;
    std::vector<Cycle> accepted_cycles;
    Py_BEGIN_ALLOW_THREADS
    const double cutoff2 = cutoff * cutoff;
    for (npy_intp left = 0; left < natoms; ++left) {
        for (npy_intp right = left + 1; right < natoms; ++right) {
            const double dx = xyz[3 * left] - xyz[3 * right];
            const double dy = xyz[3 * left + 1] - xyz[3 * right + 1];
            const double dz = xyz[3 * left + 2] - xyz[3 * right + 2];
            const double squared = dx * dx + dy * dy + dz * dz;
            if (squared > cutoff2) {
                continue;
            }
            const double distance = std::sqrt(squared);
            pair_left.push_back(left);
            pair_right.push_back(right);
            pair_distance.push_back(distance);
            if (z[left] >= 0 && z[left] <= 118 && z[right] >= 0 && z[right] <= 118) {
                const double radius_sum = radii[z[left]] + radii[z[right]];
                if (std::isfinite(radius_sum)) {
                    const double contribution =
                        0.5 * (1.0 + std::erf(cna_alpha * (radius_sum - distance)));
                    coordination[left] += contribution;
                    coordination[right] += contribution;
                }
            }
        }
    }
    for (npy_intp atom = 0; atom < natoms; ++atom) {
        effective[atom] = effective_radius(
            z[atom], coordination[atom], radii, pyykko_data, pyykko_columns);
    }
    for (std::size_t pair = 0; pair < pair_left.size(); ++pair) {
        const npy_intp left = pair_left[pair];
        const npy_intp right = pair_right[pair];
        if (z[left] < 0 || z[left] > 118 || z[right] < 0 || z[right] > 118) {
            continue;
        }
        const double standard_sum = radii[z[left]] + radii[z[right]];
        if (!std::isfinite(standard_sum) ||
            pair_distance[pair] > distance_scale * standard_sum) {
            continue;
        }
        const double reference = effective[left] + effective[right];
        double value = 0.0;
        if (reference > 1.0e-12) {
            const double reduced = (pair_distance[pair] - reference) / reference;
            const double strong_weight =
                0.5 * (1.0 - std::tanh(switch_alpha * reduced));
            const double strong =
                std::exp((reference - pair_distance[pair]) / lambda_strong);
            const double weak =
                std::exp((reference - pair_distance[pair]) / lambda_weak);
            value = strong_weight * strong + (1.0 - strong_weight) * weak;
        }
        discrete_left.push_back(left);
        discrete_right.push_back(right);
        connectivity.push_back(value);
    }
    std::vector<unsigned char> heavy_partner(natoms, 0);
    for (std::size_t pair = 0; pair < connectivity.size(); ++pair) {
        if (connectivity[pair] < 0.2) {
            continue;
        }
        const npy_intp left = discrete_left[pair];
        const npy_intp right = discrete_right[pair];
        if (z[right] != 1) {
            heavy_partner[left] = 1;
        }
        if (z[left] != 1) {
            heavy_partner[right] = 1;
        }
    }
    for (std::size_t pair = 0; pair < connectivity.size(); ++pair) {
        if (connectivity[pair] < 0.2) {
            continue;
        }
        const npy_intp left = discrete_left[pair];
        const npy_intp right = discrete_right[pair];
        if (z[left] == 1 && z[right] == 1 &&
            (heavy_partner[left] || heavy_partner[right])) {
            continue;
        }
        accepted_left.push_back(left);
        accepted_right.push_back(right);
    }
    std::vector<Edge> accepted_edges;
    accepted_edges.reserve(accepted_left.size());
    for (std::size_t index = 0; index < accepted_left.size(); ++index) {
        accepted_edges.emplace_back(accepted_left[index], accepted_right[index]);
    }
    std::vector<unsigned char> all_allowed(natoms, 1);
    accepted_cycle_rank = cycle_rank(natoms, all_allowed, accepted_edges);
    if (accepted_cycle_rank > 0) {
        accepted_candidates = horton_candidates(
            natoms, all_allowed, accepted_edges, -1);
        accepted_cycles = minimum_cycle_basis(
            accepted_candidates, accepted_edges, accepted_cycle_rank);
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(coordinates);
    Py_DECREF(numbers);
    Py_DECREF(standard);
    Py_DECREF(pyykko);
    PyObject* outputs[] = {
        numpy_vector(pair_left, NPY_INTP),
        numpy_vector(pair_right, NPY_INTP),
        numpy_vector(pair_distance, NPY_DOUBLE),
        numpy_vector(coordination, NPY_DOUBLE),
        numpy_vector(effective, NPY_DOUBLE),
        numpy_vector(discrete_left, NPY_INTP),
        numpy_vector(discrete_right, NPY_INTP),
        numpy_vector(connectivity, NPY_DOUBLE),
        numpy_vector(accepted_left, NPY_INTP),
        numpy_vector(accepted_right, NPY_INTP),
    };
    for (PyObject* output : outputs) {
        if (output == nullptr) {
            for (PyObject* owned : outputs) {
                Py_XDECREF(owned);
            }
            return nullptr;
        }
    }
    PyObject* cycles = PyTuple_New(
        static_cast<Py_ssize_t>(accepted_cycles.size()));
    if (cycles == nullptr) {
        for (PyObject* output : outputs) {
            Py_DECREF(output);
        }
        return nullptr;
    }
    for (std::size_t index = 0; index < accepted_cycles.size(); ++index) {
        PyObject* cycle = PyTuple_New(
            static_cast<Py_ssize_t>(accepted_cycles[index].size()));
        if (cycle == nullptr) {
            Py_DECREF(cycles);
            for (PyObject* output : outputs) {
                Py_DECREF(output);
            }
            return nullptr;
        }
        for (std::size_t atom = 0; atom < accepted_cycles[index].size(); ++atom) {
            PyTuple_SET_ITEM(
                cycle,
                static_cast<Py_ssize_t>(atom),
                PyLong_FromSsize_t(accepted_cycles[index][atom]));
        }
        PyTuple_SET_ITEM(cycles, static_cast<Py_ssize_t>(index), cycle);
    }
    PyObject* result = PyTuple_New(13);
    if (result == nullptr) {
        for (PyObject* output : outputs) {
            Py_DECREF(output);
        }
        return nullptr;
    }
    for (int index = 0; index < 10; ++index) {
        PyTuple_SET_ITEM(result, index, outputs[index]);
    }
    PyTuple_SET_ITEM(result, 10, cycles);
    PyTuple_SET_ITEM(
        result,
        11,
        PyLong_FromSsize_t(
            static_cast<Py_ssize_t>(accepted_candidates.size())));
    PyTuple_SET_ITEM(
        result,
        12,
        PyLong_FromSsize_t(static_cast<Py_ssize_t>(accepted_cycle_rank)));
    return result;
}

Cycle canonical_cycle(const Cycle& cycle) {
    if (cycle.empty()) {
        return {};
    }
    auto minimum = std::min_element(cycle.begin(), cycle.end());
    const std::size_t start = static_cast<std::size_t>(minimum - cycle.begin());
    Cycle forward;
    Cycle backward;
    forward.reserve(cycle.size());
    backward.reserve(cycle.size());
    for (std::size_t offset = 0; offset < cycle.size(); ++offset) {
        forward.push_back(cycle[(start + offset) % cycle.size()]);
        backward.push_back(cycle[(start + cycle.size() - offset) % cycle.size()]);
    }
    return std::min(forward, backward);
}

bool chordless(const std::vector<std::vector<npy_intp>>& adjacency, const Cycle& cycle) {
    std::vector<npy_intp> position(adjacency.size(), -1);
    for (std::size_t index = 0; index < cycle.size(); ++index) {
        position[cycle[index]] = static_cast<npy_intp>(index);
    }
    for (std::size_t index = 0; index < cycle.size(); ++index) {
        const npy_intp atom = cycle[index];
        const npy_intp previous = cycle[(index + cycle.size() - 1) % cycle.size()];
        const npy_intp following = cycle[(index + 1) % cycle.size()];
        for (const npy_intp neighbor : adjacency[atom]) {
            if (position[neighbor] >= 0 && neighbor != previous && neighbor != following) {
                return false;
            }
        }
    }
    return true;
}

std::set<Cycle, CycleOrder> split_at_chords(
    const std::vector<std::vector<npy_intp>>& adjacency, const Cycle& initial) {
    std::vector<Cycle> pending{canonical_cycle(initial)};
    std::set<Cycle, CycleOrder> result;
    while (!pending.empty()) {
        Cycle current = std::move(pending.back());
        pending.pop_back();
        std::vector<npy_intp> position(adjacency.size(), -1);
        for (std::size_t index = 0; index < current.size(); ++index) {
            position[current[index]] = static_cast<npy_intp>(index);
        }
        std::pair<npy_intp, npy_intp> chord{-1, -1};
        for (std::size_t index = 0; index < current.size() && chord.first < 0; ++index) {
            const npy_intp atom = current[index];
            const npy_intp previous =
                current[(index + current.size() - 1) % current.size()];
            const npy_intp following = current[(index + 1) % current.size()];
            for (const npy_intp neighbor : adjacency[atom]) {
                if (position[neighbor] >= 0 && neighbor != previous && neighbor != following) {
                    chord = {
                        static_cast<npy_intp>(index),
                        position[neighbor],
                    };
                    break;
                }
            }
        }
        if (chord.first < 0) {
            result.insert(canonical_cycle(current));
            continue;
        }
        npy_intp first = std::min(chord.first, chord.second);
        npy_intp second = std::max(chord.first, chord.second);
        Cycle part_a(
            current.begin() + first,
            current.begin() + second + 1);
        Cycle part_b(current.begin() + second, current.end());
        part_b.insert(part_b.end(), current.begin(), current.begin() + first + 1);
        if (part_a.size() >= 3) {
            pending.push_back(canonical_cycle(part_a));
        }
        if (part_b.size() >= 3) {
            pending.push_back(canonical_cycle(part_b));
        }
    }
    return result;
}

Cycle tree_cycle(
    npy_intp left, npy_intp right,
    const std::vector<npy_intp>& parent,
    const std::vector<npy_intp>& depth) {
    if (parent[left] == -2 || parent[right] == -2) {
        return {};
    }
    Cycle left_path;
    Cycle right_path;
    npy_intp a = left;
    npy_intp b = right;
    while (depth[a] > depth[b]) {
        left_path.push_back(a);
        if (parent[a] < 0) {
            return {};
        }
        a = parent[a];
    }
    while (depth[b] > depth[a]) {
        right_path.push_back(b);
        if (parent[b] < 0) {
            return {};
        }
        b = parent[b];
    }
    while (a != b) {
        left_path.push_back(a);
        right_path.push_back(b);
        if (parent[a] < 0 || parent[b] < 0) {
            return {};
        }
        a = parent[a];
        b = parent[b];
    }
    left_path.push_back(a);
    left_path.insert(left_path.end(), right_path.rbegin(), right_path.rend());
    return left_path;
}

std::vector<unsigned char> cyclic_core(
    npy_intp natoms,
    const std::vector<unsigned char>& allowed,
    const std::vector<Edge>& edges) {
    std::vector<std::set<npy_intp>> mutable_adjacency(natoms);
    std::vector<unsigned char> core = allowed;
    for (const auto& [left, right] : edges) {
        mutable_adjacency[left].insert(right);
        mutable_adjacency[right].insert(left);
    }
    std::vector<npy_intp> pending;
    for (npy_intp atom = 0; atom < natoms; ++atom) {
        if (core[atom] && mutable_adjacency[atom].size() < 2) {
            pending.push_back(atom);
        }
    }
    while (!pending.empty()) {
        const npy_intp atom = pending.back();
        pending.pop_back();
        if (!core[atom]) {
            continue;
        }
        core[atom] = 0;
        for (const npy_intp neighbor : mutable_adjacency[atom]) {
            if (!core[neighbor]) {
                continue;
            }
            mutable_adjacency[neighbor].erase(atom);
            if (mutable_adjacency[neighbor].size() < 2) {
                pending.push_back(neighbor);
            }
        }
    }
    return core;
}

npy_intp cycle_rank(
    npy_intp natoms,
    const std::vector<unsigned char>& allowed,
    const std::vector<Edge>& edges) {
    std::vector<npy_intp> parent(natoms);
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&parent](npy_intp atom) {
        npy_intp current = atom;
        while (parent[current] != current) {
            parent[current] = parent[parent[current]];
            current = parent[current];
        }
        return current;
    };
    for (const auto& [left, right] : edges) {
        npy_intp root_left = find(left);
        npy_intp root_right = find(right);
        if (root_left != root_right) {
            parent[root_right] = root_left;
        }
    }
    std::set<npy_intp> components;
    npy_intp allowed_count = 0;
    for (npy_intp atom = 0; atom < natoms; ++atom) {
        if (allowed[atom]) {
            ++allowed_count;
            components.insert(find(atom));
        }
    }
    return std::max<npy_intp>(
        0,
        static_cast<npy_intp>(edges.size()) - allowed_count +
            static_cast<npy_intp>(components.size()));
}

std::set<Cycle, CycleOrder> horton_candidates(
    npy_intp natoms,
    const std::vector<unsigned char>& allowed,
    const std::vector<Edge>& all_edges,
    npy_intp ring_max) {
    const auto core = cyclic_core(natoms, allowed, all_edges);
    std::vector<std::vector<npy_intp>> adjacency(natoms);
    std::vector<Edge> edges;
    for (const auto& [left, right] : all_edges) {
        if (core[left] && core[right]) {
            edges.emplace_back(left, right);
            adjacency[left].push_back(right);
            adjacency[right].push_back(left);
        }
    }
    for (auto& neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end());
    }
    std::set<npy_intp> roots;
    for (npy_intp atom = 0; atom < natoms; ++atom) {
        if (core[atom] && adjacency[atom].size() != 2) {
            roots.insert(atom);
        }
    }
    std::vector<unsigned char> unseen = core;
    for (npy_intp seed = 0; seed < natoms; ++seed) {
        if (!unseen[seed]) {
            continue;
        }
        std::vector<npy_intp> component{seed};
        unseen[seed] = 0;
        for (std::size_t index = 0; index < component.size(); ++index) {
            for (const npy_intp neighbor : adjacency[component[index]]) {
                if (unseen[neighbor]) {
                    unseen[neighbor] = 0;
                    component.push_back(neighbor);
                }
            }
        }
        bool has_root = false;
        for (const npy_intp atom : component) {
            if (roots.count(atom)) {
                has_root = true;
                break;
            }
        }
        if (!has_root) {
            roots.insert(*std::min_element(component.begin(), component.end()));
        }
    }

    std::set<Cycle, CycleOrder> candidates;
    std::set<Cycle, CycleOrder> processed;
    for (const npy_intp root : roots) {
        std::vector<npy_intp> parent(natoms, -2);
        std::vector<npy_intp> depth(natoms, -1);
        std::vector<npy_intp> queue{root};
        parent[root] = -1;
        depth[root] = 0;
        for (std::size_t index = 0; index < queue.size(); ++index) {
            const npy_intp atom = queue[index];
            for (const npy_intp neighbor : adjacency[atom]) {
                if (parent[neighbor] != -2) {
                    continue;
                }
                parent[neighbor] = atom;
                depth[neighbor] = depth[atom] + 1;
                queue.push_back(neighbor);
            }
        }
        for (const auto& [left, right] : edges) {
            if (parent[left] == right || parent[right] == left) {
                continue;
            }
            Cycle cycle = tree_cycle(left, right, parent, depth);
            if (cycle.size() < 3) {
                continue;
            }
            Cycle canonical = canonical_cycle(cycle);
            if (ring_max >= 0 &&
                static_cast<npy_intp>(canonical.size()) > ring_max) {
                continue;
            }
            if (!processed.insert(canonical).second) {
                continue;
            }
            if (chordless(adjacency, canonical)) {
                candidates.insert(canonical);
            } else {
                const auto parts = split_at_chords(adjacency, canonical);
                for (const Cycle& part : parts) {
                    if (ring_max < 0 ||
                        static_cast<npy_intp>(part.size()) <= ring_max) {
                        candidates.insert(part);
                    }
                }
            }
        }
    }
    return candidates;
}

void xor_vector(BitVector& target, const BitVector& source) {
    for (std::size_t word = 0; word < target.size(); ++word) {
        target[word] ^= source[word];
    }
}

npy_intp highest_bit(const BitVector& vector) {
    for (npy_intp word = static_cast<npy_intp>(vector.size()) - 1; word >= 0; --word) {
        if (vector[word] != 0) {
#if defined(__GNUC__) || defined(__clang__)
            const int local = 63 - __builtin_clzll(vector[word]);
#else
            int local = 0;
            std::uint64_t value = vector[word];
            while (value >>= 1U) {
                ++local;
            }
#endif
            return 64 * word + local;
        }
    }
    return -1;
}

std::vector<Cycle> minimum_cycle_basis(
    const std::set<Cycle, CycleOrder>& candidates,
    const std::vector<Edge>& edges,
    npy_intp target_rank) {
    std::map<Edge, npy_intp> edge_index;
    for (std::size_t index = 0; index < edges.size(); ++index) {
        edge_index[edges[index]] = static_cast<npy_intp>(index);
    }
    const std::size_t words = (edges.size() + 63) / 64;
    std::map<npy_intp, BitVector, std::greater<npy_intp>> basis;
    std::vector<Cycle> selected;
    for (const Cycle& cycle : candidates) {
        BitVector reduced(words, 0);
        for (std::size_t index = 0; index < cycle.size(); ++index) {
            Edge edge = std::minmax(
                cycle[index],
                cycle[(index + 1) % cycle.size()]);
            const npy_intp position = edge_index.at(edge);
            reduced[position / 64] ^= std::uint64_t{1} << (position % 64);
        }
        for (const auto& [pivot, row] : basis) {
            if (reduced[pivot / 64] & (std::uint64_t{1} << (pivot % 64))) {
                xor_vector(reduced, row);
            }
        }
        const npy_intp pivot = highest_bit(reduced);
        if (pivot < 0) {
            continue;
        }
        basis[pivot] = reduced;
        selected.push_back(cycle);
        if (static_cast<npy_intp>(selected.size()) >= target_rank) {
            break;
        }
    }
    return selected;
}

bool parse_graph(
    PyObject* edges_object,
    PyObject* allowed_object,
    npy_intp natoms,
    std::vector<Edge>& edges,
    std::vector<unsigned char>& allowed) {
    PyArrayObject* edge_array = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(edges_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* allowed_array = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(allowed_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    if (edge_array == nullptr || allowed_array == nullptr) {
        Py_XDECREF(edge_array);
        Py_XDECREF(allowed_array);
        return false;
    }
    if (natoms < 0 || PyArray_NDIM(edge_array) != 2 ||
        PyArray_DIM(edge_array, 1) != 2 || PyArray_NDIM(allowed_array) != 1) {
        Py_DECREF(edge_array);
        Py_DECREF(allowed_array);
        PyErr_SetString(PyExc_ValueError, "native topology graph arrays are invalid");
        return false;
    }
    allowed.assign(natoms, 0);
    const auto* allowed_data =
        static_cast<const npy_intp*>(PyArray_DATA(allowed_array));
    for (npy_intp index = 0; index < PyArray_DIM(allowed_array, 0); ++index) {
        const npy_intp atom = allowed_data[index];
        if (atom < 0 || atom >= natoms) {
            Py_DECREF(edge_array);
            Py_DECREF(allowed_array);
            PyErr_SetString(PyExc_ValueError, "allowed topology atom is out of range");
            return false;
        }
        allowed[atom] = 1;
    }
    const auto* edge_data = static_cast<const npy_intp*>(PyArray_DATA(edge_array));
    edges.reserve(PyArray_DIM(edge_array, 0));
    Edge previous{-1, -1};
    for (npy_intp index = 0; index < PyArray_DIM(edge_array, 0); ++index) {
        npy_intp left = edge_data[2 * index];
        npy_intp right = edge_data[2 * index + 1];
        if (left > right) {
            std::swap(left, right);
        }
        if (left < 0 || right >= natoms || left == right ||
            !allowed[left] || !allowed[right]) {
            Py_DECREF(edge_array);
            Py_DECREF(allowed_array);
            PyErr_SetString(PyExc_ValueError, "native topology edge is invalid");
            return false;
        }
        const Edge edge{left, right};
        if (!edges.empty() && edge <= previous) {
            Py_DECREF(edge_array);
            Py_DECREF(allowed_array);
            PyErr_SetString(
                PyExc_ValueError,
                "native topology edges must be unique and canonically sorted");
            return false;
        }
        edges.push_back(edge);
        previous = edge;
    }
    Py_DECREF(edge_array);
    Py_DECREF(allowed_array);
    return true;
}

PyObject* elementary_cycle_basis_native(PyObject*, PyObject* args) {
    Py_ssize_t natoms_raw;
    PyObject* edges_object;
    PyObject* allowed_object;
    PyObject* ring_max_object;
    if (!PyArg_ParseTuple(
            args, "nOOO",
            &natoms_raw, &edges_object, &allowed_object, &ring_max_object)) {
        return nullptr;
    }
    npy_intp ring_max = -1;
    if (ring_max_object != Py_None) {
        ring_max = PyLong_AsSsize_t(ring_max_object);
        if (ring_max == -1 && PyErr_Occurred()) {
            return nullptr;
        }
        if (ring_max < 3) {
            PyErr_SetString(PyExc_ValueError, "ring maximum must be at least three");
            return nullptr;
        }
    }
    const npy_intp natoms = static_cast<npy_intp>(natoms_raw);
    std::vector<Edge> edges;
    std::vector<unsigned char> allowed;
    if (!parse_graph(edges_object, allowed_object, natoms, edges, allowed)) {
        return nullptr;
    }
    npy_intp rank = 0;
    std::set<Cycle, CycleOrder> candidates;
    std::vector<Cycle> selected;
    Py_BEGIN_ALLOW_THREADS
    rank = cycle_rank(natoms, allowed, edges);
    if (rank > 0) {
        candidates = horton_candidates(natoms, allowed, edges, ring_max);
        selected = minimum_cycle_basis(candidates, edges, rank);
    }
    Py_END_ALLOW_THREADS
    PyObject* cycles = PyTuple_New(static_cast<Py_ssize_t>(selected.size()));
    if (cycles == nullptr) {
        return nullptr;
    }
    for (std::size_t index = 0; index < selected.size(); ++index) {
        PyObject* cycle = PyTuple_New(static_cast<Py_ssize_t>(selected[index].size()));
        if (cycle == nullptr) {
            Py_DECREF(cycles);
            return nullptr;
        }
        for (std::size_t atom = 0; atom < selected[index].size(); ++atom) {
            PyTuple_SET_ITEM(
                cycle,
                static_cast<Py_ssize_t>(atom),
                PyLong_FromSsize_t(selected[index][atom]));
        }
        PyTuple_SET_ITEM(cycles, static_cast<Py_ssize_t>(index), cycle);
    }
    return Py_BuildValue(
        "Nnn",
        cycles,
        static_cast<Py_ssize_t>(candidates.size()),
        static_cast<Py_ssize_t>(rank));
}

PyObject* local_primitive_values_sparse_gradients(PyObject*, PyObject* args) {
    PyObject* coordinates_object;
    PyObject* kinds_object;
    PyObject* atoms_object;
    PyObject* counts_object;
    PyObject* modes_object;
    if (!PyArg_ParseTuple(
            args, "OOOOO", &coordinates_object, &kinds_object, &atoms_object,
            &counts_object, &modes_object)) {
        return nullptr;
    }
    PyArrayObject* coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* kinds = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(kinds_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* atoms = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(atoms_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* counts = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(counts_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* modes = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(modes_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    if (coordinates == nullptr || kinds == nullptr || atoms == nullptr ||
        counts == nullptr || modes == nullptr) {
        Py_XDECREF(coordinates);
        Py_XDECREF(kinds);
        Py_XDECREF(atoms);
        Py_XDECREF(counts);
        Py_XDECREF(modes);
        return nullptr;
    }
    const npy_intp natoms = PyArray_DIM(coordinates, 0);
    const npy_intp primitive_count = PyArray_DIM(kinds, 0);
    if (PyArray_NDIM(coordinates) != 2 || PyArray_DIM(coordinates, 1) != 3 ||
        PyArray_NDIM(kinds) != 1 || PyArray_NDIM(atoms) != 2 ||
        PyArray_DIM(atoms, 0) != primitive_count || PyArray_DIM(atoms, 1) != 4 ||
        PyArray_NDIM(counts) != 1 || PyArray_DIM(counts, 0) != primitive_count ||
        PyArray_NDIM(modes) != 1 || PyArray_DIM(modes, 0) != primitive_count) {
        Py_DECREF(coordinates);
        Py_DECREF(kinds);
        Py_DECREF(atoms);
        Py_DECREF(counts);
        Py_DECREF(modes);
        PyErr_SetString(PyExc_ValueError, "native primitive batch arrays are invalid");
        return nullptr;
    }
    const auto* kind_data = static_cast<const npy_intp*>(PyArray_DATA(kinds));
    const auto* atom_data = static_cast<const npy_intp*>(PyArray_DATA(atoms));
    const auto* count_data = static_cast<const npy_intp*>(PyArray_DATA(counts));
    const auto* mode_data = static_cast<const npy_intp*>(PyArray_DATA(modes));
    for (npy_intp row = 0; row < primitive_count; ++row) {
        const npy_intp count = count_data[row];
        if (kind_data[row] < 1 || kind_data[row] > 9 || count < 2 || count > 4) {
            Py_DECREF(coordinates);
            Py_DECREF(kinds);
            Py_DECREF(atoms);
            Py_DECREF(counts);
            Py_DECREF(modes);
            PyErr_SetString(PyExc_ValueError, "unsupported native local primitive");
            return nullptr;
        }
        for (npy_intp slot = 0; slot < count; ++slot) {
            const npy_intp atom = atom_data[4 * row + slot];
            if (atom < 0 || atom >= natoms) {
                Py_DECREF(coordinates);
                Py_DECREF(kinds);
                Py_DECREF(atoms);
                Py_DECREF(counts);
                Py_DECREF(modes);
                PyErr_SetString(PyExc_ValueError, "native primitive atom is out of range");
                return nullptr;
            }
        }
    }
    npy_intp value_dimension = primitive_count;
    PyArrayObject* values = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, &value_dimension, NPY_DOUBLE));
    if (values == nullptr) {
        Py_DECREF(coordinates);
        Py_DECREF(kinds);
        Py_DECREF(atoms);
        Py_DECREF(counts);
        Py_DECREF(modes);
        return nullptr;
    }
    const auto* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    auto* value_data = static_cast<double*>(PyArray_DATA(values));
    struct PrimitiveRowResult {
        double value = 0.0;
        npy_intp nonzeros = 0;
        std::array<npy_intp, 12> indices{};
        std::array<double, 12> data{};
    };
    std::vector<PrimitiveRowResult> row_results(static_cast<std::size_t>(primitive_count));
    std::vector<double> sparse_data;
    std::vector<npy_intp> sparse_indices;
    std::vector<npy_intp> sparse_indptr;
    sparse_data.reserve(static_cast<std::size_t>(primitive_count) * 12);
    sparse_indices.reserve(static_cast<std::size_t>(primitive_count) * 12);
    sparse_indptr.reserve(static_cast<std::size_t>(primitive_count) + 1);
    sparse_indptr.push_back(0);
    auto evaluate_row = [&](npy_intp row) {
        std::array<LocalVector, 4> points{};
        const npy_intp count = count_data[row];
        for (npy_intp slot = 0; slot < count; ++slot) {
            const npy_intp atom = atom_data[4 * row + slot];
            for (npy_intp axis = 0; axis < 3; ++axis) {
                points[slot][axis] = dual_variable(
                    xyz[3 * atom + axis], static_cast<std::size_t>(3 * slot + axis));
            }
        }
        const LocalDual result = evaluate_local_primitive(
            kind_data[row], count, mode_data[row], points);
        PrimitiveRowResult& stored = row_results[static_cast<std::size_t>(row)];
        stored.value = result.value;
        for (npy_intp slot = 0; slot < count; ++slot) {
            const npy_intp atom = atom_data[4 * row + slot];
            for (npy_intp axis = 0; axis < 3; ++axis) {
                const double derivative = result.derivative[3 * slot + axis];
                if (std::abs(derivative) > 1.0e-15) {
                    const npy_intp position = stored.nonzeros++;
                    stored.indices[static_cast<std::size_t>(position)] = 3 * atom + axis;
                    stored.data[static_cast<std::size_t>(position)] = derivative;
                }
            }
        }
    };
    int worker_count = 1;
    if (primitive_count >= PARALLEL_PRIMITIVE_ROW_THRESHOLD) {
        const char* requested = std::getenv("MATRIX_COORDINATE_THREADS");
        if (requested == nullptr) requested = std::getenv("OMP_NUM_THREADS");
        if (requested != nullptr) worker_count = std::max(1, std::min(8, std::atoi(requested)));
    }
    worker_count = std::min(worker_count, static_cast<int>(primitive_count));
    Py_BEGIN_ALLOW_THREADS
    if (worker_count <= 1) {
        for (npy_intp row = 0; row < primitive_count; ++row) evaluate_row(row);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(worker_count));
        for (int worker = 0; worker < worker_count; ++worker) {
            workers.emplace_back([&, worker]() {
                for (npy_intp row = worker; row < primitive_count; row += worker_count)
                    evaluate_row(row);
            });
        }
        for (auto& worker : workers) worker.join();
    }
    Py_END_ALLOW_THREADS
    for (npy_intp row = 0; row < primitive_count; ++row) {
        const PrimitiveRowResult& stored = row_results[static_cast<std::size_t>(row)];
        value_data[row] = stored.value;
        for (npy_intp position = 0; position < stored.nonzeros; ++position) {
            sparse_indices.push_back(stored.indices[static_cast<std::size_t>(position)]);
            sparse_data.push_back(stored.data[static_cast<std::size_t>(position)]);
        }
        sparse_indptr.push_back(static_cast<npy_intp>(sparse_data.size()));
    }
    Py_DECREF(coordinates);
    Py_DECREF(kinds);
    Py_DECREF(atoms);
    Py_DECREF(counts);
    Py_DECREF(modes);
    PyObject* data_array = numpy_vector(sparse_data, NPY_DOUBLE);
    PyObject* index_array = numpy_vector(sparse_indices, NPY_INTP);
    PyObject* indptr_array = numpy_vector(sparse_indptr, NPY_INTP);
    if (data_array == nullptr || index_array == nullptr || indptr_array == nullptr) {
        Py_DECREF(values);
        Py_XDECREF(data_array);
        Py_XDECREF(index_array);
        Py_XDECREF(indptr_array);
        return nullptr;
    }
    return Py_BuildValue(
        "NNNN", reinterpret_cast<PyObject*>(values), data_array, index_array,
        indptr_array);
}

double periodic_match(double value, double reference, double period) {
    if (!(period > 0.0) || !std::isfinite(period)) return value;
    double delta = std::fmod(value - reference + 0.5 * period, period);
    if (delta < 0.0) delta += period;
    return reference + delta - 0.5 * period;
}

PyObject* local_gic_values(PyObject*, PyObject* args) {
    PyObject *coordinates, *kinds, *atoms, *counts, *modes;
    PyObject *reference_object, *combination_data_object,
        *combination_indices_object, *combination_indptr_object;
    PyObject *principal_object, *continuation_object;
    PyObject *primitive_phase_rows_object, *primitive_phase_columns_object,
        *primitive_phase_coefficients_object, *primitive_phase_references_object,
        *primitive_phase_periods_object;
    PyObject *gic_phase_rows_object, *gic_phase_references_object,
        *gic_phase_periods_object;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOOOOOOOOOOO", &coordinates, &kinds, &atoms, &counts,
            &modes, &reference_object, &combination_data_object,
            &combination_indices_object, &combination_indptr_object,
            &principal_object, &continuation_object,
            &primitive_phase_rows_object, &primitive_phase_columns_object,
            &primitive_phase_coefficients_object, &primitive_phase_references_object,
            &primitive_phase_periods_object, &gic_phase_rows_object,
            &gic_phase_references_object, &gic_phase_periods_object)) return nullptr;

    auto array = [](PyObject* object, int type) {
        return reinterpret_cast<PyArrayObject*>(
            PyArray_FROM_OTF(object, type, NPY_ARRAY_IN_ARRAY));
    };
    PyArrayObject* coordinate_array = array(coordinates, NPY_DOUBLE);
    PyArrayObject* kind_array = array(kinds, NPY_INTP);
    PyArrayObject* atom_array = array(atoms, NPY_INTP);
    PyArrayObject* count_array = array(counts, NPY_INTP);
    PyArrayObject* mode_array = array(modes, NPY_INTP);
    PyArrayObject* reference = array(reference_object, NPY_DOUBLE);
    PyArrayObject* combination_data = array(combination_data_object, NPY_DOUBLE);
    PyArrayObject* combination_indices = array(combination_indices_object, NPY_INTP);
    PyArrayObject* combination_indptr = array(combination_indptr_object, NPY_INTP);
    PyArrayObject* principal = array(principal_object, NPY_INTP);
    PyArrayObject* continuation = array(continuation_object, NPY_INTP);
    PyArrayObject* primitive_phase_rows = array(primitive_phase_rows_object, NPY_INTP);
    PyArrayObject* primitive_phase_columns = array(primitive_phase_columns_object, NPY_INTP);
    PyArrayObject* primitive_phase_coefficients = array(primitive_phase_coefficients_object, NPY_DOUBLE);
    PyArrayObject* primitive_phase_references = array(primitive_phase_references_object, NPY_DOUBLE);
    PyArrayObject* primitive_phase_periods = array(primitive_phase_periods_object, NPY_DOUBLE);
    PyArrayObject* gic_phase_rows = array(gic_phase_rows_object, NPY_INTP);
    PyArrayObject* gic_phase_references = array(gic_phase_references_object, NPY_DOUBLE);
    PyArrayObject* gic_phase_periods = array(gic_phase_periods_object, NPY_DOUBLE);
    std::vector<PyArrayObject*> held = {
        coordinate_array, kind_array, atom_array, count_array, mode_array,
        reference, combination_data, combination_indices, combination_indptr,
        principal, continuation, primitive_phase_rows, primitive_phase_columns,
        primitive_phase_coefficients, primitive_phase_references,
        primitive_phase_periods, gic_phase_rows, gic_phase_references,
        gic_phase_periods};
    if (std::any_of(held.begin(), held.end(), [](auto* value) { return value == nullptr; })) {
        for (auto* value : held) Py_XDECREF(value);
        return nullptr;
    }
    const npy_intp primitive_count = PyArray_SIZE(kind_array);
    const npy_intp row_count = PyArray_SIZE(combination_indptr) - 1;
    if (PyArray_NDIM(coordinate_array) != 2 || PyArray_DIM(coordinate_array, 1) != 3 ||
        PyArray_NDIM(atom_array) != 2 || PyArray_DIM(atom_array, 0) != primitive_count ||
        PyArray_DIM(atom_array, 1) != 4 || PyArray_SIZE(count_array) != primitive_count ||
        PyArray_SIZE(mode_array) != primitive_count || PyArray_SIZE(reference) != primitive_count ||
        row_count < 0 || PyArray_SIZE(combination_data) != PyArray_SIZE(combination_indices) ||
        PyArray_SIZE(primitive_phase_rows) != PyArray_SIZE(primitive_phase_columns) ||
        PyArray_SIZE(primitive_phase_rows) != PyArray_SIZE(primitive_phase_coefficients) ||
        PyArray_SIZE(primitive_phase_rows) != PyArray_SIZE(primitive_phase_references) ||
        PyArray_SIZE(primitive_phase_rows) != PyArray_SIZE(primitive_phase_periods) ||
        PyArray_SIZE(gic_phase_rows) != PyArray_SIZE(gic_phase_references) ||
        PyArray_SIZE(gic_phase_rows) != PyArray_SIZE(gic_phase_periods)) {
        for (auto* value : held) Py_DECREF(value);
        PyErr_SetString(PyExc_ValueError, "native GIC value-plan arrays are inconsistent");
        return nullptr;
    }
    const auto* xyz = static_cast<const double*>(PyArray_DATA(coordinate_array));
    const auto* kind_data = static_cast<const npy_intp*>(PyArray_DATA(kind_array));
    const auto* atom_data = static_cast<const npy_intp*>(PyArray_DATA(atom_array));
    const auto* count_data = static_cast<const npy_intp*>(PyArray_DATA(count_array));
    const auto* mode_data = static_cast<const npy_intp*>(PyArray_DATA(mode_array));
    const npy_intp atom_count = PyArray_DIM(coordinate_array, 0);
    std::vector<double> raw_values(static_cast<std::size_t>(primitive_count));
    for (npy_intp row = 0; row < primitive_count; ++row) {
        const npy_intp support_count = count_data[row];
        if (kind_data[row] < 1 || kind_data[row] > 6 || support_count < 2 || support_count > 4) {
            for (auto* value : held) Py_DECREF(value);
            PyErr_SetString(PyExc_ValueError, "native GIC plan contains an unsupported primitive");
            return nullptr;
        }
        std::array<ScalarVector, 4> points{};
        for (npy_intp slot = 0; slot < support_count; ++slot) {
            const npy_intp atom = atom_data[4 * row + slot];
            if (atom < 0 || atom >= atom_count) {
                for (auto* value : held) Py_DECREF(value);
                PyErr_SetString(PyExc_ValueError, "native GIC primitive atom is out of range");
                return nullptr;
            }
            points[slot] = {xyz[3 * atom], xyz[3 * atom + 1], xyz[3 * atom + 2]};
        }
        raw_values[static_cast<std::size_t>(row)] = evaluate_local_primitive_value(
            kind_data[row], support_count, mode_data[row], points);
    }
    const auto* raw_data = raw_values.data();
    const auto* ref_data = static_cast<const double*>(PyArray_DATA(reference));
    std::vector<double> primitive_values(raw_data, raw_data + primitive_count);
    const auto* principal_data = static_cast<const npy_intp*>(PyArray_DATA(principal));
    const auto* continuation_data = static_cast<const npy_intp*>(PyArray_DATA(continuation));
    auto valid_primitive_indices = [&](const npy_intp* indices, npy_intp count) {
        for (npy_intp k = 0; k < count; ++k)
            if (indices[k] < 0 || indices[k] >= primitive_count) return false;
        return true;
    };
    if (!valid_primitive_indices(principal_data, PyArray_SIZE(principal)) ||
        !valid_primitive_indices(continuation_data, PyArray_SIZE(continuation))) {
        for (auto* value : held) Py_DECREF(value);
        PyErr_SetString(PyExc_ValueError, "native GIC periodic primitive index is invalid");
        return nullptr;
    }
    for (npy_intp k = 0; k < PyArray_SIZE(principal); ++k) {
        const npy_intp index = principal_data[k];
        primitive_values[index] = periodic_match(primitive_values[index], ref_data[index], 2.0 * MATRIX_PI);
    }
    for (npy_intp k = 0; k < PyArray_SIZE(continuation); ++k) {
        const npy_intp index = continuation_data[k];
        primitive_values[index] = periodic_match(primitive_values[index], 0.0, 2.0 * MATRIX_PI);
    }
    std::vector<double> result(static_cast<size_t>(row_count), 0.0);
    const auto* cdata = static_cast<const double*>(PyArray_DATA(combination_data));
    const auto* cindices = static_cast<const npy_intp*>(PyArray_DATA(combination_indices));
    const auto* cindptr = static_cast<const npy_intp*>(PyArray_DATA(combination_indptr));
    const npy_intp combination_nonzeros = PyArray_SIZE(combination_data);
    if (cindptr[0] != 0 || cindptr[row_count] != combination_nonzeros ||
        !valid_primitive_indices(cindices, combination_nonzeros)) {
        for (auto* value : held) Py_DECREF(value);
        PyErr_SetString(PyExc_ValueError, "native GIC combination CSR is invalid");
        return nullptr;
    }
    for (npy_intp row = 0; row < row_count; ++row) {
        if (cindptr[row] > cindptr[row + 1]) {
            for (auto* value : held) Py_DECREF(value);
            PyErr_SetString(PyExc_ValueError, "native GIC combination CSR is not monotone");
            return nullptr;
        }
    }
    for (npy_intp row = 0; row < row_count; ++row)
        for (npy_intp p = cindptr[row]; p < cindptr[row + 1]; ++p)
            result[row] += cdata[p] * primitive_values[cindices[p]];

    const auto* prows = static_cast<const npy_intp*>(PyArray_DATA(primitive_phase_rows));
    const auto* pcols = static_cast<const npy_intp*>(PyArray_DATA(primitive_phase_columns));
    const auto* pcoeff = static_cast<const double*>(PyArray_DATA(primitive_phase_coefficients));
    const auto* prefs = static_cast<const double*>(PyArray_DATA(primitive_phase_references));
    const auto* pperiods = static_cast<const double*>(PyArray_DATA(primitive_phase_periods));
    if (!valid_primitive_indices(pcols, PyArray_SIZE(primitive_phase_columns))) {
        for (auto* value : held) Py_DECREF(value);
        PyErr_SetString(PyExc_ValueError, "native GIC outer primitive phase index is invalid");
        return nullptr;
    }
    for (npy_intp k = 0; k < PyArray_SIZE(primitive_phase_rows); ++k) {
        if (prows[k] < 0 || prows[k] >= row_count) {
            for (auto* value : held) Py_DECREF(value);
            PyErr_SetString(PyExc_ValueError, "native GIC primitive phase row is invalid");
            return nullptr;
        }
        const double value = raw_data[pcols[k]];
        result[prows[k]] += pcoeff[k] * (periodic_match(value, prefs[k], pperiods[k]) - value);
    }
    const auto* grows = static_cast<const npy_intp*>(PyArray_DATA(gic_phase_rows));
    const auto* grefs = static_cast<const double*>(PyArray_DATA(gic_phase_references));
    const auto* gperiods = static_cast<const double*>(PyArray_DATA(gic_phase_periods));
    for (npy_intp k = 0; k < PyArray_SIZE(gic_phase_rows); ++k) {
        const npy_intp row = grows[k];
        if (row < 0 || row >= row_count) {
            for (auto* value : held) Py_DECREF(value);
            PyErr_SetString(PyExc_ValueError, "native GIC phase row is invalid");
            return nullptr;
        }
        result[row] = periodic_match(result[row], grefs[k], gperiods[k]);
    }
    for (auto* value : held) Py_DECREF(value);
    npy_intp shape[1] = {row_count};
    PyObject* output = PyArray_SimpleNew(1, shape, NPY_DOUBLE);
    if (output != nullptr && row_count)
        std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(output)),
                    result.data(), static_cast<size_t>(row_count) * sizeof(double));
    return output;
}

PyObject* assemble_gic_csr(PyObject*, PyObject* args) {
    PyObject* primitive_data_object;
    PyObject* primitive_indices_object;
    PyObject* primitive_indptr_object;
    PyObject* combination_data_object;
    PyObject* combination_indices_object;
    PyObject* combination_indptr_object;
    Py_ssize_t column_count_raw;
    double zero_tolerance;
    if (!PyArg_ParseTuple(
            args, "OOOOOOnd", &primitive_data_object, &primitive_indices_object,
            &primitive_indptr_object, &combination_data_object,
            &combination_indices_object, &combination_indptr_object,
            &column_count_raw, &zero_tolerance)) {
        return nullptr;
    }
    PyArrayObject* primitive_data = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(primitive_data_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* primitive_indices = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(primitive_indices_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* primitive_indptr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(primitive_indptr_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* combination_data = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(combination_data_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* combination_indices = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(combination_indices_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* combination_indptr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(combination_indptr_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    auto release = [&]() {
        Py_XDECREF(primitive_data);
        Py_XDECREF(primitive_indices);
        Py_XDECREF(primitive_indptr);
        Py_XDECREF(combination_data);
        Py_XDECREF(combination_indices);
        Py_XDECREF(combination_indptr);
    };
    if (primitive_data == nullptr || primitive_indices == nullptr ||
        primitive_indptr == nullptr || combination_data == nullptr ||
        combination_indices == nullptr || combination_indptr == nullptr) {
        release();
        return nullptr;
    }
    if (column_count_raw < 0 || zero_tolerance < 0.0 ||
        PyArray_NDIM(primitive_data) != 1 ||
        PyArray_NDIM(primitive_indices) != 1 ||
        PyArray_NDIM(primitive_indptr) != 1 ||
        PyArray_NDIM(combination_data) != 1 ||
        PyArray_NDIM(combination_indices) != 1 ||
        PyArray_NDIM(combination_indptr) != 1 ||
        PyArray_SIZE(primitive_data) != PyArray_SIZE(primitive_indices) ||
        PyArray_SIZE(combination_data) != PyArray_SIZE(combination_indices) ||
        PyArray_SIZE(primitive_indptr) < 1 ||
        PyArray_SIZE(combination_indptr) < 1) {
        release();
        PyErr_SetString(PyExc_ValueError, "native GIC CSR arrays are invalid");
        return nullptr;
    }
    const auto* primitive_values =
        static_cast<const double*>(PyArray_DATA(primitive_data));
    const auto* primitive_columns =
        static_cast<const npy_intp*>(PyArray_DATA(primitive_indices));
    const auto* primitive_offsets =
        static_cast<const npy_intp*>(PyArray_DATA(primitive_indptr));
    const auto* combination_values =
        static_cast<const double*>(PyArray_DATA(combination_data));
    const auto* combination_columns =
        static_cast<const npy_intp*>(PyArray_DATA(combination_indices));
    const auto* combination_offsets =
        static_cast<const npy_intp*>(PyArray_DATA(combination_indptr));
    const npy_intp primitive_count = PyArray_SIZE(primitive_indptr) - 1;
    const npy_intp gic_count = PyArray_SIZE(combination_indptr) - 1;
    const npy_intp primitive_nnz = PyArray_SIZE(primitive_data);
    const npy_intp combination_nnz = PyArray_SIZE(combination_data);
    bool valid = primitive_offsets[0] == 0 &&
        primitive_offsets[primitive_count] == primitive_nnz &&
        combination_offsets[0] == 0 &&
        combination_offsets[gic_count] == combination_nnz;
    for (npy_intp index = 0; valid && index < primitive_count; ++index) {
        valid = primitive_offsets[index] <= primitive_offsets[index + 1];
    }
    for (npy_intp index = 0; valid && index < gic_count; ++index) {
        valid = combination_offsets[index] <= combination_offsets[index + 1];
    }
    for (npy_intp index = 0; valid && index < primitive_nnz; ++index) {
        valid = primitive_columns[index] >= 0 &&
            primitive_columns[index] < static_cast<npy_intp>(column_count_raw);
    }
    for (npy_intp index = 0; valid && index < combination_nnz; ++index) {
        valid = combination_columns[index] >= 0 &&
            combination_columns[index] < primitive_count;
    }
    if (!valid) {
        release();
        PyErr_SetString(PyExc_ValueError, "native GIC CSR structure is invalid");
        return nullptr;
    }

    std::vector<double> result_data;
    std::vector<npy_intp> result_indices;
    std::vector<npy_intp> result_indptr;
    result_data.reserve(static_cast<std::size_t>(combination_nnz) * 6);
    result_indices.reserve(static_cast<std::size_t>(combination_nnz) * 6);
    result_indptr.reserve(static_cast<std::size_t>(gic_count) + 1);
    result_indptr.push_back(0);
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp row = 0; row < gic_count; ++row) {
        std::map<npy_intp, double> accumulator;
        for (npy_intp item = combination_offsets[row];
             item < combination_offsets[row + 1]; ++item) {
            const npy_intp primitive = combination_columns[item];
            const double coefficient = combination_values[item];
            for (npy_intp entry = primitive_offsets[primitive];
                 entry < primitive_offsets[primitive + 1]; ++entry) {
                accumulator[primitive_columns[entry]] +=
                    coefficient * primitive_values[entry];
            }
        }
        for (const auto& [column, value] : accumulator) {
            if (std::abs(value) > zero_tolerance) {
                result_indices.push_back(column);
                result_data.push_back(value);
            }
        }
        result_indptr.push_back(static_cast<npy_intp>(result_data.size()));
    }
    Py_END_ALLOW_THREADS
    release();
    PyObject* data_array = numpy_vector(result_data, NPY_DOUBLE);
    PyObject* index_array = numpy_vector(result_indices, NPY_INTP);
    PyObject* indptr_array = numpy_vector(result_indptr, NPY_INTP);
    if (data_array == nullptr || index_array == nullptr || indptr_array == nullptr) {
        Py_XDECREF(data_array);
        Py_XDECREF(index_array);
        Py_XDECREF(indptr_array);
        return nullptr;
    }
    return Py_BuildValue("NNN", data_array, index_array, indptr_array);
}

PyObject* csr_product(PyObject* args, bool transpose) {
    PyObject *data_object, *indices_object, *indptr_object, *vector_object;
    Py_ssize_t column_count_raw;
    if (!PyArg_ParseTuple(args, "OOOOn", &data_object, &indices_object,
                          &indptr_object, &vector_object, &column_count_raw))
        return nullptr;
    PyArrayObject* data = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(data_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* indices = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(indices_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* indptr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(indptr_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* vector = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(vector_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto release = [&]() { Py_XDECREF(data); Py_XDECREF(indices); Py_XDECREF(indptr); Py_XDECREF(vector); };
    if (!data || !indices || !indptr || !vector) { release(); return nullptr; }
    const npy_intp rows = PyArray_SIZE(indptr) - 1;
    const npy_intp columns = static_cast<npy_intp>(column_count_raw);
    const npy_intp nnz = PyArray_SIZE(data);
    const auto* offsets = static_cast<const npy_intp*>(PyArray_DATA(indptr));
    const auto* column = static_cast<const npy_intp*>(PyArray_DATA(indices));
    const auto* value = static_cast<const double*>(PyArray_DATA(data));
    const auto* input = static_cast<const double*>(PyArray_DATA(vector));
    bool valid = columns >= 0 && PyArray_NDIM(data) == 1 && PyArray_NDIM(indices) == 1 &&
        PyArray_NDIM(indptr) == 1 && PyArray_NDIM(vector) == 1 &&
        PyArray_SIZE(indices) == nnz && rows >= 0 && offsets[0] == 0 && offsets[rows] == nnz &&
        PyArray_SIZE(vector) == (transpose ? rows : columns);
    for (npy_intp row = 0; valid && row < rows; ++row)
        valid = offsets[row] <= offsets[row + 1];
    for (npy_intp position = 0; valid && position < nnz; ++position)
        valid = column[position] >= 0 && column[position] < columns && std::isfinite(value[position]);
    if (!valid) { release(); PyErr_SetString(PyExc_ValueError, "invalid CSR product arrays"); return nullptr; }
    npy_intp shape[1] = {transpose ? columns : rows};
    PyObject* output = PyArray_ZEROS(1, shape, NPY_DOUBLE, 0);
    if (!output) { release(); return nullptr; }
    auto* result = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(output)));
    int worker_count = 1;
    const char* requested = std::getenv("MATRIX_COORDINATE_THREADS");
    if (requested == nullptr) requested = std::getenv("OMP_NUM_THREADS");
    if (requested != nullptr && nnz >= 4096)
        worker_count = std::max(1, std::min(8, std::atoi(requested)));
    worker_count = std::min(worker_count, static_cast<int>(std::max<npy_intp>(rows, 1)));
    Py_BEGIN_ALLOW_THREADS
    if (worker_count <= 1) {
        for (npy_intp row = 0; row < rows; ++row)
            for (npy_intp position = offsets[row]; position < offsets[row + 1]; ++position)
                if (transpose) result[column[position]] += value[position] * input[row];
                else result[row] += value[position] * input[column[position]];
    } else if (!transpose) {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(worker_count));
        for (int worker = 0; worker < worker_count; ++worker)
            workers.emplace_back([&, worker]() {
                const npy_intp begin = rows * worker / worker_count;
                const npy_intp end = rows * (worker + 1) / worker_count;
                for (npy_intp row = begin; row < end; ++row) {
                    double sum = 0.0;
                    for (npy_intp position = offsets[row]; position < offsets[row + 1]; ++position)
                        sum += value[position] * input[column[position]];
                    result[row] = sum;
                }
            });
        for (auto& worker : workers) worker.join();
    } else {
        std::vector<std::vector<double>> partial(
            static_cast<std::size_t>(worker_count),
            std::vector<double>(static_cast<std::size_t>(columns), 0.0));
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(worker_count));
        for (int worker = 0; worker < worker_count; ++worker)
            workers.emplace_back([&, worker]() {
                const npy_intp begin = rows * worker / worker_count;
                const npy_intp end = rows * (worker + 1) / worker_count;
                auto& local = partial[static_cast<std::size_t>(worker)];
                for (npy_intp row = begin; row < end; ++row)
                    for (npy_intp position = offsets[row]; position < offsets[row + 1]; ++position)
                        local[static_cast<std::size_t>(column[position])] +=
                            value[position] * input[row];
            });
        for (auto& worker : workers) worker.join();
        for (int worker = 0; worker < worker_count; ++worker)
            for (npy_intp item = 0; item < columns; ++item)
                result[item] += partial[static_cast<std::size_t>(worker)][static_cast<std::size_t>(item)];
    }
    Py_END_ALLOW_THREADS
    release();
    return output;
}

PyObject* csr_matvec(PyObject*, PyObject* args) { return csr_product(args, false); }
PyObject* csr_transpose_matvec(PyObject*, PyObject* args) { return csr_product(args, true); }

PyObject* csr_incremental_matching(PyObject*, PyObject* args) {
    PyObject *indices_object, *indptr_object, *matching_object;
    Py_ssize_t column_count_raw, start_row_raw;
    if (!PyArg_ParseTuple(args, "OOOnn", &indices_object, &indptr_object,
                          &matching_object, &column_count_raw, &start_row_raw)) return nullptr;
    PyArrayObject* indices = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(indices_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* indptr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(indptr_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* matching = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(matching_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    auto release = [&]() { Py_XDECREF(indices); Py_XDECREF(indptr); Py_XDECREF(matching); };
    if (!indices || !indptr || !matching) { release(); return nullptr; }
    const npy_intp rows = PyArray_SIZE(indptr) - 1;
    const npy_intp columns = static_cast<npy_intp>(column_count_raw);
    const npy_intp start_row = static_cast<npy_intp>(start_row_raw);
    const auto* offsets = static_cast<const npy_intp*>(PyArray_DATA(indptr));
    const auto* support = static_cast<const npy_intp*>(PyArray_DATA(indices));
    const auto* initial = static_cast<const npy_intp*>(PyArray_DATA(matching));
    bool valid = rows >= 0 && columns >= 0 && start_row >= 0 && start_row <= rows &&
        PyArray_NDIM(indices) == 1 && PyArray_NDIM(indptr) == 1 &&
        PyArray_NDIM(matching) == 1 && PyArray_SIZE(matching) == columns &&
        offsets[0] == 0 && offsets[rows] == PyArray_SIZE(indices);
    if (!valid) { release(); PyErr_SetString(PyExc_ValueError, "invalid incremental matching arrays"); return nullptr; }
    std::vector<npy_intp> owner(initial, initial + columns);
    bool complete = true;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp row = start_row; row < rows; ++row) {
        std::vector<char> seen_rows(static_cast<size_t>(rows), 0);
        std::vector<char> seen_columns(static_cast<size_t>(columns), 0);
        std::function<bool(npy_intp)> augment = [&](npy_intp candidate) {
            if (seen_rows[candidate]) return false;
            seen_rows[candidate] = 1;
            for (npy_intp position = offsets[candidate]; position < offsets[candidate + 1]; ++position) {
                const npy_intp column = support[position];
                if (column < 0 || column >= columns || seen_columns[column]) continue;
                seen_columns[column] = 1;
                if (owner[column] < 0 || augment(owner[column])) { owner[column] = candidate; return true; }
            }
            return false;
        };
        if (!augment(row)) { complete = false; break; }
    }
    Py_END_ALLOW_THREADS
    if (!complete) { release(); Py_RETURN_NONE; }
    npy_intp shape[1] = {columns};
    PyObject* output = PyArray_SimpleNew(1, shape, NPY_INTP);
    if (output) std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(output)),
                            owner.data(), static_cast<size_t>(columns) * sizeof(npy_intp));
    release(); return output;
}

PyObject* csr_support_certificate(PyObject*, PyObject* args) {
    PyObject *indices_object, *indptr_object;
    Py_ssize_t column_count_raw;
    if (!PyArg_ParseTuple(args, "OOn", &indices_object, &indptr_object,
                          &column_count_raw)) return nullptr;
    PyArrayObject* indices = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(indices_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* indptr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(indptr_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    auto release = [&]() { Py_XDECREF(indices); Py_XDECREF(indptr); };
    if (!indices || !indptr) { release(); return nullptr; }
    const npy_intp rows = PyArray_SIZE(indptr) - 1;
    const npy_intp columns = static_cast<npy_intp>(column_count_raw);
    const auto* offsets = static_cast<const npy_intp*>(PyArray_DATA(indptr));
    const auto* support = static_cast<const npy_intp*>(PyArray_DATA(indices));
    bool valid = rows >= 0 && columns >= 0 && PyArray_NDIM(indices) == 1 &&
        PyArray_NDIM(indptr) == 1 && offsets[0] == 0 &&
        offsets[rows] == PyArray_SIZE(indices);
    for (npy_intp row = 0; valid && row < rows; ++row)
        valid = offsets[row] <= offsets[row + 1];
    for (npy_intp item = 0; valid && item < PyArray_SIZE(indices); ++item)
        valid = support[item] >= 0 && support[item] < columns;
    if (!valid) {
        release();
        PyErr_SetString(PyExc_ValueError, "invalid CSR support arrays");
        return nullptr;
    }
    std::vector<npy_intp> owner(static_cast<size_t>(columns), -1);
    std::vector<npy_intp> unsupported;
    std::vector<npy_intp> parent(static_cast<size_t>(rows));
    std::vector<npy_intp> first_row(static_cast<size_t>(columns), -1);
    npy_intp component_count = 0;
    for (npy_intp row = 0; row < rows; ++row) parent[row] = row;
    auto find_root = [&](npy_intp item) {
        npy_intp root = item;
        while (parent[root] != root) root = parent[root];
        while (parent[item] != item) {
            const npy_intp next = parent[item]; parent[item] = root; item = next;
        }
        return root;
    };
    auto unite = [&](npy_intp left, npy_intp right) {
        const npy_intp a = find_root(left), b = find_root(right);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
    };
    npy_intp rank = 0;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp row = 0; row < rows; ++row) {
        if (offsets[row] == offsets[row + 1]) unsupported.push_back(row);
        for (npy_intp item = offsets[row]; item < offsets[row + 1]; ++item) {
            const npy_intp column = support[item];
            if (first_row[column] >= 0) unite(row, first_row[column]);
            else first_row[column] = row;
        }
        std::vector<char> seen_rows(static_cast<size_t>(rows), 0);
        std::vector<char> seen_columns(static_cast<size_t>(columns), 0);
        std::function<bool(npy_intp)> augment = [&](npy_intp candidate) {
            if (seen_rows[candidate]) return false;
            seen_rows[candidate] = 1;
            for (npy_intp item = offsets[candidate]; item < offsets[candidate + 1]; ++item) {
                const npy_intp column = support[item];
                if (seen_columns[column]) continue;
                seen_columns[column] = 1;
                if (owner[column] < 0 || augment(owner[column])) {
                    owner[column] = candidate; return true;
                }
            }
            return false;
        };
        if (augment(row)) ++rank;
    }
    std::set<npy_intp> roots;
    for (npy_intp row = 0; row < rows; ++row) roots.insert(find_root(row));
    component_count = static_cast<npy_intp>(roots.size());
    Py_END_ALLOW_THREADS
    PyObject* unsupported_list = PyList_New(static_cast<Py_ssize_t>(unsupported.size()));
    PyObject* owner_array = numpy_vector(owner, NPY_INTP);
    if (!unsupported_list || !owner_array) {
        Py_XDECREF(unsupported_list); Py_XDECREF(owner_array); release(); return nullptr;
    }
    for (size_t index = 0; index < unsupported.size(); ++index)
        PyList_SET_ITEM(unsupported_list, static_cast<Py_ssize_t>(index),
                        PyLong_FromSsize_t(unsupported[index]));
    PyObject* result = Py_BuildValue(
        "{s:n,s:n,s:N,s:N}",
        "rank", rank,
        "connected_components", component_count,
        "unsupported_rows", unsupported_list,
        "column_to_row", owner_array);
    release();
    return result;
}

PyObject* build_info(PyObject*, PyObject*) {
#if defined(__aarch64__) || defined(_M_ARM64)
    const char* architecture = "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    const char* architecture = "x86_64";
#else
    const char* architecture = "generic";
#endif
#if defined(__clang__)
    const char* compiler =
        "clang-" MATRIX_STRINGIFY(__clang_major__) "." MATRIX_STRINGIFY(__clang_minor__);
#elif defined(__GNUC__)
    const char* compiler =
        "gcc-" MATRIX_STRINGIFY(__GNUC__) "." MATRIX_STRINGIFY(__GNUC_MINOR__);
#elif defined(_MSC_VER)
    const char* compiler = "msvc-" MATRIX_STRINGIFY(_MSC_VER);
#else
    const char* compiler = "unknown";
#endif
    return Py_BuildValue(
        "{s:s,s:s,s:s,s:O,s:s}",
        "implementation", "cpp",
        "architecture", architecture,
        "compiler", compiler,
        "openmp", Py_False,
        "precision", "integer-exact");
}

PyMethodDef methods[] = {
    {"perceive_continuous_graph", perceive_continuous_graph, METH_VARARGS,
     "Continuous geometry-first graph perception primitives."},
    {"elementary_cycle_basis", elementary_cycle_basis_native, METH_VARARGS,
     "Deterministic unweighted Horton minimum cycle basis."},
    {"local_primitive_values_sparse_gradients",
     local_primitive_values_sparse_gradients, METH_VARARGS,
     "Compiled local primitive values and CSR Wilson rows."},
    {"local_gic_values", local_gic_values, METH_VARARGS,
     "Compiled local primitive, periodic-phase and GIC value evaluation."},
    {"assemble_gic_csr", assemble_gic_csr, METH_VARARGS,
     "Assemble GIC Wilson rows from primitive and combination CSR arrays."},
    {"csr_matvec", csr_matvec, METH_VARARGS, "Apply a CSR matrix to a dense vector."},
    {"csr_transpose_matvec", csr_transpose_matvec, METH_VARARGS,
     "Apply the transpose of a CSR matrix to a dense vector."},
    {"csr_incremental_matching", csr_incremental_matching, METH_VARARGS,
     "Extend an existing bipartite CSR support matching."},
    {"csr_support_certificate", csr_support_certificate, METH_VARARGS,
     "Fuse CSR support validation, matching, unsupported rows and components."},
    {"smith_primitive_values_sparse_gradients",
     smith_primitive_values_sparse_gradients, METH_VARARGS,
     "Compiled complete SMITH primitive-atlas values and CSR Wilson rows."},
    {"build_info", build_info, METH_NOARGS, "Describe this native build."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_chem_native",
    "Portable compiled MATRIX chemistry kernels.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__chem_native() {
    import_array();
    return PyModule_Create(&module);
}
