#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_24_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

namespace {

constexpr double BOHR_PER_ANGSTROM = 1.8897261254578281;

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double value = 0.0;
    for (size_t i = 0; i < a.size(); ++i) value += a[i] * b[i];
    return value;
}

double norm(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }

PyObject* numpy_vector_copy(const std::vector<double>& values) {
    npy_intp size[1] = {static_cast<npy_intp>(values.size())};
    PyObject* result = PyArray_SimpleNew(1, size, NPY_DOUBLE);
    if (result != nullptr && !values.empty())
        std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(result)),
                    values.data(), values.size() * sizeof(double));
    return result;
}

PyObject* numpy_coordinates_copy(const std::vector<double>& values, npy_intp atoms) {
    npy_intp shape[2] = {atoms, 3};
    PyObject* result = PyArray_SimpleNew(2, shape, NPY_DOUBLE);
    if (result != nullptr && !values.empty())
        std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(result)),
                    values.data(), values.size() * sizeof(double));
    return result;
}

bool finite_vector_from_object(PyObject* object, size_t expected,
                               std::vector<double>& values) {
    auto* array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (array == nullptr) return false;
    if (static_cast<size_t>(PyArray_SIZE(array)) != expected) {
        Py_DECREF(array);
        PyErr_SetString(PyExc_ValueError, "native SONIC callback returned a wrong-sized array");
        return false;
    }
    const auto* data = static_cast<const double*>(PyArray_DATA(array));
    values.assign(data, data + expected);
    Py_DECREF(array);
    if (!std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); })) {
        PyErr_SetString(PyExc_FloatingPointError,
                        "native SONIC callback returned non-finite values");
        return false;
    }
    return true;
}

bool call_values(PyObject* callback, const std::vector<double>& coordinates,
                 npy_intp atoms, size_t coordinate_count,
                 std::vector<double>& values) {
    PyObject* point = numpy_coordinates_copy(coordinates, atoms);
    if (point == nullptr) return false;
    PyObject* output = nullptr;
    // A tuple (native_function, immutable_plan...) is a callback-free native
    // value runtime.  The controller supplies only the current geometry and
    // calls the extension entry point directly, without returning through a
    // Python frame for every line-search trial.
    if (PyTuple_Check(callback) && PyTuple_Size(callback) >= 1 &&
        PyCallable_Check(PyTuple_GET_ITEM(callback, 0))) {
        const Py_ssize_t count = PyTuple_Size(callback);
        PyObject* native_args = PyTuple_New(count);
        if (native_args == nullptr) { Py_DECREF(point); return false; }
        PyTuple_SET_ITEM(native_args, 0, point);
        point = nullptr;
        for (Py_ssize_t index = 1; index < count; ++index) {
            PyObject* item = PyTuple_GET_ITEM(callback, index);
            Py_INCREF(item);
            PyTuple_SET_ITEM(native_args, index, item);
        }
        output = PyObject_CallObject(PyTuple_GET_ITEM(callback, 0), native_args);
        Py_DECREF(native_args);
    } else {
        output = PyObject_CallOneArg(callback, point);
    }
    Py_XDECREF(point);
    if (output == nullptr) return false;
    const bool valid = finite_vector_from_object(output, coordinate_count, values);
    Py_DECREF(output);
    return valid;
}

bool call_projection(PyObject* callback, std::vector<double>& coordinates,
                     npy_intp atoms) {
    if (callback == Py_None) return true;
    PyObject* point = numpy_coordinates_copy(coordinates, atoms);
    if (point == nullptr) return false;
    PyObject* output = PyObject_CallOneArg(callback, point);
    Py_DECREF(point);
    if (output == nullptr) return false;
    std::vector<double> projected;
    const bool valid = finite_vector_from_object(output, coordinates.size(), projected);
    Py_DECREF(output);
    if (valid) coordinates.swap(projected);
    return valid;
}

bool call_values_and_operator(PyObject* callback,
                              const std::vector<double>& coordinates,
                              npy_intp atoms, size_t coordinate_count,
                              std::vector<double>& values, PyObject*& operator_object) {
    PyObject* point = numpy_coordinates_copy(coordinates, atoms);
    if (point == nullptr) return false;
    PyObject* output = PyObject_CallOneArg(callback, point);
    Py_DECREF(point);
    if (output == nullptr) return false;
    if (!PyTuple_Check(output) || PyTuple_Size(output) != 2) {
        Py_DECREF(output);
        PyErr_SetString(PyExc_TypeError,
                        "native SONIC operator callback must return (values, operator)");
        return false;
    }
    if (!finite_vector_from_object(PyTuple_GET_ITEM(output, 0), coordinate_count, values)) {
        Py_DECREF(output);
        return false;
    }
    PyObject* replacement = PyTuple_GET_ITEM(output, 1);
    if (!PyObject_HasAttrString(replacement, "matvec") ||
        !PyObject_HasAttrString(replacement, "b_matvec")) {
        Py_DECREF(output);
        PyErr_SetString(PyExc_TypeError, "native SONIC callback returned an invalid operator");
        return false;
    }
    Py_INCREF(replacement);
    Py_XDECREF(operator_object);
    operator_object = replacement;
    Py_DECREF(output);
    return true;
}

bool call_operator_vector(PyObject* operator_object, const char* method,
                          const std::vector<double>& input,
                          std::vector<double>& output,
                          double damping = 0.0, bool pass_damping = false) {
    if (!pass_damping && std::strcmp(method, "matvec") == 0) {
        PyObject* plan = PyObject_GetAttrString(operator_object, "_native_inverse_plan");
        if (plan != nullptr && plan != Py_None && PyTuple_Check(plan) &&
            PyTuple_Size(plan) == 6) {
            auto* row_scale = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
                PyTuple_GET_ITEM(plan, 1), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
            auto* free_columns = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
                PyTuple_GET_ITEM(plan, 2), NPY_INTP, NPY_ARRAY_IN_ARRAY));
            auto* centered = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
                PyTuple_GET_ITEM(plan, 3), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
            auto* rotation_inverse = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
                PyTuple_GET_ITEM(plan, 4), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
            const long ncart = PyLong_AsLong(PyTuple_GET_ITEM(plan, 5));
            const bool dimensions_ok = row_scale != nullptr && free_columns != nullptr &&
                centered != nullptr && rotation_inverse != nullptr && !PyErr_Occurred() &&
                ncart == static_cast<long>(output.size()) &&
                PyArray_SIZE(row_scale) == static_cast<npy_intp>(input.size()) &&
                PyArray_SIZE(free_columns) == static_cast<npy_intp>(input.size()) &&
                PyArray_SIZE(centered) == ncart && PyArray_SIZE(rotation_inverse) == 9;
            if (dimensions_ok) {
                const auto* scale = static_cast<const double*>(PyArray_DATA(row_scale));
                std::vector<double> rhs(input.size());
                for (size_t i = 0; i < input.size(); ++i) rhs[i] = scale[i] * input[i];
                PyObject* rhs_object = numpy_vector_copy(rhs);
                PyObject* solution_object = rhs_object == nullptr ? nullptr :
                    PyObject_CallOneArg(PyTuple_GET_ITEM(plan, 0), rhs_object);
                Py_XDECREF(rhs_object);
                std::vector<double> solution;
                const bool solved = solution_object != nullptr &&
                    finite_vector_from_object(solution_object, input.size(), solution);
                Py_XDECREF(solution_object);
                if (solved) {
                    std::fill(output.begin(), output.end(), 0.0);
                    const auto* free = static_cast<const npy_intp*>(PyArray_DATA(free_columns));
                    bool valid = true;
                    for (size_t i = 0; i < solution.size(); ++i) {
                        if (free[i] < 0 || free[i] >= ncart) { valid = false; break; }
                        output[static_cast<size_t>(free[i])] = solution[i];
                    }
                    if (valid && ncart % 3 == 0) {
                        const auto* positions = static_cast<const double*>(PyArray_DATA(centered));
                        const auto* inverse = static_cast<const double*>(PyArray_DATA(rotation_inverse));
                        const size_t atoms = static_cast<size_t>(ncart / 3);
                        double translation[3] = {0.0, 0.0, 0.0};
                        for (size_t atom = 0; atom < atoms; ++atom)
                            for (size_t axis = 0; axis < 3; ++axis)
                                translation[axis] += output[3 * atom + axis];
                        for (double& value : translation) value /= static_cast<double>(atoms);
                        double angular_rhs[3] = {0.0, 0.0, 0.0};
                        for (size_t atom = 0; atom < atoms; ++atom) {
                            const double vx = output[3 * atom] - translation[0];
                            const double vy = output[3 * atom + 1] - translation[1];
                            const double vz = output[3 * atom + 2] - translation[2];
                            const double x = positions[3 * atom], y = positions[3 * atom + 1],
                                         z = positions[3 * atom + 2];
                            angular_rhs[0] += y * vz - z * vy;
                            angular_rhs[1] += z * vx - x * vz;
                            angular_rhs[2] += x * vy - y * vx;
                        }
                        double omega[3] = {0.0, 0.0, 0.0};
                        for (size_t row = 0; row < 3; ++row)
                            for (size_t column = 0; column < 3; ++column)
                                omega[row] += inverse[3 * row + column] * angular_rhs[column];
                        for (size_t atom = 0; atom < atoms; ++atom) {
                            const double x = positions[3 * atom], y = positions[3 * atom + 1],
                                         z = positions[3 * atom + 2];
                            const double rigid[3] = {
                                translation[0] + omega[1] * z - omega[2] * y,
                                translation[1] + omega[2] * x - omega[0] * z,
                                translation[2] + omega[0] * y - omega[1] * x};
                            for (size_t axis = 0; axis < 3; ++axis)
                                output[3 * atom + axis] -= rigid[axis];
                        }
                    }
                    Py_DECREF(row_scale); Py_DECREF(free_columns); Py_DECREF(centered);
                    Py_DECREF(rotation_inverse); Py_DECREF(plan);
                    if (valid) return true;
                    PyErr_SetString(PyExc_ValueError, "native SONIC inverse plan is invalid");
                    return false;
                }
            }
            Py_XDECREF(row_scale); Py_XDECREF(free_columns); Py_XDECREF(centered);
            Py_XDECREF(rotation_inverse); Py_DECREF(plan);
            if (PyErr_Occurred()) return false;
        } else {
            Py_XDECREF(plan);
            PyErr_Clear();
        }
    }
    if (!pass_damping && std::strcmp(method, "b_matvec") == 0) {
        PyObject* plan = PyObject_GetAttrString(operator_object, "_native_b_csr");
        if (plan != nullptr && PyTuple_Check(plan) && PyTuple_Size(plan) == 4) {
            auto* data = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
                PyTuple_GET_ITEM(plan, 0), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
            auto* indices = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
                PyTuple_GET_ITEM(plan, 1), NPY_INTP, NPY_ARRAY_IN_ARRAY));
            auto* indptr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
                PyTuple_GET_ITEM(plan, 2), NPY_INTP, NPY_ARRAY_IN_ARRAY));
            const long columns = PyLong_AsLong(PyTuple_GET_ITEM(plan, 3));
            if (data != nullptr && indices != nullptr && indptr != nullptr &&
                !PyErr_Occurred() && columns == static_cast<long>(input.size()) &&
                PyArray_SIZE(indptr) == static_cast<npy_intp>(output.size() + 1) &&
                PyArray_SIZE(data) == PyArray_SIZE(indices)) {
                const auto* values = static_cast<const double*>(PyArray_DATA(data));
                const auto* column = static_cast<const npy_intp*>(PyArray_DATA(indices));
                const auto* offsets = static_cast<const npy_intp*>(PyArray_DATA(indptr));
                const npy_intp nonzeros = PyArray_SIZE(data);
                bool valid = offsets[0] == 0 && offsets[output.size()] == nonzeros;
                for (size_t row = 0; valid && row < output.size(); ++row) {
                    if (offsets[row] > offsets[row + 1]) { valid = false; break; }
                    double sum = 0.0;
                    for (npy_intp p = offsets[row]; p < offsets[row + 1]; ++p) {
                        if (column[p] < 0 || column[p] >= columns) { valid = false; break; }
                        sum += values[p] * input[static_cast<size_t>(column[p])];
                    }
                    output[row] = sum;
                }
                Py_DECREF(data); Py_DECREF(indices); Py_DECREF(indptr); Py_DECREF(plan);
                if (valid && std::all_of(output.begin(), output.end(),
                                         [](double value) { return std::isfinite(value); }))
                    return true;
                PyErr_SetString(PyExc_ValueError, "native SONIC Wilson CSR plan is invalid");
                return false;
            }
            Py_XDECREF(data); Py_XDECREF(indices); Py_XDECREF(indptr);
            Py_DECREF(plan);
            if (PyErr_Occurred()) return false;
        } else {
            Py_XDECREF(plan);
            PyErr_Clear();
        }
    }
    PyObject* callable = PyObject_GetAttrString(operator_object, method);
    PyObject* vector = numpy_vector_copy(input);
    if (callable == nullptr || vector == nullptr) {
        Py_XDECREF(callable); Py_XDECREF(vector); return false;
    }
    PyObject* arguments = PyTuple_Pack(1, vector);
    Py_DECREF(vector);
    PyObject* keywords = nullptr;
    if (pass_damping) {
        keywords = Py_BuildValue("{s:d}", "damping", damping);
    }
    PyObject* result =
        arguments == nullptr ? nullptr : PyObject_Call(callable, arguments, keywords);
    Py_XDECREF(keywords); Py_XDECREF(arguments); Py_DECREF(callable);
    if (result == nullptr) return false;
    const bool valid = finite_vector_from_object(result, output.size(), output);
    Py_DECREF(result);
    return valid;
}

bool call_trust_realization(PyObject* callback, double scale,
                            double& rmsd, bool& valid) {
    PyObject* argument = PyFloat_FromDouble(scale);
    if (argument == nullptr) return false;
    PyObject* output = PyObject_CallOneArg(callback, argument);
    Py_DECREF(argument);
    if (output == nullptr) return false;
    if (!PyTuple_Check(output) || PyTuple_Size(output) != 2) {
        Py_DECREF(output);
        PyErr_SetString(PyExc_TypeError,
                        "native trust callback must return (rmsd, valid)");
        return false;
    }
    rmsd = PyFloat_AsDouble(PyTuple_GET_ITEM(output, 0));
    const int truth = PyObject_IsTrue(PyTuple_GET_ITEM(output, 1));
    Py_DECREF(output);
    if (truth < 0 || (rmsd == -1.0 && PyErr_Occurred())) return false;
    valid = truth != 0;
    return true;
}

struct Evaluation {
    double energy{};
    std::vector<double> gradient;
};

bool evaluate(PyObject* callback, const std::vector<double>& x_bohr, npy_intp atoms,
              Evaluation& result) {
    npy_intp dims[2] = {atoms, 3};
    PyObject* coordinates = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (coordinates == nullptr) return false;
    auto* data = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(coordinates)));
    for (npy_intp i = 0; i < 3 * atoms; ++i) data[i] = x_bohr[i] / BOHR_PER_ANGSTROM;
    PyObject* output = PyObject_CallOneArg(callback, coordinates);
    Py_DECREF(coordinates);
    if (output == nullptr) return false;
    if (!PyTuple_Check(output) || PyTuple_Size(output) != 2) {
        Py_DECREF(output);
        PyErr_SetString(PyExc_TypeError, "native SYCART callback must return (energy, gradient)");
        return false;
    }
    result.energy = PyFloat_AsDouble(PyTuple_GET_ITEM(output, 0));
    auto* gradient = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        PyTuple_GET_ITEM(output, 1), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (gradient == nullptr) { Py_DECREF(output); return false; }
    if (PyArray_SIZE(gradient) != 3 * atoms) {
        Py_DECREF(gradient); Py_DECREF(output);
        PyErr_SetString(PyExc_ValueError, "native SYCART gradient has wrong size");
        return false;
    }
    const auto* g = static_cast<const double*>(PyArray_DATA(gradient));
    result.gradient.assign(g, g + 3 * atoms);
    Py_DECREF(gradient); Py_DECREF(output);
    return std::isfinite(result.energy);
}

std::vector<double> project(const std::vector<double>& basis, size_t ncart,
                            size_t rank, const std::vector<double>& vector) {
    std::vector<double> result(rank, 0.0);
    for (size_t j = 0; j < rank; ++j)
        for (size_t i = 0; i < ncart; ++i) result[j] += basis[i * rank + j] * vector[i];
    return result;
}

std::vector<double> expand(const std::vector<double>& basis, size_t ncart,
                           size_t rank, const std::vector<double>& vector) {
    std::vector<double> result(ncart, 0.0);
    for (size_t i = 0; i < ncart; ++i)
        for (size_t j = 0; j < rank; ++j) result[i] += basis[i * rank + j] * vector[j];
    return result;
}

struct NativeDavidsonPreconditioner {
    size_t ncart{}, rows{}, operations{}, atoms{}, roots{}, external_count{};
    std::vector<npy_intp> indptr, indices, permutations;
    std::vector<double> entries, curvatures, external_rows, rotations;
    std::vector<double> eigenvalues, eigenvectors;
    double baseline_inverse{1.0}, shift{1.0e-2}, tolerance{1.0e-3};
    int maximum_iterations{64};

    static bool copy_double(PyObject* source, int ndim, std::vector<double>& target,
                            std::vector<npy_intp>& shape) {
        auto* array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            source, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
        if (!array) return false;
        if (PyArray_NDIM(array) != ndim) {
            Py_DECREF(array); PyErr_SetString(PyExc_ValueError, "invalid preconditioner array rank");
            return false;
        }
        shape.assign(PyArray_DIMS(array), PyArray_DIMS(array) + ndim);
        const auto* data = static_cast<const double*>(PyArray_DATA(array));
        target.assign(data, data + PyArray_SIZE(array));
        Py_DECREF(array); return true;
    }

    static bool copy_index(PyObject* source, int ndim, std::vector<npy_intp>& target,
                           std::vector<npy_intp>& shape) {
        auto* array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            source, NPY_INTP, NPY_ARRAY_IN_ARRAY));
        if (!array) return false;
        if (PyArray_NDIM(array) != ndim) {
            Py_DECREF(array); PyErr_SetString(PyExc_ValueError, "invalid preconditioner index rank");
            return false;
        }
        shape.assign(PyArray_DIMS(array), PyArray_DIMS(array) + ndim);
        const auto* data = static_cast<const npy_intp*>(PyArray_DATA(array));
        target.assign(data, data + PyArray_SIZE(array));
        Py_DECREF(array); return true;
    }

    bool load(PyObject* payload, size_t expected_ncart) {
        if (!PyDict_Check(payload)) {
            PyErr_SetString(PyExc_TypeError, "preconditioner must be a dictionary"); return false;
        }
        auto item = [payload](const char* key) -> PyObject* {
            PyObject* value = PyDict_GetItemString(payload, key);
            if (!value) PyErr_Format(PyExc_KeyError, "missing preconditioner field: %s", key);
            return value;
        };
        std::vector<npy_intp> shape;
        PyObject* value = item("indptr"); if (!value || !copy_index(value, 1, indptr, shape)) return false;
        if (indptr.size() < 2) { PyErr_SetString(PyExc_ValueError, "empty preconditioner CSR"); return false; }
        rows = indptr.size() - 1;
        value = item("indices"); if (!value || !copy_index(value, 1, indices, shape)) return false;
        value = item("data"); if (!value || !copy_double(value, 1, entries, shape)) return false;
        value = item("curvatures"); if (!value || !copy_double(value, 1, curvatures, shape)) return false;
        value = item("external_rows"); if (!value || !copy_double(value, 2, external_rows, shape)) return false;
        external_count = static_cast<size_t>(shape[0]); ncart = static_cast<size_t>(shape[1]);
        value = item("rotations"); if (!value || !copy_double(value, 3, rotations, shape)) return false;
        operations = static_cast<size_t>(shape[0]);
        if (shape[1] != 3 || shape[2] != 3) { PyErr_SetString(PyExc_ValueError, "rotations must be (n,3,3)"); return false; }
        value = item("permutations"); if (!value || !copy_index(value, 2, permutations, shape)) return false;
        atoms = static_cast<size_t>(shape[1]);
        value = item("eigenvalues"); if (!value || !copy_double(value, 1, eigenvalues, shape)) return false;
        roots = eigenvalues.size();
        value = item("eigenvectors"); if (!value || !copy_double(value, 2, eigenvectors, shape)) return false;
        if (ncart != expected_ncart || atoms * 3 != ncart || shape[0] != static_cast<npy_intp>(ncart)
            || shape[1] != static_cast<npy_intp>(roots) || operations == 0
            || permutations.size() != operations * atoms || entries.size() != indices.size()
            || curvatures.size() != rows || indptr.back() != static_cast<npy_intp>(entries.size())) {
            PyErr_SetString(PyExc_ValueError, "inconsistent native Davidson payload dimensions"); return false;
        }
        if (indptr.front() != 0) {
            PyErr_SetString(PyExc_ValueError, "native Davidson CSR must start at zero"); return false;
        }
        for (size_t row = 0; row < rows; ++row) {
            if (indptr[row] < 0 || indptr[row] > indptr[row + 1]) {
                PyErr_SetString(PyExc_ValueError, "native Davidson CSR offsets are invalid"); return false;
            }
        }
        for (npy_intp index : indices) {
            if (index < 0 || index >= static_cast<npy_intp>(ncart)) {
                PyErr_SetString(PyExc_ValueError, "native Davidson CSR index is invalid"); return false;
            }
        }
        for (npy_intp atom : permutations) {
            if (atom < 0 || atom >= static_cast<npy_intp>(atoms)) {
                PyErr_SetString(PyExc_ValueError, "native Davidson atom permutation is invalid"); return false;
            }
        }
        for (double eigenvalue : eigenvalues) {
            if (!std::isfinite(eigenvalue) || eigenvalue <= 0.0) {
                PyErr_SetString(PyExc_ValueError, "native Davidson eigenvalues must be positive"); return false;
            }
        }
        auto scalar = [&](const char* key, double& destination) -> bool {
            PyObject* object = item(key); if (!object) return false;
            destination = PyFloat_AsDouble(object); return !PyErr_Occurred();
        };
        if (!scalar("baseline_inverse", baseline_inverse) || !scalar("shift", shift)
            || !scalar("tolerance", tolerance)) return false;
        PyObject* iterations = item("maximum_iterations"); if (!iterations) return false;
        maximum_iterations = static_cast<int>(PyLong_AsLong(iterations));
        if (PyErr_Occurred() || !(baseline_inverse > 0.0) || shift < 0.0
            || !(tolerance > 0.0) || maximum_iterations <= 0) {
            PyErr_SetString(PyExc_ValueError, "invalid native Davidson controls"); return false;
        }
        return true;
    }

    std::vector<double> project(const std::vector<double>& input) const {
        std::vector<double> internal = input;
        for (size_t row = 0; row < external_count; ++row) {
            const double* mode = external_rows.data() + row * ncart;
            double coefficient = 0.0;
            for (size_t i = 0; i < ncart; ++i) coefficient += mode[i] * internal[i];
            for (size_t i = 0; i < ncart; ++i) internal[i] -= coefficient * mode[i];
        }
        std::vector<double> symmetric(ncart, 0.0);
        for (size_t op = 0; op < operations; ++op) {
            const double* rotation = rotations.data() + 9 * op;
            for (size_t source = 0; source < atoms; ++source) {
                const size_t target = static_cast<size_t>(permutations[op * atoms + source]);
                for (size_t a = 0; a < 3; ++a)
                    for (size_t b = 0; b < 3; ++b)
                        symmetric[3 * source + a] += rotation[3 * a + b] * internal[3 * target + b];
            }
        }
        for (double& element : symmetric) element /= static_cast<double>(operations);
        for (size_t row = 0; row < external_count; ++row) {
            const double* mode = external_rows.data() + row * ncart;
            double coefficient = 0.0;
            for (size_t i = 0; i < ncart; ++i) coefficient += mode[i] * symmetric[i];
            for (size_t i = 0; i < ncart; ++i) symmetric[i] -= coefficient * mode[i];
        }
        return symmetric;
    }

    std::vector<double> hessian_action(const std::vector<double>& input) const {
        auto projected = project(input);
        std::vector<double> primitive(rows, 0.0), cartesian(ncart, 0.0);
        for (size_t row = 0; row < rows; ++row)
            for (npy_intp p = indptr[row]; p < indptr[row + 1]; ++p)
                primitive[row] += entries[p] * projected[static_cast<size_t>(indices[p])];
        for (size_t row = 0; row < rows; ++row) {
            const double coefficient = curvatures[row] * primitive[row];
            for (npy_intp p = indptr[row]; p < indptr[row + 1]; ++p)
                cartesian[static_cast<size_t>(indices[p])] += entries[p] * coefficient;
        }
        return project(cartesian);
    }

    std::vector<double> deflate(const std::vector<double>& input) const {
        std::vector<double> result(input.size());
        for (size_t i = 0; i < input.size(); ++i) result[i] = baseline_inverse * input[i];
        for (size_t root = 0; root < roots; ++root) {
            double coefficient = 0.0;
            for (size_t i = 0; i < ncart; ++i)
                coefficient += eigenvectors[i * roots + root] * input[i];
            coefficient *= 1.0 / eigenvalues[root] - baseline_inverse;
            for (size_t i = 0; i < ncart; ++i)
                result[i] += eigenvectors[i * roots + root] * coefficient;
        }
        return result;
    }

    std::vector<double> apply(const std::vector<double>& input) const {
        auto rhs = project(input);
        const double rhs_norm = norm(rhs);
        if (rhs_norm <= std::numeric_limits<double>::min()) return std::vector<double>(ncart, 0.0);
        std::vector<double> solution(ncart, 0.0), residual = rhs;
        auto preconditioned = project(deflate(residual));
        auto direction = preconditioned;
        double residual_product = dot(residual, preconditioned);
        for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
            auto action = project(hessian_action(direction));
            for (size_t i = 0; i < ncart; ++i) action[i] += shift * direction[i];
            const double denominator = dot(direction, action);
            if (!std::isfinite(denominator) || denominator <= 0.0) break;
            const double step = residual_product / denominator;
            for (size_t i = 0; i < ncart; ++i) {
                solution[i] += step * direction[i]; residual[i] -= step * action[i];
            }
            residual = project(residual);
            if (norm(residual) <= tolerance * rhs_norm) return project(solution);
            preconditioned = project(deflate(residual));
            const double next_product = dot(residual, preconditioned);
            if (!std::isfinite(next_product) || next_product <= 0.0) break;
            const double beta = next_product / residual_product;
            for (size_t i = 0; i < ncart; ++i) direction[i] = preconditioned[i] + beta * direction[i];
            residual_product = next_product;
        }
        return project(solution);
    }
};

std::vector<double> lbfgs_direction(
    const std::vector<double>& gradient,
    const std::vector<std::vector<double>>& steps,
    const std::vector<std::vector<double>>& differences,
    const std::vector<double>& initial_inverse,
    bool initial_inverse_is_diagonal,
    const std::function<std::vector<double>(const std::vector<double>&)>& initial_action) {
    std::vector<double> work = gradient;
    std::vector<double> alpha(steps.size()), rho(steps.size());
    for (size_t kk = steps.size(); kk-- > 0;) {
        rho[kk] = 1.0 / dot(steps[kk], differences[kk]);
        alpha[kk] = rho[kk] * dot(steps[kk], work);
        for (size_t i = 0; i < work.size(); ++i) work[i] -= alpha[kk] * differences[kk][i];
    }
    double scale = 1.0;
    if (!steps.empty()) {
        const auto& s = steps.back(); const auto& y = differences.back();
        const double yy = dot(y, y);
        if (yy > 1.0e-14) scale = dot(s, y) / yy;
    }
    if (initial_action) {
        work = initial_action(work);
    } else if (!initial_inverse.empty()) {
        std::vector<double> transformed(work.size(), 0.0);
        if (initial_inverse_is_diagonal) {
            for (size_t i = 0; i < work.size(); ++i)
                transformed[i] = initial_inverse[i] * work[i];
        } else {
            for (size_t i = 0; i < work.size(); ++i)
                for (size_t j = 0; j < work.size(); ++j)
                    transformed[i] += initial_inverse[i * work.size() + j] * work[j];
        }
        work.swap(transformed);
    } else {
        for (double& value : work) value *= scale;
    }
    for (size_t k = 0; k < steps.size(); ++k) {
        const double beta = rho[k] * dot(differences[k], work);
        for (size_t i = 0; i < work.size(); ++i) work[i] += steps[k][i] * (alpha[k] - beta);
    }
    for (double& value : work) value = -value;
    return work;
}

std::vector<double> full_bfgs_direction(const std::vector<double>& inverse,
                                        const std::vector<double>& gradient, size_t n) {
    std::vector<double> result(n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) result[i] -= inverse[i * n + j] * gradient[j];
    return result;
}

void update_full_inverse(std::vector<double>& h, const std::vector<double>& s,
                         const std::vector<double>& y, size_t n) {
    const double sy = dot(s, y);
    if (!(sy > 1.0e-10 * std::max(1.0, norm(s) * norm(y)))) return;
    std::vector<double> hy(n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) hy[i] += h[i * n + j] * y[j];
    const double yhy = dot(y, hy);
    const double rho = 1.0 / sy;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            h[i * n + j] += (1.0 + yhy * rho) * rho * s[i] * s[j]
                - rho * (s[i] * hy[j] + hy[i] * s[j]);
}

bool solve_small_dense_system(std::vector<double> matrix, std::vector<double> rhs,
                              std::vector<double>& solution, size_t n) {
    solution = std::move(rhs);
    for (size_t column = 0; column < n; ++column) {
        size_t pivot = column;
        double pivot_size = std::abs(matrix[column * n + column]);
        for (size_t row = column + 1; row < n; ++row) {
            const double candidate = std::abs(matrix[row * n + column]);
            if (candidate > pivot_size) { pivot = row; pivot_size = candidate; }
        }
        if (!(pivot_size > 1.0e-14) || !std::isfinite(pivot_size)) return false;
        if (pivot != column) {
            for (size_t entry = column; entry < n; ++entry)
                std::swap(matrix[column * n + entry], matrix[pivot * n + entry]);
            std::swap(solution[column], solution[pivot]);
        }
        const double diagonal = matrix[column * n + column];
        for (size_t row = column + 1; row < n; ++row) {
            const double factor = matrix[row * n + column] / diagonal;
            matrix[row * n + column] = 0.0;
            for (size_t entry = column + 1; entry < n; ++entry)
                matrix[row * n + entry] -= factor * matrix[column * n + entry];
            solution[row] -= factor * solution[column];
        }
    }
    for (size_t offset = 0; offset < n; ++offset) {
        const size_t row = n - 1 - offset;
        double value = solution[row];
        for (size_t column = row + 1; column < n; ++column)
            value -= matrix[row * n + column] * solution[column];
        solution[row] = value / matrix[row * n + row];
        if (!std::isfinite(solution[row])) return false;
    }
    return true;
}

PyObject* optimize(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject *callback, *coordinates_obj, *basis_obj;
    PyObject* initial_inverse_obj = Py_None;
    PyObject* preconditioner_obj = Py_None;
    PyObject* preconditioner_refresh = Py_None;
    int preconditioner_refresh_interval = 0;
    int max_steps = 120, dense_limit = 256, history_limit = 12, dense_steps = -1;
    int gdiis_start = 0, gdiis_history = 6;
    double max_force = 4.5e-4, rms_force = 3.0e-4, trust = 0.2;
    double gradient_norm_tolerance = -1.0, energy_tolerance = -1.0;
    double gdiis_max_coefficient = 2.0;
    static const char* names[] = {"callback", "coordinates", "basis", "max_steps",
        "dense_limit", "history", "max_force", "rms_force", "trust_radius",
        "initial_inverse", "dense_steps", "gradient_norm", "energy_tolerance",
        "preconditioner", "preconditioner_refresh", "preconditioner_refresh_interval",
        "gdiis_start", "gdiis_history", "gdiis_max_coefficient", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|iiidddOiddOOiiid", const_cast<char**>(names),
            &callback, &coordinates_obj, &basis_obj, &max_steps, &dense_limit,
            &history_limit, &max_force, &rms_force, &trust, &initial_inverse_obj,
            &dense_steps, &gradient_norm_tolerance, &energy_tolerance,
            &preconditioner_obj, &preconditioner_refresh,
            &preconditioner_refresh_interval, &gdiis_start, &gdiis_history,
            &gdiis_max_coefficient)) return nullptr;
    auto* coordinates = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        coordinates_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* basis_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        basis_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!coordinates || !basis_array) { Py_XDECREF(coordinates); Py_XDECREF(basis_array); return nullptr; }
    if (PyArray_NDIM(coordinates) != 2 || PyArray_DIM(coordinates, 1) != 3 ||
        PyArray_NDIM(basis_array) != 2 || PyArray_DIM(basis_array, 0) != PyArray_SIZE(coordinates)) {
        Py_DECREF(coordinates); Py_DECREF(basis_array);
        PyErr_SetString(PyExc_ValueError, "coordinates must be (N,3) and basis (3N,R)"); return nullptr;
    }
    const size_t ncart = static_cast<size_t>(PyArray_SIZE(coordinates));
    const size_t rank = static_cast<size_t>(PyArray_DIM(basis_array, 1));
    if (rank == 0 || max_steps <= 0 || dense_limit < 0 || history_limit <= 0
        || !(max_force > 0.0) || !(rms_force > 0.0) || !(trust > 0.0)
        || gdiis_start < 0 || gdiis_history < 2 || !(gdiis_max_coefficient > 0.0)) {
        Py_DECREF(coordinates); Py_DECREF(basis_array);
        PyErr_SetString(PyExc_ValueError, "invalid native SYCART dimensions or controls");
        return nullptr;
    }
    const npy_intp atoms = PyArray_DIM(coordinates, 0);
    const auto* xc = static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* bc = static_cast<const double*>(PyArray_DATA(basis_array));
    std::vector<double> x(xc, xc + ncart), basis(bc, bc + ncart * rank);
    for (double& value : x) value *= BOHR_PER_ANGSTROM;
    Py_DECREF(coordinates); Py_DECREF(basis_array);
    std::vector<double> initial_inverse;
    bool initial_inverse_is_diagonal = false;
    if (initial_inverse_obj != Py_None) {
        auto* seed = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            initial_inverse_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
        if (seed == nullptr) return nullptr;
        if (PyArray_NDIM(seed) == 1 && PyArray_DIM(seed, 0) == static_cast<npy_intp>(rank)) {
            initial_inverse_is_diagonal = true;
        } else if (!(PyArray_NDIM(seed) == 2
                     && PyArray_DIM(seed, 0) == static_cast<npy_intp>(rank)
                     && PyArray_DIM(seed, 1) == static_cast<npy_intp>(rank))) {
            Py_DECREF(seed);
            PyErr_SetString(PyExc_ValueError,
                "initial_inverse must have shape (rank,) or (rank,rank)");
            return nullptr;
        }
        const auto* values = static_cast<const double*>(PyArray_DATA(seed));
        initial_inverse.assign(values, values + PyArray_SIZE(seed));
        Py_DECREF(seed);
        for (double value : initial_inverse) {
            if (!std::isfinite(value)) {
                PyErr_SetString(PyExc_ValueError, "initial_inverse must be finite");
                return nullptr;
            }
        }
    }
    std::unique_ptr<NativeDavidsonPreconditioner> native_preconditioner;
    if (preconditioner_obj != Py_None) {
        if (initial_inverse_obj != Py_None) {
            PyErr_SetString(PyExc_ValueError,
                "initial_inverse and preconditioner are mutually exclusive");
            return nullptr;
        }
        native_preconditioner = std::make_unique<NativeDavidsonPreconditioner>();
        if (!native_preconditioner->load(preconditioner_obj, ncart)) return nullptr;
    }
    if (preconditioner_refresh != Py_None
        && (!PyCallable_Check(preconditioner_refresh) || !native_preconditioner
            || preconditioner_refresh_interval <= 0)) {
        PyErr_SetString(PyExc_ValueError,
            "preconditioner_refresh needs an initial preconditioner and a positive interval");
        return nullptr;
    }
    std::function<std::vector<double>(const std::vector<double>&)> native_initial_action;
    if (native_preconditioner) {
        native_initial_action = [&](const std::vector<double>& reduced) {
            // Geometry is stored in bohr in this kernel, whereas the canonical
            // matrix-free SYCART chart and its Davidson compliance use
            // Cartesian displacements in angstrom.  Convert the covector from
            // Eh/bohr to Eh/angstrom before applying the compliance and the
            // resulting step back from angstrom to bohr.  Omitting these two
            // dual conversions made the native model artificially soft by
            // BOHR_PER_ANGSTROM**2 and changed the optimization trajectory.
            auto ambient = expand(basis, ncart, rank, reduced);
            for (double& value : ambient) value *= BOHR_PER_ANGSTROM;
            auto transformed = project(
                basis, ncart, rank, native_preconditioner->apply(ambient));
            for (double& value : transformed) value *= BOHR_PER_ANGSTROM;
            return transformed;
        };
    }
    Evaluation current;
    double calculator_seconds = 0.0;
    double preconditioner_refresh_seconds = 0.0;
    auto calculator_started = std::chrono::steady_clock::now();
    if (!evaluate(callback, x, atoms, current)) return nullptr;
    calculator_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - calculator_started).count();
    auto gradient = project(basis, ncart, rank, current.gradient);
    auto gradient_maximum = [](const std::vector<double>& values) {
        double maximum = 0.0;
        for (double value : values) maximum = std::max(maximum, std::abs(value));
        return maximum;
    };
    std::vector<double> gradient_norm_history;
    gradient_norm_history.push_back(gradient_maximum(gradient));
    const bool dense = rank <= static_cast<size_t>(dense_limit);
    std::vector<double> inverse;
    if (dense) {
        if (native_preconditioner) {
            inverse.assign(rank * rank, 0.0);
            std::vector<double> unit(rank, 0.0);
            for (size_t column = 0; column < rank; ++column) {
                unit[column] = 1.0;
                const auto transformed = native_initial_action(unit);
                unit[column] = 0.0;
                for (size_t row = 0; row < rank; ++row)
                    inverse[row * rank + column] = transformed[row];
            }
            for (size_t row = 0; row < rank; ++row)
                for (size_t column = row + 1; column < rank; ++column) {
                    const double symmetric = 0.5 * (
                        inverse[row * rank + column] + inverse[column * rank + row]);
                    inverse[row * rank + column] = symmetric;
                    inverse[column * rank + row] = symmetric;
                }
        } else if (initial_inverse.empty()) {
            inverse.assign(rank * rank, 0.0);
            for (size_t i=0;i<rank;++i) inverse[i*rank+i]=1.0;
        } else if (initial_inverse_is_diagonal) {
            inverse.assign(rank * rank, 0.0);
            for (size_t i=0;i<rank;++i) inverse[i*rank+i]=initial_inverse[i];
        } else {
            inverse = initial_inverse;
        }
    }
    std::vector<std::vector<double>> steps, differences;
    std::vector<std::vector<double>> gdiis_positions, gdiis_gradients;
    gdiis_positions.push_back(project(basis, ncart, rank, x));
    gdiis_gradients.push_back(gradient);
    PyObject* trace = PyList_New(0);
    int accepted_steps = 0, attempted_steps = 0, evaluations = 1;
    int gdiis_steps = 0;
    double last_energy_change = std::numeric_limits<double>::infinity();
    std::vector<double> last_cartesian_step(ncart, 0.0);
    for (int iteration = 1; iteration <= max_steps; ++iteration) {
        double maximum = 0.0; for (double v: gradient) maximum=std::max(maximum,std::abs(v));
        const double rms = norm(gradient) / std::sqrt(static_cast<double>(gradient.size()));
        const bool force_converged = gradient_norm_tolerance > 0.0
            ? norm(gradient) <= gradient_norm_tolerance
            : maximum <= max_force && rms <= rms_force;
        const bool energy_converged = energy_tolerance <= 0.0
            || std::abs(last_energy_change) <= energy_tolerance;
        if (force_converged && energy_converged) break;
        attempted_steps = iteration;
        const bool use_dense = dense && (dense_steps < 0 || accepted_steps < dense_steps);
        auto inverse_direction = [&](const std::vector<double>& values) {
            return use_dense ? full_bfgs_direction(inverse, values, rank)
                             : lbfgs_direction(values, steps, differences,
                                               initial_inverse,
                                               initial_inverse_is_diagonal,
                                               native_initial_action);
        };
        auto direction = inverse_direction(gradient);
        bool used_gdiis = false;
        // Delay GDIIS when the first steps are already descending smoothly;
        // conversely, activate it at the requested start when the early
        // gradient sequence is oscillatory.  This keeps the Farkas--Schlegel
        // extrapolation out of a well-behaved quasi-Newton basin while
        // retaining its benefit for a rough starting geometry.
        int effective_gdiis_start = gdiis_start;
        if (gdiis_start > 0 && gradient_norm_history.size() >= 4
            && gradient_norm_history[3] <= 1.15 * gradient_norm_history[2]) {
            effective_gdiis_start += 3;
        }
        if (gdiis_start > 0 && accepted_steps + 1 >= effective_gdiis_start
            && gdiis_positions.size() >= 2) {
            const size_t count = std::min(gdiis_positions.size(),
                                          static_cast<size_t>(gdiis_history));
            const size_t first = gdiis_positions.size() - count;
            std::vector<std::vector<double>> errors;
            errors.reserve(count);
            for (size_t item = 0; item < count; ++item)
                errors.push_back(inverse_direction(gdiis_gradients[first + item]));
            const size_t system_size = count + 1;
            std::vector<double> system(system_size * system_size, 0.0);
            std::vector<double> rhs(system_size, 0.0), coefficients;
            rhs[count] = 1.0;
            for (size_t row = 0; row < count; ++row) {
                for (size_t column = 0; column < count; ++column)
                    system[row * system_size + column] = dot(errors[row], errors[column]);
                system[row * system_size + count] = 1.0;
                system[count * system_size + row] = 1.0;
            }
            if (solve_small_dense_system(system, rhs, coefficients, system_size)) {
                bool bounded = true;
                for (size_t item = 0; item < count; ++item)
                    bounded = bounded && std::abs(coefficients[item]) <= gdiis_max_coefficient;
                if (bounded) {
                    std::vector<double> candidate(rank, 0.0), interpolated_gradient(rank, 0.0);
                    for (size_t item = 0; item < count; ++item)
                        for (size_t component = 0; component < rank; ++component) {
                            candidate[component] += coefficients[item]
                                * gdiis_positions[first + item][component];
                            interpolated_gradient[component] += coefficients[item]
                                * gdiis_gradients[first + item][component];
                        }
                    const auto relaxation = inverse_direction(interpolated_gradient);
                    const auto current_position = project(basis, ncart, rank, x);
                    for (size_t component = 0; component < rank; ++component)
                        candidate[component] += relaxation[component] - current_position[component];
                    if (std::all_of(candidate.begin(), candidate.end(),
                                    [](double value) { return std::isfinite(value); })
                        && dot(gradient, candidate) < -1.0e-12) {
                        direction.swap(candidate);
                        used_gdiis = true;
                    }
                }
            }
        }
        // LINK defines the Cartesian trust radius as molecular Cartesian RMSD,
        // sqrt(sum_i |dx_i|^2 / N), rather than the RMS of 3N scalar entries.
        double cart_norm = norm(direction) / std::sqrt(static_cast<double>(atoms));
        if (cart_norm > trust * BOHR_PER_ANGSTROM) {
            const double scale = trust * BOHR_PER_ANGSTROM / cart_norm;
            for (double& v: direction) v *= scale;
        }
        const auto dx = expand(basis, ncart, rank, direction);
        Evaluation trial; bool accepted=false; double factor=1.0;
        std::vector<double> trial_x(ncart);
        for (int line=0; line<4; ++line) {
            for (size_t i=0;i<ncart;++i) trial_x[i]=x[i]+factor*dx[i];
            calculator_started = std::chrono::steady_clock::now();
            if (!evaluate(callback, trial_x, atoms, trial)) { Py_DECREF(trace); return nullptr; }
            calculator_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - calculator_started).count();
            ++evaluations;
            if (trial.energy <= current.energy + 1.0e-10) { accepted=true; break; }
            factor *= 0.5;
        }
        if (!accepted) { trust=std::max(1.0e-4,trust*0.5); continue; }
        std::vector<double> actual_step=direction; for(double& v:actual_step)v*=factor;
        auto next_gradient=project(basis,ncart,rank,trial.gradient);
        std::vector<double> difference(rank); for(size_t i=0;i<rank;++i)difference[i]=next_gradient[i]-gradient[i];
        if (use_dense) update_full_inverse(inverse,actual_step,difference,rank);
        if (dot(actual_step,difference)>1.0e-10*std::max(1.0,norm(actual_step)*norm(difference))) {
            steps.push_back(actual_step); differences.push_back(difference);
            if (steps.size()>static_cast<size_t>(history_limit)) { steps.erase(steps.begin()); differences.erase(differences.begin()); }
        }
        last_energy_change = trial.energy - current.energy;
        last_cartesian_step = expand(basis, ncart, rank, actual_step);
        x.swap(trial_x); current=std::move(trial); gradient.swap(next_gradient); ++accepted_steps;
        gradient_norm_history.push_back(gradient_maximum(gradient));
        if (used_gdiis) ++gdiis_steps;
        gdiis_positions.push_back(project(basis, ncart, rank, x));
        gdiis_gradients.push_back(gradient);
        if (gdiis_positions.size() > static_cast<size_t>(gdiis_history)) {
            gdiis_positions.erase(gdiis_positions.begin());
            gdiis_gradients.erase(gdiis_gradients.begin());
        }
        if (preconditioner_refresh != Py_None
            && accepted_steps % preconditioner_refresh_interval == 0) {
            npy_intp refresh_dimensions[2] = {atoms, 3};
            PyObject* refresh_coordinates = PyArray_SimpleNew(2, refresh_dimensions, NPY_DOUBLE);
            auto* refresh_data = static_cast<double*>(PyArray_DATA(
                reinterpret_cast<PyArrayObject*>(refresh_coordinates)));
            for (size_t i = 0; i < ncart; ++i) refresh_data[i] = x[i] / BOHR_PER_ANGSTROM;
            const auto refresh_started = std::chrono::steady_clock::now();
            PyObject* refreshed_payload = PyObject_CallOneArg(
                preconditioner_refresh, refresh_coordinates);
            Py_DECREF(refresh_coordinates);
            preconditioner_refresh_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - refresh_started).count();
            if (!refreshed_payload) { Py_DECREF(trace); return nullptr; }
            auto refreshed = std::make_unique<NativeDavidsonPreconditioner>();
            const bool loaded = refreshed->load(refreshed_payload, ncart);
            Py_DECREF(refreshed_payload);
            if (!loaded) { Py_DECREF(trace); return nullptr; }
            native_preconditioner.swap(refreshed);
        }
        double next_maximum = 0.0;
        for (double v: gradient) next_maximum=std::max(next_maximum,std::abs(v));
        const double next_rms = norm(gradient) / std::sqrt(static_cast<double>(gradient.size()));
        double step_maximum = 0.0;
        for (double v: last_cartesian_step) step_maximum=std::max(step_maximum,std::abs(v));
        const double step_rms = norm(last_cartesian_step)
            / std::sqrt(static_cast<double>(last_cartesian_step.size()));
        PyObject* row=Py_BuildValue(
            "{s:i,s:d,s:d,s:d,s:d,s:d,s:d,s:O}",
            "iteration",iteration,"energy",current.energy,
            "gradient_inf",next_maximum,"gradient_rms",next_rms,
            "step_inf",step_maximum,"step_rms",step_rms,
            "energy_change",last_energy_change,"gdiis_used",used_gdiis?Py_True:Py_False);
        PyList_Append(trace,row); Py_DECREF(row);
    }
    npy_intp dims[2]={atoms,3}; PyObject* final_coordinates=PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    auto* out=static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(final_coordinates)));
    for(size_t i=0;i<ncart;++i)out[i]=x[i]/BOHR_PER_ANGSTROM;
    double maximum=0.0; for(double v:gradient)maximum=std::max(maximum,std::abs(v));
    const double rms=norm(gradient)/std::sqrt(static_cast<double>(gradient.size()));
    const double gradient_norm = norm(gradient);
    const bool force_converged = gradient_norm_tolerance > 0.0
        ? gradient_norm <= gradient_norm_tolerance
        : maximum <= max_force && rms <= rms_force;
    const bool converged = force_converged
        && (energy_tolerance <= 0.0 || std::abs(last_energy_change) <= energy_tolerance);
    PyObject* final_gradient=PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(final_gradient)),
                current.gradient.data(), ncart * sizeof(double));
    PyObject* final_displacement=PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(final_displacement)),
                last_cartesian_step.data(), ncart * sizeof(double));
    PyObject* result=Py_BuildValue("{s:O,s:O,s:O,s:d,s:i,s:i,s:i,s:O,s:s,s:d,s:d,s:d,s:d,s:O,s:d,s:d,s:i}",
        "coordinates",final_coordinates,"gradient",final_gradient,
        "last_displacement",final_displacement,"energy",current.energy,
        "iterations",accepted_steps,
        "attempted_iterations",attempted_steps,"evaluations",evaluations,"trace",trace,
        "model",dense?(dense_steps<0?"full_bfgs":"full_bfgs_then_lbfgs"):"lbfgs",
        "gradient_inf",maximum,"gradient_rms",rms,"gradient_norm",gradient_norm,
        "energy_change",last_energy_change,
        "converged",converged?Py_True:Py_False,"calculator_seconds",calculator_seconds,
        "preconditioner_refresh_seconds",preconditioner_refresh_seconds,
        "gdiis_steps",gdiis_steps);
    Py_DECREF(final_coordinates); Py_DECREF(final_gradient); Py_DECREF(final_displacement);
    Py_DECREF(trace); return result;
}

PyObject* apply_preconditioner(PyObject*, PyObject* args) {
    PyObject *payload, *vector_obj;
    if (!PyArg_ParseTuple(args, "OO", &payload, &vector_obj)) return nullptr;
    auto* vector_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        vector_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!vector_array) return nullptr;
    if (PyArray_NDIM(vector_array) != 1 || PyArray_SIZE(vector_array) == 0) {
        Py_DECREF(vector_array);
        PyErr_SetString(PyExc_ValueError, "preconditioner vector must be nonempty and one-dimensional");
        return nullptr;
    }
    const size_t dimension = static_cast<size_t>(PyArray_SIZE(vector_array));
    const auto* data = static_cast<const double*>(PyArray_DATA(vector_array));
    std::vector<double> vector(data, data + dimension);
    Py_DECREF(vector_array);
    NativeDavidsonPreconditioner preconditioner;
    if (!preconditioner.load(payload, dimension)) return nullptr;
    const auto result = preconditioner.apply(vector);
    npy_intp output_dimension[1] = {static_cast<npy_intp>(dimension)};
    PyObject* output = PyArray_SimpleNew(1, output_dimension, NPY_DOUBLE);
    std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(output)),
                result.data(), dimension * sizeof(double));
    return output;
}

PyObject* remove_rigid_motion(PyObject*, PyObject* args) {
    PyObject *vector_object, *centered_object, *inverse_object;
    if (!PyArg_ParseTuple(args, "OOO", &vector_object, &centered_object, &inverse_object))
        return nullptr;
    auto* vector_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        vector_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* centered_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        centered_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* inverse_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        inverse_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!vector_array || !centered_array || !inverse_array) {
        Py_XDECREF(vector_array); Py_XDECREF(centered_array); Py_XDECREF(inverse_array);
        return nullptr;
    }
    const npy_intp atoms = PyArray_DIM(centered_array, 0);
    if (PyArray_NDIM(vector_array) != 1 || PyArray_SIZE(vector_array) != 3 * atoms ||
        PyArray_NDIM(centered_array) != 2 || PyArray_DIM(centered_array, 1) != 3 ||
        PyArray_NDIM(inverse_array) != 2 || PyArray_DIM(inverse_array, 0) != 3 ||
        PyArray_DIM(inverse_array, 1) != 3 || atoms < 1) {
        Py_DECREF(vector_array); Py_DECREF(centered_array); Py_DECREF(inverse_array);
        PyErr_SetString(PyExc_ValueError, "rigid projection arrays are incompatible");
        return nullptr;
    }
    const auto* vector = static_cast<const double*>(PyArray_DATA(vector_array));
    const auto* centered = static_cast<const double*>(PyArray_DATA(centered_array));
    const auto* inverse = static_cast<const double*>(PyArray_DATA(inverse_array));
    npy_intp output_size[1] = {3 * atoms};
    auto* output = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, output_size, NPY_DOUBLE));
    if (!output) {
        Py_DECREF(vector_array); Py_DECREF(centered_array); Py_DECREF(inverse_array);
        return nullptr;
    }
    auto* result = static_cast<double*>(PyArray_DATA(output));
    double translation[3] = {0.0, 0.0, 0.0};
    for (npy_intp atom = 0; atom < atoms; ++atom)
        for (int axis = 0; axis < 3; ++axis)
            translation[axis] += vector[3 * atom + axis];
    for (double& value : translation) value /= static_cast<double>(atoms);
    double angular_rhs[3] = {0.0, 0.0, 0.0};
    for (npy_intp atom = 0; atom < atoms; ++atom) {
        const double x = centered[3 * atom];
        const double y = centered[3 * atom + 1];
        const double z = centered[3 * atom + 2];
        const double vx = vector[3 * atom] - translation[0];
        const double vy = vector[3 * atom + 1] - translation[1];
        const double vz = vector[3 * atom + 2] - translation[2];
        angular_rhs[0] += y * vz - z * vy;
        angular_rhs[1] += z * vx - x * vz;
        angular_rhs[2] += x * vy - y * vx;
    }
    double omega[3] = {0.0, 0.0, 0.0};
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            omega[row] += inverse[3 * row + column] * angular_rhs[column];
    for (npy_intp atom = 0; atom < atoms; ++atom) {
        const double x = centered[3 * atom];
        const double y = centered[3 * atom + 1];
        const double z = centered[3 * atom + 2];
        const double rotation[3] = {
            omega[1] * z - omega[2] * y,
            omega[2] * x - omega[0] * z,
            omega[0] * y - omega[1] * x,
        };
        for (int axis = 0; axis < 3; ++axis)
            result[3 * atom + axis] =
                vector[3 * atom + axis] - translation[axis] - rotation[axis];
    }
    Py_DECREF(vector_array); Py_DECREF(centered_array); Py_DECREF(inverse_array);
    return reinterpret_cast<PyObject*>(output);
}

PyObject* rigid_gauge_columns_valid(PyObject*, PyObject* args) {
    PyObject *centered_object, *axes_object, *columns_object;
    double maximum_condition = 1.0e10;
    if (!PyArg_ParseTuple(args, "OOO|d", &centered_object, &axes_object,
                          &columns_object, &maximum_condition)) return nullptr;
    auto* centered_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        centered_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* axes_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        axes_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* columns_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        columns_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    if (!centered_array || !axes_array || !columns_array) {
        Py_XDECREF(centered_array); Py_XDECREF(axes_array); Py_XDECREF(columns_array);
        return nullptr;
    }
    const npy_intp atoms = PyArray_DIM(centered_array, 0);
    const npy_intp rotations = PyArray_DIM(axes_array, 0);
    const npy_intp count = 3 + rotations;
    bool valid = PyArray_NDIM(centered_array) == 2 &&
        PyArray_DIM(centered_array, 1) == 3 &&
        PyArray_NDIM(axes_array) == 2 && PyArray_DIM(axes_array, 1) == 3 &&
        PyArray_NDIM(columns_array) == 1 && PyArray_SIZE(columns_array) == count &&
        count >= 3 && count <= 6 && atoms > 0 && maximum_condition > 0.0;
    const auto* centered = static_cast<const double*>(PyArray_DATA(centered_array));
    const auto* axes = static_cast<const double*>(PyArray_DATA(axes_array));
    const auto* columns = static_cast<const npy_intp*>(PyArray_DATA(columns_array));
    std::vector<double> coefficients(static_cast<size_t>(count * count), 0.0);
    for (npy_intp local = 0; valid && local < count; ++local) {
        const npy_intp column = columns[local];
        const npy_intp atom = column / 3;
        const npy_intp component = column % 3;
        if (column < 0 || atom >= atoms) { valid = false; break; }
        for (npy_intp prior = 0; prior < local; ++prior)
            if (columns[prior] == column) valid = false;
        coefficients[component * count + local] = 1.0;
        const double x = centered[3 * atom];
        const double y = centered[3 * atom + 1];
        const double z = centered[3 * atom + 2];
        for (npy_intp axis_index = 0; axis_index < rotations; ++axis_index) {
            const double ax = axes[3 * axis_index];
            const double ay = axes[3 * axis_index + 1];
            const double az = axes[3 * axis_index + 2];
            coefficients[(3 + axis_index) * count + local] =
                component == 0 ? ay * z - az * y
                : component == 1 ? az * x - ax * z
                                 : ax * y - ay * x;
        }
    }
    std::vector<double> gram(static_cast<size_t>(count * count), 0.0);
    if (valid) {
        for (npy_intp i = 0; i < count; ++i)
            for (npy_intp j = 0; j < count; ++j)
                for (npy_intp k = 0; k < count; ++k)
                    gram[i * count + j] +=
                        coefficients[k * count + i] * coefficients[k * count + j];
        for (int sweep = 0; sweep < 64; ++sweep) {
            npy_intp p = 0, q = 1;
            double largest = 0.0;
            for (npy_intp i = 0; i < count; ++i)
                for (npy_intp j = i + 1; j < count; ++j)
                    if (std::abs(gram[i * count + j]) > largest) {
                        largest = std::abs(gram[i * count + j]); p = i; q = j;
                    }
            if (largest <= 1.0e-15) break;
            const double app = gram[p * count + p];
            const double aqq = gram[q * count + q];
            const double apq = gram[p * count + q];
            const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
            const double c = std::cos(angle), s = std::sin(angle);
            for (npy_intp k = 0; k < count; ++k) {
                if (k == p || k == q) continue;
                const double akp = gram[k * count + p];
                const double akq = gram[k * count + q];
                gram[k * count + p] = gram[p * count + k] = c * akp - s * akq;
                gram[k * count + q] = gram[q * count + k] = s * akp + c * akq;
            }
            gram[p * count + p] = c*c*app - 2.0*s*c*apq + s*s*aqq;
            gram[q * count + q] = s*s*app + 2.0*s*c*apq + c*c*aqq;
            gram[p * count + q] = gram[q * count + p] = 0.0;
        }
        double smallest = std::numeric_limits<double>::infinity(), largest = 0.0;
        for (npy_intp i = 0; i < count; ++i) {
            smallest = std::min(smallest, gram[i * count + i]);
            largest = std::max(largest, gram[i * count + i]);
        }
        valid = smallest > 1.0e-24 &&
            std::sqrt(largest / smallest) <= maximum_condition;
    }
    Py_DECREF(centered_array); Py_DECREF(axes_array); Py_DECREF(columns_array);
    if (valid) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

PyObject* sonic_sparse_backtransform(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject *coordinates_object, *target_object, *values_callback,
        *operator_callback, *projection_callback, *initial_operator,
        *initial_values_object;
    double tolerance = 1.0e-9;
    int max_iterations = 60;
    double max_cartesian_step = 0.25;
    int maximum_operator_age = 4;
    double maximum_linearization_error = 0.15;
    static const char* names[] = {
        "coordinates", "target", "evaluate_values", "evaluate_operator",
        "project_coordinates", "initial_operator", "initial_values",
        "tolerance", "max_iterations", "max_cartesian_step_angstrom",
        "maximum_operator_age", "maximum_linearization_error", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOOOO|didid",
            const_cast<char**>(names), &coordinates_object, &target_object,
            &values_callback, &operator_callback, &projection_callback,
            &initial_operator, &initial_values_object, &tolerance,
            &max_iterations, &max_cartesian_step, &maximum_operator_age,
            &maximum_linearization_error)) {
        return nullptr;
    }
    auto* coordinates_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        coordinates_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* target_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        target_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (coordinates_array == nullptr || target_array == nullptr) {
        Py_XDECREF(coordinates_array); Py_XDECREF(target_array); return nullptr;
    }
    if (PyArray_NDIM(coordinates_array) != 2 ||
        PyArray_DIM(coordinates_array, 1) != 3 ||
        PyArray_NDIM(target_array) != 1 || max_iterations < 1 ||
        maximum_operator_age < 1 || tolerance <= 0.0 ||
        maximum_linearization_error <= 0.0) {
        Py_DECREF(coordinates_array); Py_DECREF(target_array);
        PyErr_SetString(PyExc_ValueError, "native SONIC back-transform arguments are invalid");
        return nullptr;
    }
    const npy_intp atoms = PyArray_DIM(coordinates_array, 0);
    const size_t ncart = static_cast<size_t>(3 * atoms);
    const size_t nq = static_cast<size_t>(PyArray_SIZE(target_array));
    const auto* coordinate_data =
        static_cast<const double*>(PyArray_DATA(coordinates_array));
    const auto* target_data = static_cast<const double*>(PyArray_DATA(target_array));
    std::vector<double> coordinates(coordinate_data, coordinate_data + ncart);
    std::vector<double> target(target_data, target_data + nq);
    Py_DECREF(coordinates_array); Py_DECREF(target_array);
    if (!std::all_of(coordinates.begin(), coordinates.end(),
                     [](double x) { return std::isfinite(x); }) ||
        !std::all_of(target.begin(), target.end(),
                     [](double x) { return std::isfinite(x); })) {
        PyErr_SetString(PyExc_FloatingPointError,
                        "native SONIC input contains non-finite values");
        return nullptr;
    }
    PyObject* operator_object = initial_operator;
    Py_INCREF(operator_object);
    std::vector<double> values;
    if (initial_values_object == Py_None) {
        if (!call_values(values_callback, coordinates, atoms, nq, values)) {
            Py_DECREF(operator_object); return nullptr;
        }
    } else if (!finite_vector_from_object(initial_values_object, nq, values)) {
        Py_DECREF(operator_object); return nullptr;
    }
    std::vector<double> residual(nq);
    for (size_t i = 0; i < nq; ++i) residual[i] = target[i] - values[i];
    const double initial_norm = norm(residual);
    const double convergence_threshold = tolerance * std::max(1.0, initial_norm);
    int operator_age = 0;
    int completed_iterations = 0;
    bool converged = initial_norm <= tolerance;
    const double scales[] = {1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625};

    auto refresh = [&]() -> bool {
        if (!call_values_and_operator(operator_callback, coordinates, atoms, nq,
                                      values, operator_object)) return false;
        for (size_t i = 0; i < nq; ++i) residual[i] = target[i] - values[i];
        operator_age = 0;
        return true;
    };

    for (int iteration = 1; !converged && iteration <= max_iterations; ++iteration) {
        completed_iterations = iteration;
        std::vector<double> dx(ncart, 0.0);
        if (!call_operator_vector(operator_object, "matvec", residual, dx)) {
            Py_DECREF(operator_object); return nullptr;
        }
        double dx_norm = norm(dx);
        if (max_cartesian_step > 0.0 && dx_norm > max_cartesian_step) {
            const double factor = max_cartesian_step / dx_norm;
            for (double& value : dx) value *= factor;
        }
        const double current_norm = norm(residual);
        bool accepted = false;
        for (double scale : scales) {
            std::vector<double> trial(coordinates);
            for (size_t i = 0; i < ncart; ++i) trial[i] += scale * dx[i];
            if (!call_projection(projection_callback, trial, atoms)) {
                Py_DECREF(operator_object); return nullptr;
            }
            std::vector<double> trial_values;
            if (!call_values(values_callback, trial, atoms, nq, trial_values)) {
                Py_DECREF(operator_object); return nullptr;
            }
            std::vector<double> trial_residual(nq);
            for (size_t i = 0; i < nq; ++i)
                trial_residual[i] = target[i] - trial_values[i];
            if (norm(trial_residual) < current_norm) {
                std::vector<double> cartesian_change(ncart);
                std::vector<double> internal_change(nq);
                for (size_t i = 0; i < ncart; ++i)
                    cartesian_change[i] = trial[i] - coordinates[i];
                for (size_t i = 0; i < nq; ++i)
                    internal_change[i] = trial_values[i] - values[i];
                std::vector<double> predicted_change(nq, 0.0);
                if (!call_operator_vector(operator_object, "b_matvec",
                                          cartesian_change, predicted_change)) {
                    Py_DECREF(operator_object); return nullptr;
                }
                std::vector<double> defect(nq);
                for (size_t i = 0; i < nq; ++i)
                    defect[i] = internal_change[i] - predicted_change[i];
                const double secant_scale = std::max(
                    {norm(internal_change), tolerance,
                     std::numeric_limits<double>::epsilon()});
                const double linearization_error = norm(defect) / secant_scale;
                coordinates.swap(trial);
                values.swap(trial_values);
                residual.swap(trial_residual);
                ++operator_age;
                if ((operator_age >= maximum_operator_age ||
                     linearization_error > maximum_linearization_error) &&
                    norm(residual) > convergence_threshold) {
                    if (!refresh()) { Py_DECREF(operator_object); return nullptr; }
                }
                accepted = true;
                break;
            }
        }
        if (!accepted && operator_age > 0) {
            if (!refresh()) { Py_DECREF(operator_object); return nullptr; }
            continue;
        }
        if (!accepted) {
            const double dampings[] = {1.0e-6, 1.0e-4, 1.0e-2, 1.0};
            for (double damping : dampings) {
                std::vector<double> fallback(ncart, 0.0);
                if (!call_operator_vector(operator_object, "matvec", residual,
                                          fallback, damping, true)) {
                    Py_DECREF(operator_object); return nullptr;
                }
                const double fallback_norm = norm(fallback);
                if (max_cartesian_step > 0.0 && fallback_norm > max_cartesian_step) {
                    const double factor = max_cartesian_step / fallback_norm;
                    for (double& value : fallback) value *= factor;
                }
                std::vector<double> trial(coordinates);
                for (size_t i = 0; i < ncart; ++i) trial[i] += fallback[i];
                if (!call_projection(projection_callback, trial, atoms)) {
                    Py_DECREF(operator_object); return nullptr;
                }
                std::vector<double> trial_values;
                if (!call_values(values_callback, trial, atoms, nq, trial_values)) {
                    Py_DECREF(operator_object); return nullptr;
                }
                std::vector<double> trial_residual(nq);
                for (size_t i = 0; i < nq; ++i)
                    trial_residual[i] = target[i] - trial_values[i];
                if (norm(trial_residual) < current_norm) {
                    std::vector<double> cartesian_change(ncart), internal_change(nq);
                    for (size_t i = 0; i < ncart; ++i)
                        cartesian_change[i] = trial[i] - coordinates[i];
                    for (size_t i = 0; i < nq; ++i)
                        internal_change[i] = trial_values[i] - values[i];
                    std::vector<double> predicted_change(nq, 0.0);
                    if (!call_operator_vector(operator_object, "b_matvec",
                                              cartesian_change, predicted_change)) {
                        Py_DECREF(operator_object); return nullptr;
                    }
                    std::vector<double> defect(nq);
                    for (size_t i = 0; i < nq; ++i)
                        defect[i] = internal_change[i] - predicted_change[i];
                    const double secant_scale = std::max(
                        {norm(internal_change), tolerance,
                         std::numeric_limits<double>::epsilon()});
                    const double linearization_error = norm(defect) / secant_scale;
                    coordinates.swap(trial); values.swap(trial_values);
                    residual.swap(trial_residual); ++operator_age;
                    if ((operator_age >= maximum_operator_age ||
                         linearization_error > maximum_linearization_error) &&
                        norm(residual) > convergence_threshold) {
                        if (!refresh()) { Py_DECREF(operator_object); return nullptr; }
                    }
                    accepted = true;
                    break;
                }
            }
        }
        if (!accepted) break;
        converged = norm(residual) <= convergence_threshold;
    }
    Py_DECREF(operator_object);
    PyObject* coordinates_result = numpy_coordinates_copy(coordinates, atoms);
    PyObject* values_result = numpy_vector_copy(values);
    PyObject* residual_result = numpy_vector_copy(residual);
    if (!coordinates_result || !values_result || !residual_result) {
        Py_XDECREF(coordinates_result); Py_XDECREF(values_result);
        Py_XDECREF(residual_result); return nullptr;
    }
    return Py_BuildValue("NNNiN", coordinates_result, values_result,
                         residual_result, completed_iterations,
                         PyBool_FromLong(converged ? 1 : 0));
}

PyObject* restrict_cartesian_trust(PyObject*, PyObject* args) {
    PyObject *vector_object, *realize_callback;
    double predictor_scale, trust_radius, tolerance;
    int maximum_iterations, sonic;
    if (!PyArg_ParseTuple(args, "OOdddii", &vector_object, &realize_callback,
                          &predictor_scale, &trust_radius, &tolerance,
                          &maximum_iterations, &sonic)) return nullptr;
    std::vector<double> vector;
    auto* vector_array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        vector_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (vector_array == nullptr) return nullptr;
    const size_t dimension = static_cast<size_t>(PyArray_SIZE(vector_array));
    const auto* vector_data = static_cast<const double*>(PyArray_DATA(vector_array));
    vector.assign(vector_data, vector_data + dimension);
    Py_DECREF(vector_array);
    if (dimension == 0 || trust_radius <= 0.0 || tolerance < 0.0 ||
        maximum_iterations < 0 || predictor_scale <= 0.0 || predictor_scale > 1.0) {
        PyErr_SetString(PyExc_ValueError, "native Cartesian trust arguments are invalid");
        return nullptr;
    }
    const double epsilon = std::numeric_limits<double>::epsilon();
    const double target = (1.0 - tolerance) * trust_radius;
    double lower_scale = 0.0, lower_rmsd = 0.0;
    double upper_scale = 1.0, upper_rmsd = std::numeric_limits<double>::infinity();
    bool upper_valid = false;
    double best_scale = 0.0, best_rmsd = 0.0;
    int iterations = 0;
    double initial_rmsd;
    bool initial_valid;
    if (!call_trust_realization(realize_callback, predictor_scale,
                                initial_rmsd, initial_valid)) return nullptr;
    if (predictor_scale < 1.0) ++iterations;
    if (predictor_scale == 1.0 && initial_valid && std::isfinite(initial_rmsd) &&
        initial_rmsd <= trust_radius) {
        PyObject* step = numpy_vector_copy(vector);
        return Py_BuildValue("Nddi", step, initial_rmsd, 1.0, 0);
    }
    if (initial_valid && std::isfinite(initial_rmsd) && initial_rmsd <= target) {
        lower_scale = best_scale = predictor_scale;
        lower_rmsd = best_rmsd = initial_rmsd;
        if (sonic) {
            std::vector<double> step(vector);
            for (double& value : step) value *= best_scale;
            return Py_BuildValue("Nddi", numpy_vector_copy(step), best_rmsd,
                                 best_scale, std::max(iterations, 1));
        }
        if (target - best_rmsd <= tolerance * trust_radius) {
            std::vector<double> step(vector);
            for (double& value : step) value *= best_scale;
            return Py_BuildValue("Nddi", numpy_vector_copy(step), best_rmsd,
                                 best_scale, std::max(iterations, 1));
        }
        const double corrected_scale = std::min(
            1.0, predictor_scale * target / std::max(best_rmsd, epsilon));
        if (corrected_scale > predictor_scale * (1.0 + tolerance)) {
            double corrected_rmsd; bool corrected_valid;
            if (!call_trust_realization(realize_callback, corrected_scale,
                                        corrected_rmsd, corrected_valid)) return nullptr;
            ++iterations;
            if (corrected_valid && std::isfinite(corrected_rmsd) &&
                corrected_rmsd <= target) {
                lower_scale = best_scale = corrected_scale;
                lower_rmsd = best_rmsd = corrected_rmsd;
                if (target - best_rmsd <= tolerance * trust_radius) {
                    std::vector<double> step(vector);
                    for (double& value : step) value *= best_scale;
                    return Py_BuildValue("Nddi", numpy_vector_copy(step), best_rmsd,
                                         best_scale, iterations);
                }
            } else {
                upper_scale = corrected_scale; upper_rmsd = corrected_rmsd;
                upper_valid = corrected_valid;
            }
        }
    } else {
        upper_scale = predictor_scale; upper_rmsd = initial_rmsd;
        upper_valid = initial_valid;
    }
    const int remaining = std::max(maximum_iterations - iterations, 0);
    for (int bracket = 0; bracket < remaining; ++bracket) {
        ++iterations;
        const double span = upper_scale - lower_scale;
        if (span <= 8.0 * epsilon) break;
        double fraction = 0.5;
        const double low = lower_rmsd - target, high = upper_rmsd - target;
        if (std::isfinite(low) && std::isfinite(high) && high > low)
            fraction = std::clamp(-low / (high - low), 0.1, 0.9);
        const double trial_scale = lower_scale + fraction * span;
        double trial_rmsd; bool trial_valid;
        if (!call_trust_realization(realize_callback, trial_scale,
                                    trial_rmsd, trial_valid)) return nullptr;
        if (trial_valid && std::isfinite(trial_rmsd) && trial_rmsd <= target) {
            lower_scale = best_scale = trial_scale;
            lower_rmsd = best_rmsd = trial_rmsd;
            if (!upper_valid || target - best_rmsd <= tolerance * trust_radius) break;
        } else {
            upper_scale = trial_scale; upper_rmsd = trial_rmsd;
            upper_valid = trial_valid;
        }
    }
    if (best_scale <= 0.0) {
        PyErr_SetString(PyExc_RuntimeError,
            "unable to realize a nonzero optimizer step inside the Cartesian trust region");
        return nullptr;
    }
    if (best_rmsd > trust_radius * (1.0 + 8.0 * epsilon)) {
        PyErr_SetString(PyExc_RuntimeError,
            "Cartesian trust solver returned an out-of-radius step");
        return nullptr;
    }
    std::vector<double> step(vector);
    for (double& value : step) value *= best_scale;
    return Py_BuildValue("Nddi", numpy_vector_copy(step), best_rmsd,
                         best_scale, iterations);
}

PyMethodDef methods[]={
    {"optimize_sycart",reinterpret_cast<PyCFunction>(optimize),METH_VARARGS|METH_KEYWORDS,
     "Run the compiled resident SYCART BFGS/L-BFGS loop."},
    {"apply_preconditioner",apply_preconditioner,METH_VARARGS,
     "Apply the compiled sparse Davidson--Krylov preconditioner."},
    {"remove_rigid_motion",remove_rigid_motion,METH_VARARGS,
     "Project translations and rotations from a Cartesian vector."},
    {"rigid_gauge_columns_valid",rigid_gauge_columns_valid,METH_VARARGS,
     "Validate a cached Cartesian gauge in the rigid-motion space."},
    {"sonic_sparse_backtransform",
     reinterpret_cast<PyCFunction>(sonic_sparse_backtransform),
     METH_VARARGS | METH_KEYWORDS,
     "Run the sparse SONIC microiteration controller."},
    {"restrict_cartesian_trust",restrict_cartesian_trust,METH_VARARGS,
     "Run the safeguarded Cartesian trust scalar controller."},
    {nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_optimizer_native",nullptr,-1,methods};
}

PyMODINIT_FUNC PyInit__optimizer_native() { import_array(); return PyModule_Create(&module); }
