#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_24_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#define MATRIX_STRINGIFY_DETAIL(value) #value
#define MATRIX_STRINGIFY(value) MATRIX_STRINGIFY_DETAIL(value)

namespace {

constexpr double kCoincidentTolerance = 1.0e-12;
constexpr double kPenetrationArgument = 8.0;

struct Inputs {
    PyArrayObject* coordinates;
    PyArrayObject* charges;
    PyArrayObject* widths;
    npy_intp atoms;
};

struct PairInputs {
    PyArrayObject* pairs;
    npy_intp count;
};

struct RadialInputs {
    PyArrayObject* coordinates;
    PyArrayObject* epsilon;
    PyArrayObject* rmin_half;
    npy_intp atoms;
};

void release_radial_inputs(RadialInputs& input) {
    Py_XDECREF(input.coordinates);
    Py_XDECREF(input.epsilon);
    Py_XDECREF(input.rmin_half);
}

void release_inputs(Inputs& input) {
    Py_XDECREF(input.coordinates);
    Py_XDECREF(input.charges);
    Py_XDECREF(input.widths);
}

bool parse_inputs(PyObject* coordinates, PyObject* charges, PyObject* widths, Inputs& out) {
    out.coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.charges = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(charges, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.widths = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(widths, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (out.coordinates == nullptr || out.charges == nullptr || out.widths == nullptr) {
        release_inputs(out);
        return false;
    }
    if (PyArray_NDIM(out.coordinates) != 2 || PyArray_DIM(out.coordinates, 1) != 3 ||
        PyArray_NDIM(out.charges) != 1 || PyArray_NDIM(out.widths) != 1) {
        PyErr_SetString(PyExc_ValueError, "native ZAFF arrays must have shapes (N,3), (N,), (N,)");
        release_inputs(out);
        return false;
    }
    out.atoms = PyArray_DIM(out.coordinates, 0);
    if (PyArray_DIM(out.charges, 0) != out.atoms || PyArray_DIM(out.widths, 0) != out.atoms) {
        PyErr_SetString(PyExc_ValueError, "native ZAFF array dimensions are inconsistent");
        release_inputs(out);
        return false;
    }
    const double* q = static_cast<const double*>(PyArray_DATA(out.charges));
    const double* w = static_cast<const double*>(PyArray_DATA(out.widths));
    const double* xyz = static_cast<const double*>(PyArray_DATA(out.coordinates));
    for (npy_intp atom = 0; atom < out.atoms; ++atom) {
        if (!std::isfinite(q[atom]) || !std::isfinite(w[atom]) || w[atom] <= 0.0) {
            PyErr_SetString(PyExc_ValueError, "native ZAFF charges and widths must be finite and widths positive");
            release_inputs(out);
            return false;
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(xyz[3 * atom + axis])) {
                PyErr_SetString(PyExc_ValueError, "native ZAFF coordinates must be finite");
                release_inputs(out);
                return false;
            }
        }
    }
    return true;
}

bool parse_pairs(PyObject* pairs, npy_intp atoms, PairInputs& out) {
    out.pairs = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(pairs, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    if (out.pairs == nullptr) {
        return false;
    }
    if (PyArray_NDIM(out.pairs) != 2 || PyArray_DIM(out.pairs, 1) != 2) {
        PyErr_SetString(PyExc_ValueError, "native ZAFF pair list must have shape (P,2)");
        Py_DECREF(out.pairs);
        out.pairs = nullptr;
        return false;
    }
    out.count = PyArray_DIM(out.pairs, 0);
    const npy_intp* indices =
        static_cast<const npy_intp*>(PyArray_DATA(out.pairs));
    for (npy_intp pair = 0; pair < out.count; ++pair) {
        const npy_intp left = indices[2 * pair];
        const npy_intp right = indices[2 * pair + 1];
        if (left < 0 || right < 0 || left >= atoms || right >= atoms ||
            left >= right) {
            PyErr_SetString(
                PyExc_ValueError,
                "native ZAFF pairs must be canonical distinct in-range indices");
            Py_DECREF(out.pairs);
            out.pairs = nullptr;
            return false;
        }
    }
    return true;
}

bool parse_radial_inputs(
    PyObject* coordinates, PyObject* epsilon, PyObject* rmin_half,
    RadialInputs& out) {
    out.coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.epsilon = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(epsilon, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.rmin_half = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(rmin_half, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (out.coordinates == nullptr || out.epsilon == nullptr ||
        out.rmin_half == nullptr) {
        release_radial_inputs(out);
        return false;
    }
    if (PyArray_NDIM(out.coordinates) != 2 ||
        PyArray_DIM(out.coordinates, 1) != 3 ||
        PyArray_NDIM(out.epsilon) != 1 ||
        PyArray_NDIM(out.rmin_half) != 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "native Exp-PE arrays must have shapes (N,3), (N,), (N,)");
        release_radial_inputs(out);
        return false;
    }
    out.atoms = PyArray_DIM(out.coordinates, 0);
    if (PyArray_DIM(out.epsilon, 0) != out.atoms ||
        PyArray_DIM(out.rmin_half, 0) != out.atoms) {
        PyErr_SetString(
            PyExc_ValueError, "native Exp-PE array dimensions are inconsistent");
        release_radial_inputs(out);
        return false;
    }
    const double* xyz =
        static_cast<const double*>(PyArray_DATA(out.coordinates));
    const double* eps = static_cast<const double*>(PyArray_DATA(out.epsilon));
    const double* rmin =
        static_cast<const double*>(PyArray_DATA(out.rmin_half));
    for (npy_intp atom = 0; atom < out.atoms; ++atom) {
        if (!std::isfinite(eps[atom]) || eps[atom] <= 0.0 ||
            !std::isfinite(rmin[atom]) || rmin[atom] <= 0.0) {
            PyErr_SetString(
                PyExc_ValueError,
                "native Exp-PE parameters must be finite and positive");
            release_radial_inputs(out);
            return false;
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(xyz[3 * atom + axis])) {
                PyErr_SetString(
                    PyExc_ValueError, "native Exp-PE coordinates must be finite");
                release_radial_inputs(out);
                return false;
            }
        }
    }
    return true;
}

inline void displacement(
    const double* xyz, npy_intp left, npy_intp right,
    double& dx, double& dy, double& dz, double& r2) {
    dx = xyz[3 * left] - xyz[3 * right];
    dy = xyz[3 * left + 1] - xyz[3 * right + 1];
    dz = xyz[3 * left + 2] - xyz[3 * right + 2];
    r2 = dx * dx + dy * dy + dz * dz;
}

inline void gaussian_correction(
    double distance, double product, double width_left, double width_right,
    double& energy, double& first, double& second) {
    const double beta = 1.0 / std::sqrt(
        2.0 * (width_left * width_left + width_right * width_right));
    const double argument = beta * distance;
    const double gaussian = std::exp(-(argument * argument));
    const double error_function = std::erf(argument);
    const double root_pi = std::sqrt(std::acos(-1.0));
    const double inverse_r = 1.0 / distance;
    const double inverse_r2 = inverse_r * inverse_r;
    const double inverse_r3 = inverse_r2 * inverse_r;
    const double gaussian_energy = product * error_function * inverse_r;
    const double gaussian_first = product * (
        2.0 * beta * gaussian * inverse_r / root_pi -
        error_function * inverse_r2);
    const double gaussian_second = product * (
        -4.0 * beta * beta * beta * gaussian / root_pi -
        4.0 * beta * gaussian * inverse_r2 / root_pi +
        2.0 * error_function * inverse_r3);
    energy = gaussian_energy - product * inverse_r;
    first = gaussian_first + product * inverse_r2;
    second = gaussian_second - 2.0 * product * inverse_r3;
}

bool accumulate_point_energy(
    const Inputs& input, double& energy, double* gradient,
    const double* direction, double* hessian_product) {
    const double* xyz = static_cast<const double*>(PyArray_DATA(input.coordinates));
    const double* q = static_cast<const double*>(PyArray_DATA(input.charges));
    for (npy_intp left = 0; left < input.atoms; ++left) {
        for (npy_intp right = left + 1; right < input.atoms; ++right) {
            double dx, dy, dz, r2;
            displacement(xyz, left, right, dx, dy, dz, r2);
            const double distance = std::sqrt(r2);
            if (distance <= kCoincidentTolerance) {
                return false;
            }
            const double product = q[left] * q[right];
            const double inverse_r = 1.0 / distance;
            const double inverse_r3 = inverse_r / r2;
            energy += product * inverse_r;
            if (gradient != nullptr) {
                const double factor = -product * inverse_r3;
                const double values[3] = {factor * dx, factor * dy, factor * dz};
                for (int axis = 0; axis < 3; ++axis) {
                    gradient[3 * left + axis] += values[axis];
                    gradient[3 * right + axis] -= values[axis];
                }
            }
            if (hessian_product != nullptr) {
                const double delta[3] = {dx, dy, dz};
                const double inverse_r5 = inverse_r3 / r2;
                double dv[3];
                for (int axis = 0; axis < 3; ++axis) {
                    dv[axis] = direction[3 * left + axis] - direction[3 * right + axis];
                }
                for (int row = 0; row < 3; ++row) {
                    double contribution = 0.0;
                    for (int column = 0; column < 3; ++column) {
                        const double block = product * (
                            3.0 * delta[row] * delta[column] * inverse_r5 -
                            (row == column ? inverse_r3 : 0.0));
                        contribution += block * dv[column];
                    }
                    hessian_product[3 * left + row] += contribution;
                    hessian_product[3 * right + row] -= contribution;
                }
            }
        }
    }
    return true;
}

npy_intp accumulate_penetration(
    const Inputs& input, double& energy, double* gradient,
    const double* direction, double* hessian_product) {
    const double* xyz = static_cast<const double*>(PyArray_DATA(input.coordinates));
    const double* q = static_cast<const double*>(PyArray_DATA(input.charges));
    const double* widths = static_cast<const double*>(PyArray_DATA(input.widths));
    npy_intp count = 0;
    for (npy_intp left = 0; left < input.atoms; ++left) {
        for (npy_intp right = left + 1; right < input.atoms; ++right) {
            double dx, dy, dz, r2;
            displacement(xyz, left, right, dx, dy, dz, r2);
            const double distance = std::sqrt(r2);
            const double beta = 1.0 / std::sqrt(
                2.0 * (widths[left] * widths[left] + widths[right] * widths[right]));
            if (beta * distance >= kPenetrationArgument) {
                continue;
            }
            double correction_energy, first, second;
            gaussian_correction(
                distance, q[left] * q[right], widths[left], widths[right],
                correction_energy, first, second);
            energy += correction_energy;
            ++count;
            const double inverse_r = 1.0 / distance;
            const double unit[3] = {dx * inverse_r, dy * inverse_r, dz * inverse_r};
            if (gradient != nullptr) {
                for (int axis = 0; axis < 3; ++axis) {
                    const double value = first * unit[axis];
                    gradient[3 * left + axis] += value;
                    gradient[3 * right + axis] -= value;
                }
            }
            if (hessian_product != nullptr) {
                double dv[3];
                for (int axis = 0; axis < 3; ++axis) {
                    dv[axis] = direction[3 * left + axis] - direction[3 * right + axis];
                }
                const double transverse = first * inverse_r;
                const double longitudinal = second - transverse;
                for (int row = 0; row < 3; ++row) {
                    double contribution = 0.0;
                    for (int column = 0; column < 3; ++column) {
                        const double block =
                            longitudinal * unit[row] * unit[column] +
                            (row == column ? transverse : 0.0);
                        contribution += block * dv[column];
                    }
                    hessian_product[3 * left + row] += contribution;
                    hessian_product[3 * right + row] -= contribution;
                }
            }
        }
    }
    return count;
}

npy_intp accumulate_penetration_pairs(
    const Inputs& input, const PairInputs& pair_input, double& energy,
    double* gradient, const double* direction, double* hessian_product) {
    const double* xyz = static_cast<const double*>(PyArray_DATA(input.coordinates));
    const double* q = static_cast<const double*>(PyArray_DATA(input.charges));
    const double* widths = static_cast<const double*>(PyArray_DATA(input.widths));
    const npy_intp* pairs =
        static_cast<const npy_intp*>(PyArray_DATA(pair_input.pairs));
    npy_intp count = 0;
    for (npy_intp pair = 0; pair < pair_input.count; ++pair) {
        const npy_intp left = pairs[2 * pair];
        const npy_intp right = pairs[2 * pair + 1];
        double dx, dy, dz, r2;
        displacement(xyz, left, right, dx, dy, dz, r2);
        const double distance = std::sqrt(r2);
        if (distance <= kCoincidentTolerance) {
            return -1;
        }
        const double beta = 1.0 / std::sqrt(
            2.0 * (widths[left] * widths[left] + widths[right] * widths[right]));
        if (beta * distance >= kPenetrationArgument) {
            continue;
        }
        double correction_energy, first, second;
        gaussian_correction(
            distance, q[left] * q[right], widths[left], widths[right],
            correction_energy, first, second);
        energy += correction_energy;
        ++count;
        const double inverse_r = 1.0 / distance;
        const double unit[3] = {dx * inverse_r, dy * inverse_r, dz * inverse_r};
        if (gradient != nullptr) {
            for (int axis = 0; axis < 3; ++axis) {
                const double value = first * unit[axis];
                gradient[3 * left + axis] += value;
                gradient[3 * right + axis] -= value;
            }
        }
        if (hessian_product != nullptr) {
            double dv[3];
            for (int axis = 0; axis < 3; ++axis) {
                dv[axis] =
                    direction[3 * left + axis] - direction[3 * right + axis];
            }
            const double transverse = first * inverse_r;
            const double longitudinal = second - transverse;
            for (int row = 0; row < 3; ++row) {
                double contribution = 0.0;
                for (int column = 0; column < 3; ++column) {
                    const double block =
                        longitudinal * unit[row] * unit[column] +
                        (row == column ? transverse : 0.0);
                    contribution += block * dv[column];
                }
                hessian_product[3 * left + row] += contribution;
                hessian_product[3 * right + row] -= contribution;
            }
        }
    }
    return count;
}

npy_intp accumulate_penetration_potential_pairs(
    const Inputs& input, const PairInputs& pair_input, double* potential) {
    const double* xyz = static_cast<const double*>(PyArray_DATA(input.coordinates));
    const double* q = static_cast<const double*>(PyArray_DATA(input.charges));
    const double* widths = static_cast<const double*>(PyArray_DATA(input.widths));
    const npy_intp* pairs =
        static_cast<const npy_intp*>(PyArray_DATA(pair_input.pairs));
    npy_intp count = 0;
    for (npy_intp pair = 0; pair < pair_input.count; ++pair) {
        const npy_intp left = pairs[2 * pair];
        const npy_intp right = pairs[2 * pair + 1];
        double dx, dy, dz, r2;
        displacement(xyz, left, right, dx, dy, dz, r2);
        const double distance = std::sqrt(r2);
        if (distance <= kCoincidentTolerance) {
            return -1;
        }
        const double beta = 1.0 / std::sqrt(
            2.0 * (widths[left] * widths[left] + widths[right] * widths[right]));
        if (beta * distance >= kPenetrationArgument) {
            continue;
        }
        double correction, unused_first, unused_second;
        gaussian_correction(
            distance, 1.0, widths[left], widths[right],
            correction, unused_first, unused_second);
        potential[left] += correction * q[right];
        potential[right] += correction * q[left];
        ++count;
    }
    return count;
}

inline void damped_exppe(
    double distance, double epsilon, double rmin,
    double& energy, double& first, double& second) {
    constexpr double alpha = 12.649110640673517;  // sqrt(160), UFF 12-6 curvature
    const double x = distance / rmin;
    const double exp_full = std::exp(alpha * (1.0 - x));
    const double exp_half = std::exp(0.5 * alpha * (1.0 - x));
    const double x2 = x * x;
    const double polynomial = x2 * x2 - 2.0 * x2 + 3.0;
    const double polynomial_1 = 4.0 * x * x2 - 4.0 * x;
    const double polynomial_2 = 12.0 * x2 - 4.0;
    const double dimensionless_0 = exp_full - polynomial * exp_half;
    const double dimensionless_1 =
        -alpha * exp_full -
        exp_half * (polynomial_1 - 0.5 * alpha * polynomial);
    const double dimensionless_2 =
        alpha * alpha * exp_full +
        exp_half * (
            -polynomial_2 + alpha * polynomial_1 -
            0.25 * alpha * alpha * polynomial);
    const double radial_energy = epsilon * dimensionless_0;
    const double radial_first = epsilon * dimensionless_1 / rmin;
    const double radial_second = epsilon * dimensionless_2 / (rmin * rmin);
    energy = radial_energy;
    first = radial_first;
    second = radial_second;
}

npy_intp accumulate_damped_exppe_pairs(
    const RadialInputs& input, const PairInputs& pair_input, double cutoff,
    double& energy, double* gradient, const double* direction,
    double* hessian_product) {
    constexpr double bohr_to_angstrom = 0.52917721092;
    const double* xyz =
        static_cast<const double*>(PyArray_DATA(input.coordinates));
    const double* epsilon =
        static_cast<const double*>(PyArray_DATA(input.epsilon));
    const double* rmin_half =
        static_cast<const double*>(PyArray_DATA(input.rmin_half));
    const npy_intp* pairs =
        static_cast<const npy_intp*>(PyArray_DATA(pair_input.pairs));
    const double cutoff2 = cutoff * cutoff;
    npy_intp count = 0;
    for (npy_intp pair = 0; pair < pair_input.count; ++pair) {
        const npy_intp left = pairs[2 * pair];
        const npy_intp right = pairs[2 * pair + 1];
        double dx, dy, dz, r2;
        displacement(xyz, left, right, dx, dy, dz, r2);
        if (r2 > cutoff2) {
            continue;
        }
        const double distance = std::sqrt(r2);
        if (distance <= kCoincidentTolerance) {
            return -1;
        }
        const double pair_epsilon =
            std::sqrt(epsilon[left] * epsilon[right]);
        const double pair_rmin = rmin_half[left] + rmin_half[right];
        double pair_energy, first, second;
        damped_exppe(
            distance, pair_epsilon, pair_rmin, pair_energy, first, second);
        energy += pair_energy;
        ++count;
        const double inverse_r = 1.0 / distance;
        const double unit[3] = {dx * inverse_r, dy * inverse_r, dz * inverse_r};
        if (gradient != nullptr) {
            for (int axis = 0; axis < 3; ++axis) {
                const double value = first * bohr_to_angstrom * unit[axis];
                gradient[3 * left + axis] += value;
                gradient[3 * right + axis] -= value;
            }
        }
        if (hessian_product != nullptr) {
            double dv[3];
            for (int axis = 0; axis < 3; ++axis) {
                dv[axis] =
                    direction[3 * left + axis] - direction[3 * right + axis];
            }
            const double transverse = first * inverse_r;
            const double longitudinal = second - transverse;
            const double scale = bohr_to_angstrom * bohr_to_angstrom;
            for (int row = 0; row < 3; ++row) {
                double contribution = 0.0;
                for (int column = 0; column < 3; ++column) {
                    const double block =
                        longitudinal * unit[row] * unit[column] +
                        (row == column ? transverse : 0.0);
                    contribution += block * dv[column];
                }
                contribution *= scale;
                hessian_product[3 * left + row] += contribution;
                hessian_product[3 * right + row] -= contribution;
            }
        }
    }
    return count;
}

PyObject* direct_gaussian_energy(PyObject*, PyObject* args) {
    PyObject *coordinates, *charges, *widths;
    if (!PyArg_ParseTuple(args, "OOO", &coordinates, &charges, &widths)) {
        return nullptr;
    }
    Inputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(coordinates, charges, widths, input)) {
        return nullptr;
    }
    double energy = 0.0;
    bool valid;
    npy_intp count = 0;
    Py_BEGIN_ALLOW_THREADS
    valid = accumulate_point_energy(input, energy, nullptr, nullptr, nullptr);
    if (valid) {
        count = accumulate_penetration(input, energy, nullptr, nullptr, nullptr);
    }
    Py_END_ALLOW_THREADS
    release_inputs(input);
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "coincident atoms in electrostatic pair");
        return nullptr;
    }
    return Py_BuildValue("dL", energy, static_cast<long long>(count));
}

PyObject* direct_gaussian_energy_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates, *charges, *widths;
    if (!PyArg_ParseTuple(args, "OOO", &coordinates, &charges, &widths)) {
        return nullptr;
    }
    Inputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(coordinates, charges, widths, input)) {
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        release_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    bool valid;
    npy_intp count = 0;
    double* gradient_data = static_cast<double*>(PyArray_DATA(gradient));
    Py_BEGIN_ALLOW_THREADS
    valid = accumulate_point_energy(input, energy, gradient_data, nullptr, nullptr);
    if (valid) {
        count = accumulate_penetration(input, energy, gradient_data, nullptr, nullptr);
    }
    Py_END_ALLOW_THREADS
    release_inputs(input);
    if (!valid) {
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in electrostatic pair");
        return nullptr;
    }
    return Py_BuildValue("dNL", energy, gradient, static_cast<long long>(count));
}

PyObject* direct_gaussian_hessian_vector(PyObject*, PyObject* args) {
    PyObject *coordinates, *charges, *widths, *direction_object;
    if (!PyArg_ParseTuple(
            args, "OOOO", &coordinates, &charges, &widths, &direction_object)) {
        return nullptr;
    }
    Inputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(coordinates, charges, widths, input)) {
        return nullptr;
    }
    PyArrayObject* direction = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(direction_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (direction == nullptr || PyArray_NDIM(direction) != 2 ||
        PyArray_DIM(direction, 0) != input.atoms || PyArray_DIM(direction, 1) != 3) {
        Py_XDECREF(direction);
        release_inputs(input);
        PyErr_SetString(PyExc_ValueError, "native ZAFF direction must have shape (N,3)");
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (product == nullptr) {
        Py_DECREF(direction);
        release_inputs(input);
        return nullptr;
    }
    double unused_energy = 0.0;
    bool valid;
    const double* direction_data = static_cast<const double*>(PyArray_DATA(direction));
    double* product_data = static_cast<double*>(PyArray_DATA(product));
    Py_BEGIN_ALLOW_THREADS
    valid = accumulate_point_energy(
        input, unused_energy, nullptr, direction_data, product_data);
    if (valid) {
        accumulate_penetration(
            input, unused_energy, nullptr, direction_data, product_data);
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(direction);
    release_inputs(input);
    if (!valid) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in electrostatic pair");
        return nullptr;
    }
    return reinterpret_cast<PyObject*>(product);
}

PyObject* direct_gaussian_potential(PyObject*, PyObject* args) {
    PyObject *coordinates, *charges, *widths;
    if (!PyArg_ParseTuple(args, "OOO", &coordinates, &charges, &widths)) {
        return nullptr;
    }
    Inputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(coordinates, charges, widths, input)) return nullptr;
    npy_intp dimensions[1] = {input.atoms};
    PyArrayObject* potential = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, dimensions, NPY_DOUBLE, 0));
    if (potential == nullptr) {
        release_inputs(input);
        return nullptr;
    }
    const auto* xyz = static_cast<const double*>(PyArray_DATA(input.coordinates));
    const auto* q = static_cast<const double*>(PyArray_DATA(input.charges));
    const auto* width = static_cast<const double*>(PyArray_DATA(input.widths));
    auto* output = static_cast<double*>(PyArray_DATA(potential));
    bool valid = true;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp atom = 0; atom < input.atoms; ++atom) {
        output[atom] = q[atom] / (std::sqrt(std::acos(-1.0)) * width[atom]);
    }
    for (npy_intp left = 0; left < input.atoms; ++left) {
        for (npy_intp right = left + 1; right < input.atoms; ++right) {
            double dx, dy, dz, r2;
            displacement(xyz, left, right, dx, dy, dz, r2);
            const double distance = std::sqrt(r2);
            if (distance <= kCoincidentTolerance) {
                valid = false;
                continue;
            }
            const double beta = 1.0 / std::sqrt(
                2.0 * (width[left] * width[left] + width[right] * width[right]));
            const double kernel = std::erf(beta * distance) / distance;
            output[left] += kernel * q[right];
            output[right] += kernel * q[left];
        }
    }
    Py_END_ALLOW_THREADS
    release_inputs(input);
    if (!valid) {
        Py_DECREF(potential);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in Gaussian potential");
        return nullptr;
    }
    return reinterpret_cast<PyObject*>(potential);
}

PyObject* direct_gaussian_directional_potential(PyObject*, PyObject* args) {
    PyObject *coordinates, *charges, *widths, *direction_object;
    if (!PyArg_ParseTuple(
            args, "OOOO", &coordinates, &charges, &widths, &direction_object)) {
        return nullptr;
    }
    Inputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(coordinates, charges, widths, input)) return nullptr;
    PyArrayObject* direction = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(direction_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (direction == nullptr || PyArray_NDIM(direction) != 2 ||
        PyArray_DIM(direction, 0) != input.atoms || PyArray_DIM(direction, 1) != 3) {
        Py_XDECREF(direction);
        release_inputs(input);
        PyErr_SetString(PyExc_ValueError, "Gaussian potential direction must have shape (N,3)");
        return nullptr;
    }
    npy_intp dimensions[1] = {input.atoms};
    PyArrayObject* potential = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, dimensions, NPY_DOUBLE, 0));
    if (potential == nullptr) {
        Py_DECREF(direction);
        release_inputs(input);
        return nullptr;
    }
    const auto* xyz = static_cast<const double*>(PyArray_DATA(input.coordinates));
    const auto* q = static_cast<const double*>(PyArray_DATA(input.charges));
    const auto* width = static_cast<const double*>(PyArray_DATA(input.widths));
    const auto* vector = static_cast<const double*>(PyArray_DATA(direction));
    auto* output = static_cast<double*>(PyArray_DATA(potential));
    bool valid = true;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp left = 0; left < input.atoms; ++left) {
        for (npy_intp right = left + 1; right < input.atoms; ++right) {
            double dx, dy, dz, r2;
            displacement(xyz, left, right, dx, dy, dz, r2);
            const double distance = std::sqrt(r2);
            if (distance <= kCoincidentTolerance) {
                valid = false;
                continue;
            }
            const double beta = 1.0 / std::sqrt(
                2.0 * (width[left] * width[left] + width[right] * width[right]));
            const double argument = beta * distance;
            const double first =
                2.0 * beta * std::exp(-argument * argument) /
                    (std::sqrt(std::acos(-1.0)) * distance) -
                std::erf(argument) / r2;
            const double rdot = (
                dx * (vector[3 * left] - vector[3 * right]) +
                dy * (vector[3 * left + 1] - vector[3 * right + 1]) +
                dz * (vector[3 * left + 2] - vector[3 * right + 2])) / distance;
            const double derivative = first * rdot;
            output[left] += derivative * q[right];
            output[right] += derivative * q[left];
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(direction);
    release_inputs(input);
    if (!valid) {
        Py_DECREF(potential);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in directional Gaussian potential");
        return nullptr;
    }
    return reinterpret_cast<PyObject*>(potential);
}

PyObject* gaussian_correction_energy(PyObject*, PyObject* args) {
    PyObject *coordinates, *charges, *widths, *pairs;
    if (!PyArg_ParseTuple(
            args, "OOOO", &coordinates, &charges, &widths, &pairs)) {
        return nullptr;
    }
    Inputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(coordinates, charges, widths, input)) {
        return nullptr;
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs, input.atoms, pair_input)) {
        release_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_penetration_pairs(
        input, pair_input, energy, nullptr, nullptr, nullptr);
    Py_END_ALLOW_THREADS
    Py_DECREF(pair_input.pairs);
    release_inputs(input);
    if (count < 0) {
        PyErr_SetString(PyExc_ValueError, "coincident atoms in electrostatic pair");
        return nullptr;
    }
    return Py_BuildValue("dL", energy, static_cast<long long>(count));
}

PyObject* gaussian_correction_energy_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates, *charges, *widths, *pairs;
    if (!PyArg_ParseTuple(
            args, "OOOO", &coordinates, &charges, &widths, &pairs)) {
        return nullptr;
    }
    Inputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(coordinates, charges, widths, input)) {
        return nullptr;
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs, input.atoms, pair_input)) {
        release_inputs(input);
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        Py_DECREF(pair_input.pairs);
        release_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    double* gradient_data = static_cast<double*>(PyArray_DATA(gradient));
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_penetration_pairs(
        input, pair_input, energy, gradient_data, nullptr, nullptr);
    Py_END_ALLOW_THREADS
    Py_DECREF(pair_input.pairs);
    release_inputs(input);
    if (count < 0) {
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in electrostatic pair");
        return nullptr;
    }
    return Py_BuildValue("dNL", energy, gradient, static_cast<long long>(count));
}

PyObject* gaussian_correction_hessian_vector(PyObject*, PyObject* args) {
    PyObject *coordinates, *charges, *widths, *pairs, *direction_object;
    if (!PyArg_ParseTuple(
            args, "OOOOO", &coordinates, &charges, &widths, &pairs,
            &direction_object)) {
        return nullptr;
    }
    Inputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(coordinates, charges, widths, input)) {
        return nullptr;
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs, input.atoms, pair_input)) {
        release_inputs(input);
        return nullptr;
    }
    PyArrayObject* direction = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(direction_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (direction == nullptr || PyArray_NDIM(direction) != 2 ||
        PyArray_DIM(direction, 0) != input.atoms ||
        PyArray_DIM(direction, 1) != 3) {
        Py_XDECREF(direction);
        Py_DECREF(pair_input.pairs);
        release_inputs(input);
        PyErr_SetString(
            PyExc_ValueError, "native ZAFF direction must have shape (N,3)");
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (product == nullptr) {
        Py_DECREF(direction);
        Py_DECREF(pair_input.pairs);
        release_inputs(input);
        return nullptr;
    }
    double unused_energy = 0.0;
    npy_intp count;
    const double* direction_data =
        static_cast<const double*>(PyArray_DATA(direction));
    double* product_data = static_cast<double*>(PyArray_DATA(product));
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_penetration_pairs(
        input, pair_input, unused_energy, nullptr, direction_data, product_data);
    Py_END_ALLOW_THREADS
    Py_DECREF(direction);
    Py_DECREF(pair_input.pairs);
    release_inputs(input);
    if (count < 0) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in electrostatic pair");
        return nullptr;
    }
    return Py_BuildValue("NL", product, static_cast<long long>(count));
}

PyObject* gaussian_correction_potential(PyObject*, PyObject* args) {
    PyObject *coordinates, *charges, *widths, *pairs;
    if (!PyArg_ParseTuple(
            args, "OOOO", &coordinates, &charges, &widths, &pairs)) {
        return nullptr;
    }
    Inputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(coordinates, charges, widths, input)) {
        return nullptr;
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs, input.atoms, pair_input)) {
        release_inputs(input);
        return nullptr;
    }
    npy_intp dimensions[1] = {input.atoms};
    PyArrayObject* potential = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, dimensions, NPY_DOUBLE, 0));
    if (potential == nullptr) {
        Py_DECREF(pair_input.pairs);
        release_inputs(input);
        return nullptr;
    }
    npy_intp count;
    double* potential_data = static_cast<double*>(PyArray_DATA(potential));
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_penetration_potential_pairs(
        input, pair_input, potential_data);
    Py_END_ALLOW_THREADS
    Py_DECREF(pair_input.pairs);
    release_inputs(input);
    if (count < 0) {
        Py_DECREF(potential);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in electrostatic pair");
        return nullptr;
    }
    return Py_BuildValue("NL", potential, static_cast<long long>(count));
}

// Shared local pair geometry for Exp-PE and the Gaussian penetration
// correction. Long-range point Coulomb remains owned by the FMM operator.
PyObject* sonic_local_pairs(PyObject*, PyObject* args) {
    PyObject *xyz, *charges, *widths, *epsilon, *rmin, *pairs;
    PyObject* direction_object = Py_None;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOOOOd|O", &xyz, &charges, &widths,
                         &epsilon, &rmin, &pairs, &cutoff, &direction_object)) return nullptr;
    if (!std::isfinite(cutoff) || cutoff <= 0) {
        PyErr_SetString(PyExc_ValueError, "local SONIC pair cutoff must be positive");
        return nullptr;
    }
    Inputs electro{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(xyz, charges, widths, electro)) return nullptr;
    RadialInputs radial{nullptr, nullptr, nullptr, 0};
    if (!parse_radial_inputs(xyz, epsilon, rmin, radial)) {
        release_inputs(electro);
        return nullptr;
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs, electro.atoms, pair_input)) {
        release_inputs(electro);
        release_radial_inputs(radial);
        return nullptr;
    }
    PyArrayObject* direction = nullptr;
    PyArrayObject* gradient = nullptr;
    PyArrayObject* product = nullptr;
    auto release = [&]() {
        release_inputs(electro);
        release_radial_inputs(radial);
        Py_DECREF(pair_input.pairs);
        Py_XDECREF(direction);
    };
    if (direction_object != Py_None) {
        direction = reinterpret_cast<PyArrayObject*>(
            PyArray_FROM_OTF(direction_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
        if (direction == nullptr || PyArray_NDIM(direction) != 2 ||
            PyArray_DIM(direction, 0) != electro.atoms || PyArray_DIM(direction, 1) != 3) {
            release();
            PyErr_SetString(PyExc_ValueError, "local SONIC pair direction must have shape (N,3)");
            return nullptr;
        }
        const double* values = static_cast<const double*>(PyArray_DATA(direction));
        for (npy_intp i = 0; i < 3 * electro.atoms; ++i) {
            if (!std::isfinite(values[i])) {
                release();
                PyErr_SetString(PyExc_ValueError, "local SONIC pair direction must be finite");
                return nullptr;
            }
        }
    }
    npy_intp shape[2] = {electro.atoms, 3};
    gradient = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2, shape, NPY_DOUBLE, 0));
    if (direction != nullptr)
        product = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2, shape, NPY_DOUBLE, 0));
    if (gradient == nullptr || (direction != nullptr && product == nullptr)) {
        Py_XDECREF(gradient);
        Py_XDECREF(product);
        release();
        return nullptr;
    }
    const auto* coordinates = static_cast<const double*>(PyArray_DATA(electro.coordinates));
    const auto* q = static_cast<const double*>(PyArray_DATA(electro.charges));
    const auto* w = static_cast<const double*>(PyArray_DATA(electro.widths));
    const auto* eps = static_cast<const double*>(PyArray_DATA(radial.epsilon));
    const auto* radii = static_cast<const double*>(PyArray_DATA(radial.rmin_half));
    const auto* indices = static_cast<const npy_intp*>(PyArray_DATA(pair_input.pairs));
    const auto* vector = direction == nullptr ? nullptr :
        static_cast<const double*>(PyArray_DATA(direction));
    auto* g = static_cast<double*>(PyArray_DATA(gradient));
    auto* hv = product == nullptr ? nullptr : static_cast<double*>(PyArray_DATA(product));
    constexpr double b = 0.52917721092;
    double energy = 0;
    npy_intp exppe_count = 0, penetration_count = 0;
    bool invalid = false;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp pair = 0; pair < pair_input.count; ++pair) {
        const npy_intp left = indices[2 * pair], right = indices[2 * pair + 1];
        double dx, dy, dz, r2;
        displacement(coordinates, left, right, dx, dy, dz, r2);
        const double r = std::sqrt(r2);
        if (r * b <= kCoincidentTolerance) { invalid = true; break; }
        const double inverse = 1.0 / r;
        const double unit[3] = {dx * inverse, dy * inverse, dz * inverse};
        double first = 0, second = 0;
        if (r * b <= cutoff) {
            double e, f, h;
            damped_exppe(r * b, std::sqrt(eps[left] * eps[right]),
                         radii[left] + radii[right], e, f, h);
            energy += e;
            first += b * f;
            second += b * b * h;
            ++exppe_count;
        }
        const double beta = 1.0 / std::sqrt(2 * (w[left] * w[left] + w[right] * w[right]));
        if (beta * r < kPenetrationArgument) {
            double e, f, h;
            gaussian_correction(r, q[left] * q[right], w[left], w[right], e, f, h);
            energy += e;
            first += f;
            second += h;
            ++penetration_count;
        }
        for (int axis = 0; axis < 3; ++axis) {
            const double value = first * unit[axis];
            g[3 * left + axis] += value;
            g[3 * right + axis] -= value;
        }
        if (hv != nullptr) {
            double delta[3], longitudinal_dot = 0;
            for (int axis = 0; axis < 3; ++axis) {
                delta[axis] = vector[3 * left + axis] - vector[3 * right + axis];
                longitudinal_dot += unit[axis] * delta[axis];
            }
            const double transverse = first * inverse;
            for (int axis = 0; axis < 3; ++axis) {
                const double value = transverse * delta[axis] +
                    (second - transverse) * longitudinal_dot * unit[axis];
                hv[3 * left + axis] += value;
                hv[3 * right + axis] -= value;
            }
        }
    }
    invalid = invalid || !std::isfinite(energy);
    for (npy_intp i = 0; i < 3 * electro.atoms; ++i)
        invalid = invalid || !std::isfinite(g[i]) || (hv != nullptr && !std::isfinite(hv[i]));
    Py_END_ALLOW_THREADS
    release();
    if (invalid) {
        Py_DECREF(gradient);
        Py_XDECREF(product);
        PyErr_SetString(PyExc_ValueError, "singular or nonfinite local SONIC pair evaluation");
        return nullptr;
    }
    PyObject* product_object = reinterpret_cast<PyObject*>(product);
    if (product == nullptr) { product_object = Py_None; Py_INCREF(Py_None); }
    return Py_BuildValue("dNNLL", energy, gradient, product_object,
                         static_cast<long long>(exppe_count), static_cast<long long>(penetration_count));
}

PyObject* damped_exppe_energy(PyObject*, PyObject* args) {
    PyObject *coordinates, *epsilon, *rmin_half, *pairs;
    double cutoff;
    if (!PyArg_ParseTuple(
            args, "OOOOd", &coordinates, &epsilon, &rmin_half, &pairs,
            &cutoff)) {
        return nullptr;
    }
    if (!std::isfinite(cutoff) || cutoff <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "native Exp-PE cutoff must be positive");
        return nullptr;
    }
    RadialInputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_radial_inputs(coordinates, epsilon, rmin_half, input)) {
        return nullptr;
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs, input.atoms, pair_input)) {
        release_radial_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_damped_exppe_pairs(
        input, pair_input, cutoff, energy, nullptr, nullptr, nullptr);
    Py_END_ALLOW_THREADS
    Py_DECREF(pair_input.pairs);
    release_radial_inputs(input);
    if (count < 0) {
        PyErr_SetString(PyExc_ValueError, "coincident atoms in Exp-PE pair");
        return nullptr;
    }
    return Py_BuildValue("dL", energy, static_cast<long long>(count));
}

PyObject* damped_exppe_energy_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates, *epsilon, *rmin_half, *pairs;
    double cutoff;
    if (!PyArg_ParseTuple(
            args, "OOOOd", &coordinates, &epsilon, &rmin_half, &pairs,
            &cutoff)) {
        return nullptr;
    }
    if (!std::isfinite(cutoff) || cutoff <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "native Exp-PE cutoff must be positive");
        return nullptr;
    }
    RadialInputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_radial_inputs(coordinates, epsilon, rmin_half, input)) {
        return nullptr;
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs, input.atoms, pair_input)) {
        release_radial_inputs(input);
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        Py_DECREF(pair_input.pairs);
        release_radial_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    double* gradient_data = static_cast<double*>(PyArray_DATA(gradient));
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_damped_exppe_pairs(
        input, pair_input, cutoff, energy, gradient_data, nullptr, nullptr);
    Py_END_ALLOW_THREADS
    Py_DECREF(pair_input.pairs);
    release_radial_inputs(input);
    if (count < 0) {
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in Exp-PE pair");
        return nullptr;
    }
    return Py_BuildValue("dNL", energy, gradient, static_cast<long long>(count));
}

PyObject* damped_exppe_hessian_vector(PyObject*, PyObject* args) {
    PyObject *coordinates, *epsilon, *rmin_half, *pairs, *direction_object;
    double cutoff;
    if (!PyArg_ParseTuple(
            args, "OOOOdO", &coordinates, &epsilon, &rmin_half, &pairs,
            &cutoff, &direction_object)) {
        return nullptr;
    }
    if (!std::isfinite(cutoff) || cutoff <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "native Exp-PE cutoff must be positive");
        return nullptr;
    }
    RadialInputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_radial_inputs(coordinates, epsilon, rmin_half, input)) {
        return nullptr;
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs, input.atoms, pair_input)) {
        release_radial_inputs(input);
        return nullptr;
    }
    PyArrayObject* direction = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(direction_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (direction == nullptr || PyArray_NDIM(direction) != 2 ||
        PyArray_DIM(direction, 0) != input.atoms ||
        PyArray_DIM(direction, 1) != 3) {
        Py_XDECREF(direction);
        Py_DECREF(pair_input.pairs);
        release_radial_inputs(input);
        PyErr_SetString(
            PyExc_ValueError, "native Exp-PE direction must have shape (N,3)");
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (product == nullptr) {
        Py_DECREF(direction);
        Py_DECREF(pair_input.pairs);
        release_radial_inputs(input);
        return nullptr;
    }
    double unused_energy = 0.0;
    npy_intp count;
    const double* direction_data =
        static_cast<const double*>(PyArray_DATA(direction));
    double* product_data = static_cast<double*>(PyArray_DATA(product));
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_damped_exppe_pairs(
        input, pair_input, cutoff, unused_energy, nullptr, direction_data,
        product_data);
    Py_END_ALLOW_THREADS
    Py_DECREF(direction);
    Py_DECREF(pair_input.pairs);
    release_radial_inputs(input);
    if (count < 0) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in Exp-PE pair");
        return nullptr;
    }
    return Py_BuildValue("NL", product, static_cast<long long>(count));
}

PyObject* switched_lj_energy_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates_object, *pairs_object;
    double sigma, epsilon, switch_distance, cutoff;
    if (!PyArg_ParseTuple(
            args, "OOdddd", &coordinates_object, &pairs_object, &sigma,
            &epsilon, &switch_distance, &cutoff)) {
        return nullptr;
    }
    if (!std::isfinite(sigma) || sigma <= 0.0 ||
        !std::isfinite(epsilon) || epsilon <= 0.0 ||
        !std::isfinite(switch_distance) || switch_distance <= 0.0 ||
        !std::isfinite(cutoff) || cutoff <= switch_distance) {
        PyErr_SetString(
            PyExc_ValueError,
            "native switched LJ parameters require positive sigma, epsilon "
            "and 0 < switch < cutoff");
        return nullptr;
    }
    PyArrayObject* coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(
            coordinates_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (coordinates == nullptr || PyArray_NDIM(coordinates) != 2 ||
        PyArray_DIM(coordinates, 1) != 3) {
        Py_XDECREF(coordinates);
        PyErr_SetString(
            PyExc_ValueError,
            "native switched LJ coordinates must have shape (N,3)");
        return nullptr;
    }
    const npy_intp atoms = PyArray_DIM(coordinates, 0);
    const double* xyz =
        static_cast<const double*>(PyArray_DATA(coordinates));
    for (npy_intp atom = 0; atom < atoms; ++atom) {
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(xyz[3 * atom + axis])) {
                Py_DECREF(coordinates);
                PyErr_SetString(
                    PyExc_ValueError,
                    "native switched LJ coordinates must be finite");
                return nullptr;
            }
        }
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs_object, atoms, pair_input)) {
        Py_DECREF(coordinates);
        return nullptr;
    }
    npy_intp dimensions[2] = {atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        Py_DECREF(pair_input.pairs);
        Py_DECREF(coordinates);
        return nullptr;
    }

    const npy_intp* pairs =
        static_cast<const npy_intp*>(PyArray_DATA(pair_input.pairs));
    double* gradient_data = static_cast<double*>(PyArray_DATA(gradient));
    const double cutoff2 = cutoff * cutoff;
    const double interval = cutoff - switch_distance;
    double energy = 0.0;
    npy_intp count = 0;
    bool coincident = false;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp pair = 0; pair < pair_input.count; ++pair) {
        const npy_intp left = pairs[2 * pair];
        const npy_intp right = pairs[2 * pair + 1];
        double dx, dy, dz, r2;
        displacement(xyz, left, right, dx, dy, dz, r2);
        if (r2 > cutoff2) {
            continue;
        }
        const double distance = std::sqrt(r2);
        if (distance <= kCoincidentTolerance) {
            coincident = true;
            break;
        }
        const double ratio = sigma / distance;
        const double ratio2 = ratio * ratio;
        const double ratio6 = ratio2 * ratio2 * ratio2;
        const double ratio12 = ratio6 * ratio6;
        const double bare = 4.0 * epsilon * (ratio12 - ratio6);
        double switching = 1.0;
        double switching_first = 0.0;
        if (distance > switch_distance) {
            const double reduced =
                (distance - switch_distance) / interval;
            const double reduced2 = reduced * reduced;
            const double reduced3 = reduced2 * reduced;
            const double reduced4 = reduced3 * reduced;
            const double reduced5 = reduced4 * reduced;
            switching =
                1.0 - 10.0 * reduced3 + 15.0 * reduced4 -
                6.0 * reduced5;
            switching_first =
                (-30.0 * reduced2 + 60.0 * reduced3 -
                 30.0 * reduced4) /
                interval;
        }
        const double radial_first =
            24.0 * epsilon * (ratio6 - 2.0 * ratio12) /
                distance * switching +
            bare * switching_first;
        const double inverse_r = 1.0 / distance;
        const double unit[3] = {
            dx * inverse_r, dy * inverse_r, dz * inverse_r};
        energy += bare * switching;
        ++count;
        for (int axis = 0; axis < 3; ++axis) {
            const double value = radial_first * unit[axis];
            gradient_data[3 * left + axis] += value;
            gradient_data[3 * right + axis] -= value;
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(pair_input.pairs);
    Py_DECREF(coordinates);
    if (coincident) {
        Py_DECREF(gradient);
        PyErr_SetString(
            PyExc_ValueError, "coincident atoms in switched LJ pair");
        return nullptr;
    }
    return Py_BuildValue(
        "dNL", energy, gradient, static_cast<long long>(count));
}

struct MorseBondInputs {
    PyArrayObject* coordinates;
    PyArrayObject* bonds;
    PyArrayObject* depths;
    PyArrayObject* alphas;
    PyArrayObject* references;
    npy_intp atoms;
    npy_intp count;
};

void release_morse_bond_inputs(MorseBondInputs& input) {
    Py_XDECREF(input.coordinates);
    Py_XDECREF(input.bonds);
    Py_XDECREF(input.depths);
    Py_XDECREF(input.alphas);
    Py_XDECREF(input.references);
}

bool parse_morse_bond_inputs(
    PyObject* coordinates, PyObject* bonds, PyObject* depths,
    PyObject* alphas, PyObject* references, MorseBondInputs& out) {
    out.coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.bonds = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(bonds, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.depths = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(depths, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.alphas = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(alphas, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.references = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(references, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (out.coordinates == nullptr || out.bonds == nullptr ||
        out.depths == nullptr || out.alphas == nullptr ||
        out.references == nullptr) {
        release_morse_bond_inputs(out);
        return false;
    }
    if (PyArray_NDIM(out.coordinates) != 2 ||
        PyArray_DIM(out.coordinates, 1) != 3 ||
        PyArray_NDIM(out.bonds) != 2 || PyArray_DIM(out.bonds, 1) != 2 ||
        PyArray_NDIM(out.depths) != 1 || PyArray_NDIM(out.alphas) != 1 ||
        PyArray_NDIM(out.references) != 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "native Morse bond arrays require (N,3), (B,2), and three (B,) arrays");
        release_morse_bond_inputs(out);
        return false;
    }
    out.atoms = PyArray_DIM(out.coordinates, 0);
    out.count = PyArray_DIM(out.bonds, 0);
    if (PyArray_DIM(out.depths, 0) != out.count ||
        PyArray_DIM(out.alphas, 0) != out.count ||
        PyArray_DIM(out.references, 0) != out.count) {
        PyErr_SetString(
            PyExc_ValueError, "native Morse bond parameter dimensions differ");
        release_morse_bond_inputs(out);
        return false;
    }
    const auto* indices =
        static_cast<const npy_intp*>(PyArray_DATA(out.bonds));
    const auto* depth = static_cast<const double*>(PyArray_DATA(out.depths));
    const auto* alpha = static_cast<const double*>(PyArray_DATA(out.alphas));
    const auto* reference =
        static_cast<const double*>(PyArray_DATA(out.references));
    for (npy_intp bond = 0; bond < out.count; ++bond) {
        const npy_intp left = indices[2 * bond];
        const npy_intp right = indices[2 * bond + 1];
        if (left < 0 || right < 0 || left >= out.atoms ||
            right >= out.atoms || left == right ||
            !std::isfinite(depth[bond]) || depth[bond] < 0.0 ||
            !std::isfinite(alpha[bond]) || alpha[bond] <= 0.0 ||
            !std::isfinite(reference[bond]) || reference[bond] <= 0.0) {
            PyErr_SetString(
                PyExc_ValueError, "native Morse bond record is invalid");
            release_morse_bond_inputs(out);
            return false;
        }
    }
    return true;
}

npy_intp accumulate_morse_bonds(
    const MorseBondInputs& input, double& energy, double* gradient,
    const double* direction, double* product) {
    constexpr double bohr_to_angstrom = 0.52917721092;
    const auto* xyz =
        static_cast<const double*>(PyArray_DATA(input.coordinates));
    const auto* bonds =
        static_cast<const npy_intp*>(PyArray_DATA(input.bonds));
    const auto* depths =
        static_cast<const double*>(PyArray_DATA(input.depths));
    const auto* alphas =
        static_cast<const double*>(PyArray_DATA(input.alphas));
    const auto* references =
        static_cast<const double*>(PyArray_DATA(input.references));
    for (npy_intp bond = 0; bond < input.count; ++bond) {
        const npy_intp left = bonds[2 * bond];
        const npy_intp right = bonds[2 * bond + 1];
        double dx, dy, dz, r2;
        displacement(xyz, left, right, dx, dy, dz, r2);
        const double distance = std::sqrt(r2);
        if (distance <= kCoincidentTolerance) {
            return -1;
        }
        const double exponential = std::exp(
            -alphas[bond] * (distance - references[bond]));
        const double difference = 1.0 - exponential;
        const double first =
            2.0 * depths[bond] * alphas[bond] * exponential * difference;
        const double second =
            2.0 * depths[bond] * alphas[bond] * alphas[bond] *
            (2.0 * exponential * exponential - exponential);
        energy += depths[bond] * difference * difference;
        const double inverse_r = 1.0 / distance;
        const double unit[3] = {
            dx * inverse_r, dy * inverse_r, dz * inverse_r};
        if (gradient != nullptr) {
            for (int axis = 0; axis < 3; ++axis) {
                const double value =
                    first * bohr_to_angstrom * unit[axis];
                gradient[3 * left + axis] += value;
                gradient[3 * right + axis] -= value;
            }
        }
        if (product != nullptr) {
            double delta_direction[3];
            for (int axis = 0; axis < 3; ++axis) {
                delta_direction[axis] =
                    direction[3 * left + axis] -
                    direction[3 * right + axis];
            }
            const double transverse = first * inverse_r;
            const double longitudinal = second - transverse;
            for (int row = 0; row < 3; ++row) {
                double value = 0.0;
                for (int column = 0; column < 3; ++column) {
                    const double block =
                        longitudinal * unit[row] * unit[column] +
                        (row == column ? transverse : 0.0);
                    value += block * delta_direction[column];
                }
                value *= bohr_to_angstrom * bohr_to_angstrom;
                product[3 * left + row] += value;
                product[3 * right + row] -= value;
            }
        }
    }
    return input.count;
}

PyObject* morse_bond_energy(PyObject*, PyObject* args) {
    PyObject *coordinates, *bonds, *depths, *alphas, *references;
    if (!PyArg_ParseTuple(
            args, "OOOOO", &coordinates, &bonds, &depths, &alphas,
            &references)) {
        return nullptr;
    }
    MorseBondInputs input{nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
    if (!parse_morse_bond_inputs(
            coordinates, bonds, depths, alphas, references, input)) {
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_morse_bonds(
        input, energy, nullptr, nullptr, nullptr);
    Py_END_ALLOW_THREADS
    release_morse_bond_inputs(input);
    if (count < 0) {
        PyErr_SetString(PyExc_ValueError, "coincident atoms in Morse bond");
        return nullptr;
    }
    return Py_BuildValue("dL", energy, static_cast<long long>(count));
}

PyObject* morse_bond_energy_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates, *bonds, *depths, *alphas, *references;
    if (!PyArg_ParseTuple(
            args, "OOOOO", &coordinates, &bonds, &depths, &alphas,
            &references)) {
        return nullptr;
    }
    MorseBondInputs input{nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
    if (!parse_morse_bond_inputs(
            coordinates, bonds, depths, alphas, references, input)) {
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        release_morse_bond_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_morse_bonds(
        input, energy, static_cast<double*>(PyArray_DATA(gradient)),
        nullptr, nullptr);
    Py_END_ALLOW_THREADS
    release_morse_bond_inputs(input);
    if (count < 0) {
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in Morse bond");
        return nullptr;
    }
    return Py_BuildValue(
        "dNL", energy, gradient, static_cast<long long>(count));
}

PyObject* morse_bond_hessian_vector(PyObject*, PyObject* args) {
    PyObject *coordinates, *bonds, *depths, *alphas, *references;
    PyObject* direction_object;
    if (!PyArg_ParseTuple(
            args, "OOOOOO", &coordinates, &bonds, &depths, &alphas,
            &references, &direction_object)) {
        return nullptr;
    }
    MorseBondInputs input{nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
    if (!parse_morse_bond_inputs(
            coordinates, bonds, depths, alphas, references, input)) {
        return nullptr;
    }
    PyArrayObject* direction = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(direction_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (direction == nullptr || PyArray_NDIM(direction) != 2 ||
        PyArray_DIM(direction, 0) != input.atoms ||
        PyArray_DIM(direction, 1) != 3) {
        Py_XDECREF(direction);
        release_morse_bond_inputs(input);
        PyErr_SetString(
            PyExc_ValueError, "native Morse direction must have shape (N,3)");
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (product == nullptr) {
        Py_DECREF(direction);
        release_morse_bond_inputs(input);
        return nullptr;
    }
    double unused_energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_morse_bonds(
        input, unused_energy, nullptr,
        static_cast<const double*>(PyArray_DATA(direction)),
        static_cast<double*>(PyArray_DATA(product)));
    Py_END_ALLOW_THREADS
    Py_DECREF(direction);
    release_morse_bond_inputs(input);
    if (count < 0) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in Morse bond");
        return nullptr;
    }
    return Py_BuildValue("NL", product, static_cast<long long>(count));
}

struct GaussianBondInputs {
    PyArrayObject* coordinates;
    PyArrayObject* bonds;
    PyArrayObject* amplitudes;
    PyArrayObject* exponents;
    PyArrayObject* references;
    npy_intp atoms;
    npy_intp count;
};

void release_gaussian_bond_inputs(GaussianBondInputs& input) {
    Py_XDECREF(input.coordinates);
    Py_XDECREF(input.bonds);
    Py_XDECREF(input.amplitudes);
    Py_XDECREF(input.exponents);
    Py_XDECREF(input.references);
}

bool parse_gaussian_bond_inputs(
    PyObject* coordinates, PyObject* bonds, PyObject* amplitudes,
    PyObject* exponents, PyObject* references, GaussianBondInputs& out) {
    out.coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.bonds = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(bonds, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.amplitudes = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(amplitudes, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.exponents = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(exponents, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.references = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(references, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (out.coordinates == nullptr || out.bonds == nullptr ||
        out.amplitudes == nullptr || out.exponents == nullptr ||
        out.references == nullptr) {
        release_gaussian_bond_inputs(out);
        return false;
    }
    if (PyArray_NDIM(out.coordinates) != 2 ||
        PyArray_DIM(out.coordinates, 1) != 3 ||
        PyArray_NDIM(out.bonds) != 2 || PyArray_DIM(out.bonds, 1) != 2 ||
        PyArray_NDIM(out.amplitudes) != 1 ||
        PyArray_NDIM(out.exponents) != 1 ||
        PyArray_NDIM(out.references) != 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "native Gaussian bond arrays require (N,3), (B,2), and three (B,) arrays");
        release_gaussian_bond_inputs(out);
        return false;
    }
    out.atoms = PyArray_DIM(out.coordinates, 0);
    out.count = PyArray_DIM(out.bonds, 0);
    if (PyArray_DIM(out.amplitudes, 0) != out.count ||
        PyArray_DIM(out.exponents, 0) != out.count ||
        PyArray_DIM(out.references, 0) != out.count) {
        PyErr_SetString(
            PyExc_ValueError, "native Gaussian bond parameter dimensions differ");
        release_gaussian_bond_inputs(out);
        return false;
    }
    const auto* indices =
        static_cast<const npy_intp*>(PyArray_DATA(out.bonds));
    const auto* amplitude =
        static_cast<const double*>(PyArray_DATA(out.amplitudes));
    const auto* exponent =
        static_cast<const double*>(PyArray_DATA(out.exponents));
    const auto* reference =
        static_cast<const double*>(PyArray_DATA(out.references));
    for (npy_intp bond = 0; bond < out.count; ++bond) {
        const npy_intp left = indices[2 * bond];
        const npy_intp right = indices[2 * bond + 1];
        if (left < 0 || right < 0 || left >= out.atoms ||
            right >= out.atoms || left == right ||
            !std::isfinite(amplitude[bond]) ||
            !std::isfinite(exponent[bond]) || exponent[bond] <= 0.0 ||
            !std::isfinite(reference[bond]) || reference[bond] <= 0.0) {
            PyErr_SetString(
                PyExc_ValueError, "native Gaussian bond record is invalid");
            release_gaussian_bond_inputs(out);
            return false;
        }
    }
    return true;
}

npy_intp accumulate_gaussian_bonds(
    const GaussianBondInputs& input, double& energy, double* gradient,
    const double* direction, double* product) {
    constexpr double bohr_to_angstrom = 0.52917721092;
    const auto* xyz =
        static_cast<const double*>(PyArray_DATA(input.coordinates));
    const auto* bonds =
        static_cast<const npy_intp*>(PyArray_DATA(input.bonds));
    const auto* amplitudes =
        static_cast<const double*>(PyArray_DATA(input.amplitudes));
    const auto* exponents =
        static_cast<const double*>(PyArray_DATA(input.exponents));
    const auto* references =
        static_cast<const double*>(PyArray_DATA(input.references));
    for (npy_intp bond = 0; bond < input.count; ++bond) {
        const npy_intp left = bonds[2 * bond];
        const npy_intp right = bonds[2 * bond + 1];
        double dx, dy, dz, r2;
        displacement(xyz, left, right, dx, dy, dz, r2);
        const double distance = std::sqrt(r2);
        if (distance <= kCoincidentTolerance) {
            return -1;
        }
        const double delta = distance - references[bond];
        const double exponent = exponents[bond];
        const double value =
            amplitudes[bond] * std::exp(-exponent * delta * delta);
        const double first = -2.0 * exponent * delta * value;
        const double second =
            (4.0 * exponent * exponent * delta * delta - 2.0 * exponent) * value;
        energy += value;
        const double inverse_r = 1.0 / distance;
        const double unit[3] = {
            dx * inverse_r, dy * inverse_r, dz * inverse_r};
        if (gradient != nullptr) {
            for (int axis = 0; axis < 3; ++axis) {
                const double component =
                    first * bohr_to_angstrom * unit[axis];
                gradient[3 * left + axis] += component;
                gradient[3 * right + axis] -= component;
            }
        }
        if (product != nullptr) {
            double delta_direction[3];
            for (int axis = 0; axis < 3; ++axis) {
                delta_direction[axis] =
                    direction[3 * left + axis] -
                    direction[3 * right + axis];
            }
            const double transverse = first * inverse_r;
            const double longitudinal = second - transverse;
            for (int row = 0; row < 3; ++row) {
                double component = 0.0;
                for (int column = 0; column < 3; ++column) {
                    const double block =
                        longitudinal * unit[row] * unit[column] +
                        (row == column ? transverse : 0.0);
                    component += block * delta_direction[column];
                }
                component *= bohr_to_angstrom * bohr_to_angstrom;
                product[3 * left + row] += component;
                product[3 * right + row] -= component;
            }
        }
    }
    return input.count;
}

PyObject* gaussian_bond_energy(PyObject*, PyObject* args) {
    PyObject *coordinates, *bonds, *amplitudes, *exponents, *references;
    if (!PyArg_ParseTuple(
            args, "OOOOO", &coordinates, &bonds, &amplitudes,
            &exponents, &references)) {
        return nullptr;
    }
    GaussianBondInputs input{
        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
    if (!parse_gaussian_bond_inputs(
            coordinates, bonds, amplitudes, exponents, references, input)) {
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_gaussian_bonds(
        input, energy, nullptr, nullptr, nullptr);
    Py_END_ALLOW_THREADS
    release_gaussian_bond_inputs(input);
    if (count < 0) {
        PyErr_SetString(PyExc_ValueError, "coincident atoms in Gaussian bond");
        return nullptr;
    }
    return Py_BuildValue("dL", energy, static_cast<long long>(count));
}

PyObject* gaussian_bond_energy_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates, *bonds, *amplitudes, *exponents, *references;
    if (!PyArg_ParseTuple(
            args, "OOOOO", &coordinates, &bonds, &amplitudes,
            &exponents, &references)) {
        return nullptr;
    }
    GaussianBondInputs input{
        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
    if (!parse_gaussian_bond_inputs(
            coordinates, bonds, amplitudes, exponents, references, input)) {
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        release_gaussian_bond_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_gaussian_bonds(
        input, energy, static_cast<double*>(PyArray_DATA(gradient)),
        nullptr, nullptr);
    Py_END_ALLOW_THREADS
    release_gaussian_bond_inputs(input);
    if (count < 0) {
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in Gaussian bond");
        return nullptr;
    }
    return Py_BuildValue(
        "dNL", energy, gradient, static_cast<long long>(count));
}

PyObject* gaussian_bond_hessian_vector(PyObject*, PyObject* args) {
    PyObject *coordinates, *bonds, *amplitudes, *exponents, *references;
    PyObject* direction_object;
    if (!PyArg_ParseTuple(
            args, "OOOOOO", &coordinates, &bonds, &amplitudes,
            &exponents, &references, &direction_object)) {
        return nullptr;
    }
    GaussianBondInputs input{
        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
    if (!parse_gaussian_bond_inputs(
            coordinates, bonds, amplitudes, exponents, references, input)) {
        return nullptr;
    }
    PyArrayObject* direction = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(direction_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (direction == nullptr || PyArray_NDIM(direction) != 2 ||
        PyArray_DIM(direction, 0) != input.atoms ||
        PyArray_DIM(direction, 1) != 3) {
        Py_XDECREF(direction);
        release_gaussian_bond_inputs(input);
        PyErr_SetString(
            PyExc_ValueError, "native Gaussian direction must have shape (N,3)");
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (product == nullptr) {
        Py_DECREF(direction);
        release_gaussian_bond_inputs(input);
        return nullptr;
    }
    double unused_energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_gaussian_bonds(
        input, unused_energy, nullptr,
        static_cast<const double*>(PyArray_DATA(direction)),
        static_cast<double*>(PyArray_DATA(product)));
    Py_END_ALLOW_THREADS
    Py_DECREF(direction);
    release_gaussian_bond_inputs(input);
    if (count < 0) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in Gaussian bond");
        return nullptr;
    }
    return Py_BuildValue("NL", product, static_cast<long long>(count));
}

struct DynamicGaussianInputs {
    PyArrayObject* coordinates;
    PyArrayObject* coordination_pairs;
    PyArrayObject* coordination_radii;
    PyArrayObject* bonds;
    PyArrayObject* amplitudes;
    PyArrayObject* exponents;
    PyArrayObject* reference_bases;
    PyArrayObject* reference_coefficients;
    npy_intp atoms;
    npy_intp pair_count;
    npy_intp bond_count;
    double cn_max;
    double steepness;
};

void release_dynamic_gaussian_inputs(DynamicGaussianInputs& input) {
    Py_XDECREF(input.coordinates);
    Py_XDECREF(input.coordination_pairs);
    Py_XDECREF(input.coordination_radii);
    Py_XDECREF(input.bonds);
    Py_XDECREF(input.amplitudes);
    Py_XDECREF(input.exponents);
    Py_XDECREF(input.reference_bases);
    Py_XDECREF(input.reference_coefficients);
}

bool parse_dynamic_gaussian_inputs(
    PyObject* coordinates, PyObject* coordination_pairs,
    PyObject* coordination_radii, PyObject* bonds, PyObject* amplitudes,
    PyObject* exponents, PyObject* reference_bases,
    PyObject* reference_coefficients, double cn_max, double steepness,
    DynamicGaussianInputs& out) {
    out.coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.coordination_pairs = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordination_pairs, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.coordination_radii = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordination_radii, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.bonds = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(bonds, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.amplitudes = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(amplitudes, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.exponents = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(exponents, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.reference_bases = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(reference_bases, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.reference_coefficients = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(reference_coefficients, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (out.coordinates == nullptr || out.coordination_pairs == nullptr ||
        out.coordination_radii == nullptr || out.bonds == nullptr ||
        out.amplitudes == nullptr || out.exponents == nullptr ||
        out.reference_bases == nullptr || out.reference_coefficients == nullptr) {
        release_dynamic_gaussian_inputs(out);
        return false;
    }
    if (PyArray_NDIM(out.coordinates) != 2 ||
        PyArray_DIM(out.coordinates, 1) != 3 ||
        PyArray_NDIM(out.coordination_pairs) != 2 ||
        PyArray_DIM(out.coordination_pairs, 1) != 2 ||
        PyArray_NDIM(out.coordination_radii) != 1 ||
        PyArray_NDIM(out.bonds) != 2 || PyArray_DIM(out.bonds, 1) != 2 ||
        PyArray_NDIM(out.amplitudes) != 1 ||
        PyArray_NDIM(out.exponents) != 1 ||
        PyArray_NDIM(out.reference_bases) != 1 ||
        PyArray_NDIM(out.reference_coefficients) != 2 ||
        PyArray_DIM(out.reference_coefficients, 1) != 2) {
        PyErr_SetString(PyExc_ValueError, "invalid dynamic GFN-FF bond array rank");
        release_dynamic_gaussian_inputs(out);
        return false;
    }
    out.atoms = PyArray_DIM(out.coordinates, 0);
    out.pair_count = PyArray_DIM(out.coordination_pairs, 0);
    out.bond_count = PyArray_DIM(out.bonds, 0);
    if (PyArray_DIM(out.coordination_radii, 0) != out.pair_count ||
        PyArray_DIM(out.amplitudes, 0) != out.bond_count ||
        PyArray_DIM(out.exponents, 0) != out.bond_count ||
        PyArray_DIM(out.reference_bases, 0) != out.bond_count ||
        PyArray_DIM(out.reference_coefficients, 0) != out.bond_count ||
        !std::isfinite(cn_max) || cn_max <= 0.0 ||
        !std::isfinite(steepness) || steepness <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "invalid dynamic GFN-FF bond dimensions");
        release_dynamic_gaussian_inputs(out);
        return false;
    }
    const auto* cpairs = static_cast<const npy_intp*>(
        PyArray_DATA(out.coordination_pairs));
    const auto* radii = static_cast<const double*>(
        PyArray_DATA(out.coordination_radii));
    const auto* bindices = static_cast<const npy_intp*>(PyArray_DATA(out.bonds));
    const auto* amplitude = static_cast<const double*>(PyArray_DATA(out.amplitudes));
    const auto* exponent = static_cast<const double*>(PyArray_DATA(out.exponents));
    const auto* bases = static_cast<const double*>(PyArray_DATA(out.reference_bases));
    const auto* coefficients = static_cast<const double*>(
        PyArray_DATA(out.reference_coefficients));
    for (npy_intp pair = 0; pair < out.pair_count; ++pair) {
        const npy_intp left = cpairs[2 * pair];
        const npy_intp right = cpairs[2 * pair + 1];
        if (left < 0 || right < 0 || left >= out.atoms || right >= out.atoms ||
            left == right || !std::isfinite(radii[pair]) || radii[pair] <= 0.0) {
            PyErr_SetString(PyExc_ValueError, "invalid GFN-FF coordination pair");
            release_dynamic_gaussian_inputs(out);
            return false;
        }
    }
    for (npy_intp bond = 0; bond < out.bond_count; ++bond) {
        const npy_intp left = bindices[2 * bond];
        const npy_intp right = bindices[2 * bond + 1];
        if (left < 0 || right < 0 || left >= out.atoms || right >= out.atoms ||
            left == right || !std::isfinite(amplitude[bond]) ||
            !std::isfinite(exponent[bond]) || exponent[bond] <= 0.0 ||
            !std::isfinite(bases[bond]) || bases[bond] <= 0.0 ||
            !std::isfinite(coefficients[2 * bond]) ||
            !std::isfinite(coefficients[2 * bond + 1])) {
            PyErr_SetString(PyExc_ValueError, "invalid dynamic GFN-FF bond record");
            release_dynamic_gaussian_inputs(out);
            return false;
        }
    }
    out.cn_max = cn_max;
    out.steepness = steepness;
    return true;
}

npy_intp accumulate_dynamic_gaussian_bonds(
    const DynamicGaussianInputs& input, double& energy, double* gradient,
    const double* direction, double* product,
    double* coordination_atoms = nullptr, double* coordination_pairs = nullptr) {
    constexpr double bohr_to_angstrom = 0.52917721092;
    constexpr double inverse_sqrt_pi = 0.5641895835477563;
    const auto* xyz = static_cast<const double*>(PyArray_DATA(input.coordinates));
    const auto* cpairs = static_cast<const npy_intp*>(
        PyArray_DATA(input.coordination_pairs));
    const auto* radii = static_cast<const double*>(
        PyArray_DATA(input.coordination_radii));
    const auto* bonds = static_cast<const npy_intp*>(PyArray_DATA(input.bonds));
    const auto* amplitudes = static_cast<const double*>(PyArray_DATA(input.amplitudes));
    const auto* exponents = static_cast<const double*>(PyArray_DATA(input.exponents));
    const auto* bases = static_cast<const double*>(PyArray_DATA(input.reference_bases));
    const auto* coefficients = static_cast<const double*>(
        PyArray_DATA(input.reference_coefficients));

    std::vector<double> raw(input.atoms, 0.0), cn(input.atoms), hp(input.atoms),
        hpp(input.atoms), raw_dot(direction ? input.atoms : 0, 0.0),
        cn_dot(direction ? input.atoms : 0, 0.0),
        adjoint(input.atoms, 0.0), adjoint_dot(direction ? input.atoms : 0, 0.0);
    std::vector<double> pair_distance(input.pair_count), pair_first(input.pair_count),
        pair_second(input.pair_count), pair_rdot(direction ? input.pair_count : 0, 0.0);
    std::vector<std::array<double, 3>> pair_unit(input.pair_count),
        pair_udot(direction ? input.pair_count : 0);
    const double k = input.steepness;
    for (npy_intp pair = 0; pair < input.pair_count; ++pair) {
        const npy_intp left = cpairs[2 * pair];
        const npy_intp right = cpairs[2 * pair + 1];
        double dx, dy, dz, r2;
        displacement(xyz, left, right, dx, dy, dz, r2);
        const double r = std::sqrt(r2);
        if (r <= kCoincidentTolerance) return -1;
        pair_distance[pair] = r;
        pair_unit[pair] = {dx / r, dy / r, dz / r};
        const double reduced = (r - radii[pair]) / radii[pair];
        const double gaussian = std::exp(-k * k * reduced * reduced);
        const double value = 0.5 * (1.0 - std::erf(k * reduced));
        pair_first[pair] = -k * inverse_sqrt_pi * gaussian / radii[pair];
        pair_second[pair] = 2.0 * k * k * k * (r - radii[pair]) *
            inverse_sqrt_pi * gaussian /
            (radii[pair] * radii[pair] * radii[pair]);
        if (coordination_pairs != nullptr) {
            double* row = coordination_pairs + 6 * pair;
            row[0] = r;
            for (int axis = 0; axis < 3; ++axis) row[1 + axis] = pair_unit[pair][axis];
            row[4] = pair_first[pair];
            row[5] = pair_second[pair];
        }
        raw[left] += value;
        raw[right] += value;
        if (direction != nullptr) {
            double dot = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                dot += pair_unit[pair][axis] *
                    (direction[3 * left + axis] - direction[3 * right + axis]);
            }
            pair_rdot[pair] = dot;
            const double contribution = pair_first[pair] * dot;
            raw_dot[left] += contribution;
            raw_dot[right] += contribution;
            for (int axis = 0; axis < 3; ++axis) {
                pair_udot[pair][axis] =
                    (direction[3 * left + axis] - direction[3 * right + axis] -
                     pair_unit[pair][axis] * dot) / r;
            }
        }
    }
    const double softplus_max = std::log1p(std::exp(input.cn_max));
    for (npy_intp atom = 0; atom < input.atoms; ++atom) {
        cn[atom] = softplus_max - std::log1p(std::exp(input.cn_max - raw[atom]));
        hp[atom] = 1.0 / (1.0 + std::exp(raw[atom] - input.cn_max));
        hpp[atom] = -hp[atom] * (1.0 - hp[atom]);
        if (coordination_atoms != nullptr) {
            double* row = coordination_atoms + 4 * atom;
            row[0] = cn[atom];
            row[1] = raw[atom];
            row[2] = hp[atom];
            row[3] = hpp[atom];
        }
        if (direction != nullptr) cn_dot[atom] = hp[atom] * raw_dot[atom];
    }

    std::vector<double> bond_first(input.bond_count), bond_second(input.bond_count),
        bond_sdot(direction ? input.bond_count : 0, 0.0), bond_distance(input.bond_count);
    std::vector<std::array<double, 3>> bond_unit(input.bond_count),
        bond_udot(direction ? input.bond_count : 0);
    for (npy_intp bond = 0; bond < input.bond_count; ++bond) {
        const npy_intp left = bonds[2 * bond];
        const npy_intp right = bonds[2 * bond + 1];
        double dx, dy, dz, r2;
        displacement(xyz, left, right, dx, dy, dz, r2);
        const double r = std::sqrt(r2);
        if (r <= kCoincidentTolerance) return -1;
        bond_distance[bond] = r;
        bond_unit[bond] = {dx / r, dy / r, dz / r};
        const double reference = bases[bond] + coefficients[2 * bond] * cn[left] +
            coefficients[2 * bond + 1] * cn[right];
        const double delta = r - reference;
        const double exponent = exponents[bond];
        const double value = amplitudes[bond] * std::exp(-exponent * delta * delta);
        const double first = -2.0 * exponent * delta * value;
        const double second =
            (4.0 * exponent * exponent * delta * delta - 2.0 * exponent) * value;
        energy += value;
        bond_first[bond] = first;
        bond_second[bond] = second;
        for (int axis = 0; axis < 3; ++axis) {
            if (gradient != nullptr) {
                const double component = first * bond_unit[bond][axis];
                gradient[3 * left + axis] += component;
                gradient[3 * right + axis] -= component;
            }
        }
        adjoint[left] -= first * coefficients[2 * bond];
        adjoint[right] -= first * coefficients[2 * bond + 1];
        if (direction != nullptr) {
            double rdot = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                rdot += bond_unit[bond][axis] *
                    (direction[3 * left + axis] - direction[3 * right + axis]);
            }
            const double sdot = rdot - coefficients[2 * bond] * cn_dot[left] -
                coefficients[2 * bond + 1] * cn_dot[right];
            bond_sdot[bond] = sdot;
            adjoint_dot[left] -= second * sdot * coefficients[2 * bond];
            adjoint_dot[right] -= second * sdot * coefficients[2 * bond + 1];
            for (int axis = 0; axis < 3; ++axis) {
                bond_udot[bond][axis] =
                    (direction[3 * left + axis] - direction[3 * right + axis] -
                     bond_unit[bond][axis] * rdot) / r;
                const double component = second * sdot * bond_unit[bond][axis] +
                    first * bond_udot[bond][axis];
                product[3 * left + axis] += component;
                product[3 * right + axis] -= component;
            }
        }
    }

    for (npy_intp pair = 0; pair < input.pair_count; ++pair) {
        const npy_intp left = cpairs[2 * pair];
        const npy_intp right = cpairs[2 * pair + 1];
        const double saturated = adjoint[left] * hp[left] + adjoint[right] * hp[right];
        const double weight = pair_first[pair] * saturated;
        for (int axis = 0; axis < 3; ++axis) {
            if (gradient != nullptr) {
                const double component = weight * pair_unit[pair][axis];
                gradient[3 * left + axis] += component;
                gradient[3 * right + axis] -= component;
            }
        }
        if (direction != nullptr) {
            const double saturated_dot =
                adjoint_dot[left] * hp[left] +
                adjoint[left] * hpp[left] * raw_dot[left] +
                adjoint_dot[right] * hp[right] +
                adjoint[right] * hpp[right] * raw_dot[right];
            const double weight_dot = pair_second[pair] * pair_rdot[pair] * saturated +
                pair_first[pair] * saturated_dot;
            for (int axis = 0; axis < 3; ++axis) {
                const double component = weight_dot * pair_unit[pair][axis] +
                    weight * pair_udot[pair][axis];
                product[3 * left + axis] += component;
                product[3 * right + axis] -= component;
            }
        }
    }
    if (gradient != nullptr) {
        for (npy_intp value = 0; value < 3 * input.atoms; ++value) {
            gradient[value] *= bohr_to_angstrom;
        }
    }
    if (product != nullptr) {
        for (npy_intp value = 0; value < 3 * input.atoms; ++value) {
            product[value] *= bohr_to_angstrom * bohr_to_angstrom;
        }
    }
    return input.bond_count;
}

bool parse_dynamic_gaussian_call(PyObject* args, DynamicGaussianInputs& input) {
    PyObject *coordinates, *coordination_pairs, *coordination_radii, *bonds,
        *amplitudes, *exponents, *bases, *coefficients;
    double cn_max, steepness;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOOdd", &coordinates, &coordination_pairs,
            &coordination_radii, &bonds, &amplitudes, &exponents, &bases,
            &coefficients, &cn_max, &steepness)) return false;
    return parse_dynamic_gaussian_inputs(
        coordinates, coordination_pairs, coordination_radii, bonds, amplitudes,
        exponents, bases, coefficients, cn_max, steepness, input);
}

PyObject* dynamic_gaussian_bond_energy_gradient(PyObject*, PyObject* args) {
    DynamicGaussianInputs input{nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, 0, 0, 0, 0.0, 0.0};
    if (!parse_dynamic_gaussian_call(args, input)) return nullptr;
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        release_dynamic_gaussian_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_dynamic_gaussian_bonds(
        input, energy, static_cast<double*>(PyArray_DATA(gradient)), nullptr, nullptr);
    Py_END_ALLOW_THREADS
    release_dynamic_gaussian_inputs(input);
    if (count < 0) {
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in dynamic GFN-FF bond model");
        return nullptr;
    }
    return Py_BuildValue("dNL", energy, gradient, static_cast<long long>(count));
}

PyObject* dynamic_gaussian_bond_shared_coordination(PyObject*, PyObject* args) {
    DynamicGaussianInputs input{};
    if (!parse_dynamic_gaussian_call(args, input)) return nullptr;
    npy_intp gdims[2] = {input.atoms, 3};
    npy_intp adims[2] = {input.atoms, 4};
    npy_intp pdims[2] = {input.pair_count, 6};
    auto* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, gdims, NPY_DOUBLE, 0));
    auto* atoms = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2, adims, NPY_DOUBLE));
    auto* pairs = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2, pdims, NPY_DOUBLE));
    if (gradient == nullptr || atoms == nullptr || pairs == nullptr) {
        Py_XDECREF(gradient);
        Py_XDECREF(atoms);
        Py_XDECREF(pairs);
        release_dynamic_gaussian_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_dynamic_gaussian_bonds(
        input, energy, static_cast<double*>(PyArray_DATA(gradient)), nullptr, nullptr,
        static_cast<double*>(PyArray_DATA(atoms)), static_cast<double*>(PyArray_DATA(pairs)));
    Py_END_ALLOW_THREADS
    release_dynamic_gaussian_inputs(input);
    if (count < 0) {
        Py_DECREF(gradient);
        Py_DECREF(atoms);
        Py_DECREF(pairs);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in dynamic GFN-FF bond model");
        return nullptr;
    }
    return Py_BuildValue("dNNN", energy, gradient, atoms, pairs);
}

PyObject* dynamic_gaussian_bond_hessian_vector(PyObject*, PyObject* args) {
    PyObject* base_args;
    PyObject* direction_object;
    if (!PyArg_ParseTuple(args, "OO", &base_args, &direction_object)) return nullptr;
    DynamicGaussianInputs input{nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, 0, 0, 0, 0.0, 0.0};
    if (!PyTuple_Check(base_args) || !parse_dynamic_gaussian_call(base_args, input)) {
        return nullptr;
    }
    PyArrayObject* direction = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(direction_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (direction == nullptr || PyArray_NDIM(direction) != 2 ||
        PyArray_DIM(direction, 0) != input.atoms || PyArray_DIM(direction, 1) != 3) {
        Py_XDECREF(direction);
        release_dynamic_gaussian_inputs(input);
        PyErr_SetString(PyExc_ValueError, "dynamic GFN-FF direction must have shape (N,3)");
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (product == nullptr) {
        Py_DECREF(direction);
        release_dynamic_gaussian_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_dynamic_gaussian_bonds(
        input, energy, nullptr, static_cast<const double*>(PyArray_DATA(direction)),
        static_cast<double*>(PyArray_DATA(product)));
    Py_END_ALLOW_THREADS
    Py_DECREF(direction);
    release_dynamic_gaussian_inputs(input);
    if (count < 0) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in dynamic GFN-FF bond model");
        return nullptr;
    }
    return Py_BuildValue("NL", product, static_cast<long long>(count));
}

struct GFNFFRepulsionInputs {
    PyArrayObject* coordinates;
    PyArrayObject* pairs;
    PyArrayObject* amplitudes;
    PyArrayObject* exponents;
    npy_intp atoms;
    npy_intp count;
};

void release_gfnff_repulsion_inputs(GFNFFRepulsionInputs& input) {
    Py_XDECREF(input.coordinates);
    Py_XDECREF(input.pairs);
    Py_XDECREF(input.amplitudes);
    Py_XDECREF(input.exponents);
}

bool parse_gfnff_repulsion_inputs(
    PyObject* coordinates, PyObject* pairs, PyObject* amplitudes,
    PyObject* exponents, GFNFFRepulsionInputs& out) {
    out.coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.pairs = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(pairs, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.amplitudes = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(amplitudes, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.exponents = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(exponents, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (out.coordinates == nullptr || out.pairs == nullptr ||
        out.amplitudes == nullptr || out.exponents == nullptr) {
        release_gfnff_repulsion_inputs(out);
        return false;
    }
    if (PyArray_NDIM(out.coordinates) != 2 ||
        PyArray_DIM(out.coordinates, 1) != 3 ||
        PyArray_NDIM(out.pairs) != 2 || PyArray_DIM(out.pairs, 1) != 2 ||
        PyArray_NDIM(out.amplitudes) != 1 || PyArray_NDIM(out.exponents) != 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "native GFN-FF repulsion requires (N,3), (P,2), and two (P,) arrays");
        release_gfnff_repulsion_inputs(out);
        return false;
    }
    out.atoms = PyArray_DIM(out.coordinates, 0);
    out.count = PyArray_DIM(out.pairs, 0);
    if (PyArray_DIM(out.amplitudes, 0) != out.count ||
        PyArray_DIM(out.exponents, 0) != out.count) {
        PyErr_SetString(
            PyExc_ValueError, "native GFN-FF repulsion dimensions differ");
        release_gfnff_repulsion_inputs(out);
        return false;
    }
    const auto* indices = static_cast<const npy_intp*>(PyArray_DATA(out.pairs));
    const auto* amplitude = static_cast<const double*>(PyArray_DATA(out.amplitudes));
    const auto* exponent = static_cast<const double*>(PyArray_DATA(out.exponents));
    for (npy_intp pair = 0; pair < out.count; ++pair) {
        const npy_intp left = indices[2 * pair];
        const npy_intp right = indices[2 * pair + 1];
        if (left < 0 || right < 0 || left >= out.atoms || right >= out.atoms ||
            left == right || !std::isfinite(amplitude[pair]) ||
            amplitude[pair] <= 0.0 || !std::isfinite(exponent[pair]) ||
            exponent[pair] <= 0.0) {
            PyErr_SetString(
                PyExc_ValueError, "native GFN-FF repulsion record is invalid");
            release_gfnff_repulsion_inputs(out);
            return false;
        }
    }
    return true;
}

npy_intp accumulate_gfnff_repulsion(
    const GFNFFRepulsionInputs& input, double& energy, double* gradient,
    const double* direction, double* product) {
    const auto* xyz = static_cast<const double*>(PyArray_DATA(input.coordinates));
    const auto* pairs = static_cast<const npy_intp*>(PyArray_DATA(input.pairs));
    const auto* amplitudes =
        static_cast<const double*>(PyArray_DATA(input.amplitudes));
    const auto* exponents =
        static_cast<const double*>(PyArray_DATA(input.exponents));
    for (npy_intp pair = 0; pair < input.count; ++pair) {
        const npy_intp left = pairs[2 * pair];
        const npy_intp right = pairs[2 * pair + 1];
        double dx, dy, dz, r2;
        displacement(xyz, left, right, dx, dy, dz, r2);
        const double distance = std::sqrt(r2);
        if (distance <= kCoincidentTolerance) return -1;
        const double root = std::sqrt(distance);
        const double r32 = distance * root;
        const double exponent = exponents[pair];
        const double value =
            amplitudes[pair] * std::exp(-exponent * r32) / distance;
        const double log_first = -1.0 / distance - 1.5 * exponent * root;
        const double log_second =
            1.0 / r2 - 0.75 * exponent / root;
        const double first = value * log_first;
        const double second = value * (log_first * log_first + log_second);
        energy += value;
        const double inverse_r = 1.0 / distance;
        const double unit[3] = {
            dx * inverse_r, dy * inverse_r, dz * inverse_r};
        if (gradient != nullptr) {
            for (int axis = 0; axis < 3; ++axis) {
                const double component = first * unit[axis];
                gradient[3 * left + axis] += component;
                gradient[3 * right + axis] -= component;
            }
        }
        if (product != nullptr) {
            double vector[3];
            double projection = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                vector[axis] = direction[3 * left + axis] -
                    direction[3 * right + axis];
                projection += unit[axis] * vector[axis];
            }
            const double transverse = first * inverse_r;
            const double longitudinal = second - transverse;
            for (int axis = 0; axis < 3; ++axis) {
                const double component = transverse * vector[axis] +
                    longitudinal * projection * unit[axis];
                product[3 * left + axis] += component;
                product[3 * right + axis] -= component;
            }
        }
    }
    return input.count;
}

bool parse_gfnff_repulsion_call(
    PyObject* args, GFNFFRepulsionInputs& input) {
    PyObject *coordinates, *pairs, *amplitudes, *exponents;
    if (!PyArg_ParseTuple(
            args, "OOOO", &coordinates, &pairs, &amplitudes, &exponents)) {
        return false;
    }
    return parse_gfnff_repulsion_inputs(
        coordinates, pairs, amplitudes, exponents, input);
}

PyObject* gfnff_repulsion_energy_gradient(PyObject*, PyObject* args) {
    GFNFFRepulsionInputs input{nullptr, nullptr, nullptr, nullptr, 0, 0};
    if (!parse_gfnff_repulsion_call(args, input)) return nullptr;
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        release_gfnff_repulsion_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_gfnff_repulsion(
        input, energy, static_cast<double*>(PyArray_DATA(gradient)), nullptr, nullptr);
    Py_END_ALLOW_THREADS
    release_gfnff_repulsion_inputs(input);
    if (count < 0) {
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in GFN-FF repulsion");
        return nullptr;
    }
    return Py_BuildValue("dNL", energy, gradient, static_cast<long long>(count));
}

PyObject* gfnff_repulsion_hessian_vector(PyObject*, PyObject* args) {
    PyObject* base_args;
    PyObject* direction_object;
    if (!PyArg_ParseTuple(args, "OO", &base_args, &direction_object)) return nullptr;
    GFNFFRepulsionInputs input{nullptr, nullptr, nullptr, nullptr, 0, 0};
    if (!PyTuple_Check(base_args) || !parse_gfnff_repulsion_call(base_args, input)) {
        return nullptr;
    }
    PyArrayObject* direction = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(direction_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (direction == nullptr || PyArray_NDIM(direction) != 2 ||
        PyArray_DIM(direction, 0) != input.atoms ||
        PyArray_DIM(direction, 1) != 3) {
        Py_XDECREF(direction);
        release_gfnff_repulsion_inputs(input);
        PyErr_SetString(
            PyExc_ValueError, "GFN-FF repulsion direction must have shape (N,3)");
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (product == nullptr) {
        Py_DECREF(direction);
        release_gfnff_repulsion_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_gfnff_repulsion(
        input, energy, nullptr,
        static_cast<const double*>(PyArray_DATA(direction)),
        static_cast<double*>(PyArray_DATA(product)));
    Py_END_ALLOW_THREADS
    Py_DECREF(direction);
    release_gfnff_repulsion_inputs(input);
    if (count < 0) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in GFN-FF repulsion");
        return nullptr;
    }
    return Py_BuildValue("NL", product, static_cast<long long>(count));
}

struct GFNFFD3ReferenceInputs {
    PyArrayObject* coordination;
    PyArrayObject* pairs;
    PyArrayObject* reference_counts;
    PyArrayObject* reference_cn;
    PyArrayObject* pair_c6;
    npy_intp atoms;
    npy_intp count;
    npy_intp maximum_references;
    double weighting_factor;
};

void release_gfnff_d3_reference_inputs(GFNFFD3ReferenceInputs& input) {
    Py_XDECREF(input.coordination);
    Py_XDECREF(input.pairs);
    Py_XDECREF(input.reference_counts);
    Py_XDECREF(input.reference_cn);
    Py_XDECREF(input.pair_c6);
}

bool parse_gfnff_d3_reference_inputs(
    PyObject* args, GFNFFD3ReferenceInputs& out) {
    PyObject *coordination, *pairs, *reference_counts, *reference_cn,
        *pair_c6;
    if (!PyArg_ParseTuple(
            args, "OOOOOd", &coordination, &pairs, &reference_counts,
            &reference_cn, &pair_c6, &out.weighting_factor)) {
        return false;
    }
    out.coordination = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordination, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.pairs = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(pairs, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.reference_counts = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(reference_counts, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.reference_cn = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(reference_cn, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.pair_c6 = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(pair_c6, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (out.coordination == nullptr || out.pairs == nullptr ||
        out.reference_counts == nullptr || out.reference_cn == nullptr ||
        out.pair_c6 == nullptr) {
        release_gfnff_d3_reference_inputs(out);
        return false;
    }
    if (PyArray_NDIM(out.coordination) != 1 ||
        PyArray_NDIM(out.pairs) != 2 || PyArray_DIM(out.pairs, 1) != 2 ||
        PyArray_NDIM(out.reference_counts) != 1 ||
        PyArray_NDIM(out.reference_cn) != 2 ||
        PyArray_NDIM(out.pair_c6) != 3 ||
        !std::isfinite(out.weighting_factor) ||
        out.weighting_factor <= 0.0) {
        PyErr_SetString(
            PyExc_ValueError, "native GFN-FF D3 reference arrays are invalid");
        release_gfnff_d3_reference_inputs(out);
        return false;
    }
    out.atoms = PyArray_DIM(out.coordination, 0);
    out.count = PyArray_DIM(out.pairs, 0);
    out.maximum_references = PyArray_DIM(out.reference_cn, 1);
    if (PyArray_DIM(out.reference_counts, 0) != out.atoms ||
        PyArray_DIM(out.reference_cn, 0) != out.atoms ||
        PyArray_DIM(out.pair_c6, 0) != out.count ||
        PyArray_DIM(out.pair_c6, 1) != out.maximum_references ||
        PyArray_DIM(out.pair_c6, 2) != out.maximum_references ||
        out.maximum_references < 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "native GFN-FF D3 reference dimensions are inconsistent");
        release_gfnff_d3_reference_inputs(out);
        return false;
    }
    const auto* cn =
        static_cast<const double*>(PyArray_DATA(out.coordination));
    const auto* indices =
        static_cast<const npy_intp*>(PyArray_DATA(out.pairs));
    const auto* counts =
        static_cast<const npy_intp*>(PyArray_DATA(out.reference_counts));
    const auto* references =
        static_cast<const double*>(PyArray_DATA(out.reference_cn));
    for (npy_intp atom = 0; atom < out.atoms; ++atom) {
        if (!std::isfinite(cn[atom]) || counts[atom] < 1 ||
            counts[atom] > out.maximum_references) {
            PyErr_SetString(
                PyExc_ValueError, "native GFN-FF D3 atom reference is invalid");
            release_gfnff_d3_reference_inputs(out);
            return false;
        }
        for (npy_intp ref = 0; ref < counts[atom]; ++ref) {
            if (!std::isfinite(references[atom * out.maximum_references + ref])) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "native GFN-FF D3 reference CN is not finite");
                release_gfnff_d3_reference_inputs(out);
                return false;
            }
        }
    }
    for (npy_intp pair = 0; pair < out.count; ++pair) {
        const npy_intp left = indices[2 * pair];
        const npy_intp right = indices[2 * pair + 1];
        if (left < 0 || right < 0 || left >= out.atoms ||
            right >= out.atoms || left == right) {
            PyErr_SetString(
                PyExc_ValueError, "native GFN-FF D3 pair is invalid");
            release_gfnff_d3_reference_inputs(out);
            return false;
        }
    }
    return true;
}

PyObject* gfnff_d3_reference_derivatives(PyObject*, PyObject* args) {
    GFNFFD3ReferenceInputs input{
        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0.0};
    if (!parse_gfnff_d3_reference_inputs(args, input)) return nullptr;
    npy_intp dimension[1] = {input.count};
    std::array<PyArrayObject*, 6> output{};
    for (auto& array : output) {
        array = reinterpret_cast<PyArrayObject*>(
            PyArray_ZEROS(1, dimension, NPY_DOUBLE, 0));
        if (array == nullptr) {
            for (auto* allocated : output) Py_XDECREF(allocated);
            release_gfnff_d3_reference_inputs(input);
            return nullptr;
        }
    }
    const auto* cn =
        static_cast<const double*>(PyArray_DATA(input.coordination));
    const auto* pairs =
        static_cast<const npy_intp*>(PyArray_DATA(input.pairs));
    const auto* counts =
        static_cast<const npy_intp*>(PyArray_DATA(input.reference_counts));
    const auto* references =
        static_cast<const double*>(PyArray_DATA(input.reference_cn));
    const auto* c6 = static_cast<const double*>(PyArray_DATA(input.pair_c6));
    std::array<double*, 6> result{};
    for (int index = 0; index < 6; ++index) {
        result[index] = static_cast<double*>(PyArray_DATA(output[index]));
    }
    const npy_intp width = input.maximum_references;
    std::vector<double> weight(3 * input.atoms * width, 0.0);
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp atom = 0; atom < input.atoms; ++atom) {
        const npy_intp count = counts[atom];
        double maximum_log = -HUGE_VAL;
        for (npy_intp ref = 0; ref < count; ++ref) {
            const double delta =
                cn[atom] - references[atom * width + ref];
            maximum_log = std::max(
                maximum_log, -input.weighting_factor * delta * delta);
        }
        double normalization = 0.0;
        for (npy_intp ref = 0; ref < count; ++ref) {
            const double delta =
                cn[atom] - references[atom * width + ref];
            const double value = std::exp(
                -input.weighting_factor * delta * delta - maximum_log);
            weight[atom * width + ref] = value;
            normalization += value;
        }
        double mean = 0.0;
        for (npy_intp ref = 0; ref < count; ++ref) {
            const npy_intp offset = atom * width + ref;
            weight[offset] /= normalization;
            const double log_first = 2.0 * input.weighting_factor *
                (references[offset] - cn[atom]);
            mean += weight[offset] * log_first;
        }
        double variance = 0.0;
        for (npy_intp ref = 0; ref < count; ++ref) {
            const npy_intp offset = atom * width + ref;
            const double centered = 2.0 * input.weighting_factor *
                (references[offset] - cn[atom]) - mean;
            weight[input.atoms * width + offset] =
                weight[offset] * centered;
            variance += weight[offset] * centered * centered;
        }
        for (npy_intp ref = 0; ref < count; ++ref) {
            const npy_intp offset = atom * width + ref;
            const double centered = 2.0 * input.weighting_factor *
                (references[offset] - cn[atom]) - mean;
            weight[2 * input.atoms * width + offset] =
                weight[offset] * (centered * centered - variance);
        }
    }
    for (npy_intp pair = 0; pair < input.count; ++pair) {
        const npy_intp left = pairs[2 * pair];
        const npy_intp right = pairs[2 * pair + 1];
        const npy_intp left_count = counts[left];
        const npy_intp right_count = counts[right];
        for (npy_intp i = 0; i < left_count; ++i) {
            const npy_intp left_offset = left * width + i;
            const double wi = weight[left_offset];
            const double dwi = weight[input.atoms * width + left_offset];
            const double d2wi = weight[2 * input.atoms * width + left_offset];
            for (npy_intp j = 0; j < right_count; ++j) {
                const npy_intp right_offset = right * width + j;
                const double wj = weight[right_offset];
                const double dwj = weight[input.atoms * width + right_offset];
                const double d2wj =
                    weight[2 * input.atoms * width + right_offset];
                const double coefficient =
                    c6[(pair * width + i) * width + j];
                result[0][pair] += wi * coefficient * wj;
                result[1][pair] += dwi * coefficient * wj;
                result[2][pair] += wi * coefficient * dwj;
                result[3][pair] += d2wi * coefficient * wj;
                result[4][pair] += dwi * coefficient * dwj;
                result[5][pair] += wi * coefficient * d2wj;
            }
        }
    }
    Py_END_ALLOW_THREADS
    release_gfnff_d3_reference_inputs(input);
    return Py_BuildValue(
        "NNNNNN", output[0], output[1], output[2], output[3], output[4],
        output[5]);
}

constexpr int kLocalDimension = 24;
constexpr double kBohrToAngstrom = 0.52917721092;

struct DirectionalJet {
    double value = 0.0;
    std::array<double, kLocalDimension> gradient{};
    double dot = 0.0;
    std::array<double, kLocalDimension> gradient_dot{};
};

DirectionalJet jet_constant(double value) {
    DirectionalJet result;
    result.value = value;
    return result;
}

DirectionalJet jet_variable(double value, int index, double dot) {
    DirectionalJet result;
    result.value = value;
    result.gradient[index] = 1.0;
    result.dot = dot;
    return result;
}

DirectionalJet operator+(const DirectionalJet& left, const DirectionalJet& right) {
    DirectionalJet result;
    result.value = left.value + right.value;
    result.dot = left.dot + right.dot;
    for (int index = 0; index < kLocalDimension; ++index) {
        result.gradient[index] = left.gradient[index] + right.gradient[index];
        result.gradient_dot[index] =
            left.gradient_dot[index] + right.gradient_dot[index];
    }
    return result;
}

DirectionalJet operator-(const DirectionalJet& left, const DirectionalJet& right) {
    DirectionalJet result;
    result.value = left.value - right.value;
    result.dot = left.dot - right.dot;
    for (int index = 0; index < kLocalDimension; ++index) {
        result.gradient[index] = left.gradient[index] - right.gradient[index];
        result.gradient_dot[index] =
            left.gradient_dot[index] - right.gradient_dot[index];
    }
    return result;
}

DirectionalJet operator*(const DirectionalJet& left, const DirectionalJet& right) {
    DirectionalJet result;
    result.value = left.value * right.value;
    result.dot = left.dot * right.value + left.value * right.dot;
    for (int index = 0; index < kLocalDimension; ++index) {
        result.gradient[index] =
            left.value * right.gradient[index] +
            right.value * left.gradient[index];
        result.gradient_dot[index] =
            left.dot * right.gradient[index] +
            left.value * right.gradient_dot[index] +
            right.dot * left.gradient[index] +
            right.value * left.gradient_dot[index];
    }
    return result;
}

DirectionalJet operator*(double left, const DirectionalJet& right) {
    return jet_constant(left) * right;
}

DirectionalJet operator*(const DirectionalJet& left, double right) {
    return left * jet_constant(right);
}

DirectionalJet operator+(double left, const DirectionalJet& right) {
    return jet_constant(left) + right;
}

DirectionalJet operator+(const DirectionalJet& left, double right) {
    return left + jet_constant(right);
}

DirectionalJet operator-(const DirectionalJet& left, double right) {
    return left - jet_constant(right);
}

DirectionalJet operator-(double left, const DirectionalJet& right) {
    return jet_constant(left) - right;
}

DirectionalJet operator-(const DirectionalJet& value) {
    return -1.0 * value;
}

DirectionalJet jet_unary(
    const DirectionalJet& value, double result_value, double first,
    double second) {
    DirectionalJet result;
    result.value = result_value;
    result.dot = first * value.dot;
    for (int index = 0; index < kLocalDimension; ++index) {
        result.gradient[index] = first * value.gradient[index];
        result.gradient_dot[index] =
            second * value.dot * value.gradient[index] +
            first * value.gradient_dot[index];
    }
    return result;
}

DirectionalJet jet_inverse(const DirectionalJet& value) {
    const double inverse = 1.0 / value.value;
    return jet_unary(
        value, inverse, -inverse * inverse,
        2.0 * inverse * inverse * inverse);
}

DirectionalJet operator/(const DirectionalJet& left, const DirectionalJet& right) {
    return left * jet_inverse(right);
}

DirectionalJet operator/(const DirectionalJet& left, double right) {
    return left * (1.0 / right);
}

DirectionalJet operator/(double left, const DirectionalJet& right) {
    return jet_constant(left) / right;
}

DirectionalJet jet_sqrt(const DirectionalJet& value) {
    const double root = std::sqrt(value.value);
    return jet_unary(
        value, root, 0.5 / root, -0.25 / (value.value * root));
}

DirectionalJet jet_exp(const DirectionalJet& value) {
    const double exponential = std::exp(value.value);
    return jet_unary(value, exponential, exponential, exponential);
}

DirectionalJet jet_log(const DirectionalJet& value) {
    return jet_unary(
        value, std::log(value.value), 1.0 / value.value,
        -1.0 / (value.value * value.value));
}

DirectionalJet jet_pow(const DirectionalJet& value, double power) {
    return jet_exp(power * jet_log(value));
}

DirectionalJet jet_tanh(const DirectionalJet& value) {
    const double hyperbolic = std::tanh(value.value);
    const double first = 1.0 - hyperbolic * hyperbolic;
    return jet_unary(value, hyperbolic, first, -2.0 * hyperbolic * first);
}

DirectionalJet jet_sin(const DirectionalJet& value) {
    return jet_unary(
        value, std::sin(value.value), std::cos(value.value),
        -std::sin(value.value));
}

DirectionalJet jet_cos(const DirectionalJet& value) {
    return jet_unary(
        value, std::cos(value.value), -std::sin(value.value),
        -std::cos(value.value));
}

DirectionalJet jet_acos(const DirectionalJet& value) {
    const double bounded = std::max(-1.0, std::min(1.0, value.value));
    const double denominator = std::sqrt(1.0 - bounded * bounded);
    return jet_unary(
        value, std::acos(bounded), -1.0 / denominator,
        -bounded / (denominator * denominator * denominator));
}

DirectionalJet jet_atan2(
    const DirectionalJet& y, const DirectionalJet& x) {
    DirectionalJet result;
    const double denominator = x.value * x.value + y.value * y.value;
    result.value = std::atan2(y.value, x.value);
    result.dot =
        (x.value * y.dot - y.value * x.dot) / denominator;
    const double denominator_dot =
        2.0 * (x.value * x.dot + y.value * y.dot);
    for (int index = 0; index < kLocalDimension; ++index) {
        const double numerator =
            x.value * y.gradient[index] - y.value * x.gradient[index];
        const double numerator_dot =
            x.dot * y.gradient[index] +
            x.value * y.gradient_dot[index] -
            y.dot * x.gradient[index] -
            y.value * x.gradient_dot[index];
        result.gradient[index] = numerator / denominator;
        result.gradient_dot[index] =
            numerator_dot / denominator -
            numerator * denominator_dot / (denominator * denominator);
    }
    return result;
}

using JetVector = std::array<DirectionalJet, 3>;

JetVector jet_subtract(const JetVector& left, const JetVector& right) {
    return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}

JetVector jet_add(const JetVector& left, const JetVector& right) {
    return {left[0] + right[0], left[1] + right[1], left[2] + right[2]};
}

JetVector jet_scale(const JetVector& value, double scale) {
    return {scale * value[0], scale * value[1], scale * value[2]};
}

DirectionalJet jet_dot(const JetVector& left, const JetVector& right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

JetVector jet_cross(const JetVector& left, const JetVector& right) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

DirectionalJet jet_norm(const JetVector& value) {
    return jet_sqrt(jet_dot(value, value));
}

JetVector jet_divide(const JetVector& value, const DirectionalJet& scale) {
    return {value[0] / scale, value[1] / scale, value[2] / scale};
}

DirectionalJet bond_order_factor(
    const DirectionalJet& distance, double reference, double radius_sum) {
    constexpr double strong_decay = 0.20;
    constexpr double weak_decay = 0.40;
    constexpr double sharpness = 20.0;
    const auto raw = [&](const DirectionalJet& r) {
        const DirectionalJet x = (r - radius_sum) / radius_sum;
        const DirectionalJet weight =
            0.5 * (1.0 - jet_tanh(sharpness * x));
        const DirectionalJet strong =
            jet_exp((radius_sum - r) / strong_decay);
        const DirectionalJet weak =
            jet_exp((radius_sum - r) / weak_decay);
        return weak + weight * (strong - weak);
    };
    const double reference_x = (reference - radius_sum) / radius_sum;
    const double reference_weight =
        0.5 * (1.0 - std::tanh(sharpness * reference_x));
    const double reference_strong =
        std::exp((radius_sum - reference) / strong_decay);
    const double reference_weak =
        std::exp((radius_sum - reference) / weak_decay);
    const double normalization =
        reference_weak +
        reference_weight * (reference_strong - reference_weak);
    return raw(distance) / normalization;
}

DirectionalJet gfnff_distance_damping(
    const DirectionalJet& distance, double cutoff_squared) {
    const DirectionalJet ratio =
        (distance * distance) / cutoff_squared;
    return jet_constant(1.0) / (1.0 + ratio * ratio);
}

DirectionalJet proper_dihedral(
    const std::array<JetVector, 4>& position) {
    const JetVector b1 = jet_subtract(position[0], position[1]);
    const JetVector b2 = jet_subtract(position[2], position[1]);
    const JetVector b3 = jet_subtract(position[3], position[2]);
    const DirectionalJet r2 = jet_norm(b2);
    const JetVector n1 = jet_cross(b1, b2);
    const JetVector n2 = jet_cross(b2, b3);
    return jet_atan2(
        jet_dot(jet_cross(n1, n2), jet_divide(b2, r2)),
        jet_dot(n1, n2));
}

DirectionalJet improper_angle(
    const std::array<JetVector, 4>& position) {
    const JetVector re = jet_subtract(position[0], position[1]);
    const JetVector rd = jet_subtract(position[2], position[1]);
    const JetVector normal = jet_cross(re, rd);
    const JetVector rv = jet_subtract(position[3], position[0]);
    const DirectionalJet sine =
        jet_dot(normal, rv) / (jet_norm(normal) * jet_norm(rv));
    return 0.5 * std::acos(-1.0) - jet_acos(sine);
}

struct GFNFFValenceInputs {
    PyArrayObject* coordinates = nullptr;
    PyArrayObject* angle_atoms = nullptr;
    PyArrayObject* angle_parameters = nullptr;
    PyArrayObject* torsion_atoms = nullptr;
    PyArrayObject* torsion_parameters = nullptr;
    PyArrayObject* special_atoms = nullptr;
    PyArrayObject* special_amplitudes = nullptr;
    PyArrayObject* direction = nullptr;
    npy_intp atoms = 0;
    npy_intp angle_count = 0;
    npy_intp torsion_count = 0;
    npy_intp special_count = 0;
};

void release_gfnff_valence_inputs(GFNFFValenceInputs& input) {
    Py_XDECREF(input.coordinates);
    Py_XDECREF(input.angle_atoms);
    Py_XDECREF(input.angle_parameters);
    Py_XDECREF(input.torsion_atoms);
    Py_XDECREF(input.torsion_parameters);
    Py_XDECREF(input.special_atoms);
    Py_XDECREF(input.special_amplitudes);
    Py_XDECREF(input.direction);
}

bool parse_gfnff_valence_inputs(
    PyObject* coordinates, PyObject* angle_atoms,
    PyObject* angle_parameters, PyObject* torsion_atoms,
    PyObject* torsion_parameters, PyObject* special_atoms,
    PyObject* special_amplitudes, PyObject* direction,
    GFNFFValenceInputs& out) {
    out.coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.angle_atoms = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(angle_atoms, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.angle_parameters = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(angle_parameters, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.torsion_atoms = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(torsion_atoms, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.torsion_parameters = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(torsion_parameters, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.special_atoms = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(special_atoms, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.special_amplitudes = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(special_amplitudes, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (direction != Py_None) {
        out.direction = reinterpret_cast<PyArrayObject*>(
            PyArray_FROM_OTF(direction, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    }
    if (out.coordinates == nullptr || out.angle_atoms == nullptr ||
        out.angle_parameters == nullptr || out.torsion_atoms == nullptr ||
        out.torsion_parameters == nullptr || out.special_atoms == nullptr ||
        out.special_amplitudes == nullptr ||
        (direction != Py_None && out.direction == nullptr)) {
        release_gfnff_valence_inputs(out);
        return false;
    }
    if (PyArray_NDIM(out.coordinates) != 2 ||
        PyArray_DIM(out.coordinates, 1) != 3 ||
        PyArray_NDIM(out.angle_atoms) != 2 ||
        PyArray_DIM(out.angle_atoms, 1) != 3 ||
        PyArray_NDIM(out.angle_parameters) != 2 ||
        PyArray_DIM(out.angle_parameters, 1) != 4 ||
        PyArray_NDIM(out.torsion_atoms) != 2 ||
        PyArray_DIM(out.torsion_atoms, 1) != 4 ||
        PyArray_NDIM(out.torsion_parameters) != 2 ||
        PyArray_DIM(out.torsion_parameters, 1) != 7 ||
        PyArray_NDIM(out.special_atoms) != 2 ||
        PyArray_DIM(out.special_atoms, 1) != 4 ||
        PyArray_NDIM(out.special_amplitudes) != 1) {
        PyErr_SetString(
            PyExc_ValueError, "native GFN-FF valence array shapes are invalid");
        release_gfnff_valence_inputs(out);
        return false;
    }
    out.atoms = PyArray_DIM(out.coordinates, 0);
    out.angle_count = PyArray_DIM(out.angle_atoms, 0);
    out.torsion_count = PyArray_DIM(out.torsion_atoms, 0);
    out.special_count = PyArray_DIM(out.special_atoms, 0);
    if (PyArray_DIM(out.angle_parameters, 0) != out.angle_count ||
        PyArray_DIM(out.torsion_parameters, 0) != out.torsion_count ||
        PyArray_DIM(out.special_amplitudes, 0) != out.special_count ||
        (out.direction != nullptr &&
         (PyArray_NDIM(out.direction) != 2 ||
          PyArray_DIM(out.direction, 0) != out.atoms ||
          PyArray_DIM(out.direction, 1) != 3))) {
        PyErr_SetString(
            PyExc_ValueError, "native GFN-FF valence dimensions differ");
        release_gfnff_valence_inputs(out);
        return false;
    }
    const auto valid_atoms = [&](PyArrayObject* array, int width) {
        const auto* indices =
            static_cast<const npy_intp*>(PyArray_DATA(array));
        const npy_intp count = PyArray_DIM(array, 0);
        for (npy_intp item = 0; item < count; ++item) {
            for (int atom = 0; atom < width; ++atom) {
                const npy_intp value = indices[width * item + atom];
                if (value < 0 || value >= out.atoms) return false;
                for (int previous = 0; previous < atom; ++previous) {
                    if (value == indices[width * item + previous]) return false;
                }
            }
        }
        return true;
    };
    if (!valid_atoms(out.angle_atoms, 3) ||
        !valid_atoms(out.torsion_atoms, 4) ||
        !valid_atoms(out.special_atoms, 4)) {
        PyErr_SetString(
            PyExc_ValueError, "native GFN-FF valence atom is out of range");
        release_gfnff_valence_inputs(out);
        return false;
    }
    const auto* angle_values =
        static_cast<const double*>(PyArray_DATA(out.angle_parameters));
    for (npy_intp item = 0; item < out.angle_count; ++item) {
        const double* p = angle_values + 4 * item;
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || p[1] < 0.0 ||
            !std::isfinite(p[2]) || p[2] <= 0.0 ||
            !std::isfinite(p[3]) || p[3] <= 0.0) {
            PyErr_SetString(PyExc_ValueError, "native GFN-FF angle is invalid");
            release_gfnff_valence_inputs(out);
            return false;
        }
    }
    const auto* torsion_values =
        static_cast<const double*>(PyArray_DATA(out.torsion_parameters));
    for (npy_intp item = 0; item < out.torsion_count; ++item) {
        const double* p = torsion_values + 7 * item;
        const int mode = static_cast<int>(std::round(p[0]));
        if (!std::isfinite(p[0]) || std::abs(p[0] - mode) > 1.0e-12 ||
            mode < -1 || !std::isfinite(p[1]) || p[1] < 0.0 ||
            !std::isfinite(p[2]) || !std::isfinite(p[3]) || p[3] < 0.0 ||
            !std::isfinite(p[4]) || p[4] <= 0.0 ||
            !std::isfinite(p[5]) || p[5] <= 0.0 ||
            !std::isfinite(p[6]) || p[6] <= 0.0) {
            PyErr_SetString(PyExc_ValueError, "native GFN-FF torsion is invalid");
            release_gfnff_valence_inputs(out);
            return false;
        }
    }
    const auto* amplitudes =
        static_cast<const double*>(PyArray_DATA(out.special_amplitudes));
    for (npy_intp item = 0; item < out.special_count; ++item) {
        if (!std::isfinite(amplitudes[item]) || amplitudes[item] < 0.0) {
            PyErr_SetString(
                PyExc_ValueError, "native GFN-FF special torsion is invalid");
            release_gfnff_valence_inputs(out);
            return false;
        }
    }
    return true;
}

bool accumulate_gfnff_valence(
    const GFNFFValenceInputs& input, double& energy, double* gradient,
    double* product) {
    const auto* xyz =
        static_cast<const double*>(PyArray_DATA(input.coordinates));
    const auto* direction = input.direction == nullptr
        ? nullptr
        : static_cast<const double*>(PyArray_DATA(input.direction));
    const auto make_positions = [&](const npy_intp* atoms, int count) {
        std::array<JetVector, 4> positions{};
        for (int atom = 0; atom < count; ++atom) {
            for (int axis = 0; axis < 3; ++axis) {
                const int local = 3 * atom + axis;
                const npy_intp global = atoms[atom];
                positions[atom][axis] = jet_variable(
                    xyz[3 * global + axis], local,
                    direction == nullptr ? 0.0 : direction[3 * global + axis]);
            }
        }
        return positions;
    };
    const auto scatter = [&](const DirectionalJet& value,
                             const npy_intp* atoms, int count) {
        energy += value.value;
        for (int atom = 0; atom < count; ++atom) {
            const npy_intp global = atoms[atom];
            for (int axis = 0; axis < 3; ++axis) {
                const int local = 3 * atom + axis;
                if (gradient != nullptr) {
                    gradient[3 * global + axis] += value.gradient[local];
                }
                if (product != nullptr) {
                    product[3 * global + axis] += value.gradient_dot[local];
                }
            }
        }
    };
    const auto* angle_atoms =
        static_cast<const npy_intp*>(PyArray_DATA(input.angle_atoms));
    const auto* angle_parameters =
        static_cast<const double*>(PyArray_DATA(input.angle_parameters));
    for (npy_intp item = 0; item < input.angle_count; ++item) {
        const npy_intp* atoms = angle_atoms + 3 * item;
        const double* p = angle_parameters + 4 * item;
        const auto position = make_positions(atoms, 3);
        const JetVector left = jet_subtract(position[0], position[1]);
        const JetVector right = jet_subtract(position[2], position[1]);
        const DirectionalJet r1 = jet_norm(left);
        const DirectionalJet r2 = jet_norm(right);
        if (r1.value <= kCoincidentTolerance ||
            r2.value <= kCoincidentTolerance) return false;
        const DirectionalJet cosine = jet_dot(left, right) / (r1 * r2);
        const DirectionalJet damping =
            gfnff_distance_damping(r1, p[2]) *
            gfnff_distance_damping(r2, p[3]);
        DirectionalJet shape;
        if (std::abs(std::acos(-1.0) - p[0]) < 1.0e-6) {
            if (std::abs(cosine.value) >= 1.0 - 1.0e-12) return false;
            const DirectionalJet theta = jet_acos(cosine);
            const DirectionalJet delta = theta - p[0];
            shape = delta * delta;
        } else {
            const DirectionalJet delta = cosine - std::cos(p[0]);
            shape = delta * delta;
        }
        scatter(p[1] * damping * shape, atoms, 3);
    }
    const auto* torsion_atoms =
        static_cast<const npy_intp*>(PyArray_DATA(input.torsion_atoms));
    const auto* torsion_parameters =
        static_cast<const double*>(PyArray_DATA(input.torsion_parameters));
    for (npy_intp item = 0; item < input.torsion_count; ++item) {
        const npy_intp* atoms = torsion_atoms + 4 * item;
        const double* p = torsion_parameters + 7 * item;
        const auto position = make_positions(atoms, 4);
        const int mode = static_cast<int>(std::round(p[0]));
        const DirectionalJet r1 = jet_norm(
            mode > 0 ? jet_subtract(position[0], position[1])
                     : jet_subtract(position[1], position[0]));
        const DirectionalJet r2 = jet_norm(
            mode > 0 ? jet_subtract(position[2], position[1])
                     : jet_subtract(position[1], position[2]));
        const DirectionalJet r3 = jet_norm(
            mode > 0 ? jet_subtract(position[3], position[2])
                     : jet_subtract(position[1], position[3]));
        if (r1.value <= kCoincidentTolerance ||
            r2.value <= kCoincidentTolerance ||
            r3.value <= kCoincidentTolerance) return false;
        const DirectionalJet damping =
            gfnff_distance_damping(r1, p[4]) *
            gfnff_distance_damping(r2, p[5]) *
            gfnff_distance_damping(r3, p[6]);
        DirectionalJet shape;
        if (mode > 0) {
            const DirectionalJet phi = proper_dihedral(position);
            shape = 1.0 + jet_cos(
                p[1] * (phi - p[2]) + jet_constant(std::acos(-1.0)));
        } else {
            const DirectionalJet phi = improper_angle(position);
            if (mode == 0) {
                shape = 1.0 + jet_cos(
                    phi - p[2] + jet_constant(std::acos(-1.0)));
            } else {
                const DirectionalJet delta = jet_cos(phi) - std::cos(p[2]);
                shape = delta * delta;
            }
        }
        scatter(p[3] * damping * shape, atoms, 4);
    }
    const auto* special_atoms =
        static_cast<const npy_intp*>(PyArray_DATA(input.special_atoms));
    const auto* amplitudes =
        static_cast<const double*>(PyArray_DATA(input.special_amplitudes));
    for (npy_intp item = 0; item < input.special_count; ++item) {
        const npy_intp* atoms = special_atoms + 4 * item;
        const auto position = make_positions(atoms, 4);
        const DirectionalJet phi = proper_dihedral(position);
        scatter(amplitudes[item] * (1.0 - jet_cos(2.0 * phi)), atoms, 4);
    }
    return true;
}

bool parse_gfnff_valence_call(
    PyObject* args, PyObject*& coordinates, PyObject*& angle_atoms,
    PyObject*& angle_parameters, PyObject*& torsion_atoms,
    PyObject*& torsion_parameters, PyObject*& special_atoms,
    PyObject*& special_amplitudes) {
    return PyArg_ParseTuple(
        args, "OOOOOOO", &coordinates, &angle_atoms, &angle_parameters,
        &torsion_atoms, &torsion_parameters, &special_atoms,
        &special_amplitudes);
}

PyObject* gfnff_valence_energy_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates, *angle_atoms, *angle_parameters, *torsion_atoms,
        *torsion_parameters, *special_atoms, *special_amplitudes;
    if (!parse_gfnff_valence_call(
            args, coordinates, angle_atoms, angle_parameters, torsion_atoms,
            torsion_parameters, special_atoms, special_amplitudes)) return nullptr;
    GFNFFValenceInputs input;
    if (!parse_gfnff_valence_inputs(
            coordinates, angle_atoms, angle_parameters, torsion_atoms,
            torsion_parameters, special_atoms, special_amplitudes, Py_None,
            input)) return nullptr;
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        release_gfnff_valence_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    bool valid;
    Py_BEGIN_ALLOW_THREADS
    valid = accumulate_gfnff_valence(
        input, energy, static_cast<double*>(PyArray_DATA(gradient)), nullptr);
    Py_END_ALLOW_THREADS
    const long long count = static_cast<long long>(
        input.angle_count + input.torsion_count + input.special_count);
    release_gfnff_valence_inputs(input);
    if (!valid) {
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_ValueError, "singular GFN-FF valence geometry");
        return nullptr;
    }
    return Py_BuildValue("dNL", energy, gradient, count);
}

PyObject* gfnff_valence_hessian_vector(PyObject*, PyObject* args) {
    PyObject* base_args;
    PyObject* direction;
    if (!PyArg_ParseTuple(args, "OO", &base_args, &direction) ||
        !PyTuple_Check(base_args)) return nullptr;
    PyObject *coordinates, *angle_atoms, *angle_parameters, *torsion_atoms,
        *torsion_parameters, *special_atoms, *special_amplitudes;
    if (!parse_gfnff_valence_call(
            base_args, coordinates, angle_atoms, angle_parameters,
            torsion_atoms, torsion_parameters, special_atoms,
            special_amplitudes)) return nullptr;
    GFNFFValenceInputs input;
    if (!parse_gfnff_valence_inputs(
            coordinates, angle_atoms, angle_parameters, torsion_atoms,
            torsion_parameters, special_atoms, special_amplitudes, direction,
            input)) return nullptr;
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (product == nullptr) {
        release_gfnff_valence_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    bool valid;
    Py_BEGIN_ALLOW_THREADS
    valid = accumulate_gfnff_valence(
        input, energy, nullptr, static_cast<double*>(PyArray_DATA(product)));
    Py_END_ALLOW_THREADS
    const long long count = static_cast<long long>(
        input.angle_count + input.torsion_count + input.special_count);
    release_gfnff_valence_inputs(input);
    if (!valid) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_ValueError, "singular GFN-FF valence geometry");
        return nullptr;
    }
    return Py_BuildValue("NL", product, count);
}

struct GFNFFNCIInputs {
    PyArrayObject* coordinates = nullptr;
    PyArrayObject* hbond_atoms = nullptr;
    PyArrayObject* hbond_parameters = nullptr;
    PyArrayObject* xbond_atoms = nullptr;
    PyArrayObject* xbond_parameters = nullptr;
    PyArrayObject* atm_atoms = nullptr;
    PyArrayObject* atm_amplitudes = nullptr;
    PyArrayObject* direction = nullptr;
    npy_intp atoms = 0;
    npy_intp hbond_count = 0;
    npy_intp xbond_count = 0;
    npy_intp atm_count = 0;
};

void release_gfnff_nci_inputs(GFNFFNCIInputs& input) {
    Py_XDECREF(input.coordinates);
    Py_XDECREF(input.hbond_atoms);
    Py_XDECREF(input.hbond_parameters);
    Py_XDECREF(input.xbond_atoms);
    Py_XDECREF(input.xbond_parameters);
    Py_XDECREF(input.atm_atoms);
    Py_XDECREF(input.atm_amplitudes);
    Py_XDECREF(input.direction);
}

bool parse_gfnff_nci_inputs(
    PyObject* coordinates, PyObject* hbond_atoms,
    PyObject* hbond_parameters, PyObject* xbond_atoms,
    PyObject* xbond_parameters, PyObject* atm_atoms,
    PyObject* atm_amplitudes, PyObject* direction, GFNFFNCIInputs& out) {
    out.coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.hbond_atoms = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(hbond_atoms, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.hbond_parameters = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(hbond_parameters, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.xbond_atoms = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(xbond_atoms, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.xbond_parameters = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(xbond_parameters, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.atm_atoms = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(atm_atoms, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.atm_amplitudes = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(atm_amplitudes, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (direction != Py_None) {
        out.direction = reinterpret_cast<PyArrayObject*>(
            PyArray_FROM_OTF(direction, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    }
    if (out.coordinates == nullptr || out.hbond_atoms == nullptr ||
        out.hbond_parameters == nullptr || out.xbond_atoms == nullptr ||
        out.xbond_parameters == nullptr || out.atm_atoms == nullptr ||
        out.atm_amplitudes == nullptr ||
        (direction != Py_None && out.direction == nullptr)) {
        release_gfnff_nci_inputs(out);
        return false;
    }
    if (PyArray_NDIM(out.coordinates) != 2 ||
        PyArray_DIM(out.coordinates, 1) != 3 ||
        PyArray_NDIM(out.hbond_atoms) != 2 ||
        PyArray_DIM(out.hbond_atoms, 1) != 7 ||
        PyArray_NDIM(out.hbond_parameters) != 2 ||
        PyArray_DIM(out.hbond_parameters, 1) != 22 ||
        PyArray_NDIM(out.xbond_atoms) != 2 ||
        PyArray_DIM(out.xbond_atoms, 1) != 3 ||
        PyArray_NDIM(out.xbond_parameters) != 2 ||
        PyArray_DIM(out.xbond_parameters, 1) != 7 ||
        PyArray_NDIM(out.atm_atoms) != 2 ||
        PyArray_DIM(out.atm_atoms, 1) != 3 ||
        PyArray_NDIM(out.atm_amplitudes) != 1) {
        PyErr_SetString(
            PyExc_ValueError, "native GFN-FF NCI array shapes are invalid");
        release_gfnff_nci_inputs(out);
        return false;
    }
    out.atoms = PyArray_DIM(out.coordinates, 0);
    out.hbond_count = PyArray_DIM(out.hbond_atoms, 0);
    out.xbond_count = PyArray_DIM(out.xbond_atoms, 0);
    out.atm_count = PyArray_DIM(out.atm_atoms, 0);
    if (PyArray_DIM(out.hbond_parameters, 0) != out.hbond_count ||
        PyArray_DIM(out.xbond_parameters, 0) != out.xbond_count ||
        PyArray_DIM(out.atm_amplitudes, 0) != out.atm_count ||
        (out.direction != nullptr &&
         (PyArray_NDIM(out.direction) != 2 ||
          PyArray_DIM(out.direction, 0) != out.atoms ||
          PyArray_DIM(out.direction, 1) != 3))) {
        PyErr_SetString(
            PyExc_ValueError, "native GFN-FF NCI dimensions differ");
        release_gfnff_nci_inputs(out);
        return false;
    }
    const auto* h_atoms =
        static_cast<const npy_intp*>(PyArray_DATA(out.hbond_atoms));
    const auto* h_parameters =
        static_cast<const double*>(PyArray_DATA(out.hbond_parameters));
    for (npy_intp item = 0; item < out.hbond_count; ++item) {
        const double* p = h_parameters + 22 * item;
        const int mode = static_cast<int>(std::round(p[0]));
        const int neighbors = static_cast<int>(std::round(p[14]));
        const int torsions = static_cast<int>(std::round(p[21]));
        const int active_count = mode == 1 ? 3 :
            (mode == 2 ? 3 + neighbors :
             (mode == 3 ? 5 : 4 + torsions));
        if (mode < 1 || mode > 4 || neighbors < 0 || neighbors > 4 ||
            (mode == 1 && neighbors != 0) ||
            (mode == 3 && neighbors != 2) ||
            (mode == 4 && (neighbors != 1 || torsions < 0 || torsions > 3)) ||
            std::abs(p[14] - neighbors) > 1.0e-12 ||
            std::abs(p[21] - torsions) > 1.0e-12) {
            PyErr_SetString(
                PyExc_ValueError, "unsupported GFN-FF hydrogen-bond mode");
            release_gfnff_nci_inputs(out);
            return false;
        }
        for (int slot = 0; slot < 7; ++slot) {
            const npy_intp atom = h_atoms[7 * item + slot];
            const bool active = slot < active_count;
            if ((active && (atom < 0 || atom >= out.atoms)) ||
                (!active && atom != -1)) {
                PyErr_SetString(
                    PyExc_ValueError, "native GFN-FF hydrogen-bond atom is invalid");
                release_gfnff_nci_inputs(out);
                return false;
            }
        }
        for (int parameter = 0; parameter < 22; ++parameter) {
            if (!std::isfinite(p[parameter])) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "native GFN-FF hydrogen-bond parameter is invalid");
                release_gfnff_nci_inputs(out);
                return false;
            }
        }
        if (p[1] < 0.0 || p[2] <= 0.0 || p[3] <= 0.0 || p[4] <= 0.0 ||
            p[5] <= 0.0 || p[6] <= 0.0 || p[15] < 0.0 ||
            (mode == 3 && (p[16] <= 0.0 || p[17] <= 0.0))) {
            PyErr_SetString(
                PyExc_ValueError, "native GFN-FF hydrogen-bond scale is invalid");
            release_gfnff_nci_inputs(out);
            return false;
        }
    }
    const auto validate_fixed_atoms = [&](PyArrayObject* array, int width) {
        const auto* indices =
            static_cast<const npy_intp*>(PyArray_DATA(array));
        for (npy_intp item = 0; item < PyArray_DIM(array, 0); ++item) {
            for (int slot = 0; slot < width; ++slot) {
                const npy_intp atom = indices[width * item + slot];
                if (atom < 0 || atom >= out.atoms) return false;
            }
        }
        return true;
    };
    if (!validate_fixed_atoms(out.xbond_atoms, 3) ||
        !validate_fixed_atoms(out.atm_atoms, 3)) {
        PyErr_SetString(PyExc_ValueError, "native GFN-FF NCI atom is invalid");
        release_gfnff_nci_inputs(out);
        return false;
    }
    const auto* x_parameters =
        static_cast<const double*>(PyArray_DATA(out.xbond_parameters));
    for (npy_intp item = 0; item < 7 * out.xbond_count; ++item) {
        if (!std::isfinite(x_parameters[item])) {
            PyErr_SetString(
                PyExc_ValueError, "native GFN-FF halogen-bond parameter is invalid");
            release_gfnff_nci_inputs(out);
            return false;
        }
    }
    const auto* atm_values =
        static_cast<const double*>(PyArray_DATA(out.atm_amplitudes));
    for (npy_intp item = 0; item < out.atm_count; ++item) {
        if (!std::isfinite(atm_values[item])) {
            PyErr_SetString(
                PyExc_ValueError, "native GFN-FF bonded ATM parameter is invalid");
            release_gfnff_nci_inputs(out);
            return false;
        }
    }
    return true;
}

bool accumulate_gfnff_nci(
    const GFNFFNCIInputs& input, double& energy, double* gradient,
    double* product) {
    const auto* xyz =
        static_cast<const double*>(PyArray_DATA(input.coordinates));
    const auto* direction = input.direction == nullptr
        ? nullptr
        : static_cast<const double*>(PyArray_DATA(input.direction));
    const auto position = [&](npy_intp atom, int slot) {
        JetVector result{};
        for (int axis = 0; axis < 3; ++axis) {
            result[axis] = jet_variable(
                xyz[3 * atom + axis], 3 * slot + axis,
                direction == nullptr ? 0.0 : direction[3 * atom + axis]);
        }
        return result;
    };
    const auto scatter = [&](const DirectionalJet& value,
                             const npy_intp* atoms, int count) {
        energy += value.value;
        for (int slot = 0; slot < count; ++slot) {
            const npy_intp atom = atoms[slot];
            if (atom < 0) continue;
            for (int axis = 0; axis < 3; ++axis) {
                if (gradient != nullptr) {
                    gradient[3 * atom + axis] +=
                        value.gradient[3 * slot + axis];
                }
                if (product != nullptr) {
                    product[3 * atom + axis] +=
                        value.gradient_dot[3 * slot + axis];
                }
            }
        }
    };
    const auto distance = [&](const JetVector& left, const JetVector& right) {
        return jet_norm(jet_subtract(left, right));
    };
    const auto* h_atoms =
        static_cast<const npy_intp*>(PyArray_DATA(input.hbond_atoms));
    const auto* h_parameters =
        static_cast<const double*>(PyArray_DATA(input.hbond_parameters));
    for (npy_intp item = 0; item < input.hbond_count; ++item) {
        const npy_intp* atoms = h_atoms + 7 * item;
        const double* p = h_parameters + 22 * item;
        const int mode = static_cast<int>(std::round(p[0]));
        const int neighbor_count = static_cast<int>(std::round(p[14]));
        const int torsion_count = static_cast<int>(std::round(p[21]));
        const JetVector a = position(atoms[0], 0);
        const JetVector b = position(atoms[1], 1);
        const JetVector h = position(atoms[2], 2);
        const DirectionalJet rab = distance(a, b);
        const DirectionalJet rah = distance(a, h);
        const DirectionalJet rbh = distance(b, h);
        if (rab.value <= kCoincidentTolerance ||
            rah.value <= kCoincidentTolerance ||
            rbh.value <= kCoincidentTolerance) return false;
        const DirectionalJet rab2 = rab * rab;
        const DirectionalJet rah2 = rah * rah;
        const DirectionalJet rbh2 = rbh * rbh;
        const DirectionalJet expo =
            (p[3] / p[2]) * ((rah + rbh + 1.0e-12) / rab - 1.0);
        if (expo.value > 15.0) continue;
        const DirectionalJet out_of_line = 2.0 / (1.0 + jet_exp(expo));
        const DirectionalJet long_ratio = jet_pow(rab2 / p[4], p[5]);
        const DirectionalJet short_ratio = jet_pow(
            (p[6] * p[2]) / rab2, mode == 4 ? 6.0 : p[5]);
        const DirectionalJet damping =
            (1.0 / (1.0 + long_ratio)) *
            (1.0 / (1.0 + short_ratio));
        DirectionalJet term;
        if (mode == 1) {
            const DirectionalJet rah4 = rah2 * rah2;
            const DirectionalJet rbh4 = rbh2 * rbh2;
            const DirectionalJet denominator = rah4 + rbh4;
            const DirectionalJet base =
                (p[7] * rah4 + p[8] * rbh4) / denominator;
            const DirectionalJet acid =
                (p[10] * rah4 + p[9] * rbh4) / denominator;
            term = -p[15] * acid * base * damping * p[1] * out_of_line /
                (rab2 * rab);
        } else {
            DirectionalJet neighbor_orientation = jet_constant(1.0);
            for (int neighbor = 0; neighbor < neighbor_count; ++neighbor) {
                const JetVector n = position(atoms[3 + neighbor], 3 + neighbor);
                const DirectionalJet ran = distance(a, n);
                const DirectionalJet rbn = distance(b, n);
                if (ran.value <= kCoincidentTolerance ||
                    rbn.value <= kCoincidentTolerance) return false;
                const DirectionalJet neighbor_expo =
                    (p[13] / p[2]) * ((ran + rbn + 1.0e-12) / rab - 1.0);
                neighbor_orientation = neighbor_orientation *
                    (2.0 / (1.0 + jet_exp(-1.0 * neighbor_expo)) - 1.0);
            }
            const DirectionalJet radial = damping *
                (p[11] / (rbh2 * rbh) + p[12] / (rab2 * rab));
            DirectionalJet special_orientation = jet_constant(1.0);
            if (mode == 3) {
                const JetVector n1 = position(atoms[3], 3);
                const JetVector n2 = position(atoms[4], 4);
                const JetVector neighbor_sum = jet_add(
                    jet_subtract(n1, b), jet_subtract(n2, b));
                const DirectionalJet norm = jet_norm(neighbor_sum);
                if (norm.value <= kCoincidentTolerance) return false;
                const JetVector lone_pair = jet_subtract(
                    b, jet_scale(jet_divide(neighbor_sum, norm), p[16]));
                const DirectionalJet ralp = distance(a, lone_pair);
                const DirectionalJet rblp = distance(b, lone_pair);
                const DirectionalJet lp_expo =
                    (p[17] / p[2]) * ((ralp + rblp + 1.0e-12) / rab - 1.0);
                special_orientation = 2.0 / (1.0 + jet_exp(lp_expo));
            } else if (mode == 4) {
                const JetVector c = position(atoms[3], 3);
                const JetVector cb = jet_subtract(b, c);
                const JetVector ch = jet_subtract(h, c);
                const DirectionalJet cb_norm = jet_norm(cb);
                const DirectionalJet ch_norm = jet_norm(ch);
                if (cb_norm.value <= kCoincidentTolerance ||
                    ch_norm.value <= kCoincidentTolerance) return false;
                const DirectionalJet cosine =
                    jet_dot(cb, ch) / (cb_norm * ch_norm);
                const double reference_cosine = std::cos(p[18]);
                const double bend_fc = 1.0 - p[19];
                const double bend_denominator =
                    (1.0 - reference_cosine) * (1.0 - reference_cosine);
                const DirectionalJet bend_delta = cosine - reference_cosine;
                special_orientation =
                    1.0 - (bend_fc / bend_denominator) *
                    bend_delta * bend_delta;
                DirectionalJet torsion_product = jet_constant(1.0);
                for (int torsion = 0; torsion < torsion_count; ++torsion) {
                    const JetVector r = position(atoms[4 + torsion], 4 + torsion);
                    const std::array<JetVector, 4> torsion_positions = {r, b, c, h};
                    const DirectionalJet phi = proper_dihedral(torsion_positions);
                    const double torsion_fc = 0.5 * (1.0 - p[20]);
                    constexpr double pi = 3.141592653589793238462643383279502884;
                    const DirectionalJet phase =
                        2.0 * (phi - 0.5 * pi) + pi;
                    torsion_product = torsion_product *
                        ((1.0 + jet_cos(phase)) * torsion_fc + p[20]);
                }
                special_orientation = special_orientation * torsion_product;
            }
            term = -p[15] * p[7] * p[1] * out_of_line *
                neighbor_orientation * special_orientation * radial;
        }
        const int active_count = mode == 1 ? 3 :
            (mode == 2 ? 3 + neighbor_count :
             (mode == 3 ? 5 : 4 + torsion_count));
        scatter(term, atoms, active_count);
    }
    const auto* x_atoms =
        static_cast<const npy_intp*>(PyArray_DATA(input.xbond_atoms));
    const auto* x_parameters =
        static_cast<const double*>(PyArray_DATA(input.xbond_parameters));
    for (npy_intp item = 0; item < input.xbond_count; ++item) {
        const npy_intp* atoms = x_atoms + 3 * item;
        const double* p = x_parameters + 7 * item;
        const JetVector a = position(atoms[0], 0);
        const JetVector b = position(atoms[1], 1);
        const JetVector x = position(atoms[2], 2);
        const DirectionalJet rab = distance(a, b);
        // libxtb regularizes the two X-containing distances before using
        // them in the out-of-line and radial factors.  Keep the same tiny
        // offset so termwise energy parity is not lost at ~1e-13 Eh.
        const DirectionalJet rax = distance(a, x) + 1.0e-12;
        const DirectionalJet rbx = distance(b, x) + 1.0e-12;
        if (rab.value <= kCoincidentTolerance ||
            rax.value <= kCoincidentTolerance ||
            rbx.value <= kCoincidentTolerance) return false;
        const DirectionalJet expo = p[2] * ((rax + rbx) / rab - 1.0);
        if (expo.value > 15.0) continue;
        const DirectionalJet out_of_line = 2.0 / (1.0 + jet_exp(expo));
        const DirectionalJet rbx2 = rbx * rbx;
        const DirectionalJet long_ratio = jet_pow(rbx2 / p[3], p[4]);
        const DirectionalJet short_ratio =
            jet_pow((p[5] * p[1]) / rbx2, p[4]);
        const DirectionalJet damping =
            (1.0 / (1.0 + long_ratio)) *
            (1.0 / (1.0 + short_ratio));
        const DirectionalJet term =
            -p[6] * p[0] * damping * out_of_line / (rbx2 * rbx);
        scatter(term, atoms, 3);
    }
    const auto* atm_atoms =
        static_cast<const npy_intp*>(PyArray_DATA(input.atm_atoms));
    const auto* atm_amplitudes =
        static_cast<const double*>(PyArray_DATA(input.atm_amplitudes));
    for (npy_intp item = 0; item < input.atm_count; ++item) {
        const npy_intp* atoms = atm_atoms + 3 * item;
        const JetVector i = position(atoms[0], 0);
        const JetVector j = position(atoms[1], 1);
        const JetVector k = position(atoms[2], 2);
        const DirectionalJet rij = distance(i, j);
        const DirectionalJet rjk = distance(j, k);
        const DirectionalJet rik = distance(i, k);
        if (rij.value <= kCoincidentTolerance ||
            rjk.value <= kCoincidentTolerance ||
            rik.value <= kCoincidentTolerance) return false;
        const DirectionalJet r2ij = rij * rij;
        const DirectionalJet r2jk = rjk * rjk;
        const DirectionalJet r2ik = rik * rik;
        const DirectionalJet product_r2 = r2ij * r2jk * r2ik;
        const DirectionalJet angular =
            0.375 * (-r2ij + r2jk + r2ik) *
            (r2ij - r2jk + r2ik) * (r2ij + r2jk - r2ik) /
            product_r2;
        const DirectionalJet r9 = product_r2 * rij * rjk * rik;
        scatter(atm_amplitudes[item] * (angular + 1.0) / r9, atoms, 3);
    }
    return true;
}

bool parse_gfnff_nci_call(
    PyObject* args, PyObject*& coordinates, PyObject*& hbond_atoms,
    PyObject*& hbond_parameters, PyObject*& xbond_atoms,
    PyObject*& xbond_parameters, PyObject*& atm_atoms,
    PyObject*& atm_amplitudes) {
    return PyArg_ParseTuple(
        args, "OOOOOOO", &coordinates, &hbond_atoms, &hbond_parameters,
        &xbond_atoms, &xbond_parameters, &atm_atoms, &atm_amplitudes);
}

PyObject* gfnff_nci_energy_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates, *hbond_atoms, *hbond_parameters, *xbond_atoms,
        *xbond_parameters, *atm_atoms, *atm_amplitudes;
    if (!parse_gfnff_nci_call(
            args, coordinates, hbond_atoms, hbond_parameters, xbond_atoms,
            xbond_parameters, atm_atoms, atm_amplitudes)) return nullptr;
    GFNFFNCIInputs input;
    if (!parse_gfnff_nci_inputs(
            coordinates, hbond_atoms, hbond_parameters, xbond_atoms,
            xbond_parameters, atm_atoms, atm_amplitudes, Py_None, input)) {
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        release_gfnff_nci_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    bool valid;
    Py_BEGIN_ALLOW_THREADS
    valid = accumulate_gfnff_nci(
        input, energy, static_cast<double*>(PyArray_DATA(gradient)), nullptr);
    Py_END_ALLOW_THREADS
    const long long count = static_cast<long long>(
        input.hbond_count + input.xbond_count + input.atm_count);
    release_gfnff_nci_inputs(input);
    if (!valid) {
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_ValueError, "singular GFN-FF NCI geometry");
        return nullptr;
    }
    return Py_BuildValue("dNL", energy, gradient, count);
}

PyObject* gfnff_nci_hessian_vector(PyObject*, PyObject* args) {
    PyObject* base_args;
    PyObject* direction;
    if (!PyArg_ParseTuple(args, "OO", &base_args, &direction) ||
        !PyTuple_Check(base_args)) return nullptr;
    PyObject *coordinates, *hbond_atoms, *hbond_parameters, *xbond_atoms,
        *xbond_parameters, *atm_atoms, *atm_amplitudes;
    if (!parse_gfnff_nci_call(
            base_args, coordinates, hbond_atoms, hbond_parameters,
            xbond_atoms, xbond_parameters, atm_atoms, atm_amplitudes)) {
        return nullptr;
    }
    GFNFFNCIInputs input;
    if (!parse_gfnff_nci_inputs(
            coordinates, hbond_atoms, hbond_parameters, xbond_atoms,
            xbond_parameters, atm_atoms, atm_amplitudes, direction, input)) {
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (product == nullptr) {
        release_gfnff_nci_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    bool valid;
    Py_BEGIN_ALLOW_THREADS
    valid = accumulate_gfnff_nci(
        input, energy, nullptr, static_cast<double*>(PyArray_DATA(product)));
    Py_END_ALLOW_THREADS
    const long long count = static_cast<long long>(
        input.hbond_count + input.xbond_count + input.atm_count);
    release_gfnff_nci_inputs(input);
    if (!valid) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_ValueError, "singular GFN-FF NCI geometry");
        return nullptr;
    }
    return Py_BuildValue("NL", product, count);
}

struct LocalValenceInputs {
    PyArrayObject* coordinates = nullptr;
    PyArrayObject* angle_atoms = nullptr;
    PyArrayObject* angle_parameters = nullptr;
    PyArrayObject* torsion_atoms = nullptr;
    PyArrayObject* torsion_parameters = nullptr;
    PyArrayObject* term_offsets = nullptr;
    PyArrayObject* terms = nullptr;
    PyArrayObject* direction = nullptr;
    npy_intp atoms = 0;
    npy_intp angle_count = 0;
    npy_intp torsion_count = 0;
    npy_intp term_count = 0;
};

void release_local_valence_inputs(LocalValenceInputs& input) {
    Py_XDECREF(input.coordinates);
    Py_XDECREF(input.angle_atoms);
    Py_XDECREF(input.angle_parameters);
    Py_XDECREF(input.torsion_atoms);
    Py_XDECREF(input.torsion_parameters);
    Py_XDECREF(input.term_offsets);
    Py_XDECREF(input.terms);
    Py_XDECREF(input.direction);
}

bool parse_local_valence_inputs(
    PyObject* coordinates, PyObject* angle_atoms,
    PyObject* angle_parameters, PyObject* torsion_atoms,
    PyObject* torsion_parameters, PyObject* term_offsets, PyObject* terms,
    PyObject* direction, LocalValenceInputs& out) {
    out.coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.angle_atoms = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(angle_atoms, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.angle_parameters = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(angle_parameters, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.torsion_atoms = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(torsion_atoms, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.torsion_parameters = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(torsion_parameters, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.term_offsets = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(term_offsets, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.terms = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(terms, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (direction != Py_None) {
        out.direction = reinterpret_cast<PyArrayObject*>(
            PyArray_FROM_OTF(direction, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    }
    if (out.coordinates == nullptr || out.angle_atoms == nullptr ||
        out.angle_parameters == nullptr || out.torsion_atoms == nullptr ||
        out.torsion_parameters == nullptr || out.term_offsets == nullptr ||
        out.terms == nullptr || (direction != Py_None && out.direction == nullptr)) {
        release_local_valence_inputs(out);
        return false;
    }
    if (PyArray_NDIM(out.coordinates) != 2 ||
        PyArray_DIM(out.coordinates, 1) != 3 ||
        PyArray_NDIM(out.angle_atoms) != 2 ||
        PyArray_DIM(out.angle_atoms, 1) != 3 ||
        PyArray_NDIM(out.angle_parameters) != 2 ||
        PyArray_DIM(out.angle_parameters, 1) != 6 ||
        PyArray_NDIM(out.torsion_atoms) != 2 ||
        PyArray_DIM(out.torsion_atoms, 1) != 4 ||
        PyArray_NDIM(out.torsion_parameters) != 2 ||
        PyArray_DIM(out.torsion_parameters, 1) != 6 ||
        PyArray_NDIM(out.term_offsets) != 1 ||
        PyArray_NDIM(out.terms) != 2 || PyArray_DIM(out.terms, 1) != 3) {
        PyErr_SetString(
            PyExc_ValueError, "native local-valence array shapes are invalid");
        release_local_valence_inputs(out);
        return false;
    }
    out.atoms = PyArray_DIM(out.coordinates, 0);
    out.angle_count = PyArray_DIM(out.angle_atoms, 0);
    out.torsion_count = PyArray_DIM(out.torsion_atoms, 0);
    out.term_count = PyArray_DIM(out.terms, 0);
    if (PyArray_DIM(out.angle_parameters, 0) != out.angle_count ||
        PyArray_DIM(out.torsion_parameters, 0) != out.torsion_count ||
        PyArray_DIM(out.term_offsets, 0) != out.torsion_count + 1 ||
        (out.direction != nullptr &&
         (PyArray_NDIM(out.direction) != 2 ||
          PyArray_DIM(out.direction, 0) != out.atoms ||
          PyArray_DIM(out.direction, 1) != 3))) {
        PyErr_SetString(
            PyExc_ValueError, "native local-valence dimensions differ");
        release_local_valence_inputs(out);
        return false;
    }
    const auto* angle_indices =
        static_cast<const npy_intp*>(PyArray_DATA(out.angle_atoms));
    const auto* torsion_indices =
        static_cast<const npy_intp*>(PyArray_DATA(out.torsion_atoms));
    const auto* angle_values =
        static_cast<const double*>(PyArray_DATA(out.angle_parameters));
    const auto* torsion_values =
        static_cast<const double*>(PyArray_DATA(out.torsion_parameters));
    const auto* offsets =
        static_cast<const npy_intp*>(PyArray_DATA(out.term_offsets));
    const auto* term_values =
        static_cast<const double*>(PyArray_DATA(out.terms));
    const auto* xyz =
        static_cast<const double*>(PyArray_DATA(out.coordinates));
    const auto* direction_values =
        out.direction == nullptr
            ? nullptr
            : static_cast<const double*>(PyArray_DATA(out.direction));
    if (offsets[0] != 0 || offsets[out.torsion_count] != out.term_count) {
        PyErr_SetString(PyExc_ValueError, "native torsion term offsets are invalid");
        release_local_valence_inputs(out);
        return false;
    }
    for (npy_intp item = 0; item < out.atoms * 3; ++item) {
        if (!std::isfinite(xyz[item]) ||
            (direction_values != nullptr &&
             !std::isfinite(direction_values[item]))) {
            PyErr_SetString(
                PyExc_ValueError,
                "native local-valence coordinates and direction must be finite");
            release_local_valence_inputs(out);
            return false;
        }
    }
    for (npy_intp item = 0; item < out.angle_count; ++item) {
        const npy_intp* atoms = angle_indices + 3 * item;
        const double* parameters = angle_values + 6 * item;
        if (atoms[0] < 0 || atoms[1] < 0 || atoms[2] < 0 ||
            atoms[0] >= out.atoms || atoms[1] >= out.atoms ||
            atoms[2] >= out.atoms || atoms[0] == atoms[1] ||
            atoms[1] == atoms[2] || atoms[0] == atoms[2] ||
            !std::isfinite(parameters[0]) || parameters[0] < 0.0 ||
            !std::isfinite(parameters[1])) {
            PyErr_SetString(PyExc_ValueError, "native angle atom is out of range");
            release_local_valence_inputs(out);
            return false;
        }
        for (int parameter = 2; parameter < 6; ++parameter) {
            if (!std::isfinite(parameters[parameter]) ||
                parameters[parameter] <= 0.0) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "native angle radial parameters must be finite and positive");
                release_local_valence_inputs(out);
                return false;
            }
        }
    }
    for (npy_intp item = 0; item < out.torsion_count; ++item) {
        const npy_intp* atoms = torsion_indices + 4 * item;
        const double* parameters = torsion_values + 6 * item;
        bool invalid = false;
        for (int atom = 0; atom < 4; ++atom) {
            invalid = invalid || atoms[atom] < 0 || atoms[atom] >= out.atoms;
            for (int other = 0; other < atom; ++other) {
                invalid = invalid || atoms[atom] == atoms[other];
            }
        }
        for (int parameter = 0; parameter < 6; ++parameter) {
            invalid = invalid || !std::isfinite(parameters[parameter]) ||
                parameters[parameter] <= 0.0;
        }
        if (invalid) {
            PyErr_SetString(PyExc_ValueError, "native torsion atom is out of range");
            release_local_valence_inputs(out);
            return false;
        }
    }
    for (npy_intp item = 0; item < out.torsion_count; ++item) {
        if (offsets[item] < 0 || offsets[item] >= offsets[item + 1] ||
            offsets[item + 1] > out.term_count) {
            PyErr_SetString(PyExc_ValueError, "native torsion term span is invalid");
            release_local_valence_inputs(out);
            return false;
        }
    }
    for (npy_intp term = 0; term < out.term_count; ++term) {
        const double amplitude = term_values[3 * term];
        const double periodicity = term_values[3 * term + 1];
        const double phase = term_values[3 * term + 2];
        if (!std::isfinite(amplitude) || !std::isfinite(periodicity) ||
            periodicity <= 0.0 ||
            std::abs(periodicity - std::round(periodicity)) > 1.0e-12 ||
            !std::isfinite(phase)) {
            PyErr_SetString(
                PyExc_ValueError, "native torsion Fourier term is invalid");
            release_local_valence_inputs(out);
            return false;
        }
    }
    return true;
}

bool accumulate_local_valence(
    const LocalValenceInputs& input, double& energy, double* gradient,
    double* product) {
    const auto* xyz =
        static_cast<const double*>(PyArray_DATA(input.coordinates));
    const auto* direction =
        input.direction == nullptr
            ? nullptr
            : static_cast<const double*>(PyArray_DATA(input.direction));
    const auto* angle_atoms =
        static_cast<const npy_intp*>(PyArray_DATA(input.angle_atoms));
    const auto* angle_parameters =
        static_cast<const double*>(PyArray_DATA(input.angle_parameters));
    const auto* torsion_atoms =
        static_cast<const npy_intp*>(PyArray_DATA(input.torsion_atoms));
    const auto* torsion_parameters =
        static_cast<const double*>(PyArray_DATA(input.torsion_parameters));
    const auto* offsets =
        static_cast<const npy_intp*>(PyArray_DATA(input.term_offsets));
    const auto* terms =
        static_cast<const double*>(PyArray_DATA(input.terms));
    const auto make_positions = [&](const npy_intp* atoms, int count) {
        std::array<JetVector, 4> positions{};
        for (int atom = 0; atom < count; ++atom) {
            for (int axis = 0; axis < 3; ++axis) {
                const int local = 3 * atom + axis;
                const npy_intp global = atoms[atom];
                positions[atom][axis] = jet_variable(
                    xyz[3 * global + axis], local,
                    direction == nullptr
                        ? 0.0
                        : kBohrToAngstrom * direction[3 * global + axis]);
            }
        }
        return positions;
    };
    const auto scatter = [&](const DirectionalJet& value, const npy_intp* atoms,
                             int count) {
        energy += value.value;
        for (int atom = 0; atom < count; ++atom) {
            const npy_intp global = atoms[atom];
            for (int axis = 0; axis < 3; ++axis) {
                const int local = 3 * atom + axis;
                if (gradient != nullptr) {
                    gradient[3 * global + axis] +=
                        kBohrToAngstrom * value.gradient[local];
                }
                if (product != nullptr) {
                    product[3 * global + axis] +=
                        kBohrToAngstrom * value.gradient_dot[local];
                }
            }
        }
    };
    for (npy_intp item = 0; item < input.angle_count; ++item) {
        const npy_intp* atoms = angle_atoms + 3 * item;
        const double* parameters = angle_parameters + 6 * item;
        const auto position = make_positions(atoms, 3);
        const JetVector left = jet_subtract(position[0], position[1]);
        const JetVector right = jet_subtract(position[2], position[1]);
        const DirectionalJet r1 = jet_norm(left);
        const DirectionalJet r2 = jet_norm(right);
        if (r1.value <= kCoincidentTolerance ||
            r2.value <= kCoincidentTolerance) {
            return false;
        }
        const DirectionalJet cosine = jet_dot(left, right) / (r1 * r2);
        if (std::abs(cosine.value) >= 1.0 - 1.0e-12) {
            return false;
        }
        const DirectionalJet theta = jet_acos(cosine);
        const double force = parameters[0];
        const double theta0 = parameters[1];
        const DirectionalJet radial =
            bond_order_factor(r1, parameters[2], parameters[3]) *
            bond_order_factor(r2, parameters[4], parameters[5]);
        DirectionalJet shape;
        double amplitude;
        if (std::abs(std::sin(theta0)) <= 1.0e-5) {
            const DirectionalJet sine = jet_sin(theta);
            shape = sine * sine;
            amplitude = 0.5 * force;
        } else {
            const DirectionalJet difference =
                jet_cos(theta) - std::cos(theta0);
            shape = difference * difference;
            const double sine0 = std::sin(theta0);
            amplitude = force / (2.0 * sine0 * sine0);
        }
        scatter(amplitude * radial * shape, atoms, 3);
    }
    for (npy_intp item = 0; item < input.torsion_count; ++item) {
        const npy_intp* atoms = torsion_atoms + 4 * item;
        const double* parameters = torsion_parameters + 6 * item;
        const auto position = make_positions(atoms, 4);
        const JetVector b1 = jet_subtract(position[0], position[1]);
        const JetVector b2 = jet_subtract(position[2], position[1]);
        const JetVector b3 = jet_subtract(position[3], position[2]);
        const DirectionalJet r1 = jet_norm(b1);
        const DirectionalJet r2 = jet_norm(b2);
        const DirectionalJet r3 = jet_norm(b3);
        const JetVector n1 = jet_cross(b1, b2);
        const JetVector n2 = jet_cross(b2, b3);
        if (r1.value <= kCoincidentTolerance ||
            r2.value <= kCoincidentTolerance ||
            r3.value <= kCoincidentTolerance ||
            jet_norm(n1).value <= kCoincidentTolerance ||
            jet_norm(n2).value <= kCoincidentTolerance) {
            return false;
        }
        const DirectionalJet phi = jet_atan2(
            jet_dot(jet_cross(n1, n2), jet_divide(b2, r2)),
            jet_dot(n1, n2));
        DirectionalJet fourier = jet_constant(0.0);
        for (npy_intp term = offsets[item]; term < offsets[item + 1]; ++term) {
            const double amplitude = terms[3 * term];
            const double periodicity = terms[3 * term + 1];
            const double phase = terms[3 * term + 2];
            fourier = fourier +
                amplitude * (1.0 + jet_cos(periodicity * phi - phase));
        }
        const DirectionalJet radial =
            bond_order_factor(r1, parameters[0], parameters[1]) *
            bond_order_factor(r2, parameters[2], parameters[3]) *
            bond_order_factor(r3, parameters[4], parameters[5]);
        scatter(radial * fourier, atoms, 4);
    }
    return true;
}

PyObject* local_valence_energy(PyObject*, PyObject* args) {
    PyObject *coordinates, *angle_atoms, *angle_parameters, *torsion_atoms;
    PyObject *torsion_parameters, *term_offsets, *terms;
    if (!PyArg_ParseTuple(
            args, "OOOOOOO", &coordinates, &angle_atoms, &angle_parameters,
            &torsion_atoms, &torsion_parameters, &term_offsets, &terms)) {
        return nullptr;
    }
    LocalValenceInputs input;
    if (!parse_local_valence_inputs(
            coordinates, angle_atoms, angle_parameters, torsion_atoms,
            torsion_parameters, term_offsets, terms, Py_None, input)) {
        return nullptr;
    }
    double energy = 0.0;
    bool valid;
    Py_BEGIN_ALLOW_THREADS
    valid = accumulate_local_valence(input, energy, nullptr, nullptr);
    Py_END_ALLOW_THREADS
    const npy_intp count = input.angle_count + input.torsion_count;
    release_local_valence_inputs(input);
    if (!valid) {
        PyErr_SetString(PyExc_FloatingPointError, "singular native local valence geometry");
        return nullptr;
    }
    return Py_BuildValue(
        "dL", energy, static_cast<long long>(count));
}

PyObject* local_valence_energy_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates, *angle_atoms, *angle_parameters, *torsion_atoms;
    PyObject *torsion_parameters, *term_offsets, *terms;
    if (!PyArg_ParseTuple(
            args, "OOOOOOO", &coordinates, &angle_atoms, &angle_parameters,
            &torsion_atoms, &torsion_parameters, &term_offsets, &terms)) {
        return nullptr;
    }
    LocalValenceInputs input;
    if (!parse_local_valence_inputs(
            coordinates, angle_atoms, angle_parameters, torsion_atoms,
            torsion_parameters, term_offsets, terms, Py_None, input)) {
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        release_local_valence_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    bool valid;
    Py_BEGIN_ALLOW_THREADS
    valid = accumulate_local_valence(
        input, energy, static_cast<double*>(PyArray_DATA(gradient)), nullptr);
    Py_END_ALLOW_THREADS
    const npy_intp count = input.angle_count + input.torsion_count;
    release_local_valence_inputs(input);
    if (!valid) {
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_FloatingPointError, "singular native local valence geometry");
        return nullptr;
    }
    return Py_BuildValue("dNL", energy, gradient, static_cast<long long>(count));
}

PyObject* local_valence_hessian_vector(PyObject*, PyObject* args) {
    PyObject *coordinates, *angle_atoms, *angle_parameters, *torsion_atoms;
    PyObject *torsion_parameters, *term_offsets, *terms, *direction;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOO", &coordinates, &angle_atoms, &angle_parameters,
            &torsion_atoms, &torsion_parameters, &term_offsets, &terms,
            &direction)) {
        return nullptr;
    }
    LocalValenceInputs input;
    if (!parse_local_valence_inputs(
            coordinates, angle_atoms, angle_parameters, torsion_atoms,
            torsion_parameters, term_offsets, terms, direction, input)) {
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (product == nullptr) {
        release_local_valence_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    bool valid;
    Py_BEGIN_ALLOW_THREADS
    valid = accumulate_local_valence(
        input, energy, nullptr, static_cast<double*>(PyArray_DATA(product)));
    Py_END_ALLOW_THREADS
    const npy_intp count = input.angle_count + input.torsion_count;
    release_local_valence_inputs(input);
    if (!valid) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_FloatingPointError, "singular native local valence geometry");
        return nullptr;
    }
    return Py_BuildValue("NL", product, static_cast<long long>(count));
}

PyObject* planar_image_potential_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates_object, *charges_object, *origin_object, *normal_object;
    if (!PyArg_ParseTuple(
            args, "OOOO", &coordinates_object, &charges_object, &origin_object,
            &normal_object)) {
        return nullptr;
    }
    PyArrayObject* coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* charges = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(charges_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* origin = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(origin_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* normal = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(normal_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (coordinates == nullptr || charges == nullptr || origin == nullptr ||
        normal == nullptr) {
        Py_XDECREF(coordinates);
        Py_XDECREF(charges);
        Py_XDECREF(origin);
        Py_XDECREF(normal);
        return nullptr;
    }
    const bool valid_shapes =
        PyArray_NDIM(coordinates) == 2 && PyArray_DIM(coordinates, 1) == 3 &&
        PyArray_NDIM(charges) == 1 &&
        PyArray_DIM(charges, 0) == PyArray_DIM(coordinates, 0) &&
        PyArray_NDIM(origin) == 1 && PyArray_DIM(origin, 0) == 3 &&
        PyArray_NDIM(normal) == 1 && PyArray_DIM(normal, 0) == 3;
    if (!valid_shapes) {
        PyErr_SetString(
            PyExc_ValueError,
            "native planar image arrays require (N,3), (N,), (3,), (3,)");
        Py_DECREF(coordinates);
        Py_DECREF(charges);
        Py_DECREF(origin);
        Py_DECREF(normal);
        return nullptr;
    }
    const npy_intp atoms = PyArray_DIM(coordinates, 0);
    npy_intp potential_dimension[1] = {atoms};
    npy_intp gradient_dimensions[2] = {atoms, 3};
    PyArrayObject* potential = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, potential_dimension, NPY_DOUBLE, 0));
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, gradient_dimensions, NPY_DOUBLE, 0));
    if (potential == nullptr || gradient == nullptr) {
        Py_XDECREF(potential);
        Py_XDECREF(gradient);
        Py_DECREF(coordinates);
        Py_DECREF(charges);
        Py_DECREF(origin);
        Py_DECREF(normal);
        return nullptr;
    }
    const double* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const double* q = static_cast<const double*>(PyArray_DATA(charges));
    const double* plane = static_cast<const double*>(PyArray_DATA(origin));
    const double* n = static_cast<const double*>(PyArray_DATA(normal));
    double* output_potential = static_cast<double*>(PyArray_DATA(potential));
    double* output_gradient = static_cast<double*>(PyArray_DATA(gradient));
    bool valid = true;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp target = 0; target < atoms; ++target) {
        double target_potential = 0.0;
        double gx = 0.0, gy = 0.0, gz = 0.0;
        for (npy_intp source = 0; source < atoms; ++source) {
            const double height =
                (xyz[3 * source] - plane[0]) * n[0] +
                (xyz[3 * source + 1] - plane[1]) * n[1] +
                (xyz[3 * source + 2] - plane[2]) * n[2];
            const double dx =
                xyz[3 * target] - (xyz[3 * source] - 2.0 * height * n[0]);
            const double dy =
                xyz[3 * target + 1] -
                (xyz[3 * source + 1] - 2.0 * height * n[1]);
            const double dz =
                xyz[3 * target + 2] -
                (xyz[3 * source + 2] - 2.0 * height * n[2]);
            const double radius2 = dx * dx + dy * dy + dz * dz;
            if (radius2 <= kCoincidentTolerance * kCoincidentTolerance) {
                valid = false;
                continue;
            }
            const double inverse_radius = 1.0 / std::sqrt(radius2);
            const double weighted = q[source] * inverse_radius;
            const double weighted_inverse_radius2 = weighted / radius2;
            target_potential += weighted;
            gx -= weighted_inverse_radius2 * dx;
            gy -= weighted_inverse_radius2 * dy;
            gz -= weighted_inverse_radius2 * dz;
        }
        output_potential[target] = target_potential;
        output_gradient[3 * target] = gx;
        output_gradient[3 * target + 1] = gy;
        output_gradient[3 * target + 2] = gz;
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(coordinates);
    Py_DECREF(charges);
    Py_DECREF(origin);
    Py_DECREF(normal);
    if (!valid) {
        Py_DECREF(potential);
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_FloatingPointError, "singular planar image geometry");
        return nullptr;
    }
    return Py_BuildValue("NN", potential, gradient);
}

PyObject* planar_image_hessian_vector(PyObject*, PyObject* args) {
    PyObject *coordinates_object, *charges_object, *origin_object, *normal_object;
    PyObject* direction_object;
    double factor;
    if (!PyArg_ParseTuple(
            args, "OOOOOd", &coordinates_object, &charges_object, &origin_object,
            &normal_object, &direction_object, &factor)) {
        return nullptr;
    }
    PyArrayObject* coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* charges = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(charges_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* origin = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(origin_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* normal = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(normal_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* direction = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(direction_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (coordinates == nullptr || charges == nullptr || origin == nullptr ||
        normal == nullptr || direction == nullptr) {
        Py_XDECREF(coordinates);
        Py_XDECREF(charges);
        Py_XDECREF(origin);
        Py_XDECREF(normal);
        Py_XDECREF(direction);
        return nullptr;
    }
    const npy_intp atoms =
        PyArray_NDIM(coordinates) == 2 ? PyArray_DIM(coordinates, 0) : -1;
    const int direction_rank = PyArray_NDIM(direction);
    const npy_intp direction_count =
        direction_rank == 2 ? 1 :
        (direction_rank == 3 ? PyArray_DIM(direction, 0) : -1);
    const npy_intp direction_atoms =
        direction_rank == 2 ? PyArray_DIM(direction, 0) :
        (direction_rank == 3 ? PyArray_DIM(direction, 1) : -1);
    const npy_intp direction_axes =
        direction_rank == 2 ? PyArray_DIM(direction, 1) :
        (direction_rank == 3 ? PyArray_DIM(direction, 2) : -1);
    const bool valid_shapes =
        atoms >= 0 && PyArray_DIM(coordinates, 1) == 3 &&
        PyArray_NDIM(charges) == 1 && PyArray_DIM(charges, 0) == atoms &&
        PyArray_NDIM(origin) == 1 && PyArray_DIM(origin, 0) == 3 &&
        PyArray_NDIM(normal) == 1 && PyArray_DIM(normal, 0) == 3 &&
        direction_count >= 1 && direction_atoms == atoms &&
        direction_axes == 3 && std::isfinite(factor);
    if (!valid_shapes) {
        PyErr_SetString(PyExc_ValueError, "native planar image HVP arrays are inconsistent");
        Py_DECREF(coordinates);
        Py_DECREF(charges);
        Py_DECREF(origin);
        Py_DECREF(normal);
        Py_DECREF(direction);
        return nullptr;
    }
    npy_intp dimensions[3] = {direction_count, atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(
            direction_rank,
            direction_rank == 2 ? dimensions + 1 : dimensions,
            NPY_DOUBLE,
            0));
    if (product == nullptr) {
        Py_DECREF(coordinates);
        Py_DECREF(charges);
        Py_DECREF(origin);
        Py_DECREF(normal);
        Py_DECREF(direction);
        return nullptr;
    }
    const double* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const double* q = static_cast<const double*>(PyArray_DATA(charges));
    const double* plane = static_cast<const double*>(PyArray_DATA(origin));
    const double* n = static_cast<const double*>(PyArray_DATA(normal));
    const double* vector = static_cast<const double*>(PyArray_DATA(direction));
    double* output = static_cast<double*>(PyArray_DATA(product));
    bool valid = true;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp density = 0; density < direction_count; ++density) {
      const npy_intp density_offset = 3 * density * atoms;
      for (npy_intp target = 0; target < atoms; ++target) {
        double hx = 0.0, hy = 0.0, hz = 0.0;
        for (npy_intp source = 0; source < atoms; ++source) {
            const double height =
                (xyz[3 * source] - plane[0]) * n[0] +
                (xyz[3 * source + 1] - plane[1]) * n[1] +
                (xyz[3 * source + 2] - plane[2]) * n[2];
            const double dx =
                xyz[3 * target] - (xyz[3 * source] - 2.0 * height * n[0]);
            const double dy =
                xyz[3 * target + 1] -
                (xyz[3 * source + 1] - 2.0 * height * n[1]);
            const double dz =
                xyz[3 * target + 2] -
                (xyz[3 * source + 2] - 2.0 * height * n[2]);
            const double radius2 = dx * dx + dy * dy + dz * dz;
            if (radius2 <= kCoincidentTolerance * kCoincidentTolerance) {
                valid = false;
                continue;
            }
            const double projection =
                vector[density_offset + 3 * source] * n[0] +
                vector[density_offset + 3 * source + 1] * n[1] +
                vector[density_offset + 3 * source + 2] * n[2];
            const double ex =
                vector[density_offset + 3 * target] -
                (vector[density_offset + 3 * source] -
                 2.0 * projection * n[0]);
            const double ey =
                vector[density_offset + 3 * target + 1] -
                (vector[density_offset + 3 * source + 1] -
                 2.0 * projection * n[1]);
            const double ez =
                vector[density_offset + 3 * target + 2] -
                (vector[density_offset + 3 * source + 2] -
                 2.0 * projection * n[2]);
            const double inverse_radius = 1.0 / std::sqrt(radius2);
            const double inverse_radius3 = inverse_radius / radius2;
            const double dot = dx * ex + dy * ey + dz * ez;
            const double coefficient =
                3.0 * dot * inverse_radius3 / radius2;
            hx += q[source] * (coefficient * dx - inverse_radius3 * ex);
            hy += q[source] * (coefficient * dy - inverse_radius3 * ey);
            hz += q[source] * (coefficient * dz - inverse_radius3 * ez);
        }
        const double scale = factor * q[target];
        output[density_offset + 3 * target] = scale * hx;
        output[density_offset + 3 * target + 1] = scale * hy;
        output[density_offset + 3 * target + 2] = scale * hz;
      }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(coordinates);
    Py_DECREF(charges);
    Py_DECREF(origin);
    Py_DECREF(normal);
    Py_DECREF(direction);
    if (!valid) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_FloatingPointError, "singular planar image geometry");
        return nullptr;
    }
    return reinterpret_cast<PyObject*>(product);
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
        "precision", "float64");
}

PyMethodDef methods[] = {
    {"direct_gaussian_energy", direct_gaussian_energy, METH_VARARGS,
     "Direct Gaussian electrostatic energy."},
    {"direct_gaussian_energy_gradient", direct_gaussian_energy_gradient, METH_VARARGS,
     "Direct Gaussian electrostatic energy and analytic gradient."},
    {"direct_gaussian_hessian_vector", direct_gaussian_hessian_vector, METH_VARARGS,
     "Direct Gaussian electrostatic analytic Hessian-vector product."},
    {"direct_gaussian_potential", direct_gaussian_potential, METH_VARARGS,
     "Direct Gaussian electrostatic matrix-vector product including self terms."},
    {"direct_gaussian_directional_potential", direct_gaussian_directional_potential,
     METH_VARARGS, "Directional derivative of a direct Gaussian matrix-vector product."},
    {"gaussian_correction_energy", gaussian_correction_energy, METH_VARARGS,
     "Gaussian penetration correction energy on an explicit pair list."},
    {"gaussian_correction_energy_gradient", gaussian_correction_energy_gradient,
     METH_VARARGS,
     "Gaussian penetration correction energy and gradient on a pair list."},
    {"gaussian_correction_hessian_vector", gaussian_correction_hessian_vector,
     METH_VARARGS,
     "Gaussian penetration correction Hessian-vector product on a pair list."},
    {"gaussian_correction_potential", gaussian_correction_potential, METH_VARARGS,
     "Gaussian penetration correction potential on a pair list."},
    {"sonic_local_pairs", sonic_local_pairs, METH_VARARGS,
     "Shared Exp-PE and Gaussian-penetration E/G/Hv pair loop."},
    {"exppe_energy", damped_exppe_energy, METH_VARARGS,
     "Undamped Exp-PE energy on a cutoff pair list."},
    {"exppe_energy_gradient", damped_exppe_energy_gradient, METH_VARARGS,
     "Undamped Exp-PE energy and analytic gradient."},
    {"exppe_hessian_vector", damped_exppe_hessian_vector, METH_VARARGS,
     "Undamped Exp-PE analytic Hessian-vector product."},
    {"damped_exppe_energy", damped_exppe_energy, METH_VARARGS,
     "Undamped Exp-PE energy on a cutoff pair list."},
    {"damped_exppe_energy_gradient", damped_exppe_energy_gradient, METH_VARARGS,
     "Undamped Exp-PE energy and analytic gradient."},
    {"damped_exppe_hessian_vector", damped_exppe_hessian_vector, METH_VARARGS,
     "Undamped Exp-PE analytic Hessian-vector product."},
    {"switched_lj_energy_gradient", switched_lj_energy_gradient, METH_VARARGS,
     "C2-switched Lennard-Jones energy and analytic gradient."},
    {"morse_bond_energy", morse_bond_energy, METH_VARARGS,
     "Morse bond energy."},
    {"morse_bond_energy_gradient", morse_bond_energy_gradient, METH_VARARGS,
     "Morse bond energy and analytic Cartesian gradient."},
    {"morse_bond_hessian_vector", morse_bond_hessian_vector, METH_VARARGS,
     "Morse bond analytic Cartesian Hessian-vector product."},
    {"gaussian_bond_energy", gaussian_bond_energy, METH_VARARGS,
     "GFN-FF Gaussian bond energy."},
    {"gaussian_bond_energy_gradient", gaussian_bond_energy_gradient, METH_VARARGS,
     "GFN-FF Gaussian bond energy and analytic Cartesian gradient."},
    {"gaussian_bond_hessian_vector", gaussian_bond_hessian_vector, METH_VARARGS,
     "GFN-FF Gaussian bond analytic Cartesian Hessian-vector product."},
    {"dynamic_gaussian_bond_energy_gradient", dynamic_gaussian_bond_energy_gradient,
     METH_VARARGS, "Dynamic-CN GFN-FF Gaussian bond energy and gradient."},
    {"dynamic_gaussian_bond_shared_coordination", dynamic_gaussian_bond_shared_coordination,
     METH_VARARGS, "GFN-FF bond E/G and shared coordination state for EEQ and D3."},
    {"dynamic_gaussian_bond_hessian_vector", dynamic_gaussian_bond_hessian_vector,
     METH_VARARGS, "Dynamic-CN GFN-FF Gaussian bond Hessian-vector product."},
    {"gfnff_repulsion_energy_gradient", gfnff_repulsion_energy_gradient,
     METH_VARARGS, "Exact sparse GFN-FF repulsion energy and gradient."},
    {"gfnff_repulsion_hessian_vector", gfnff_repulsion_hessian_vector,
     METH_VARARGS, "Exact sparse GFN-FF repulsion Hessian-vector product."},
    {"gfnff_d3_reference_derivatives", gfnff_d3_reference_derivatives,
     METH_VARARGS, "Sparse GFN-FF D3 reference-C6 interpolation derivatives."},
    {"gfnff_valence_energy_gradient", gfnff_valence_energy_gradient,
     METH_VARARGS, "Exact local GFN-FF angle and torsion energy and gradient."},
    {"gfnff_valence_hessian_vector", gfnff_valence_hessian_vector,
     METH_VARARGS, "Exact local GFN-FF angle and torsion Hessian-vector product."},
    {"gfnff_nci_energy_gradient", gfnff_nci_energy_gradient,
     METH_VARARGS, "Sparse GFN-FF hydrogen/halogen-bond and bonded-ATM energy and gradient."},
    {"gfnff_nci_hessian_vector", gfnff_nci_hessian_vector,
     METH_VARARGS, "Sparse GFN-FF hydrogen/halogen-bond and bonded-ATM Hessian-vector product."},
    {"local_valence_energy", local_valence_energy, METH_VARARGS,
     "Bond-order-damped angle and torsion energy."},
    {"local_valence_energy_gradient", local_valence_energy_gradient, METH_VARARGS,
     "Bond-order-damped angle and torsion energy and analytic gradient."},
    {"local_valence_hessian_vector", local_valence_hessian_vector, METH_VARARGS,
     "Bond-order-damped angle and torsion analytic Hessian-vector product."},
    {"planar_image_potential_gradient", planar_image_potential_gradient, METH_VARARGS,
     "Direct planar image potential and target gradient."},
    {"planar_image_hessian_vector", planar_image_hessian_vector, METH_VARARGS,
     "Direct planar image analytic Hessian-vector product."},
    {"build_info", build_info, METH_NOARGS, "Describe this native build."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_zaff_native",
    "Portable compiled ZAFF numerical kernels.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__zaff_native() {
    import_array();
    return PyModule_Create(&module);
}
