#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_24_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <vector>

namespace {

using Complex = std::complex<double>;

double wigner_3j_value(const std::array<int, 6>& v) {
    const auto [j1, j2, j3, m1, m2, m3] = v;
    if (m1 + m2 + m3 != 0) return 0.0;
    const std::array<std::pair<int, int>, 3> jm_pair{{{j1,m1},{j2,m2},{j3,m3}}};
    for (const auto& [j, m] : jm_pair)
        if (j < 0 || std::abs(m) > j || ((j - m) & 1)) return 0.0;
    if (j3 < std::abs(j1-j2) || j3 > j1+j2 || ((j1+j2+j3)&1)) return 0.0;
    const int a=(j1+j2-j3)/2, b=(j1-j2+j3)/2, c=(-j1+j2+j3)/2;
    const int sum=(j1+j2+j3)/2;
    const std::array<int,6> jm{{(j1+m1)/2,(j1-m1)/2,(j2+m2)/2,
                               (j2-m2)/2,(j3+m3)/2,(j3-m3)/2}};
    long double log_norm=0.5L*(std::lgammal(a+1.0L)+std::lgammal(b+1.0L)+
        std::lgammal(c+1.0L)-std::lgammal(sum+2.0L));
    for (int x: jm) log_norm += 0.5L*std::lgammal(x+1.0L);
    const int bb=(j1-m1)/2, bc=(j2+m2)/2;
    const int od=(j3-j2+m1)/2, oe=(j3-j1-m2)/2;
    const int zmin=std::max({0,-od,-oe}), zmax=std::min({a,bb,bc});
    if (zmin>zmax) return 0.0;
    std::vector<std::pair<int,long double>> terms;
    long double maximum=-INFINITY;
    for (int z=zmin; z<=zmax; ++z) {
        const std::array<int,6> denominator{{z,a-z,bb-z,bc-z,od+z,oe+z}};
        long double value=0.0L;
        for (int x: denominator) value -= std::lgammal(x+1.0L);
        terms.push_back({z,value}); maximum=std::max(maximum,value);
    }
    long double signed_sum=0.0L;
    for (const auto& [z,value]: terms)
        signed_sum += (z&1 ? -1.0L : 1.0L)*std::exp(value-maximum);
    const int exponent=(j1-j2-m3)/2;
    return static_cast<double>((exponent&1 ? -1.0L : 1.0L)*
        std::exp(log_norm+maximum)*signed_sum);
}

long double log_delta(const int x, const int y, const int z) {
    const int p=(x+y-z)/2, q=(x-y+z)/2, r=(-x+y+z)/2, s=(x+y+z)/2;
    return 0.5L*(std::lgammal(p+1.0L)+std::lgammal(q+1.0L)+
        std::lgammal(r+1.0L)-std::lgammal(s+2.0L));
}

double wigner_6j_value(const std::array<int, 6>& v) {
    const auto [a,b,c,d,e,f]=v;
    for (int x: v) if (x<0) return 0.0;
    const std::array<std::array<int,3>,4> triangles{{{{a,b,c}},{{a,e,f}},{{d,b,f}},{{d,e,c}}}};
    for (const auto& t: triangles)
        if (t[2]<std::abs(t[0]-t[1]) || t[2]>t[0]+t[1] || ((t[0]+t[1]+t[2])&1)) return 0.0;
    const std::array<int,4> xs{{(a+b+c)/2,(a+e+f)/2,(d+b+f)/2,(d+e+c)/2}};
    const std::array<int,3> ys{{(a+b+d+e)/2,(a+c+d+f)/2,(b+c+e+f)/2}};
    const int zmin=*std::max_element(xs.begin(),xs.end());
    const int zmax=*std::min_element(ys.begin(),ys.end());
    if (zmin>zmax) return 0.0;
    std::vector<std::pair<int,long double>> terms;
    long double maximum=-INFINITY;
    for (int z=zmin; z<=zmax; ++z) {
        long double value=std::lgammal(z+2.0L);
        for (int x: xs) value-=std::lgammal(z-x+1.0L);
        for (int y: ys) value-=std::lgammal(y-z+1.0L);
        terms.push_back({z,value}); maximum=std::max(maximum,value);
    }
    long double signed_sum=0.0L;
    for (const auto& [z,value]: terms)
        signed_sum+=(z&1 ? -1.0L : 1.0L)*std::exp(value-maximum);
    long double prefactor=0.0L;
    for (const auto& t: triangles) prefactor+=log_delta(t[0],t[1],t[2]);
    return static_cast<double>(std::exp(prefactor+maximum)*signed_sum);
}

PyObject* wigner_3j_doubled_native(PyObject*, PyObject* args) {
    std::array<int,6> values;
    if (!PyArg_ParseTuple(args,"iiiiii",&values[0],&values[1],&values[2],
                          &values[3],&values[4],&values[5])) return nullptr;
    return PyFloat_FromDouble(wigner_3j_value(values));
}

PyObject* wigner_6j_doubled_native(PyObject*, PyObject* args) {
    std::array<int,6> values;
    if (!PyArg_ParseTuple(args,"iiiiii",&values[0],&values[1],&values[2],
                          &values[3],&values[4],&values[5])) return nullptr;
    return PyFloat_FromDouble(wigner_6j_value(values));
}

inline Complex load_complex(const npy_cdouble value) {
    return {npy_creal(value), npy_cimag(value)};
}

inline npy_cdouble store_complex(const Complex value) {
    return npy_cpack(value.real(), value.imag());
}

struct BlockTerm {
    npy_intp row_offset;
    npy_intp column_offset;
    npy_intp rows;
    npy_intp columns;
    Complex scale;
    PyArrayObject* matrix;
};

void release_block_terms(std::vector<BlockTerm>& terms) {
    for (auto& term : terms) {
        Py_XDECREF(term.matrix);
        term.matrix = nullptr;
    }
}

bool append_block_terms(
    PyObject* source,
    const npy_intp dimension,
    const bool scaled,
    std::vector<BlockTerm>& terms
) {
    PyObject* sequence = PySequence_Fast(source, "LOCK block terms must be a sequence");
    if (sequence == nullptr) {
        return false;
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence);
    for (Py_ssize_t index = 0; index < count; ++index) {
        PyObject* item = PySequence_Fast_GET_ITEM(sequence, index);
        const Py_ssize_t expected = scaled ? 4 : 2;
        if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != expected) {
            PyErr_SetString(
                PyExc_ValueError,
                scaled
                    ? "coupling terms must be (row, column, scale, matrix)"
                    : "diagonal terms must be (offset, matrix)"
            );
            Py_DECREF(sequence);
            return false;
        }
        const npy_intp row = PyLong_AsSsize_t(PyTuple_GET_ITEM(item, 0));
        const npy_intp column = scaled
            ? PyLong_AsSsize_t(PyTuple_GET_ITEM(item, 1))
            : row;
        if (PyErr_Occurred()) {
            Py_DECREF(sequence);
            return false;
        }
        Complex scale(1.0, 0.0);
        PyObject* matrix_object = PyTuple_GET_ITEM(item, scaled ? 3 : 1);
        if (scaled) {
            PyObject* scale_object = PyTuple_GET_ITEM(item, 2);
            const double real = PyComplex_RealAsDouble(scale_object);
            const double imag = PyComplex_ImagAsDouble(scale_object);
            if (PyErr_Occurred()) {
                Py_DECREF(sequence);
                return false;
            }
            scale = Complex(real, imag);
        }
        PyArrayObject* matrix = reinterpret_cast<PyArrayObject*>(
            PyArray_FROM_OTF(matrix_object, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
        if (matrix == nullptr) {
            Py_DECREF(sequence);
            return false;
        }
        const bool valid =
            PyArray_NDIM(matrix) == 2 && row >= 0 && column >= 0 &&
            row + PyArray_DIM(matrix, 0) <= dimension &&
            column + PyArray_DIM(matrix, 1) <= dimension;
        if (!valid) {
            PyErr_SetString(PyExc_ValueError, "LOCK block term exceeds Hamiltonian dimensions");
            Py_DECREF(matrix);
            Py_DECREF(sequence);
            return false;
        }
        terms.push_back(BlockTerm{
            row,
            column,
            PyArray_DIM(matrix, 0),
            PyArray_DIM(matrix, 1),
            scale,
            matrix,
        });
    }
    Py_DECREF(sequence);
    return true;
}

PyObject* assemble_spin_hyperfine(PyObject*, PyObject* args) {
    Py_ssize_t dimension_value = 0;
    PyObject* diagonal_object = nullptr;
    PyObject* coupling_object = nullptr;
    if (!PyArg_ParseTuple(
            args, "nOO", &dimension_value, &diagonal_object, &coupling_object)) {
        return nullptr;
    }
    if (dimension_value <= 0) {
        PyErr_SetString(PyExc_ValueError, "LOCK spin/hyperfine dimension must be positive");
        return nullptr;
    }
    const npy_intp dimension = static_cast<npy_intp>(dimension_value);
    std::vector<BlockTerm> terms;
    if (!append_block_terms(diagonal_object, dimension, false, terms) ||
        !append_block_terms(coupling_object, dimension, true, terms)) {
        release_block_terms(terms);
        return nullptr;
    }
    npy_intp shape[2] = {dimension, dimension};
    PyArrayObject* output = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, shape, NPY_CDOUBLE, 0));
    if (output == nullptr) {
        release_block_terms(terms);
        return nullptr;
    }
    npy_cdouble* output_data = static_cast<npy_cdouble*>(PyArray_DATA(output));

    Py_BEGIN_ALLOW_THREADS
    for (const auto& term : terms) {
        const npy_cdouble* block =
            static_cast<const npy_cdouble*>(PyArray_DATA(term.matrix));
        for (npy_intp row = 0; row < term.rows; ++row) {
            for (npy_intp column = 0; column < term.columns; ++column) {
                const npy_intp output_index =
                    (term.row_offset + row) * dimension + term.column_offset + column;
                const npy_intp block_index = row * term.columns + column;
                const Complex value =
                    load_complex(output_data[output_index]) +
                    term.scale * load_complex(block[block_index]);
                output_data[output_index] = store_complex(value);
            }
        }
    }
    Py_END_ALLOW_THREADS

    release_block_terms(terms);
    return reinterpret_cast<PyObject*>(output);
}

PyObject* assemble_lam_rotation(PyObject*, PyObject* args) {
    PyObject* energies_object = nullptr;
    PyObject* linear_object = nullptr;
    PyObject* quadratic_object = nullptr;
    PyObject* rotational_object = nullptr;
    PyObject* distortion_object = Py_None;
    if (!PyArg_ParseTuple(
            args, "OOOO|O", &energies_object, &linear_object, &quadratic_object,
            &rotational_object, &distortion_object)) {
        return nullptr;
    }
    PyArrayObject* energies = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(energies_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* linear = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(linear_object, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* quadratic = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(quadratic_object, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* rotational = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(rotational_object, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* distortion = nullptr;
    if (distortion_object != Py_None) {
        distortion = reinterpret_cast<PyArrayObject*>(
            PyArray_FROM_OTF(distortion_object, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
    }
    if (energies == nullptr || linear == nullptr || quadratic == nullptr ||
        rotational == nullptr || (distortion_object != Py_None && distortion == nullptr)) {
        Py_XDECREF(energies);
        Py_XDECREF(linear);
        Py_XDECREF(quadratic);
        Py_XDECREF(rotational);
        Py_XDECREF(distortion);
        return nullptr;
    }
    const npy_intp nlam = PyArray_DIM(energies, 0);
    const npy_intp nrot = PyArray_DIM(rotational, 1);
    const bool valid =
        PyArray_NDIM(energies) == 1 &&
        PyArray_NDIM(linear) == 3 && PyArray_DIM(linear, 0) == 3 &&
        PyArray_DIM(linear, 1) == nlam && PyArray_DIM(linear, 2) == nlam &&
        PyArray_NDIM(quadratic) == 4 && PyArray_DIM(quadratic, 0) == 3 &&
        PyArray_DIM(quadratic, 1) == 3 && PyArray_DIM(quadratic, 2) == nlam &&
        PyArray_DIM(quadratic, 3) == nlam &&
        PyArray_NDIM(rotational) == 3 && PyArray_DIM(rotational, 0) == 3 &&
        PyArray_DIM(rotational, 2) == nrot &&
        (distortion == nullptr ||
         (PyArray_NDIM(distortion) == 2 && PyArray_DIM(distortion, 0) == nrot &&
          PyArray_DIM(distortion, 1) == nrot));
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "inconsistent LOCK Hamiltonian array shapes");
        Py_DECREF(energies);
        Py_DECREF(linear);
        Py_DECREF(quadratic);
        Py_DECREF(rotational);
        Py_XDECREF(distortion);
        return nullptr;
    }
    const npy_intp dimension = nlam * nrot;
    npy_intp shape[2] = {dimension, dimension};
    PyArrayObject* output = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, shape, NPY_CDOUBLE, 0));
    if (output == nullptr) {
        Py_DECREF(energies);
        Py_DECREF(linear);
        Py_DECREF(quadratic);
        Py_DECREF(rotational);
        Py_XDECREF(distortion);
        return nullptr;
    }
    const double* energy_data = static_cast<const double*>(PyArray_DATA(energies));
    const npy_cdouble* linear_data =
        static_cast<const npy_cdouble*>(PyArray_DATA(linear));
    const npy_cdouble* quadratic_data =
        static_cast<const npy_cdouble*>(PyArray_DATA(quadratic));
    const npy_cdouble* rotational_data =
        static_cast<const npy_cdouble*>(PyArray_DATA(rotational));
    const npy_cdouble* distortion_data = distortion == nullptr
        ? nullptr
        : static_cast<const npy_cdouble*>(PyArray_DATA(distortion));
    npy_cdouble* output_data = static_cast<npy_cdouble*>(PyArray_DATA(output));
    std::vector<Complex> symrot(9 * nrot * nrot, Complex(0.0, 0.0));

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp a = 0; a < 3; ++a) {
        for (npy_intp b = 0; b < 3; ++b) {
            for (npy_intp r = 0; r < nrot; ++r) {
                for (npy_intp s = 0; s < nrot; ++s) {
                    Complex value(0.0, 0.0);
                    for (npy_intp t = 0; t < nrot; ++t) {
                        const Complex art = load_complex(
                            rotational_data[(a * nrot + r) * nrot + t]);
                        const Complex bts = load_complex(
                            rotational_data[(b * nrot + t) * nrot + s]);
                        const Complex brt = load_complex(
                            rotational_data[(b * nrot + r) * nrot + t]);
                        const Complex ats = load_complex(
                            rotational_data[(a * nrot + t) * nrot + s]);
                        value += 0.5 * (art * bts + brt * ats);
                    }
                    symrot[((a * 3 + b) * nrot + r) * nrot + s] = value;
                }
            }
        }
    }
    for (npy_intp lm = 0; lm < nlam; ++lm) {
        for (npy_intp ln = 0; ln < nlam; ++ln) {
            for (npy_intp r = 0; r < nrot; ++r) {
                for (npy_intp s = 0; s < nrot; ++s) {
                    Complex value(0.0, 0.0);
                    if (lm == ln && r == s) {
                        value += energy_data[lm];
                    }
                    if (lm == ln && distortion_data != nullptr) {
                        value += load_complex(distortion_data[r * nrot + s]);
                    }
                    for (npy_intp a = 0; a < 3; ++a) {
                        value += load_complex(
                            linear_data[(a * nlam + lm) * nlam + ln]) *
                            load_complex(rotational_data[(a * nrot + r) * nrot + s]);
                        for (npy_intp b = 0; b < 3; ++b) {
                            value += load_complex(
                                quadratic_data[
                                    (((a * 3 + b) * nlam + lm) * nlam + ln)]) *
                                symrot[((a * 3 + b) * nrot + r) * nrot + s];
                        }
                    }
                    const npy_intp row = lm * nrot + r;
                    const npy_intp column = ln * nrot + s;
                    output_data[row * dimension + column] = store_complex(value);
                }
            }
        }
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(energies);
    Py_DECREF(linear);
    Py_DECREF(quadratic);
    Py_DECREF(rotational);
    Py_XDECREF(distortion);
    return reinterpret_cast<PyObject*>(output);
}

PyObject* hellmann_feynman_diagonal(PyObject*, PyObject* args) {
    PyObject* vectors_object = nullptr;
    PyObject* operators_object = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &vectors_object, &operators_object)) {
        return nullptr;
    }
    PyArrayObject* vectors = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(vectors_object, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* operators = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(operators_object, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
    if (vectors == nullptr || operators == nullptr) {
        Py_XDECREF(vectors);
        Py_XDECREF(operators);
        return nullptr;
    }
    const bool valid =
        PyArray_NDIM(vectors) == 2 && PyArray_NDIM(operators) == 3 &&
        PyArray_DIM(operators, 1) == PyArray_DIM(vectors, 0) &&
        PyArray_DIM(operators, 2) == PyArray_DIM(vectors, 0);
    if (!valid) {
        PyErr_SetString(
            PyExc_ValueError,
            "Hellmann-Feynman operators must have shape (noperator, nbasis, nbasis)"
        );
        Py_DECREF(vectors);
        Py_DECREF(operators);
        return nullptr;
    }
    const npy_intp nbasis = PyArray_DIM(vectors, 0);
    const npy_intp nstate = PyArray_DIM(vectors, 1);
    const npy_intp noperator = PyArray_DIM(operators, 0);
    npy_intp shape[2] = {noperator, nstate};
    PyArrayObject* output = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, shape, NPY_DOUBLE, 0));
    if (output == nullptr) {
        Py_DECREF(vectors);
        Py_DECREF(operators);
        return nullptr;
    }
    const npy_cdouble* v = static_cast<const npy_cdouble*>(PyArray_DATA(vectors));
    const npy_cdouble* op = static_cast<const npy_cdouble*>(PyArray_DATA(operators));
    double* result = static_cast<double*>(PyArray_DATA(output));
    Py_BEGIN_ALLOW_THREADS
    std::vector<Complex> accumulated(nstate);
    for (npy_intp p = 0; p < noperator; ++p) {
        std::fill(accumulated.begin(), accumulated.end(), Complex(0.0, 0.0));
        for (npy_intp row = 0; row < nbasis; ++row) {
            for (npy_intp column = 0; column < nbasis; ++column) {
                const Complex operator_value =
                    load_complex(op[(p * nbasis + row) * nbasis + column]);
                if (operator_value == Complex(0.0, 0.0)) {
                    continue;
                }
                for (npy_intp state = 0; state < nstate; ++state) {
                    accumulated[state] +=
                        std::conj(load_complex(v[row * nstate + state])) *
                        operator_value * load_complex(v[column * nstate + state]);
                }
            }
        }
        for (npy_intp state = 0; state < nstate; ++state) {
            result[p * nstate + state] = accumulated[state].real();
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(vectors);
    Py_DECREF(operators);
    return reinterpret_cast<PyObject*>(output);
}

struct RAMFactorTerm {
    Complex scale;
    PyArrayObject* lam;
    PyArrayObject* rot;
};

void release_ram_terms(std::vector<RAMFactorTerm>& terms) {
    for (auto& term : terms) {
        Py_XDECREF(term.lam);
        Py_XDECREF(term.rot);
    }
}

bool parse_ram_terms(
    PyObject* source,
    const npy_intp nprim,
    const npy_intp nrot,
    std::vector<RAMFactorTerm>& terms
) {
    PyObject* sequence = PySequence_Fast(source, "RAM factors must be a sequence");
    if (sequence == nullptr) return false;
    for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(sequence); ++index) {
        PyObject* item = PySequence_Fast_GET_ITEM(sequence, index);
        if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 3) {
            PyErr_SetString(PyExc_ValueError, "each RAM factor must be (scale, lam, rot)");
            Py_DECREF(sequence);
            return false;
        }
        const Complex scale(
            PyComplex_RealAsDouble(PyTuple_GET_ITEM(item, 0)),
            PyComplex_ImagAsDouble(PyTuple_GET_ITEM(item, 0)));
        if (PyErr_Occurred()) {
            Py_DECREF(sequence);
            return false;
        }
        auto* lam = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            PyTuple_GET_ITEM(item, 1), NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
        auto* rot = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            PyTuple_GET_ITEM(item, 2), NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
        const bool valid = lam != nullptr && rot != nullptr &&
            PyArray_NDIM(lam) == 2 && PyArray_DIM(lam, 0) == nprim &&
            PyArray_DIM(lam, 1) == nprim && PyArray_NDIM(rot) == 2 &&
            PyArray_DIM(rot, 0) == nrot && PyArray_DIM(rot, 1) == nrot;
        if (!valid) {
            PyErr_SetString(PyExc_ValueError, "inconsistent RAM factor dimensions");
            Py_XDECREF(lam);
            Py_XDECREF(rot);
            Py_DECREF(sequence);
            return false;
        }
        terms.push_back({scale, lam, rot});
    }
    Py_DECREF(sequence);
    return true;
}

inline Complex contracted_ram_element(
    const npy_cdouble* contractions,
    const std::vector<Complex>& projected,
    const npy_intp nprim,
    const npy_intp nstate,
    const npy_intp left,
    const npy_intp right,
    const npy_intp a,
    const npy_intp b
) {
    Complex value(0.0, 0.0);
    for (npy_intp mu = 0; mu < nprim; ++mu) {
        value += std::conj(load_complex(
            contractions[(left * nprim + mu) * nstate + a])) *
            projected[(right * nprim + mu) * nstate + b];
    }
    return value;
}

std::vector<Complex> project_ram_contractions(
    const npy_cdouble* contractions,
    const npy_cdouble* lam,
    const npy_intp nrot,
    const npy_intp nprim,
    const npy_intp nstate
) {
    std::vector<Complex> projected(nrot * nprim * nstate, Complex(0.0, 0.0));
    for (npy_intp right = 0; right < nrot; ++right)
        for (npy_intp mu = 0; mu < nprim; ++mu)
            for (npy_intp nu = 0; nu < nprim; ++nu) {
                const Complex element = load_complex(lam[mu * nprim + nu]);
                if (element == Complex(0.0, 0.0)) continue;
                for (npy_intp b = 0; b < nstate; ++b)
                    projected[(right * nprim + mu) * nstate + b] += element *
                        load_complex(contractions[(right * nprim + nu) * nstate + b]);
            }
    return projected;
}

PyObject* assemble_ram_factorized(PyObject*, PyObject* args) {
    PyObject* contractions_object = nullptr;
    PyObject* terms_object = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &contractions_object, &terms_object)) return nullptr;
    auto* contractions = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        contractions_object, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
    if (contractions == nullptr) return nullptr;
    if (PyArray_NDIM(contractions) != 3) {
        PyErr_SetString(PyExc_ValueError, "RAM contractions must have shape (nrot,nprim,nstate)");
        Py_DECREF(contractions);
        return nullptr;
    }
    const npy_intp nrot = PyArray_DIM(contractions, 0);
    const npy_intp nprim = PyArray_DIM(contractions, 1);
    const npy_intp nstate = PyArray_DIM(contractions, 2);
    std::vector<RAMFactorTerm> terms;
    if (!parse_ram_terms(terms_object, nprim, nrot, terms)) {
        release_ram_terms(terms);
        Py_DECREF(contractions);
        return nullptr;
    }
    const npy_intp dimension = nrot * nstate;
    npy_intp shape[2] = {dimension, dimension};
    auto* output = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2, shape, NPY_CDOUBLE, 0));
    if (output == nullptr) {
        release_ram_terms(terms);
        Py_DECREF(contractions);
        return nullptr;
    }
    const auto* c = static_cast<const npy_cdouble*>(PyArray_DATA(contractions));
    auto* result = static_cast<npy_cdouble*>(PyArray_DATA(output));
    Py_BEGIN_ALLOW_THREADS
    for (const auto& term : terms) {
        const auto* lam = static_cast<const npy_cdouble*>(PyArray_DATA(term.lam));
        const auto* rot = static_cast<const npy_cdouble*>(PyArray_DATA(term.rot));
        const auto projected = project_ram_contractions(c, lam, nrot, nprim, nstate);
        for (npy_intp left = 0; left < nrot; ++left) {
            for (npy_intp right = 0; right < nrot; ++right) {
                const Complex rotational = load_complex(rot[left * nrot + right]);
                if (std::abs(rotational) <= 1.0e-14) continue;
                for (npy_intp a = 0; a < nstate; ++a) {
                    for (npy_intp b = 0; b < nstate; ++b) {
                        const npy_intp row = left * nstate + a;
                        const npy_intp column = right * nstate + b;
                        const Complex value = load_complex(result[row * dimension + column]) +
                            term.scale * rotational * contracted_ram_element(
                                c, projected, nprim, nstate, left, right, a, b);
                        result[row * dimension + column] = store_complex(value);
                    }
                }
            }
        }
    }
    Py_END_ALLOW_THREADS
    release_ram_terms(terms);
    Py_DECREF(contractions);
    return reinterpret_cast<PyObject*>(output);
}

PyObject* ram_factorized_expectation(PyObject*, PyObject* args) {
    PyObject* vectors_object = nullptr;
    PyObject* contractions_object = nullptr;
    PyObject* lam_object = nullptr;
    PyObject* rot_object = nullptr;
    if (!PyArg_ParseTuple(
            args, "OOOO", &vectors_object, &contractions_object, &lam_object, &rot_object)) {
        return nullptr;
    }
    auto* vectors = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        vectors_object, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* contractions = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        contractions_object, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
    if (vectors == nullptr || contractions == nullptr || PyArray_NDIM(vectors) != 2 ||
        PyArray_NDIM(contractions) != 3) {
        PyErr_SetString(PyExc_ValueError, "invalid RAM vectors or contractions");
        Py_XDECREF(vectors);
        Py_XDECREF(contractions);
        return nullptr;
    }
    const npy_intp nrot = PyArray_DIM(contractions, 0);
    const npy_intp nprim = PyArray_DIM(contractions, 1);
    const npy_intp ncontracted = PyArray_DIM(contractions, 2);
    std::vector<RAMFactorTerm> terms;
    PyObject* single = Py_BuildValue("[(dOO)]", 1.0, lam_object, rot_object);
    const bool parsed = single != nullptr && parse_ram_terms(single, nprim, nrot, terms);
    Py_XDECREF(single);
    const bool valid = parsed && PyArray_DIM(vectors, 0) == nrot * ncontracted;
    if (!valid) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "inconsistent RAM expectation dimensions");
        release_ram_terms(terms);
        Py_DECREF(vectors);
        Py_DECREF(contractions);
        return nullptr;
    }
    const npy_intp nstates = PyArray_DIM(vectors, 1);
    npy_intp shape[1] = {nstates};
    auto* output = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(1, shape, NPY_DOUBLE, 0));
    if (output == nullptr) {
        release_ram_terms(terms);
        Py_DECREF(vectors);
        Py_DECREF(contractions);
        return nullptr;
    }
    const auto* c = static_cast<const npy_cdouble*>(PyArray_DATA(contractions));
    const auto* v = static_cast<const npy_cdouble*>(PyArray_DATA(vectors));
    const auto* lam = static_cast<const npy_cdouble*>(PyArray_DATA(terms[0].lam));
    const auto* rot = static_cast<const npy_cdouble*>(PyArray_DATA(terms[0].rot));
    auto* result = static_cast<double*>(PyArray_DATA(output));
    Py_BEGIN_ALLOW_THREADS
    std::vector<Complex> accumulated(nstates, Complex(0.0, 0.0));
    const auto projected = project_ram_contractions(
        c, lam, nrot, nprim, ncontracted);
    for (npy_intp left = 0; left < nrot; ++left) {
        for (npy_intp right = 0; right < nrot; ++right) {
            const Complex rotational = load_complex(rot[left * nrot + right]);
            if (std::abs(rotational) <= 1.0e-14) continue;
            for (npy_intp a = 0; a < ncontracted; ++a) {
                for (npy_intp b = 0; b < ncontracted; ++b) {
                    const Complex factor = rotational * contracted_ram_element(
                        c, projected, nprim, ncontracted, left, right, a, b);
                    const npy_intp row = left * ncontracted + a;
                    const npy_intp column = right * ncontracted + b;
                    for (npy_intp state = 0; state < nstates; ++state) {
                        accumulated[state] += std::conj(load_complex(v[row * nstates + state])) *
                            factor * load_complex(v[column * nstates + state]);
                    }
                }
            }
        }
    }
    for (npy_intp state = 0; state < nstates; ++state) result[state] = accumulated[state].real();
    Py_END_ALLOW_THREADS
    release_ram_terms(terms);
    Py_DECREF(vectors);
    Py_DECREF(contractions);
    return reinterpret_cast<PyObject*>(output);
}

PyObject* ram_factorized_expectations(PyObject*, PyObject* args) {
    PyObject* vectors_object = nullptr;
    PyObject* contractions_object = nullptr;
    PyObject* groups_object = nullptr;
    if (!PyArg_ParseTuple(
            args, "OOO", &vectors_object, &contractions_object, &groups_object)) {
        return nullptr;
    }
    auto* vectors = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        vectors_object, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* contractions = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        contractions_object, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
    if (vectors == nullptr || contractions == nullptr || PyArray_NDIM(vectors) != 2 ||
        PyArray_NDIM(contractions) != 3) {
        PyErr_SetString(PyExc_ValueError, "invalid RAM vectors or contractions");
        Py_XDECREF(vectors);
        Py_XDECREF(contractions);
        return nullptr;
    }
    const npy_intp nrot = PyArray_DIM(contractions, 0);
    const npy_intp nprim = PyArray_DIM(contractions, 1);
    const npy_intp ncontracted = PyArray_DIM(contractions, 2);
    if (PyArray_DIM(vectors, 0) != nrot * ncontracted) {
        PyErr_SetString(PyExc_ValueError, "inconsistent RAM expectation dimensions");
        Py_DECREF(vectors);
        Py_DECREF(contractions);
        return nullptr;
    }
    PyObject* groups = PySequence_Fast(groups_object, "RAM factor groups must be a sequence");
    if (groups == nullptr) {
        Py_DECREF(vectors);
        Py_DECREF(contractions);
        return nullptr;
    }
    const Py_ssize_t ngroups = PySequence_Fast_GET_SIZE(groups);
    std::vector<std::vector<RAMFactorTerm>> parsed_groups(static_cast<size_t>(ngroups));
    bool parsed = true;
    for (Py_ssize_t group = 0; group < ngroups; ++group) {
        if (!parse_ram_terms(
                PySequence_Fast_GET_ITEM(groups, group), nprim, nrot,
                parsed_groups[static_cast<size_t>(group)])) {
            parsed = false;
            break;
        }
    }
    Py_DECREF(groups);
    if (!parsed) {
        for (auto& terms : parsed_groups) release_ram_terms(terms);
        Py_DECREF(vectors);
        Py_DECREF(contractions);
        return nullptr;
    }

    const npy_intp nstates = PyArray_DIM(vectors, 1);
    npy_intp shape[2] = {static_cast<npy_intp>(ngroups), nstates};
    auto* output = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2, shape, NPY_DOUBLE, 0));
    if (output == nullptr) {
        for (auto& terms : parsed_groups) release_ram_terms(terms);
        Py_DECREF(vectors);
        Py_DECREF(contractions);
        return nullptr;
    }

    const auto* c = static_cast<const npy_cdouble*>(PyArray_DATA(contractions));
    const auto* v = static_cast<const npy_cdouble*>(PyArray_DATA(vectors));
    auto* result = static_cast<double*>(PyArray_DATA(output));
    Py_BEGIN_ALLOW_THREADS
    const npy_intp primitive_vector_size = nrot * nprim * nstates;
    std::vector<Complex> primitive_vectors(
        primitive_vector_size, Complex(0.0, 0.0));
    for (npy_intp rot_index = 0; rot_index < nrot; ++rot_index)
        for (npy_intp mu = 0; mu < nprim; ++mu)
            for (npy_intp state = 0; state < nstates; ++state) {
                Complex value(0.0, 0.0);
                for (npy_intp contracted = 0; contracted < ncontracted; ++contracted)
                    value += load_complex(
                        c[(rot_index * nprim + mu) * ncontracted + contracted]) *
                        load_complex(v[
                            (rot_index * ncontracted + contracted) * nstates + state]);
                primitive_vectors[(rot_index * nprim + mu) * nstates + state] = value;
            }
    for (npy_intp group = 0; group < static_cast<npy_intp>(ngroups); ++group) {
        std::vector<Complex> operated(primitive_vector_size, Complex(0.0, 0.0));
        for (const auto& term : parsed_groups[static_cast<size_t>(group)]) {
            const auto* lam = static_cast<const npy_cdouble*>(PyArray_DATA(term.lam));
            const auto* rot = static_cast<const npy_cdouble*>(PyArray_DATA(term.rot));
            std::vector<Complex> projected(primitive_vector_size, Complex(0.0, 0.0));
            for (npy_intp right = 0; right < nrot; ++right)
                for (npy_intp mu = 0; mu < nprim; ++mu)
                    for (npy_intp nu = 0; nu < nprim; ++nu) {
                        const Complex element = load_complex(lam[mu * nprim + nu]);
                        if (element == Complex(0.0, 0.0)) continue;
                        for (npy_intp state = 0; state < nstates; ++state)
                            projected[(right * nprim + mu) * nstates + state] += element *
                                primitive_vectors[(right * nprim + nu) * nstates + state];
                    }
            for (npy_intp left = 0; left < nrot; ++left) {
                for (npy_intp right = 0; right < nrot; ++right) {
                    const Complex rotational = term.scale *
                        load_complex(rot[left * nrot + right]);
                    if (std::abs(rotational) <= 1.0e-14) continue;
                    for (npy_intp mu = 0; mu < nprim; ++mu)
                        for (npy_intp state = 0; state < nstates; ++state)
                            operated[(left * nprim + mu) * nstates + state] += rotational *
                                projected[(right * nprim + mu) * nstates + state];
                }
            }
        }
        for (npy_intp state = 0; state < nstates; ++state) {
            Complex value(0.0, 0.0);
            for (npy_intp index = 0; index < nrot * nprim; ++index)
                value += std::conj(primitive_vectors[index * nstates + state]) *
                    operated[index * nstates + state];
            result[group * nstates + state] = value.real();
        }
    }
    Py_END_ALLOW_THREADS
    for (auto& terms : parsed_groups) release_ram_terms(terms);
    Py_DECREF(vectors);
    Py_DECREF(contractions);
    return reinterpret_cast<PyObject*>(output);
}

PyObject* ram_contracted_basis_responses(PyObject*, PyObject* args) {
    PyObject* vectors_object = nullptr;
    PyObject* contractions_object = nullptr;
    PyObject* derivatives_object = nullptr;
    PyObject* terms_object = nullptr;
    if (!PyArg_ParseTuple(
            args, "OOOO", &vectors_object, &contractions_object,
            &derivatives_object, &terms_object)) {
        return nullptr;
    }
    auto* vectors = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        vectors_object, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* contractions = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        contractions_object, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* derivatives = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        derivatives_object, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
    if (vectors == nullptr || contractions == nullptr || derivatives == nullptr ||
        PyArray_NDIM(vectors) != 2 || PyArray_NDIM(contractions) != 3 ||
        PyArray_NDIM(derivatives) != 4) {
        PyErr_SetString(PyExc_ValueError, "invalid RAM contracted-basis response arrays");
        Py_XDECREF(vectors);
        Py_XDECREF(contractions);
        Py_XDECREF(derivatives);
        return nullptr;
    }
    const npy_intp nresponse = PyArray_DIM(derivatives, 0);
    const npy_intp nrot = PyArray_DIM(contractions, 0);
    const npy_intp nprim = PyArray_DIM(contractions, 1);
    const npy_intp ncontracted = PyArray_DIM(contractions, 2);
    const bool valid =
        PyArray_DIM(vectors, 0) == nrot * ncontracted &&
        PyArray_DIM(derivatives, 1) == nrot &&
        PyArray_DIM(derivatives, 2) == nprim &&
        PyArray_DIM(derivatives, 3) == ncontracted;
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "inconsistent RAM contracted-basis response dimensions");
        Py_DECREF(vectors);
        Py_DECREF(contractions);
        Py_DECREF(derivatives);
        return nullptr;
    }
    std::vector<RAMFactorTerm> terms;
    if (!parse_ram_terms(terms_object, nprim, nrot, terms)) {
        release_ram_terms(terms);
        Py_DECREF(vectors);
        Py_DECREF(contractions);
        Py_DECREF(derivatives);
        return nullptr;
    }
    const npy_intp nstates = PyArray_DIM(vectors, 1);
    npy_intp shape[2] = {nresponse, nstates};
    auto* output = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2, shape, NPY_DOUBLE, 0));
    if (output == nullptr) {
        release_ram_terms(terms);
        Py_DECREF(vectors);
        Py_DECREF(contractions);
        Py_DECREF(derivatives);
        return nullptr;
    }

    const auto* c = static_cast<const npy_cdouble*>(PyArray_DATA(contractions));
    const auto* d = static_cast<const npy_cdouble*>(PyArray_DATA(derivatives));
    const auto* v = static_cast<const npy_cdouble*>(PyArray_DATA(vectors));
    auto* result = static_cast<double*>(PyArray_DATA(output));
    const npy_intp derivative_stride = nrot * nprim * ncontracted;
    Py_BEGIN_ALLOW_THREADS
    // Transform the eigenvectors into the primitive torsional basis once,
    // apply the complete factorized Hamiltonian there, and contract each
    // basis derivative with that common H|psi>.  This evaluates
    // 2 Re <dC psi | H | C psi> and avoids rebuilding a contracted block for
    // every factor, K pair, response parameter, and eigenstate.
    const npy_intp primitive_vector_size = nrot * nprim * nstates;
    std::vector<Complex> primitive_vectors(
        primitive_vector_size, Complex(0.0, 0.0));
    for (npy_intp rot_index = 0; rot_index < nrot; ++rot_index)
        for (npy_intp mu = 0; mu < nprim; ++mu)
            for (npy_intp state = 0; state < nstates; ++state) {
                Complex value(0.0, 0.0);
                for (npy_intp contracted = 0; contracted < ncontracted; ++contracted)
                    value += load_complex(
                        c[(rot_index * nprim + mu) * ncontracted + contracted]) *
                        load_complex(v[
                            (rot_index * ncontracted + contracted) * nstates + state]);
                primitive_vectors[(rot_index * nprim + mu) * nstates + state] = value;
            }
    std::vector<Complex> operated(primitive_vector_size, Complex(0.0, 0.0));
    for (const auto& term : terms) {
        const auto* lam = static_cast<const npy_cdouble*>(PyArray_DATA(term.lam));
        const auto* rot = static_cast<const npy_cdouble*>(PyArray_DATA(term.rot));
        std::vector<Complex> projected(primitive_vector_size, Complex(0.0, 0.0));
        for (npy_intp right = 0; right < nrot; ++right)
            for (npy_intp mu = 0; mu < nprim; ++mu)
                for (npy_intp nu = 0; nu < nprim; ++nu) {
                    const Complex element = load_complex(lam[mu * nprim + nu]);
                    if (element == Complex(0.0, 0.0)) continue;
                    for (npy_intp state = 0; state < nstates; ++state)
                        projected[(right * nprim + mu) * nstates + state] += element *
                            primitive_vectors[(right * nprim + nu) * nstates + state];
                }
        for (npy_intp left = 0; left < nrot; ++left) {
            for (npy_intp right = 0; right < nrot; ++right) {
                const Complex rotational = term.scale *
                    load_complex(rot[left * nrot + right]);
                if (std::abs(rotational) <= 1.0e-14) continue;
                for (npy_intp mu = 0; mu < nprim; ++mu)
                    for (npy_intp state = 0; state < nstates; ++state)
                        operated[(left * nprim + mu) * nstates + state] += rotational *
                            projected[(right * nprim + mu) * nstates + state];
            }
        }
    }
    for (npy_intp response = 0; response < nresponse; ++response) {
        const auto* derivative = d + response * derivative_stride;
        for (npy_intp state = 0; state < nstates; ++state) {
            Complex value(0.0, 0.0);
            for (npy_intp rot_index = 0; rot_index < nrot; ++rot_index)
                for (npy_intp mu = 0; mu < nprim; ++mu) {
                    Complex derivative_vector(0.0, 0.0);
                    for (npy_intp contracted = 0; contracted < ncontracted; ++contracted)
                        derivative_vector += load_complex(
                            derivative[(rot_index * nprim + mu) * ncontracted + contracted]) *
                            load_complex(v[
                                (rot_index * ncontracted + contracted) * nstates + state]);
                    value += std::conj(derivative_vector) *
                        operated[(rot_index * nprim + mu) * nstates + state];
                }
            result[response * nstates + state] = 2.0 * value.real();
        }
    }
    Py_END_ALLOW_THREADS
    release_ram_terms(terms);
    Py_DECREF(vectors);
    Py_DECREF(contractions);
    Py_DECREF(derivatives);
    return reinterpret_cast<PyObject*>(output);
}

PyObject* build_info(PyObject*, PyObject*) {
    return Py_BuildValue("{s:s,s:s}", "implementation", "compiled-c++17", "kernel", "lock");
}

PyObject* kronecker_sum_matvec(PyObject*, PyObject* args) {
    PyObject* terms_object = nullptr;
    PyObject* vector_object = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &terms_object, &vector_object)) return nullptr;
    PyArrayObject* vector = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(vector_object, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
    if (vector == nullptr) return nullptr;
    const int vector_rank = PyArray_NDIM(vector);
    if (vector_rank != 1 && vector_rank != 2) {
        PyErr_SetString(PyExc_ValueError, "LOCK Kronecker input must be a vector or column matrix");
        Py_DECREF(vector);
        return nullptr;
    }
    const npy_intp dimension = PyArray_DIM(vector, 0);
    const npy_intp vector_count = vector_rank == 1 ? 1 : PyArray_DIM(vector, 1);
    npy_intp shape[2] = {dimension, vector_count};
    PyArrayObject* output = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(vector_rank, shape, NPY_CDOUBLE, 0));
    PyObject* terms = PySequence_Fast(terms_object, "LOCK Kronecker terms must be a sequence");
    if (output == nullptr || terms == nullptr) {
        Py_XDECREF(output); Py_XDECREF(terms); Py_DECREF(vector); return nullptr;
    }
    const auto* source = static_cast<const npy_cdouble*>(PyArray_DATA(vector));
    auto* destination = static_cast<npy_cdouble*>(PyArray_DATA(output));
    for (Py_ssize_t term_index = 0; term_index < PySequence_Fast_GET_SIZE(terms); ++term_index) {
        PyObject* item = PySequence_Fast_GET_ITEM(terms, term_index);
        if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 2) {
            PyErr_SetString(PyExc_ValueError, "each Kronecker term must be (scale, factors)");
            Py_DECREF(terms); Py_DECREF(vector); Py_DECREF(output); return nullptr;
        }
        const Complex scale(
            PyComplex_RealAsDouble(PyTuple_GET_ITEM(item, 0)),
            PyComplex_ImagAsDouble(PyTuple_GET_ITEM(item, 0)));
        if (PyErr_Occurred()) {
            Py_DECREF(terms); Py_DECREF(vector); Py_DECREF(output); return nullptr;
        }
        PyObject* factor_sequence = PySequence_Fast(
            PyTuple_GET_ITEM(item, 1), "Kronecker factors must be a sequence");
        if (factor_sequence == nullptr) {
            Py_DECREF(terms); Py_DECREF(vector); Py_DECREF(output); return nullptr;
        }
        std::vector<PyArrayObject*> factors;
        std::vector<npy_intp> dims;
        npy_intp product = 1;
        bool valid = true;
        for (Py_ssize_t axis = 0; axis < PySequence_Fast_GET_SIZE(factor_sequence); ++axis) {
            auto* factor = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
                PySequence_Fast_GET_ITEM(factor_sequence, axis), NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY));
            if (factor == nullptr) { valid = false; break; }
            factors.push_back(factor);
            if (PyArray_NDIM(factor) != 2 || PyArray_DIM(factor, 0) != PyArray_DIM(factor, 1)) {
                PyErr_SetString(PyExc_ValueError, "Kronecker factors must be square");
                valid = false; break;
            }
            dims.push_back(PyArray_DIM(factor, 0));
            product *= dims.back();
        }
        Py_DECREF(factor_sequence);
        if (valid && product != dimension) {
            PyErr_SetString(PyExc_ValueError, "Kronecker factors and vector dimensions disagree");
            valid = false;
        }
        if (!valid) {
            for (auto* factor : factors) Py_XDECREF(factor);
            Py_DECREF(terms); Py_DECREF(vector); Py_DECREF(output); return nullptr;
        }
        std::vector<Complex> current(dimension * vector_count), next(dimension * vector_count);
        Py_BEGIN_ALLOW_THREADS
        for (npy_intp index = 0; index < dimension * vector_count; ++index)
            current[index] = load_complex(source[index]);
        for (size_t axis = 0; axis < factors.size(); ++axis) {
            const npy_intp n = dims[axis];
            npy_intp stride = 1;
            for (size_t right = axis + 1; right < dims.size(); ++right) stride *= dims[right];
            const npy_intp outer_count = dimension / (n * stride);
            const auto* factor = static_cast<const npy_cdouble*>(PyArray_DATA(factors[axis]));
            std::fill(next.begin(), next.end(), Complex(0.0, 0.0));
            for (npy_intp outer = 0; outer < outer_count; ++outer)
                for (npy_intp row = 0; row < n; ++row)
                    for (npy_intp inner = 0; inner < stride; ++inner)
                        for (npy_intp vector_index = 0; vector_index < vector_count; ++vector_index) {
                            Complex value(0.0, 0.0);
                            for (npy_intp column = 0; column < n; ++column)
                                value += load_complex(factor[row * n + column]) *
                                    current[((outer * n + column) * stride + inner) *
                                            vector_count + vector_index];
                            next[((outer * n + row) * stride + inner) * vector_count +
                                 vector_index] = value;
                        }
            current.swap(next);
        }
        for (npy_intp index = 0; index < dimension * vector_count; ++index)
            destination[index] = store_complex(load_complex(destination[index]) + scale * current[index]);
        Py_END_ALLOW_THREADS
        for (auto* factor : factors) Py_DECREF(factor);
    }
    Py_DECREF(terms);
    Py_DECREF(vector);
    return reinterpret_cast<PyObject*>(output);
}

PyMethodDef methods[] = {
    {"wigner_3j_doubled", wigner_3j_doubled_native, METH_VARARGS,
     "Evaluate a Wigner 3-j symbol from doubled integer quantum numbers."},
    {"wigner_6j_doubled", wigner_6j_doubled_native, METH_VARARGS,
     "Evaluate a Wigner 6-j symbol from doubled integer quantum numbers."},
    {"assemble_lam_rotation", assemble_lam_rotation, METH_VARARGS,
     "Assemble a contracted LOCK LAM-rotation Hamiltonian."},
    {"assemble_spin_hyperfine", assemble_spin_hyperfine, METH_VARARGS,
     "Assemble a symmetry-blocked LOCK spin/hyperfine Hamiltonian."},
    {"hellmann_feynman_diagonal", hellmann_feynman_diagonal, METH_VARARGS,
     "Evaluate diagonal Hellmann-Feynman expectation values."},
    {"assemble_ram_factorized", assemble_ram_factorized, METH_VARARGS,
     "Assemble a K-dependent contracted RAM Hamiltonian."},
    {"ram_factorized_expectation", ram_factorized_expectation, METH_VARARGS,
     "Evaluate a factorized RAM Hellmann-Feynman diagonal."},
    {"ram_factorized_expectations", ram_factorized_expectations, METH_VARARGS,
     "Evaluate a batch of factorized RAM Hellmann-Feynman diagonals."},
    {"ram_contracted_basis_responses", ram_contracted_basis_responses, METH_VARARGS,
     "Evaluate contracted-basis derivative responses for RAM factors."},
    {"kronecker_sum_matvec", kronecker_sum_matvec, METH_VARARGS,
     "Apply a sum of Kronecker products to one or many vectors without assembling it."},
    {"build_info", build_info, METH_NOARGS, "Describe this native LOCK build."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_lock_native",
    "Portable compiled LOCK Hamiltonian kernels.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__lock_native() {
    import_array();
    return PyModule_Create(&module);
}
