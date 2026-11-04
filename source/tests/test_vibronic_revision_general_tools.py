"""General process timing, Hessian export and product-state regressions."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

from matrix_core.process_timing import time_process
from matrix_dvr.product_mean_field import solve_product_mean_field
from matrix_orca.hessian_io import write_orca_hessian
from matrix_orca.parsers import hessian_input_from_orca_hessian_file


class ProcessTimingTests(unittest.TestCase):
    def test_success_and_failure(self):
        result = time_process([sys.executable, '-c', 'pass'], timeout_seconds=2)
        self.assertEqual(result.returncode, 0)
        self.assertGreater(result.wall_seconds, 0)
        with self.assertRaises(subprocess.CalledProcessError):
            time_process([sys.executable, '-c', 'raise SystemExit(3)'])
        result = time_process([sys.executable, '-c', 'raise SystemExit(3)'], check=False)
        self.assertEqual(result.returncode, 3)

    def test_timeout_and_bad_arguments(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            time_process([sys.executable, '-c', 'import time;time.sleep(10)'], timeout_seconds=.05)
        for kwargs in ({'timeout_seconds': 0}, {'stdout': subprocess.PIPE}):
            with self.assertRaises(ValueError):
                time_process([sys.executable, '-c', 'pass'], **kwargs)


class ProductMeanFieldTests(unittest.TestCase):
    def test_separable_spectrum(self):
        a = np.array([[2., .4], [.4, 3.]])
        b = np.array([[1., .2, 0.], [.2, 2., .1], [0., .1, 4.]])
        h = np.kron(a, np.eye(3)) + np.kron(np.eye(2), b)
        result = solve_product_mean_field(h, (2, 3))
        energies, vectors = result.product_spectrum()
        np.testing.assert_allclose(energies, np.linalg.eigvalsh(h), atol=1e-12)
        np.testing.assert_allclose(h @ vectors, vectors * energies, atol=1e-12)

    def test_complex_three_axes(self):
        a = np.array([[2., .2j], [-.2j, 3.]])
        h = np.kron(np.kron(a, np.eye(2)), np.eye(2))
        h += np.kron(np.kron(np.eye(2), a), np.eye(2))
        h += np.kron(np.kron(np.eye(2), np.eye(2)), a)
        result = solve_product_mean_field(h, (2, 2, 2))
        self.assertAlmostEqual(result.energy, np.linalg.eigvalsh(h)[0], places=12)

    def test_coupled_variational_bound_and_validation(self):
        a = np.array([[0., .2], [.2, 1.]])
        x = np.array([[0., 1.], [1., 0.]])
        h = np.kron(a, np.eye(2)) + np.kron(np.eye(2), a) + .1*np.kron(x, x)
        result = solve_product_mean_field(h, (2, 2))
        self.assertGreaterEqual(result.energy + 1e-12, np.linalg.eigvalsh(h)[0])
        self.assertLess(result.stationary_residual, 1e-8)
        self.assertTrue(np.all(np.diff(result.energy_history) <= 1e-12))
        with self.assertRaises(ValueError):
            solve_product_mean_field(h, (2, 3))
        with self.assertRaises(RuntimeError):
            solve_product_mean_field(h, (2, 2), maximum_iterations=1)


class HessianExportTests(unittest.TestCase):
    def test_roundtrip_and_curvature_sign(self):
        # Diatomic relative stretch; translation and rotation are null.
        u = np.array([0., 0., -1., 0., 0., 1.]) / np.sqrt(2)
        for curvature in (1., -1.):
            with tempfile.TemporaryDirectory() as folder:
                hessian = curvature * np.outer(u, u)
                path = write_orca_hessian(Path(folder)/'field.hess',
                    hessian_hartree_per_bohr2=hessian,
                    coordinates_bohr=[[0., 0., -1.], [0., 0., 1.]],
                    atomic_numbers=[1, 1], masses_amu=[1., 1.], energy_hartree=-1.)
                parsed = hessian_input_from_orca_hessian_file(path)
                np.testing.assert_allclose(parsed.cartesian_hessian, hessian, atol=1e-15)
                self.assertIn('$vibrational_frequencies', path.read_text())
                if curvature < 0:
                    frequencies = path.read_text().split('$vibrational_frequencies')[1].split('$atoms')[0]
                    self.assertIn('-', frequencies)


class CartesianAHTests(unittest.TestCase):
    def field(self, curvature=1.):
        u=np.array([0.,0.,-1.,0.,0.,1.])/np.sqrt(2)
        return dict(atomic_numbers=np.array([1,1]), masses_amu=np.ones(2),
            coordinates_bohr=np.array([[0.,0.,-1.],[0.,0.,1.]]),
            gradient_hartree_per_bohr=np.zeros(6),
            hessian_hartree_per_bohr2=curvature*np.outer(u,u),
            energy_hartree=-1., sha256='controlled-model', electronic=dict(
                transition_dipole_au=np.array([1.,0.,0.]),
                electric_gradient_cartesian=None))

    def test_identical_fields(self):
        from matrix_trinity.vibronic_gaussian import gaussian_cartesian_vibronic_model
        data=self.field()
        model=gaussian_cartesian_vibronic_model(data,data,property_data=data)
        np.testing.assert_allclose(model.mode_map,[[1.]],atol=1e-12)
        np.testing.assert_allclose(model.final_shift_dimensionless,[0.],atol=1e-12)
        self.assertEqual(model.origin_cm,0.)

    def test_unstable_and_missing_property_derivatives(self):
        from matrix_trinity.vibronic_gaussian import gaussian_cartesian_vibronic_model
        data=self.field()
        with self.assertRaises(ValueError):
            gaussian_cartesian_vibronic_model(data,self.field(-1.),property_data=data)
        with self.assertRaises(ValueError):
            gaussian_cartesian_vibronic_model(data,data,property_data=data,herzberg_teller=True)


class HarmonicSubspaceTests(unittest.TestCase):
    def test_induced_mass_metric(self):
        from matrix_trinity.vibronic_harmonic import ESALCHarmonicState
        from matrix_trinity.vibronic_subspace import project_esalc_harmonic_state
        state=ESALCHarmonicState(np.diag([3.,5.]),np.diag([1.,4.]),
            np.array([1.,3.]),'controlled-common-chart',np.array([.1,.2]))
        basis=np.array([[1.],[1.]])/np.sqrt(2)
        projected=project_esalc_harmonic_state(state,basis)
        np.testing.assert_allclose(projected.metric_inverse_amu,[[1.6]],atol=1e-14)
        np.testing.assert_allclose(projected.force_constants_hartree,[[4.]],atol=1e-14)
        np.testing.assert_allclose(projected.reference_esalc,[2*np.sqrt(2)],atol=1e-14)

    def test_incompatible_projection(self):
        from matrix_trinity.vibronic_harmonic import ESALCHarmonicState
        from matrix_trinity.vibronic_subspace import project_esalc_harmonic_state
        state=ESALCHarmonicState(np.eye(2),np.eye(2),np.zeros(2),'test')
        with self.assertRaises(ValueError):
            project_esalc_harmonic_state(state,[[1.],[1.]])
        a=project_esalc_harmonic_state(state,np.eye(2))
        np.testing.assert_allclose(a.metric_inverse_amu,np.eye(2))


if __name__ == '__main__':
    unittest.main()
