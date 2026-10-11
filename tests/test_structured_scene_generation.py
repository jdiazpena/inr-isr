"""Focused non-neural mathematical tests; run directly with unittest."""
import unittest
import numpy as np
from scipy.integrate import quad

from inr_isr.synthetic.field import StructuredScene, average_scene


def fixture():
    return {'schema_version': 1, 'duration_sec': 43200, 'truth_margin_sec': 300,
            'front': {'low_ne_m3': 5e10, 'high_ne_m3': 1e11, 'center_y_km': 255,
                      'amplitude_km': 60, 'wavelength_km': 360,
                      'vertical_width_10_90_km': 60, 'phase_rad': 0},
            'patches': [{'name': 'patch', 'x0_km': -400, 'y0_km': 255,
                         'sigma_x_km': 50, 'sigma_y_km': 60, 'amplitude_m3': 4e10}]}


class GenerationMath(unittest.TestCase):
    def test_derivatives_and_common_translation(self):
        for speed in (0.4, 1.5):
            scene = StructuredScene(fixture(), speed)
            x, y, t = np.array([125., 150., 180.]), np.array([240., 255., 285.]), 550 / speed
            v = scene.evaluate(x, y, t)
            for axis, label in enumerate(('x_km', 'y_km', 't_sec')):
                plus, minus = [x, y, t], [x, y, t]
                h = 0.001
                plus[axis] = plus[axis] + h
                minus[axis] = minus[axis] - h
                fd = (scene.evaluate(*plus)['log10_Ne'] - scene.evaluate(*minus)['log10_Ne']) / (2 * h)
                np.testing.assert_allclose(fd, v[f'true_dlog10Ne_d{label}'], rtol=1e-6, atol=2e-10)
            for axis, label in ((0, 'xx_km2'), (1, 'yy_km2'), (2, 'tt_sec2')):
                plus, minus = [x, y, t], [x, y, t]
                h = 0.02
                plus[axis] = plus[axis] + h
                minus[axis] = minus[axis] - h
                fd = (scene.evaluate(*plus)['log10_Ne'] - 2 * v['log10_Ne'] + scene.evaluate(*minus)['log10_Ne']) / h**2
                np.testing.assert_allclose(fd, v[f'true_d2log10Ne_d{label}'], rtol=1e-4, atol=2e-9)
            moved = scene.evaluate(x + speed * 10, y, t + 10)
            np.testing.assert_allclose(moved['Ne'], v['Ne'], rtol=3e-15)
            np.testing.assert_allclose(v['true_dlog10Ne_dt_sec'], -speed * v['true_dlog10Ne_dx_km'], rtol=2e-15)

    def test_linear_exposure_average_and_margin(self):
        for speed in (0.4, 1.5):
            scene = StructuredScene(fixture(), speed)
            np.testing.assert_array_equal(average_scene(scene, 150., 255., 0., 0)['Ne'], scene.evaluate(150., 255., 0.)['Ne'])
            for exposure in (60, 120, 300):
                t = 550 / speed
                reference = quad(lambda s: float(scene.evaluate(150, 255, s, derivatives=False)['Ne']),
                                 t - exposure / 2, t + exposure / 2, epsabs=1, epsrel=1e-11)[0] / exposure
                fine = average_scene(scene, 150, 255, t, exposure, max_spacing_sec=0.1)['Ne']
                self.assertLess(abs(float(fine) / reference - 1), 1e-6)
            # Differentiate the averaged log field independently in physical units.
            position = [150., 255., 550 / speed]
            averaged = average_scene(scene, *position, 300.)
            for axis, label in enumerate(('x_km', 'y_km', 't_sec')):
                plus, minus = position.copy(), position.copy()
                h = 0.01
                plus[axis] += h
                minus[axis] -= h
                fd = (average_scene(scene, *plus, 300.)['log10_Ne'] -
                      average_scene(scene, *minus, 300.)['log10_Ne']) / (2 * h)
                np.testing.assert_allclose(fd, averaged[f'true_dlog10Ne_d{label}'], rtol=1e-5, atol=2e-10)
            self.assertTrue(np.isfinite(average_scene(scene, 150, 255, 0, 600)['Ne']))
            with self.assertRaises(ValueError):
                average_scene(scene, 150, 255, 0, 601)
            with self.assertRaises(ValueError):
                average_scene(scene, 150, 255, 100, -1)


if __name__ == '__main__':
    unittest.main()
