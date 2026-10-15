"""Non-neural checks of independent edge widths, C2 joins and analytic truth."""
import copy
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from inr_isr.synthetic import scene_from_definition, average_scene
from inr_isr.synthetic.profiles import LogPatch


def fixture():
    return dict(schema_version=2,duration_sec=43200,truth_margin_sec=300,
        front=dict(low_log10_ne=9.5,high_log10_ne=11.5,center_y_km=255,
                   amplitude_km=60,wavelength_km=720,vertical_width_10_90_km=180,
                   phase_rad=-2*np.pi*150/720),
        patches=[dict(name='sharp',pair_id='pair',region='front',edge_class='sharp',
                      x0_km=150,y0_km=255,half_height_diameter_km=360,
                      edge_width_10_90_km=40,enhancement_dex=1)])


class LogGeneration(unittest.TestCase):
    def test_size_edge_and_core_are_independent(self):
        for width in (40.,120.):
            item=fixture()['patches'][0];item['edge_width_10_90_km']=width
            p=LogPatch(**item)
            value=lambda r:float(p.evaluate(r,0,derivatives=False)['enhancement_dex'])
            radii=[brentq(lambda r:value(r)-level,p.inner_radius_km,p.outer_radius_km) for level in (0.9,0.5,0.1)]
            self.assertAlmostEqual(radii[2]-radii[0],width,places=9)
            self.assertAlmostEqual(2*radii[1],360,places=9)
            v=p.evaluate(0.,0.)
            self.assertEqual(float(v['enhancement_dex']),1)
            for key in ('x','y','xx','xy','yy'):self.assertEqual(float(v[key]),0)
            for radius in (p.inner_radius_km,p.outer_radius_km):
                v=p.evaluate(radius,0)
                for key in ('x','y','xx','xy','yy'):self.assertAlmostEqual(float(v[key]),0,places=12)
                for h in (1e-3,1e-4):
                    a,b=p.evaluate(radius-h,0),p.evaluate(radius+h,0)
                    self.assertLess(abs(float(a['enhancement_dex']-b['enhancement_dex'])),1e-10)
                    self.assertLess(abs(float(a['xx']-b['xx'])),2e-6)

    def test_first_and_second_derivatives_translation(self):
        for speed in (0.4,1.5):
            scene=scene_from_definition(fixture(),speed)
            # Patch annulus, front transition, interior and exterior.
            pos=[np.array([310.,20.,150.,400.]),np.array([280.,240.,255.,490.]),123.]
            v=scene.evaluate(*pos)
            for axis,label in enumerate(('x_km','y_km','t_sec')):
                plus,minus=pos.copy(),pos.copy();h=0.001
                plus[axis]=plus[axis]+h;minus[axis]=minus[axis]-h
                fd=(scene.evaluate(*plus)['log10_Ne']-scene.evaluate(*minus)['log10_Ne'])/(2*h)
                np.testing.assert_allclose(fd,v['true_dlog10Ne_d'+label],rtol=2e-6,atol=2e-10)
            for axis,label in ((0,'xx_km2'),(1,'yy_km2'),(2,'tt_sec2')):
                plus,minus=pos.copy(),pos.copy();h=0.02
                plus[axis]=plus[axis]+h;minus[axis]=minus[axis]-h
                fd=(scene.evaluate(*plus)['log10_Ne']-2*v['log10_Ne']+scene.evaluate(*minus)['log10_Ne'])/h**2
                np.testing.assert_allclose(fd,v['true_d2log10Ne_d'+label],rtol=2e-4,atol=2e-9)
            moved=scene.evaluate(pos[0]+10*speed,pos[1],pos[2]+10)
            np.testing.assert_allclose(moved['log10_Ne'],v['log10_Ne'],rtol=0,atol=1e-13)
            h=0.02
            mixed=(scene.evaluate(pos[0]+h,pos[1]+h,pos[2])['log10_Ne']-
                   scene.evaluate(pos[0]+h,pos[1]-h,pos[2])['log10_Ne']-
                   scene.evaluate(pos[0]-h,pos[1]+h,pos[2])['log10_Ne']+
                   scene.evaluate(pos[0]-h,pos[1]-h,pos[2])['log10_Ne'])/(4*h*h)
            np.testing.assert_allclose(mixed,v['true_d2log10Ne_dxy_km2'],rtol=2e-4,atol=2e-9)

    def test_exposure_derivatives_and_margin(self):
        for speed in (0.4,1.5):
            scene=scene_from_definition(fixture(),speed)
            pos=[330.,255.,0.]
            for exposure in (60.,120.,300.,600.):
                reference=quad(lambda t:float(scene.evaluate(pos[0],pos[1],t,derivatives=False)['Ne']),
                               -exposure/2,exposure/2,epsabs=1,epsrel=1e-11,limit=200)[0]/exposure
                v=float(average_scene(scene,*pos,exposure)['Ne'])
                self.assertLess(abs(v/reference-1),1e-6)
            averaged=average_scene(scene,*pos,120.)
            for axis,label in enumerate(('x_km','y_km','t_sec')):
                plus,minus=pos.copy(),pos.copy();h=0.01
                plus[axis]+=h;minus[axis]-=h
                fd=(average_scene(scene,*plus,120.)['log10_Ne']-
                    average_scene(scene,*minus,120.)['log10_Ne'])/(2*h)
                np.testing.assert_allclose(fd,averaged['true_dlog10Ne_d'+label],rtol=2e-5,atol=2e-10)
            with self.assertRaises(ValueError):average_scene(scene,*pos,601.)

    def test_invalid_recipe_and_overlap_fail(self):
        definition=fixture();definition['patches'].append(copy.deepcopy(definition['patches'][0]))
        definition['patches'][1]['name']='other'
        with self.assertRaises(ValueError):scene_from_definition(definition,0.4)
        definition=fixture();definition['patches'][0]['half_height_diameter_km']=20
        with self.assertRaises(ValueError):scene_from_definition(definition,0.4)


if __name__=='__main__':unittest.main()
