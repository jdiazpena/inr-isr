"""Configured-window checks with data/analytic functions only; no neural fits."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from inr_isr.experiments.window import ExperimentWindow
from inr_isr.experiments.contract import Contract, digest
from inr_isr.experiments.analysis import feature_analysis, analyze
from inr_isr.synthetic.field import scene_from_definition


def analytic_scene_definition():
    return {"schema_version":2,"duration_sec":43200,"truth_margin_sec":300,
            "domain":{"x_bounds_km":[-100,400],"y_bounds_km":[0,510]},
            "front":{"low_log10_ne":9.5,"high_log10_ne":11.5,"center_y_km":255,
                     "amplitude_km":60,"wavelength_km":720,"vertical_width_10_90_km":180,"phase_rad":0},
            "patches":[{"name":"p1","pair_id":"pair","region":"low","edge_class":"sharp",
                        "x0_km":-2000,"y0_km":100,"half_height_diameter_km":300,
                        "edge_width_10_90_km":40,"enhancement_dex":1}]}


class ConfiguredWindowChecks(unittest.TestCase):
    def test_bounds_are_configured_and_limit_is_enforced(self):
        for start in (0, 3600, 21600):
            window = ExperimentWindow(start, start + 3600, 3600)
            times = window.sample_times(30, 5)
            self.assertEqual(times[0], start + 5)
            self.assertLess(times[-1], start + 3600)
            window.assert_times([start, start + 3600])
            window.assert_scalers({'t_sec':{'min':start,'max':start+3600}},True)
            with self.assertRaises(ValueError):window.assert_scalers({'t_sec':{'min':start,'max':start+600}},True)
            with self.assertRaises(ValueError):window.assert_scalers({'t_sec':{'min':start,'max':start+10800}},True)
            for invalid in ([start - 1], [start + 3601], [np.nan]):
                with self.assertRaises(ValueError): window.assert_times(invalid)
        for args in [(0, 10800, 3600), (0, 0, 3600), (0, np.inf, 3600), (0, 3600, -1)]:
            with self.assertRaises(ValueError): ExperimentWindow(*args)

    def test_every_input_and_map_checked_before_model_or_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('train', 'development', 'probes', 'queries'):
                pd.DataFrame({'t_sec': [100, 3700]}).to_csv(root/f'{name}.csv', index=False)
            data = {'schema':'shared-xyt-development-1', 'roles':{'train':[1], 'development':[2], 'reserved':[3]},
                    'time_bounds_sec':[100,3700], 'window_policy':{'maximum_duration_sec':3600, 'require_observed_endpoints':True},
                    'analysis':{'map_times_sec':[105,3695]},
                    'inputs':{n:{'path':f'{n}.csv','sha256':digest(root/f'{n}.csv')} for n in ('train','development','probes','queries')}}
            path = root/'contract.json'
            path.write_text(json.dumps(data))
            ExperimentWindow.from_contract(Contract.load(path)).validate_inputs(Contract.load(path))
            for name in ('train', 'development', 'probes', 'queries'):
                original = (root/f'{name}.csv').read_bytes()
                pd.DataFrame({'t_sec':[100, 3701]}).to_csv(root/f'{name}.csv', index=False)
                data['inputs'][name]['sha256'] = digest(root/f'{name}.csv')
                path.write_text(json.dumps(data));contract = Contract.load(path)
                with patch('inr_isr.experiments.analysis.resolve_device') as device, \
                     patch('inr_isr.experiments.analysis.torch_load_checkpoint') as load:
                    with self.assertRaises(ValueError): analyze(contract, root/'absent-run', root/'absent-output')
                    device.assert_not_called();load.assert_not_called()
                (root/f'{name}.csv').write_bytes(original)
                data['inputs'][name]['sha256'] = digest(root/f'{name}.csv')
            data['analysis']['map_times_sec'] = [3701];path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): ExperimentWindow.from_contract(Contract.load(path)).validate_inputs(Contract.load(path))

    def test_feature_profiles_never_query_outside_window(self):
        scene = scene_from_definition(analytic_scene_definition(), .4)
        window = ExperimentWindow(3600, 7200, 3600)
        observed = []
        def truth_only(x,y,t):
            window.assert_times(t, 'analytic callback')
            observed.extend(np.asarray(t).ravel())
            return scene.evaluate(x,y,t,derivatives=False)['log10_Ne']
        settings = {'front_fraction_bounds':[.1,.9], 'front_time_step_sec':300,
                    'query_time_offset_sec':5, 'front_profile_samples':205}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            feature_analysis(scene, truth_only, settings, output, window=window,
                             domain={'x_bounds_km':[-100,400], 'y_bounds_km':[0,510]})
            fronts = pd.read_csv(output/'front_metrics.csv')
            self.assertEqual(len(fronts), 60)
            self.assertTrue(fronts.t_sec.between(window.start,window.end).all())
            patches = pd.read_csv(output/'patch_metrics.csv')
            self.assertTrue(patches.t_sec.between(window.start,window.end).all())
        self.assertTrue(len(observed)>0)

    def test_complete_analysis_uses_one_hour_with_an_analytic_function(self):
        class AnalyticFunction:
            def __call__(self,q):
                if bool((q[:,2].abs()>1.00001).any()):
                    raise AssertionError('Analysis attempted temporal extrapolation')
                return q[:,:1]**2+2*q[:,1:2]**2+3*q[:,2:3]**2+4*q[:,:1]*q[:,1:2]
            def parameters(self):return iter(())
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp);run = output/'run';run.mkdir()
            (output/'scene.json').write_text(json.dumps(analytic_scene_definition()))
            observations=pd.DataFrame({'x_km':[0.,100.],'y_km':[100.,300.],
                                       't_sec':[3600.,7200.],'log10_Ne':[9.5,11.5],'beam_id':[1,1]})
            observations.to_csv(output/'train.csv',index=False)
            observations.assign(beam_id=2).to_csv(output/'development.csv',index=False)
            queries=pd.DataFrame(np.random.default_rng(0).uniform([-100,0,3600],[400,510,7200],(16,3)),
                                 columns=['x_km','y_km','t_sec'])
            queries['inside_training_hull']=[True]*8+[False]*8
            queries['nearest_training_km']=1.
            queries.to_csv(output/'queries.csv',index=False)
            pd.DataFrame({'x_km':[0.,100.,200.],'y_km':[100.,300.,150.],
                          'role':['train','development','reserved']}).to_csv(output/'roles.csv',index=False)
            names=('train','development','queries','roles','scene')
            data={'schema':'shared-xyt-development-1','case':'analytic-one-hour',
                  'roles':{'train':[1],'development':[2],'reserved':[3]},
                  'time_bounds_sec':[3600,7200],'window_policy':{'maximum_duration_sec':3600,'require_observed_endpoints':True},
                  'domain':analytic_scene_definition()['domain'],'speed_km_s':.4,'prediction_chunk':1024,
                  'inputs':{n:{'path':str(output/f"{n}.{'json' if n=='scene' else 'csv'}"),
                               'sha256':digest(output/f"{n}.{'json' if n=='scene' else 'csv'}")} for n in names},
                  'analysis':{'derivative_query_count':16,'grid_nx':8,'grid_ny':9,
                              'map_times_sec':[3605,4805,6005,7195],'cut_y_km':255,
                              'patch_threshold_dex':.1,'front_fraction_bounds':[.1,.9]}}
            path=output/'contract.json';path.write_text(json.dumps(data));contract=Contract.load(path)
            case={'coord_scalers':{axis:{'min':lo,'max':hi} for axis,lo,hi in
                                  [('x_km',-100,400),('y_km',0,510),('t_sec',3600,7200)]},
                  'target_scaler':{'column':'log10_Ne','min':9.5,'max':11.5}}
            checkpoint = {'coord_scalers':case['coord_scalers'],'target_scaler':case['target_scaler'],
                          'experiment_contract_sha256':digest(path),'step':3}
            (run/'model_best_development.pt').write_bytes(b'analytic fixture;never loaded as a model')
            pd.DataFrame({'step':[1,3],'data_loss':[1.,.5],**{k:[1.,1.] for k in ('fxx_rms','fxy_rms','fyy_rms','ftt_rms')}}).to_csv(run/'history.csv',index=False)
            pd.DataFrame({'step':[3],'macro_rmse_dex':[1.]}).to_csv(run/'development_history.csv',index=False)
            with patch('inr_isr.experiments.analysis.torch_load_checkpoint',return_value=checkpoint), \
                 patch('inr_isr.experiments.analysis.build_model_from_checkpoint',return_value=AnalyticFunction()):
                analyze(contract,run,output/'analysis',cpu=True)
            for file in (output/'analysis').glob('*.csv'):
                if file.stat().st_size<=1:continue
                frame=pd.read_csv(file)
                if 't_sec' in frame:self.assertTrue(frame.t_sec.between(3600,7200).all(),file.name)
            for file in (output/'analysis').glob('*.npz'):
                with np.load(file) as arrays:
                    if 't_sec' in arrays:self.assertTrue(((arrays['t_sec']>=3600)&(arrays['t_sec']<=7200)).all())
            summary=json.loads((output/'analysis/summary.json').read_text())
            self.assertEqual(summary['time_window_sec'],[3600,7200])


if __name__ == '__main__': unittest.main()
