"""Native real-data contracts and analysis, using HDF fixtures and analytic functions.

No neural model or optimizer is constructed; no training or neural inference.
"""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch

from inr_isr.experiments.contract import Contract, digest
from inr_isr.experiments.radar import native_dataset, source_summary
from inr_isr.experiments.validation import ReconstructionMonitor
from inr_isr.experiments.radar_analysis import analyze_radar, final_checkpoint
from train_radar_3d_window_reference_reg_diagnostic import build_parser


class RadarChecks(TestCase):
    def fixture(self, root):
        origin = 1327402800.
        times = origin + np.array([60.,191.,420.])
        path = root/'radar.h5'
        with h5py.File(path,'w') as f:
            f['Time/UnixTime'] = np.column_stack([times-60,times+60])
            f['BeamCodes'] = np.array([[1.,0.,60.],[2.,90.,60.],[3.,180.,60.],[4.,270.,60.]])
            ne = np.arange(12,dtype=float).reshape(3,4,1)*1e9+1e10
            ne[1,1,0] = np.nan
            f['FittedParams/Ne'] = ne
            f['FittedParams/dNe'] = np.full_like(ne,np.nan)  # dNe does not filter native data.
            f['FittedParams/Range'] = np.full((4,1),350.)
            f['FittedParams/Altitude'] = np.full((4,1),300.)
        args = vars(build_parser().parse_args([])) | {'h5_path':str(path),'h0_km':300.,
            'time_start_utc':'2012-01-24T11:00:00Z','time_end_utc':'2012-01-24T12:00:00Z',
            'window_start_index':None,'window_size_records':None,'num_steps':5000,'num_diagnostic_collocation':4,
            'target_xy_ratio':0.,'target_t_ratio':0.,'no_plots':True}
        ds = native_dataset(path,args)
        ds.df.to_csv(root/'train.csv',index=False)
        geometry = ds.df[['beam_index','beamcode','az_deg','el_deg','range_index','altitude_km','x_km','y_km']].drop_duplicates().sort_values('beam_index')
        geometry['role'] = 'train';geometry.to_csv(root/'roles.csv',index=False)
        q = pd.DataFrame({'x_km':np.linspace(-100,100,8),'y_km':np.linspace(-50,50,8),'t_sec':np.linspace(1,359,8)})
        q['unix_mid'] = q.t_sec+times[0]
        q.to_csv(root/'queries.csv',index=False);q.iloc[:4].to_csv(root/'probes.csv',index=False)
        data = {'schema':'shared-xyt-radar-reconstruction-1','id':'analytic-radar','case':'analytic-radar',
            'roles':{'train':[0,1,2,3],'development':[],'reserved':[]},'checkpoint_policy':'final_update',
            'selection_every':0,'objective':'data_only','measurement_policy':'native_exposure_average_at_midpoint',
            'nominal_integration_sec':120,'observation_summary':source_summary(path,args,ds),'training':args,
            'preprocessing':{'coord_scalers':ds.coord_scalers,'target_scaler':ds.target_scaler},
            'time_bounds_sec':[0.,360.],'window_policy':{'maximum_duration_sec':3600,'require_observed_endpoints':True},
            'domain':{'x_bounds_km':[-180.,180.],'y_bounds_km':[-180.,180.]},'prediction_chunk':1024,
            'analysis':{'field_query_count':8,'derivative_query_count':8,'derivative_chunk':4,
                'grid_nx':8,'grid_ny':9,'map_times_sec':[10.,300.],'cut_y_km':0.,'cut_time_samples':7,
                'cut_x_samples':8,'log_density_limits':[9.,12.],'comparison_unix_bounds':[times[0]+1,times[-1]-1]},
            'inputs':{name:{'path':str(root/file),'sha256':digest(root/file)} for name,file in
                      [('hdf5','radar.h5'),('train','train.csv'),('roles','roles.csv'),('probes','probes.csv'),('queries','queries.csv')]}}
        return data,ds

    def load(self,root,data):
        path = root/'contract.json';path.write_text(json.dumps(data));return Contract.load(path)

    def test_irregular_times_and_declared_invalid_density_are_native(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);data,ds=self.fixture(root)
            before=torch.random.get_rng_state().clone()
            contract=self.load(root,data)
            self.assertTrue(contract.radar and contract.final_update)
            self.assertFalse(contract.all_beam or contract.withholding)
            self.assertEqual(ds.n_samples,11)
            self.assertEqual(data['observation_summary']['excluded_rows'],[
                {'time_index':1,'beam_index':1,'range_index':0,'reason':'nonfinite_Ne'}])
            self.assertTrue(torch.equal(before,torch.random.get_rng_state()))

    def test_rejects_hidden_row_beam_and_time_removals_even_after_rehash(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);data,ds=self.fixture(root);full=ds.df
            for table in [full.iloc[:-1],full[full.beam_index!=1],full[full.time_index!=1],pd.concat([full,full.iloc[:1]])]:
                table.to_csv(root/'train.csv',index=False);data['inputs']['train']['sha256']=digest(root/'train.csv')
                with self.assertRaisesRegex(ValueError,'every retained native observation'):self.load(root,data)

    def test_rejects_index_precedence_averaging_stride_holdouts_and_long_windows(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);data,ds=self.fixture(root)
            for key,value in [('window_start_index',0),('window_size_records',11),('record_stride',2),
                              ('time_end_utc','2012-01-24T14:00:00Z')]:
                changed=copy.deepcopy(data);changed['training'][key]=value
                with self.assertRaises(ValueError):self.load(root,changed)
            for change in [{'measurement_policy':'average_averages'},{'selection_every':100}]:
                with self.assertRaises(ValueError):self.load(root,data|change)
            changed=copy.deepcopy(data);changed['roles']['development']=[0]
            with self.assertRaises(ValueError):self.load(root,changed)

    def test_rejects_changed_exclusion_utc_and_probe_count(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);data,ds=self.fixture(root)
            changed=copy.deepcopy(data);changed['observation_summary']['excluded_rows']=[]
            with self.assertRaises(ValueError):self.load(root,changed)
            q=pd.read_csv(root/'probes.csv');q.unix_mid+=100
            q.to_csv(root/'probes.csv',index=False);data['inputs']['probes']['sha256']=digest(root/'probes.csv')
            with self.assertRaises(ValueError):self.load(root,data)

    def test_native_dispatch_and_diagnostics_without_running_engine(self):
        from inr_isr.experiments.__main__ import train
        with TemporaryDirectory() as tmp:
            root=Path(tmp);data,ds=self.fixture(root);contract=self.load(root,data)
            with patch('training_engine.train_window') as engine:
                train(contract,root/'run',smoke=True,cpu=True)
            args=engine.call_args.args[0];kwargs=engine.call_args.kwargs
            self.assertEqual(args.h5_path,str(root/'radar.h5'))
            self.assertIsNone(args.window_start_index);self.assertEqual(args.num_steps,3)
            self.assertTrue(kwargs['diagnostics_without_regularization'])
            self.assertEqual(kwargs['dataset_factory'].__module__,'train_radar_3d_window_reference_reg_diagnostic')

    def test_final_saving_and_fixed_probes_do_not_select_or_consume_rng(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);data,ds=self.fixture(root);contract=self.load(root,data);run=root/'run';run.mkdir()
            monitor=ReconstructionMonitor(contract,ds,run)
            before=torch.random.get_rng_state().clone()
            self.assertEqual(tuple(monitor.diagnostic_points(torch.device('cpu')).shape),(4,3))
            self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
            holder=SimpleNamespace(state_dict=lambda:{'fixture':'no neural parameters'})
            with patch('inr_isr.experiments.validation.predict_log10_on_coords',side_effect=AssertionError('No selection')):
                monitor.after_update(1,3,holder,data['training'],torch.device('cpu'))
                self.assertFalse((run/'model_reconstruction.pt').exists())
                monitor.after_update(3,3,holder,data['training'],torch.device('cpu'))
            saved=torch.load(run/'model_reconstruction.pt',weights_only=False)
            self.assertEqual(saved['step'],3)
            self.assertEqual(json.loads((run/'selection.json').read_text())['withheld_beams'],0)

    def test_real_analysis_with_analytic_function_has_no_truth_or_temporal_extrapolation(self):
        class AnalyticFunction:
            def __call__(self,q):
                if bool((q[:,2].abs()>1.00001).any()):raise AssertionError('Temporal extrapolation')
                return q[:,:1]**2+q[:,1:2]**2+q[:,2:3]**2
            def parameters(self):return iter(())
        with TemporaryDirectory() as tmp:
            root=Path(tmp);data,ds=self.fixture(root);contract=self.load(root,data);run=root/'run';run.mkdir()
            effective=data['training']|{'num_steps':3,'output_dir':str(run),'cpu':True}
            (run/'run_config.json').write_text(json.dumps(effective))
            (run/'selection.json').write_text(json.dumps({'rule':'final_update','step':3,'completed_updates':3,
                'score_based_selection':False,'withheld_beams':0,'withheld_times':0}))
            (run/'model_reconstruction.pt').write_bytes(b'analytic fixture, no network')
            checkpoint={'step':3,'checkpoint_type':'final_reconstruction_after_update','experiment_id':data['id'],
                'experiment_contract_sha256':digest(contract.path),'coord_scalers':ds.coord_scalers,'target_scaler':ds.target_scaler}
            columns=['fxx_rms','fxy_rms','fyy_rms','ftt_rms','grad_norm_total','grad_norm_data']
            pd.DataFrame({'step':[1,3],'data_loss':[1.,.5],'total_loss':[1.,.5],**{c:[1.,1.] for c in columns}}).to_csv(run/'history.csv',index=False)
            with patch('inr_isr.experiments.radar_analysis.torch_load_checkpoint',return_value=checkpoint), \
                 patch('inr_isr.experiments.radar_analysis.build_model_from_checkpoint',return_value=AnalyticFunction()):
                analyze_radar(contract,run,root/'analysis',cpu=True)
                broken=checkpoint|{'step':2}
                with patch('inr_isr.experiments.radar_analysis.torch_load_checkpoint',return_value=broken):
                    with self.assertRaises(ValueError):final_checkpoint(contract,run,torch.device('cpu'))
            result=json.loads((root/'analysis/summary.json').read_text())
            self.assertTrue(result['execution_check_only']);self.assertEqual(result['withheld_beams'],0)
            self.assertEqual(result['measurement_fit']['per_beam'][1]['count'],2)
            self.assertNotIn('primary_field_score',result)
            for file in (root/'analysis').glob('*.csv'):
                frame=pd.read_csv(file)
                self.assertFalse(any('truth' in col or col.startswith('error_') for col in frame))
                if 't_sec' in frame:self.assertTrue(frame.t_sec.between(0,360).all())
            self.assertTrue((root/'analysis/measurement_fit.png').exists())
