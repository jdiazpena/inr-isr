"""All-beam coverage/saving/analysis checks; analytic functions only, no networks."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from inr_isr.experiments.contract import Contract, digest
from inr_isr.experiments.validation import ReconstructionMonitor
from inr_isr.experiments.analysis import analyze
from synthetic_dataset import SyntheticPlasmaTimeDataset
from test_configured_experiment_window import analytic_scene_definition


class AllBeamChecks(TestCase):
    def fixture(self, root):
        geometry = pd.DataFrame({'beam_index':[0,1,2], 'x_km':[0.,100.,0.],
                                 'y_km':[100.,100.,300.], 'role':['train']*3})
        geometry.to_csv(root/'roles.csv',index=False)
        obs = pd.DataFrame([{'beam_id':int(r.beam_index),'x_km':r.x_km,'y_km':r.y_km,
                             't_sec':t,'time_index':i,'log10_Ne':9.5+i+.1*r.beam_index}
                            for i,t in enumerate((3600,7200)) for r in geometry.itertuples()])
        obs.to_csv(root/'train.csv',index=False)
        pd.DataFrame({'x_km':[10.,20.],'y_km':[120.,130.], 't_sec':[4000.,6000.]}).to_csv(root/'probes.csv',index=False)
        q=pd.DataFrame({'x_km':[10.,20.],'y_km':[120.,130.],'t_sec':[4500.,6500.],
                        'inside_training_hull':[True,True],'nearest_training_km':[10.,20.]})
        q.to_csv(root/'queries.csv',index=False)
        (root/'scene.json').write_text(json.dumps(analytic_scene_definition()))
        data={'schema':'shared-xyt-reconstruction-1','id':'analytic-all-beam','case':'analytic-all-beam',
              'roles':{'train':[0,1,2],'development':[],'reserved':[]},
              'checkpoint_policy':'final_update','selection_every':0,'time_bounds_sec':[3600,7200],
              'observation_cadence_sec':3600,'speed_km_s':.4,'prediction_chunk':1024,
              'window_policy':{'maximum_duration_sec':3600,'require_observed_endpoints':True},
              'domain':analytic_scene_definition()['domain'],
              'analysis':{'derivative_query_count':2,'grid_nx':8,'grid_ny':9,'map_times_sec':[3600,4050],
                          'cut_y_km':255,'patch_threshold_dex':.1,'front_fraction_bounds':[.1,.9]},
              'inputs':{n:{'path':str(root/f"{n}.{'json' if n=='scene' else 'csv'}"),
                           'sha256':digest(root/f"{n}.{'json' if n=='scene' else 'csv'}")}
                        for n in ('train','roles','queries','probes','scene')}}
        return data

    def load(self,root,data):
        p=root/'contract.json';p.write_text(json.dumps(data));return Contract.load(p)

    def test_rejects_missing_beams_times_duplicates_and_nonfinite_rows(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.fixture(root);full=pd.read_csv(root/'train.csv')
            self.assertTrue(self.load(root,data).all_beam)
            damaged=[full[full.beam_id!=1],full[full.t_sec!=7200],full.iloc[:-1],
                     pd.concat([full,full.iloc[:1]]),full.assign(log10_Ne=np.nan),
                     full.assign(x_km=99.)]
            for table in damaged:
                table.to_csv(root/'train.csv',index=False)
                data['inputs']['train']['sha256']=digest(root/'train.csv')
                with self.assertRaises(ValueError):self.load(root,data)

    def test_rejects_withholding_and_ambiguous_selection(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.fixture(root)
            for name,value in [('development',[1]),('reserved',[1])]:
                changed=copy.deepcopy(data);changed['roles'][name]=value
                with self.assertRaises(ValueError):self.load(root,changed)
            changed=copy.deepcopy(data);changed['checkpoint_policy']='best_truth'
            with self.assertRaises(ValueError):self.load(root,changed)
            changed=copy.deepcopy(data);changed['inputs']['development']=changed['inputs']['train']
            with self.assertRaises(ValueError):self.load(root,changed)
            # Legacy studies still require their original nonempty disjoint groups.
            changed=copy.deepcopy(data);changed['schema']='shared-xyt-development-1'
            with self.assertRaises(ValueError):self.load(root,changed)

    def test_final_save_has_no_model_evaluation_or_score_selection(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);contract=self.load(root,self.fixture(root));run=root/'run';run.mkdir()
            ds=SyntheticPlasmaTimeDataset(root/'train.csv',verbose=False)
            monitor=ReconstructionMonitor(contract,ds,run)
            state={'fixture':'ordinary text; no parameters or neural model'}
            holder=SimpleNamespace(state_dict=lambda:state)
            before=torch.random.get_rng_state().clone()
            monitor.diagnostic_points(torch.device('cpu'))
            self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
            with patch('inr_isr.experiments.validation.predict_log10_on_coords',side_effect=AssertionError('No selection prediction allowed')):
                monitor.after_update(1,3,holder,{},torch.device('cpu'))
                self.assertFalse((run/'model_reconstruction.pt').exists())
                monitor.after_update(3,3,holder,{},torch.device('cpu'))
            saved=torch.load(run/'model_reconstruction.pt',weights_only=False)
            self.assertEqual(saved['step'],3)
            self.assertEqual(saved['model_state_dict'],state)
            self.assertEqual(saved['experiment_contract_sha256'],digest(contract.path))
            self.assertFalse((run/'development_history.csv').exists())
            self.assertFalse((run/'model_best_development.pt').exists())

    def test_full_analysis_without_withheld_tables_or_histories(self):
        class AnalyticFunction:
            def __call__(self,q):return q[:,:1]**2+q[:,1:2]**2+q[:,2:3]**2
            def parameters(self):return iter(())
        with TemporaryDirectory() as tmp:
            root=Path(tmp);contract=self.load(root,self.fixture(root));run=root/'run';run.mkdir()
            ds=SyntheticPlasmaTimeDataset(root/'train.csv',verbose=False)
            checkpoint={'coord_scalers':ds.coord_scalers,'target_scaler':ds.target_scaler,'step':3,
                        'checkpoint_type':'final_reconstruction_after_update','experiment_contract_sha256':digest(contract.path)}
            (run/'model_reconstruction.pt').write_bytes(b'analytic fixture; never loaded as a network')
            (run/'selection.json').write_text(json.dumps({'rule':'final_update','completed_updates':3}))
            pd.DataFrame({'step':[1,3],'data_loss':[1.,.5],**{k:[1.,1.] for k in ('fxx_rms','fxy_rms','fyy_rms','ftt_rms')}}).to_csv(run/'history.csv',index=False)
            with patch('inr_isr.experiments.analysis.torch_load_checkpoint',return_value=checkpoint), \
                 patch('inr_isr.experiments.analysis.build_model_from_checkpoint',return_value=AnalyticFunction()):
                analyze(contract,run,root/'analysis',cpu=True)
            summary=json.loads((root/'analysis/summary.json').read_text())
            self.assertEqual(set(summary['observations']),{'train'})
            self.assertEqual(summary['observations']['train']['pooled_rmse_dex']>=0,True)
            self.assertEqual(summary['checkpoint']['step'],3)
            self.assertTrue((root/'analysis/training_and_diagnostics.png').exists())
            self.assertFalse((root/'analysis/development_predictions.csv').exists())
