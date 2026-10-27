"""Partition/final-save/support checks with CSVs and analytic functions; no networks."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from inr_isr.experiments.contract import Contract, digest
from inr_isr.experiments.validation import ReconstructionMonitor
from inr_isr.experiments.analysis import analyze, withholding_support
from synthetic_dataset import SyntheticPlasmaTimeDataset
from test_all_beam_experiment import AllBeamChecks


class WithholdingChecks(TestCase):
    def fixture(self, root, kind):
        data = AllBeamChecks().fixture(root)
        geometry = pd.read_csv(root/'roles.csv')
        geometry.loc[len(geometry)] = [3, 25., 150., 'train']
        source = pd.DataFrame([{'beam_id':int(g.beam_index),'x_km':g.x_km,'y_km':g.y_km,
            't_sec':t,'time_index':i,'log10_Ne':9.5+.2*i+.1*g.beam_index}
            for i,t in enumerate([3600,4500,5400,6300,7200]) for g in geometry.itertuples()])
        mask = source.beam_id.eq(3) if kind=='beams' else source.t_sec.eq(5400)
        if kind=='beams':geometry.loc[geometry.beam_index.eq(3),'role']='withheld'
        for name, frame in [('roles',geometry),('source_observations',source),('train',source.loc[~mask]),('withheld',source.loc[mask])]:
            frame.to_csv(root/f'{name}.csv',index=False)
            data['inputs'][name]={'path':str(root/f'{name}.csv'),'sha256':digest(root/f'{name}.csv')}
        data.update(schema='shared-xyt-withholding-1', observation_cadence_sec=900,
                    withholding={'kind':kind,'beam_indices':[3] if kind=='beams' else [],
                                 'times_sec':[5400] if kind=='times' else []})
        data['roles']['train']=[0,1,2] if kind=='beams' else [0,1,2,3]
        return data

    def load(self,root,data):
        p=root/'contract.json';p.write_text(json.dumps(data));return Contract.load(p)

    def test_separate_beam_and_time_partitions(self):
        for kind in ['beams','times']:
            with TemporaryDirectory() as tmp:
                root=Path(tmp);c=self.load(root,self.fixture(root,kind))
                self.assertTrue(c.withholding and c.final_update)
                self.assertFalse(c.all_beam)
                train=pd.read_csv(c.file('train'));held=pd.read_csv(c.file('withheld'))
                self.assertEqual(len(train)+len(held),20)
                self.assertEqual(bool(set(train.beam_id)&set(held.beam_id)),kind=='times')

    def test_partition_overlap_extra_removal_and_wrong_targets_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.fixture(root,'times');train=pd.read_csv(root/'train.csv');held=pd.read_csv(root/'withheld.csv')
            damaged=[pd.concat([train,held.iloc[:1]]),train.iloc[:-1],train.assign(log10_Ne=train.log10_Ne+1)]
            for frame in damaged:
                frame.to_csv(root/'train.csv',index=False);data['inputs']['train']['sha256']=digest(root/'train.csv')
                with self.assertRaises(ValueError):self.load(root,data)

    def test_endpoints_mixed_masks_selection_and_role_errors_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.fixture(root,'times')
            for mutate in [lambda c:c['withholding'].update(times_sec=[3600]),
                           lambda c:c['withholding'].update(beam_indices=[3]),
                           lambda c:c.update(checkpoint_policy='best_withheld'),
                           lambda c:c.update(selection_every=10),
                           lambda c:c['roles'].update(train=[0,1,2])]:
                changed=copy.deepcopy(data);mutate(changed)
                with self.assertRaises(ValueError):self.load(root,changed)

    def test_final_monitor_never_predicts_or_selects_on_withheld_values(self):
        for kind in ['beams','times']:
            with TemporaryDirectory() as tmp:
                root=Path(tmp);c=self.load(root,self.fixture(root,kind));out=root/'run';out.mkdir()
                ds=SyntheticPlasmaTimeDataset(c.file('train'),verbose=False)
                m=ReconstructionMonitor(c,ds,out);holder=SimpleNamespace(state_dict=lambda:{'fixture':'no model'})
                with patch('inr_isr.experiments.validation.predict_log10_on_coords',side_effect=AssertionError('No held predictions in fitting')),patch('inr_isr.experiments.validation.torch.save') as save:
                    m.after_update(1,3,holder,{},'cpu');save.assert_not_called()
                    m.after_update(3,3,holder,{},'cpu');save.assert_called_once()
                selection=json.loads((out/'selection.json').read_text())
                self.assertFalse(selection['score_based_selection'])
                self.assertEqual(selection['withheld_beams'],int(kind=='beams'))
                self.assertEqual(selection['withheld_times'],int(kind=='times'))
                self.assertFalse((out/'development_history.csv').exists())

    def test_support_uses_retained_geometry_and_actual_timestamps(self):
        for kind in ['beams','times']:
            with TemporaryDirectory() as tmp:
                root=Path(tmp);c=self.load(root,self.fixture(root,kind))
                q=pd.DataFrame({'x_km':[25.],'y_km':[150.],'t_sec':[5400.],
                                'nearest_training_km':[0.],'inside_training_hull':[False]})
                result=withholding_support(q,pd.read_csv(c.file('train')),c.data['withholding'],900,pd.read_csv(c.file('withheld')))
                self.assertTrue(result.inside_training_hull.iloc[0])
                if kind=='beams':
                    self.assertGreater(result.nearest_training_km.iloc[0],0)
                    self.assertTrue(result.closer_to_withheld_beam.iloc[0])
                else:
                    self.assertEqual(result.nearest_training_time_sec.iloc[0],900)
                    self.assertEqual(result.bracketing_training_gap_sec.iloc[0],1800)
                    self.assertEqual(result.time_gap_fraction.iloc[0],.5)
                    self.assertTrue(result.inside_withheld_time_gap.iloc[0])
                self.assertEqual(q.nearest_training_km.iloc[0],0) # Original frozen queries unchanged.

    def test_complete_analysis_uses_final_model_and_assesses_withheld_rows(self):
        class AnalyticFunction:
            def __call__(self,q):return q[:,:1]**2+q[:,1:2]**2+q[:,2:3]**2
            def parameters(self):return iter(())
        for kind in ['beams','times']:
            with TemporaryDirectory() as tmp:
                root=Path(tmp);c=self.load(root,self.fixture(root,kind));run=root/'run';run.mkdir()
                ds=SyntheticPlasmaTimeDataset(c.file('train'),verbose=False)
                checkpoint={'coord_scalers':ds.coord_scalers,'target_scaler':ds.target_scaler,'step':3,
                    'checkpoint_type':'final_reconstruction_after_update','experiment_contract_sha256':digest(c.path)}
                (run/'model_reconstruction.pt').write_bytes(b'analytic fixture; no neural checkpoint')
                (run/'selection.json').write_text(json.dumps({'rule':'final_update','completed_updates':3,
                    'score_based_selection':False,'withheld_beams':int(kind=='beams'),'withheld_times':int(kind=='times')}))
                pd.DataFrame({'step':[1,3],'data_loss':[1.,.5],**{k:[1.,1.] for k in ('fxx_rms','fxy_rms','fyy_rms','ftt_rms')}}).to_csv(run/'history.csv',index=False)
                with patch('inr_isr.experiments.analysis.torch_load_checkpoint',return_value=checkpoint),patch('inr_isr.experiments.analysis.build_model_from_checkpoint',return_value=AnalyticFunction()):
                    analyze(c,run,root/'analysis',cpu=True)
                summary=json.loads((root/'analysis/summary.json').read_text())
                self.assertEqual(set(summary['observations']),{'train','withheld'})
                self.assertFalse(summary['score_based_selection'])
                self.assertEqual(summary['checkpoint']['step'],3)
                q=pd.read_csv(root/'analysis/field_queries_predictions.csv')
                self.assertIn('nearest_training_time_sec',q)
                self.assertTrue((root/'analysis/withheld_predictions.csv').exists())
