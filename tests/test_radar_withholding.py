"""Native partition and analytic assessment checks; no neural model or training."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

import test_radar_experiment as fixtures
from inr_isr.experiments.contract import Contract, digest
from inr_isr.experiments.radar import native_dataset, source_summary
from inr_isr.experiments.radar_withholding import (
    retained_dataset, training_dataset, partition_summary, annotate_support, validate_assessment_inputs)
from inr_isr.experiments.radar_analysis import support, analyze_radar, final_checkpoint
from inr_isr.experiments.validation import ReconstructionMonitor
from inr_isr.training.uncertainties import weighting_definition


def fixture(root, kind='beams', domain='log'):
    data, _ = fixtures.RadarChecks().fixture(root)
    with h5py.File(root/'radar.h5', 'r+') as stream:
        ne = stream['FittedParams/Ne'][()]
        stream['FittedParams/dNe'][...] = np.where(np.isfinite(ne), ne*.1, 1.)
    source = native_dataset(root/'radar.h5', data['training'])
    full = source.df.copy()
    geometry = pd.read_csv(root/'roles.csv', float_precision='round_trip')
    rule = {'kind': kind, 'beam_indices': [3] if kind == 'beams' else [],
            'time_indices': [1] if kind == 'times' else [],
            'times_sec': [float(full.loc[full.time_index.eq(1), 't_sec'].iloc[0])] if kind == 'times' else [],
            'require_same_hull': False}
    data['source_preprocessing'] = {'coord_scalers': source.coord_scalers, 'target_scaler': source.target_scaler}
    source = retained_dataset(source, rule)
    mask = full.beam_index.isin(rule['beam_indices']) if kind == 'beams' else full.time_index.isin(rule['time_indices'])
    labelled = geometry.copy()
    labelled['role'] = np.where(labelled.beam_index.isin(rule['beam_indices']), 'withheld', 'train')
    for name, frame in [('source_roles', geometry), ('source_observations', full),
                        ('roles', labelled), ('train', full.loc[~mask]), ('withheld', full.loc[mask])]:
        path = root/f'{name}.csv';frame.to_csv(path, index=False)
        data['inputs'][name] = {'path': str(path), 'sha256': digest(path)}
    data['inputs']['hdf5']['sha256'] = digest(root/'radar.h5')
    data.update(schema='shared-xyt-radar-withholding-1', withholding=rule,
                partition_summary=partition_summary(full, full.loc[~mask], full.loc[mask]),
                preprocessing={'coord_scalers': source.coord_scalers, 'target_scaler': source.target_scaler})
    data['roles']['train'] = sorted(map(int, source.df.beam_index.unique()))
    data['training'].update(data_loss_domain=domain, linear_density_reference=1e12 if domain == 'linear' else None,
                            data_weighting='reported_dNe')
    data['data_loss_definition'] = {k: data['training'][k] for k in ('data_loss_domain', 'linear_density_reference')}
    data['data_weighting_definition'] = weighting_definition('reported_dNe')
    path = root/'contract.json';path.write_text(json.dumps(data))
    return Contract.load(path), source


@pytest.mark.parametrize('kind', ['beams', 'times'])
@pytest.mark.parametrize('domain', ['log', 'linear'])
def test_native_partition_and_aligned_uncertainties(tmp_path, kind, domain):
    before = torch.random.get_rng_state().clone()
    contract, expected = fixture(tmp_path, kind, domain)
    actual = training_dataset(contract)
    assert actual.h5_path == expected.h5_path
    pd.testing.assert_frame_equal(actual.df, expected.df)
    assert actual.n_samples == 8 and contract.data['partition_summary']['withheld_rows'] == 3
    assert actual.coord_scalers == expected.coord_scalers and actual.target_scaler == expected.target_scaler
    assert torch.equal(before, torch.random.get_rng_state())
    validate_assessment_inputs(contract, dataset=actual)


def test_held_extrema_do_not_fit_normalization_or_targets(tmp_path):
    contract, dataset = fixture(tmp_path)
    full = pd.read_csv(contract.file('source_observations'))
    assert dataset.target_scaler['max'] == dataset.df.log10_Ne.max()
    assert dataset.target_scaler['max'] < full.log10_Ne.max()
    dataset.values[0, 0] += 1
    with pytest.raises(ValueError, match='training tensors'):
        validate_assessment_inputs(contract, dataset)


@pytest.mark.parametrize('change', ['extra_row', 'missing_row', 'changed_dNe', 'held_target'])
def test_exact_native_partition_rejects_leaks_and_tampering(tmp_path, change):
    contract, _ = fixture(tmp_path)
    name = 'withheld' if change == 'held_target' else 'train'
    frame = pd.read_csv(contract.file(name), float_precision='round_trip')
    if change == 'extra_row':frame = pd.concat([frame, pd.read_csv(contract.file('withheld')).iloc[:1]])
    if change == 'missing_row':frame = frame.iloc[:-1]
    if change == 'changed_dNe':frame.loc[0, 'dNe'] *= 2
    if change == 'held_target':frame.loc[0, 'log10_Ne'] += 1
    path = Path(contract.file(name));frame.to_csv(path, index=False)
    contract.data['inputs'][name]['sha256'] = digest(path)
    with pytest.raises(ValueError, match='observation partition'):
        validate_assessment_inputs(contract)


@pytest.mark.parametrize('change', ['endpoint', 'mixed', 'false_count', 'role', 'hull'])
def test_declared_rule_counts_roles_and_hull_are_enforced(tmp_path, change):
    contract, _ = fixture(tmp_path, 'times' if change in ('endpoint', 'mixed') else 'beams')
    if change == 'endpoint':contract.data['withholding'].update(time_indices=[0], times_sec=[0.])
    if change == 'mixed':contract.data['withholding']['beam_indices'] = [1]
    if change == 'false_count':contract.data['partition_summary']['training_rows'] += 1
    if change == 'role':contract.data['roles']['train'] = [0]
    if change == 'hull':contract.data['withholding']['require_same_hull'] = True
    with pytest.raises(ValueError):validate_assessment_inputs(contract)


@pytest.mark.parametrize('kind', ['beams', 'times'])
def test_native_dispatch_and_final_save_never_select_on_held_values(tmp_path, kind):
    from inr_isr.experiments.__main__ import train
    contract, dataset = fixture(tmp_path, kind)
    with patch('training_engine.train_window') as engine:
        train(contract, tmp_path/'dispatch', smoke=True, cpu=True)
    args = engine.call_args.args[0]
    pd.testing.assert_frame_equal(engine.call_args.kwargs['dataset_factory'](args).df, dataset.df)
    assert args.num_steps == 3 and args.data_weighting == 'reported_dNe'
    output = tmp_path/'save';output.mkdir()
    monitor = ReconstructionMonitor(contract, dataset, output)
    with patch('inr_isr.experiments.validation.predict_log10_on_coords', side_effect=AssertionError('No assessment during fitting')):
        monitor.after_update(3, 3, SimpleNamespace(state_dict=lambda: {'fixture': 'no neural parameters'}), {}, 'cpu')
    selection = json.loads((output/'selection.json').read_text())
    assert selection['score_based_selection'] is False
    assert selection['withheld_beams'] == int(kind == 'beams')
    assert selection['withheld_times'] == int(kind == 'times')


def test_real_gap_annotations_use_native_timestamps_not_nominal_cadence(tmp_path):
    contract, dataset = fixture(tmp_path, 'times')
    held = pd.read_csv(contract.file('withheld'))
    query = held.iloc[:1].copy()
    result = annotate_support(support(query, dataset.df, 1327402860.), dataset.df, held, contract.data['withholding'])
    assert result.bracketing_training_gap_sec.iloc[0] == 360.
    assert result.nearest_training_time_sec.iloc[0] == 131.
    assert result.inside_withheld_time_gap.iloc[0]


@pytest.mark.parametrize('kind', ['beams', 'times'])
def test_analytic_full_analysis_assesses_held_rows_and_preserves_native_device_policy(tmp_path, kind):
    class AnalyticFunction:
        def __call__(self, q):return q[:, :1]**2 + q[:, 1:2]**2 + q[:, 2:3]**2
        def parameters(self):return iter(())
    contract, ds = fixture(tmp_path, kind)
    run = tmp_path/'run';run.mkdir()
    config = contract.data['training'] | {'num_steps': 3, 'output_dir': str(run), 'cpu': True,
        'data_weighting_definition': contract.weighting_definition(),
        'data_loss_definition': contract.loss_definition()}
    (run/'run_config.json').write_text(json.dumps(config))
    (run/'selection.json').write_text(json.dumps({'rule': 'final_update', 'completed_updates': 3,
        'score_based_selection': False, 'withheld_beams': int(kind == 'beams'), 'withheld_times': int(kind == 'times')}))
    (run/'model_reconstruction.pt').write_bytes(b'analytic fixture; no neural checkpoint')
    checkpoint = {'step': 3, 'config': config, 'checkpoint_type': 'final_reconstruction_after_update',
        'experiment_id': contract.data['id'], 'experiment_contract_sha256': digest(contract.path),
        'coord_scalers': ds.coord_scalers, 'target_scaler': ds.target_scaler}
    pd.DataFrame({'step': [1, 3], 'data_loss': [1., .5], 'total_loss': [1., .5],
                 **{c: [1., 1.] for c in ('fxx_rms', 'fxy_rms', 'fyy_rms', 'ftt_rms', 'grad_norm_total', 'grad_norm_data')}}).to_csv(run/'history.csv', index=False)
    with patch('inr_isr.experiments.radar_analysis.torch_load_checkpoint', return_value=checkpoint), \
         patch('inr_isr.experiments.radar_analysis.build_model_from_checkpoint', return_value=AnalyticFunction()):
        analyze_radar(contract, run, tmp_path/'analysis', cpu=True)
    summary = json.loads((tmp_path/'analysis/summary.json').read_text())
    assert summary['assessment_only'] and summary['withheld_assessment']['rows'] == 3
    assert summary['withheld_beams'] == int(kind == 'beams')
    assert summary['withheld_times'] == int(kind == 'times')
    assert (tmp_path/'analysis/withheld_measurements.png').is_file()
    assert 'primary_field_score' not in summary
