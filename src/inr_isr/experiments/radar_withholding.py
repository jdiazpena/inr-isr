"""Explicit native-radar observation partitions; withheld targets never fit scalers.

The original all-observation validator is applied to the complete source before
partitioning. Retained rows use the existing dataset normalization kernel and
the same training engine. No model or optimizer is defined here.
"""
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree


def observation_mask(frame, rule):
    beams = rule['beam_indices']
    indices = rule['time_indices']
    times = rule['times_sec']
    if any(len(set(values)) != len(values) for values in (beams, indices, times)):
        raise ValueError('Duplicate radar exclusion declarations')
    if rule['kind'] == 'beams':
        if not beams or indices or times or not set(beams) < set(frame.beam_index):
            raise ValueError('Beam assessment needs a proper native beam subset and no removed times')
        return frame.beam_index.isin(beams)
    if rule['kind'] == 'times':
        native = frame[['time_index', 't_sec']].drop_duplicates().sort_values('time_index')
        selected = native[native.time_index.isin(indices)]
        if (beams or not indices or len(selected) != len(indices)
                or selected.t_sec.tolist() != sorted(times)
                or set(indices) & {int(native.time_index.min()), int(native.time_index.max())}):
            raise ValueError('Time assessment needs declared native interior records and no removed beams')
        return frame.time_index.isin(indices)
    raise ValueError('Declare separate beam-only or time-only radar assessment')


def retained_dataset(dataset, rule):
    from datasets import normalize_minus1_plus1
    from inr_isr.data.window import prepare_selected_window
    retained = dataset.df.loc[~observation_mask(dataset.df, rule)].copy()
    # The temporary full-source scalers are replaced, never reused by a model.
    prepare_selected_window(dataset, retained, drop_bad_rows=False, report_drops=False,
        dropped_prefix='Radar assessment', empty_message='No retained radar observations',
        normalize=normalize_minus1_plus1)
    return dataset


def training_dataset(contract):
    from .radar import native_dataset
    return retained_dataset(native_dataset(contract.file('hdf5'), contract.data['training']),
                            contract.data['withholding'])


def partition_summary(source, train, held):
    return {'source_rows': len(source), 'training_rows': len(train), 'withheld_rows': len(held),
            'available_beams': int(source.beam_index.nunique()),
            'training_beams': int(train.beam_index.nunique()),
            'withheld_beams': int(len(set(source.beam_index) - set(train.beam_index))),
            'source_records': int(source.time_index.nunique()),
            'training_records': int(train.time_index.nunique()),
            'withheld_records': int(len(set(source.time_index) - set(train.time_index))),
            'training_midpoint_gaps_sec': np.diff(np.sort(train.unix_mid.unique())).tolist()}


def _same_table(actual, expected, label):
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise ValueError(f'{label} differs from the declared native observation partition') from error


def validate_assessment_inputs(contract, dataset=None):
    from .contract import Contract
    from .radar import native_dataset, validate_radar_inputs
    from .window import ExperimentWindow
    data = contract.data
    if not contract.radar_withholding or data.get('selection_every') != 0:
        raise ValueError('Radar assessment requires explicit final-update saving without score selection')
    if any(name in data['inputs'] for name in ('development', 'reserved', 'scene')):
        raise ValueError('Radar assessment has no selection groups or synthetic truth')
    source = native_dataset(contract.file('hdf5'), data['training'])
    complete = deepcopy(data)
    complete['schema'] = 'shared-xyt-radar-reconstruction-1'
    complete['inputs'] = {name: deepcopy(data['inputs'][name])
                          for name in ('hdf5', 'queries', 'probes')}
    complete['inputs'].update(train=deepcopy(data['inputs']['source_observations']),
                              roles=deepcopy(data['inputs']['source_roles']))
    complete['roles'] = {'train': sorted(map(int, source.df.beam_index.unique())),
                         'development': [], 'reserved': []}
    complete['preprocessing'] = data['source_preprocessing']
    # This preserves every original all-beam completeness/UTC/quality guard.
    validate_radar_inputs(Contract(contract.path, complete), dataset=source)
    full = pd.read_csv(contract.file('source_observations'), float_precision='round_trip')
    train = pd.read_csv(contract.file('train'), float_precision='round_trip')
    held = pd.read_csv(contract.file('withheld'), float_precision='round_trip')
    mask = observation_mask(full, data['withholding'])
    if not mask.any() or mask.all():
        raise ValueError('Assessment must retain and withhold observations')
    _same_table(train, full.loc[~mask], 'Training table')
    _same_table(held, full.loc[mask], 'Withheld table')
    geometry = pd.read_csv(contract.file('roles'), float_precision='round_trip')
    original_geometry = pd.read_csv(contract.file('source_roles'), float_precision='round_trip')
    expected_geometry = original_geometry.copy()
    expected_geometry['role'] = np.where(
        expected_geometry.beam_index.isin(data['withholding']['beam_indices']), 'withheld', 'train')
    _same_table(geometry, expected_geometry, 'Beam roles')
    if set(data['roles']['train']) != set(train.beam_index):
        raise ValueError('Training beam declaration differs from retained observations')
    if data['withholding'].get('require_same_hull', False):
        vertices = ConvexHull(original_geometry[['x_km', 'y_km']].to_numpy()).vertices
        if not set(original_geometry.iloc[vertices].beam_index).issubset(set(train.beam_index)):
            raise ValueError('This assessment must preserve every original hull vertex')
    actual_summary = partition_summary(full, train, held)
    if data['partition_summary'] != actual_summary:
        raise ValueError('Declared radar partition counts or retained timestamp gaps differ')
    expected = retained_dataset(source, data['withholding'])
    if data['preprocessing'] != {'coord_scalers': expected.coord_scalers,
                                 'target_scaler': expected.target_scaler}:
        raise ValueError('Radar assessment scalers must fit retained observations only')
    if dataset is not None:
        _same_table(dataset.df, expected.df, 'Actual training dataset')
        for name in ('coords_raw', 'values_raw', 'coords', 'values'):
            if not np.array_equal(np.asarray(getattr(dataset, name)), np.asarray(getattr(expected, name))):
                raise ValueError('Actual training tensors differ from retained native normalization')
        if dataset.coord_scalers != expected.coord_scalers or dataset.target_scaler != expected.target_scaler:
            raise ValueError('Actual training scalers use a different observation set')
    ExperimentWindow.from_contract(contract).validate_inputs(contract).assert_scalers(expected.coord_scalers, True)
    return expected if dataset is None else dataset


def annotate_support(table, train, held, rule):
    """Add assessment labels to support already computed from retained data."""
    out = table.copy()
    if rule['kind'] == 'beams':
        locations = held[['x_km', 'y_km']].drop_duplicates().to_numpy()
        out['nearest_withheld_beam_km'] = cKDTree(locations).query(out[['x_km', 'y_km']].to_numpy())[0]
        out['closer_to_withheld_beam'] = out.nearest_withheld_beam_km < out.nearest_training_km
    else:
        times = np.sort(train.t_sec.unique())
        q = out.t_sec.to_numpy()
        i = np.clip(np.searchsorted(times, q, side='right') - 1, 0, len(times) - 2)
        left, right = times[i], times[i+1]
        removed = np.sort(held.t_sec.unique())
        contains = np.any((removed[None, :] > left[:, None]) &
                          (removed[None, :] < right[:, None]), axis=1)
        out['inside_withheld_time_gap'] = contains & (q > left) & (q < right)
        out['time_gap_fraction'] = (q - left) / (right - left)
    return out


def assessment_results(contract, prediction, output):
    """Evaluate withheld measurements once after fitting; never select a model."""
    from .contract import grouped_metrics
    from .analysis import error_metrics
    held = pd.read_csv(contract.file('withheld'), float_precision='round_trip')
    held['prediction_log10'] = prediction(held)
    held['residual_dex'] = held.prediction_log10 - held.log10_Ne
    if contract.weighting_definition()['data_weighting'] == 'reported_dNe':
        held['sigma_log10'] = held.dNe / (held.Ne * np.log(10.))
        held['standardized_log_residual'] = held.residual_dex / held.sigma_log10
        with np.errstate(over='raise', invalid='raise'):
            held['standardized_linear_residual'] = (np.power(10., held.prediction_log10) - held.Ne) / held.dNe
    derived = ['prediction_log10', 'residual_dex']
    if 'standardized_log_residual' in held:
        derived += ['sigma_log10', 'standardized_log_residual', 'standardized_linear_residual']
    if not np.isfinite(held[derived].to_numpy()).all():
        raise ValueError('Nonfinite withheld measurement prediction or residual')
    held.to_csv(output / 'withheld_predictions.csv', index=False)
    metrics = grouped_metrics(held.prediction_log10.to_numpy(), held.log10_Ne.to_numpy(), held.beam_index.to_numpy())
    pd.DataFrame(metrics['per_beam']).rename(columns={'beam_id': 'beam_index'}).to_csv(
        output / 'withheld_residuals_by_beam.csv', index=False)
    pd.DataFrame([{'time_index': int(i), 'unix_mid': float(g.unix_mid.iloc[0]),
                   **error_metrics(g.residual_dex.to_numpy())} for i, g in held.groupby('time_index')]).to_csv(
        output / 'withheld_residuals_by_time.csv', index=False)
    result = {'metrics': metrics, 'rows': len(held),
              'meaning': 'Prediction of explicitly withheld noisy radar estimates by the reduced-input model; not plasma truth or the all-beam model error.'}
    if 'standardized_log_residual' in held:
        result['standardized_log_mean_square'] = float(np.mean(held.standardized_log_residual ** 2))
        result['standardized_linear_mean_square'] = float(np.mean(held.standardized_linear_residual ** 2))
    return held, result
