"""User-executed matched development analysis; final evaluation is explicit."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from .common import fresh_directory, sha256, write_json
from .data import field_values, load_prepared, make_queries, observation_truth, read_tables
from .math import LinearBaseline, feature_summary, metrics


def frozen_final(path, config_identity, checkpoint):
    """Require a separate user-supplied freeze record, never created by training."""
    if path is None:
        raise ValueError("Final evaluation requires an explicit comparison freeze record")
    record = json.loads(Path(path).read_text())
    required = {"comparison_frozen", "config_identity", "checkpoint_sha256", "operator", "frozen_utc"}
    if set(record) != required or record["comparison_frozen"] is not True or not record["operator"] or not record["frozen_utc"]:
        raise ValueError("Invalid comparison freeze record")
    if record["config_identity"] != config_identity or record["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("Freeze record does not bind this comparison/checkpoint")
    return sha256(path)


def score_comparison(truth, predictions, supported, groups=None):
    common = np.isfinite(truth).copy()
    for name in predictions:
        common &= supported[name] & np.isfinite(predictions[name])
    if not common.any():
        raise ValueError("No common supported comparison queries")
    results, group_rows = {}, []
    for name, values in predictions.items():
        results[name], rows = metrics(truth[common], values[common], None if groups is None else groups[common])
        for row in rows:
            group_rows.append({"method": name, **row})
    return {"total_rows": len(truth), "common_supported_rows": int(common.sum()),
            "excluded_rows": int((~common).sum()), "unsupported_by_method": {k: int((~v).sum()) for k, v in supported.items()},
            "methods": results}, group_rows, common


def analyze(case, config_identity, data_path, checkpoint, output, final=False, freeze=None):
    # The final gate precedes every final-target/query-truth read and model call.
    freeze_hash = frozen_final(freeze, config_identity, checkpoint) if final else None
    rows, data_record = load_prepared(data_path, config_identity)
    geometry, times = read_tables(case, verify_input=False)
    training = rows[rows.role == "train"]
    if final:
        observed = pd.read_csv(Path(data_path) / "reserved_metadata.csv", float_precision="round_trip")
        observed["log10_Ne"] = observation_truth(case, observed, allow_final=True)
        coordinates = make_queries(case, geometry, times, "final", allow_final=True)
    else:
        observed = rows[rows.role == "validation"].copy()
        coordinates = np.load(Path(data_path) / "development_coordinates.npy", allow_pickle=False)
    from .training import load_model, predict_observations, predict_field
    from inr_isr.device import device_record
    import torch
    model, state, device = load_model(checkpoint, config_identity, data_path, case["training"]["device"])
    if state["fit_observation_ids"] != training.observation_id.tolist() or state["validation_observation_ids"] != rows[rows.role == "validation"].observation_id.tolist():
        raise ValueError("Checkpoint fitted/scored different observation identities")
    baseline = LinearBaseline(training)
    interpolation, interp_support, extended = baseline.observations(observed)
    predictions = {"INR": predict_observations(model, observed, state["target_scaler"], state["bounds"], device),
                   "interpolation": interpolation, "constant": np.full(len(observed), baseline.constant)}
    support = {"INR": np.ones(len(observed), dtype=bool), "interpolation": interp_support, "constant": np.ones(len(observed), dtype=bool)}
    obs_scores, group_rows, common = score_comparison(observed.log10_Ne.to_numpy(), predictions, support, observed.beam_index.to_numpy())
    obs_scores["endpoint_extended_rows"] = int(extended.sum())
    near = common & observed.within_one_width_of_path.to_numpy(dtype=bool)
    obs_scores["geometry_near_path_rows"] = int(near.sum())
    obs_scores["geometry_near_path_scores"] = {name: metrics(observed.log10_Ne.to_numpy()[near], values[near],
                                                               observed.beam_index.to_numpy()[near])[0]
                                             for name, values in predictions.items()} if near.any() else {}
    obs_scores["endpoint_extended_scores"] = {}
    obs_scores["ordinary_integration_scores"] = {}
    for mask_name, selected in (("endpoint_extended_scores", common & extended), ("ordinary_integration_scores", common & ~extended)):
        if selected.any():
            for name, values in predictions.items():
                obs_scores[mask_name][name] = metrics(observed.log10_Ne.to_numpy()[selected], values[selected], observed.beam_index.to_numpy()[selected])[0]
    truth = field_values(case, coordinates)
    interpolation, interp_support, _ = baseline.predict(coordinates)
    field_predictions = {"INR": predict_field(model, coordinates, state["target_scaler"], state["bounds"], device),
                         "interpolation": interpolation, "constant": np.full(len(coordinates), baseline.constant)}
    field_support = {"INR": np.ones(len(coordinates), dtype=bool), "interpolation": interp_support, "constant": np.ones(len(coordinates), dtype=bool)}
    field_scores, _, field_common = score_comparison(truth, field_predictions, field_support)
    output = fresh_directory(output)
    observed["common_support"] = common
    observed["endpoint_extended"] = extended
    for name, values in predictions.items():
        observed[f"{name}_supported"] = support[name]
        observed[f"{name}_log10_prediction"] = values
        observed[f"{name}_log10_error"] = values - observed.log10_Ne
    observed.to_csv(output / "observation_predictions.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output / "observation_group_metrics.csv", index=False)
    np.savez_compressed(output / "field_queries.npz", coordinates=coordinates, truth_log10=truth,
                        common_support=field_common, **{f"{k}_log10": v for k, v in field_predictions.items()},
                        **{f"{k}_supported": v for k, v in field_support.items()})
    execution = device_record(case["training"]["device"], device, model=model,
                              batch=torch.as_tensor(coordinates[:1], dtype=torch.float32, device=device), amp=False)
    write_json(output / "execution_device.json", execution)
    report = {"role": "final" if final else "development", "config_identity": config_identity,
              "checkpoint_sha256": sha256(checkpoint), "selected_step": state["step"], "freeze_record_sha256": freeze_hash,
              "observations": obs_scores, "instantaneous_field": field_scores,
              "baseline_limit": "Linear interpolation treats training integration averages as midpoint nodal values; endpoint extensions are flagged",
              "final_targets_used": final}
    if not final:
        report["plots"] = development_plots(case, model, state, device, baseline, geometry, times, rows, checkpoint, output)
    write_json(output / "metrics.json", report)
    return report


def development_plots(case, model, state, device, baseline, geometry, times, rows, checkpoint, output):
    """Four compact figures with their full numerical inputs in ignored outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .training import predict_field, predict_observations
    bounds, size = np.asarray(case["bounds"]), case["diagnostics"]["grid_size"]
    x = bounds[0, 0] + (np.arange(size) + 0.5) * (bounds[0, 1] - bounds[0, 0]) / size
    y = bounds[1, 0] + (np.arange(size) + 0.5) * (bounds[1, 1] - bounds[1, 0]) / size
    X, Y = np.meshgrid(x, y)
    inside = Delaunay(geometry.loc[geometry.spatial_role == "train", ["x_km", "y_km"]]).find_simplex(np.c_[X.ravel(), Y.ravel()]) >= 0
    selected_times = times.t_sec.min() + np.asarray(case["diagnostics"]["slice_fractions"]) * (times.t_sec.max() - times.t_sec.min())
    all_values = {name: [] for name in ("truth", "INR", "interpolation", "constant")}
    for t in selected_times:
        coordinates = np.c_[X.ravel()[inside], Y.ravel()[inside], np.full(inside.sum(), t)]
        values = {"truth": field_values(case, coordinates),
                  "INR": predict_field(model, coordinates, state["target_scaler"], state["bounds"], device),
                  "interpolation": baseline.predict(coordinates)[0], "constant": np.full(len(coordinates), baseline.constant)}
        for name in all_values:
            image = np.full(X.size, np.nan)
            image[inside] = values[name]
            all_values[name].append(image.reshape(X.shape))
    arrays = {name: np.asarray(value) for name, value in all_values.items()}
    features = []
    for i, t in enumerate(selected_times):
        reference = feature_summary(x, y, arrays["truth"][i], case["field"]["background_ne_m3"])
        for name, images in arrays.items():
            entry = feature_summary(x, y, images[i], case["field"]["background_ne_m3"])
            entry.update(method=name, t_sec=float(t),
                         peak_position_error_km=float(np.hypot(entry["peak_x_km"] - reference["peak_x_km"],
                                                              entry["peak_y_km"] - reference["peak_y_km"])),
                         excess_amplitude_error_m3=entry["excess_amplitude_m3"] - reference["excess_amplitude_m3"])
            features.append(entry)
    pd.DataFrame(features).to_csv(output / "feature_diagnostics.csv", index=False)
    np.savez_compressed(output / "slice_inputs.npz", x_km=X, y_km=Y, times_sec=selected_times, support=inside.reshape(X.shape), **arrays)
    limits = [min(np.nanmin(a) for a in arrays.values()), max(np.nanmax(a) for a in arrays.values())]
    names = ["truth", "INR", "interpolation", "constant"]
    fig, axes = plt.subplots(len(selected_times), 4, figsize=(13, 3.4 * len(selected_times)), squeeze=False, constrained_layout=True)
    for i, t in enumerate(selected_times):
        for j, name in enumerate(names):
            mesh = axes[i, j].pcolormesh(X, Y, arrays[name][i], shading="nearest", vmin=limits[0], vmax=limits[1])
            for role, marker in (("train", "o"), ("validation", "^"), ("final_test", "s")):
                locations = geometry[geometry.spatial_role == role]
                axes[i, j].scatter(locations.x_km, locations.y_km, s=16, marker=marker,
                                   facecolors="none", edgecolors="white", linewidths=.6, label=role + " locations")
            if i == 0 and j == 0:
                axes[i, j].legend(fontsize=6, loc="upper right")
            axes[i, j].set(title=f"{name}, t={t / 60:.1f} min", xlabel="x [km]", ylabel="y [km]", aspect="equal")
    fig.colorbar(mesh, ax=axes, label="instantaneous log10(Ne / 1 m^-3)")
    fig.savefig(output / "reconstruction.png", dpi=130); plt.close(fig)
    errors = {name: arrays[name] - arrays["truth"] for name in names[1:]}
    limit = max(max(np.nanmax(np.abs(a)) for a in errors.values()), 1e-12)
    fig, axes = plt.subplots(len(selected_times), 3, figsize=(11, 3.4 * len(selected_times)), squeeze=False, constrained_layout=True)
    for i, t in enumerate(selected_times):
        for j, name in enumerate(names[1:]):
            mesh = axes[i, j].pcolormesh(X, Y, errors[name][i], shading="nearest", cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[i, j].set(title=f"{name} error, t={t / 60:.1f} min", xlabel="x [km]", ylabel="y [km]", aspect="equal")
    fig.colorbar(mesh, ax=axes, label="prediction minus instantaneous truth [log10]")
    fig.savefig(output / "errors.png", dpi=130); plt.close(fig)
    history = pd.read_csv(Path(checkpoint).parent / "history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), constrained_layout=True)
    axes[0].plot(history.step, history.training_normalized_mse, label="before-update full training mean")
    valid = history.validation_macro_log10_rmse.notna()
    axes[1].plot(history.step[valid], history.validation_macro_log10_rmse[valid], marker=".", label="after-update validation")
    axes[0].set(xlabel="completed update", ylabel="normalized MSE")
    axes[1].set(xlabel="completed update", ylabel="macro log10 RMSE")
    for axis in axes: axis.legend(fontsize=8)
    fig.savefig(output / "learning.png", dpi=130); plt.close(fig)
    traces = []
    fig, axes = plt.subplots(1, len(case["diagnostics"]["trace_beam_indices"]), figsize=(10, 3.3), squeeze=False, constrained_layout=True)
    for i, beam in enumerate(case["diagnostics"]["trace_beam_indices"]):
        trace = rows[rows.beam_index == beam].copy()
        if not len(trace): raise ValueError("Diagnostic trace must use a training/validation beam")
        trace["INR"] = predict_observations(model, trace, state["target_scaler"], state["bounds"], device)
        trace["interpolation"] = baseline.observations(trace)[0]
        trace["constant"] = baseline.constant
        for name, values in [("truth", trace.log10_Ne), *[(n, trace[n]) for n in names[1:]]]:
            axes[0, i].plot(trace.t_sec / 60, 10 ** values / 1e11, label=name)
        axes[0, i].set(title=f"beam {beam}: {trace.role.iloc[0]}", xlabel="minutes", ylabel="integration-averaged Ne [1e11 m^-3]")
        axes[0, i].legend(fontsize=8)
        traces.append(trace)
    pd.concat(traces).to_csv(output / "trace_inputs.csv", index=False)
    fig.savefig(output / "traces.png", dpi=130); plt.close(fig)
    return ["learning.png", "reconstruction.png", "errors.png", "traces.png"]
