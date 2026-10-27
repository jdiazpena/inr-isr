"""Development reconstruction analysis against a frozen analytic log-density field."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial import Delaunay

from inr_isr.checkpoints import torch_load_checkpoint, build_model_from_checkpoint
from inr_isr.device import resolve_device, device_record
from inr_isr.prediction import predict_log10_on_coords
from inr_isr.transforms import normalize_coords_from_scalers, denormalize_target
from inr_isr.training.derivatives import coordinate_gradient, hessian_rows
from inr_isr.synthetic.field import scene_from_definition
from .contract import digest, grouped_metrics
from .window import ExperimentWindow


def error_metrics(error):
    e = np.asarray(error, dtype=float)
    if not np.isfinite(e).all():
        raise ValueError(f"Nonfinite prediction/error: {int((~np.isfinite(e)).sum())} values")
    if not len(e):
        return {"count": 0, "rmse": None, "mae": None, "bias": None}
    return {"count": len(e), "rmse": float(np.sqrt(np.mean(e**2))),
            "mae": float(np.mean(abs(e))), "bias": float(np.mean(e))}


def physical_derivatives(model, normalized, coord_scalers, target_scaler, device, chunk=512):
    """Reuse shared autograd kernels; apply separate physical chain factors."""
    scale = .5 * (target_scaler["max"] - target_scaler["min"])
    factors = np.array([2 / (coord_scalers[c]["max"] - coord_scalers[c]["min"])
                        for c in ("x_km", "y_km", "t_sec")])
    values = {name: [] for name in ("x", "y", "t", "xx", "xy", "yy", "tt")}
    for start in range(0, len(normalized), chunk):
        points = torch.from_numpy(normalized[start:start + chunk]).to(device).detach().clone().requires_grad_(True)
        first = coordinate_gradient(model(points), points, only_inputs=True)
        rows = hessian_rows(first, points, axes=("x", "y", "t"), selected=("x", "y", "t"), only_inputs=True)
        for i, name in enumerate(("x", "y", "t")):
            values[name].append(first[:, i].detach().cpu().numpy().astype(float) * scale * factors[i])
        for name, i, j in (("xx", 0, 0), ("xy", 0, 1), ("yy", 1, 1), ("tt", 2, 2)):
            values[name].append(rows[("x", "y", "t")[i]][:, j].detach().cpu().numpy().astype(float)
                                * scale * factors[i] * factors[j])
    return {key: np.concatenate(parts) for key, parts in values.items()}


def crossings(coordinates, values, level):
    """Linear crossing coordinates; exact sampled zeros are counted once."""
    x, e = np.asarray(coordinates), np.asarray(values) - level
    exact = x[e == 0].tolist()
    i = np.flatnonzero(e[:-1] * e[1:] < 0)
    return np.unique(np.r_[exact, x[i] - e[i] * (x[i + 1] - x[i]) / (e[i + 1] - e[i])])


def half_height_interval(x, enhancement, level=.5):
    """Interval around the largest enhancement; plateau maxima need no unique x."""
    peak = int(np.argmax(enhancement))
    found = crossings(x, enhancement, level)
    left, right = found[found < x[peak]], found[found > x[peak]]
    return (float(left[-1]), float(right[0]), len(found)) if len(left) and len(right) else (None, None, len(found))


def width_around_peak(x, enhancement, level=.5):
    """Width at the prescribed increment's half-height, not half a fitted maximum."""
    left, right, count = half_height_interval(x, enhancement, level)
    return (None if left is None else right - left), count


def feature_analysis(scene, predict, settings, output, training_geometry=None, *, window, domain):
    front_rows, profile_rows = [], []
    xy = None if training_geometry is None else training_geometry[["x_km", "y_km"]].to_numpy()
    hull = Delaunay(xy) if xy is not None and len(xy) >= 3 and np.linalg.matrix_rank(xy - xy.mean(axis=0)) == 2 else None
    xmin, xmax = domain["x_bounds_km"]
    ymin, ymax = domain["y_bounds_km"]
    y = np.linspace(ymin, ymax, settings.get("front_profile_samples", 205))
    locations = settings.get("front_x_fractions", [.1, .3, .5, .7, .9])
    times = window.sample_times(settings.get("front_time_step_sec", 300), settings.get("query_time_offset_sec", 5))
    for t in times:
        for fraction in locations:
            x = xmin + fraction * (xmax - xmin)
            truth = scene.evaluate(np.full_like(y, x), y, t, derivatives=False)
            center = float(scene.boundary_y(x, t))
            contaminated = bool((truth["patch_enhancement_dex"][abs(y - center) <= settings.get("front_patch_exclusion_km", 180)] > settings.get("front_contamination_threshold_dex", .05)).any())
            record = {"x_km": x, "t_sec": float(t), "truth_center_y_km": center,
                      "center_inside_training_hull": None if hull is None else bool(hull.find_simplex([[x, center]])[0] >= 0),
                      "profile_inside_training_hull_fraction": None if hull is None else float(np.mean(hull.find_simplex(np.column_stack([np.full_like(y, x), y])) >= 0)),
                      "patch_contaminated": contaminated, "missing_center": None,
                      "pred_center_y_km": None, "center_error_km": None, "crossing_count": None,
                      "width_error_km": None}
            if not contaminated:
                prediction = predict(np.full_like(y, x), y, np.full_like(y, t))
                low_level, high_level = scene.front["low_log10_ne"], scene.front["high_log10_ne"]
                middle = crossings(y, prediction, (low_level + high_level) / 2)
                record.update(missing_center=not bool(len(middle)), crossing_count=len(middle))
                if len(middle):
                    pcenter = float(middle[np.argmin(abs(middle - center))])
                    low, high = (crossings(y, prediction, low_level + fraction * (high_level - low_level)) for fraction in settings["front_fraction_bounds"])
                    low, high = low[low < pcenter], high[high > pcenter]
                    record.update(pred_center_y_km=pcenter, center_error_km=pcenter - center)
                    if len(low) and len(high):
                        record["width_error_km"] = float(high[0] - low[-1] - scene.front["vertical_width_10_90_km"])
                profile_rows.extend({"x_km": x, "y_km": float(yy), "t_sec": float(t),
                                     "truth_log10": float(u), "prediction_log10": float(p)}
                                    for yy, u, p in zip(y, truth["log10_Ne"], prediction))
            front_rows.append(record)
    fronts = pd.DataFrame(front_rows)
    fronts.to_csv(output / "front_metrics.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(output / "front_profiles.csv", index=False)
    patch_rows, patch_profiles = [], []
    patch_center = xmin + settings.get("patch_reference_x_fraction", .5) * (xmax - xmin)
    for patch in scene.patches:
        t = (patch_center - patch.x0_km) / scene.speed
        if not window.start <= t <= window.end or not ymin <= patch.y0_km <= ymax:
            continue
        radius = patch.outer_radius_km + settings.get("patch_profile_padding_km", 30)
        x = np.linspace(max(xmin, patch_center - radius), min(xmax, patch_center + radius), settings.get("patch_profile_samples", 501))
        y = np.full_like(x, patch.y0_km)
        truth = scene.evaluate(x, y, t, derivatives=False)
        prediction = predict(x, y, np.full_like(x, t))
        enhancement = prediction - truth["background_log10_Ne"]
        left, right, count = half_height_interval(x, enhancement, patch.enhancement_dex / 2)
        width = None if left is None else right - left
        peak = int(np.argmax(enhancement))
        row = {"patch_id": patch.name, "region": patch.region, "edge_class": patch.edge_class,
               "t_sec": float(t), "height_dex": float(enhancement[peak]),
               "height_error_dex": float(enhancement[peak] - patch.enhancement_dex),
               "position_error_km": None if left is None else (left + right) / 2 - patch_center,
               "maximum_x_km": float(x[peak]), "missing_half_height": width is None,
               "profile_inside_training_hull_fraction": None if hull is None else float(np.mean(hull.find_simplex(np.column_stack([x, y])) >= 0)),
               "half_height_crossings": count, "width_error_km": None if width is None else width - patch.half_height_diameter_km,
               "profile_clipped": patch_center - radius < xmin or patch_center + radius > xmax}
        patch_rows.append(row)
        patch_profiles.extend({"patch_id": patch.name, "x_km": float(xx), "y_km": patch.y0_km,
            "t_sec": float(t), "truth_log10": float(u), "prediction_log10": float(p),
            "truth_enhancement_dex": float(e), "prediction_enhancement_dex": float(ep)}
            for xx, u, p, e, ep in zip(x, truth["log10_Ne"], prediction, truth["patch_enhancement_dex"], enhancement))
    patches = pd.DataFrame(patch_rows)
    patches.to_csv(output / "patch_metrics.csv", index=False)
    profiles = pd.DataFrame(patch_profiles)
    profiles.to_csv(output / "patch_profiles.csv", index=False)
    if len(profiles):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for ax, edge in zip(axes, ("sharp", "gradual")):
            candidates = patches[patches.edge_class == edge]
            if len(candidates):
                name = candidates.iloc[0].patch_id
                p = profiles[profiles.patch_id == name]
                ax.plot(p.x_km, p.truth_log10, label="truth")
                ax.plot(p.x_km, p.prediction_log10, label="reconstruction")
                ax.set_title(f"{edge}: {name}")
            ax.set(xlabel="x (km)", ylabel="log10 density", ylim=settings.get("log_density_limits", [9.5, 12.5]))
            if len(candidates):
                ax.legend()
            else:
                ax.text(.5, .5, f"No {edge} profile in this window", transform=ax.transAxes,
                        ha="center", va="center")
        fig.savefig(output / "patch_profiles.png", dpi=140)
        plt.close(fig)
    valid_front = fronts[~fronts.patch_contaminated]
    return {"front_profiles": len(fronts), "patch_contaminated_front_profiles": int(fronts.patch_contaminated.sum()),
        "missing_front_crossings": int(valid_front.missing_center.eq(True).sum()),
        "multiple_front_crossings": int(valid_front.crossing_count.gt(1).sum()),
        "missing_front_widths": int(valid_front.width_error_km.isna().sum()),
        "front_center_error_km": error_metrics(valid_front.center_error_km.dropna().to_numpy()),
        "front_centres_inside_training_hull": int(valid_front.center_inside_training_hull.eq(True).sum()),
        "front_width_error_km": error_metrics(valid_front.width_error_km.dropna().to_numpy()),
        "patch_profiles": len(patches), "missing_patch_half_heights": int(patches.missing_half_height.sum()) if len(patches) else 0,
        "clipped_patch_profiles": int(patches.profile_clipped.sum()) if len(patches) else 0,
        "multiple_patch_half_heights": int(patches.half_height_crossings.gt(2).sum()) if len(patches) else 0,
        "patch_height_error_dex": error_metrics(patches.height_error_dex.to_numpy() if len(patches) else []),
        "patch_position_error_km": error_metrics(patches.position_error_km.dropna().to_numpy() if len(patches) else []),
        "patch_width_error_km": error_metrics(patches.width_error_km.dropna().to_numpy() if len(patches) else []),
        "caution": "Truth-assisted associations; successful-error summaries must be read with all failure counts."}


def analyze(contract, run, output, *, cpu=False):
    window = ExperimentWindow.from_contract(contract).validate_inputs(contract)
    device = resolve_device("cpu" if cpu else "cuda")
    checkpoint_path = Path(run) / "model_best_development.pt"
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    if checkpoint.get("experiment_contract_sha256") != digest(contract.path):
        raise ValueError("Selected model belongs to another experiment contract")
    scalers, target = checkpoint["coord_scalers"], checkpoint["target_scaler"]
    window.assert_scalers(scalers, contract.data.get('window_policy', {}).get('require_observed_endpoints', False))
    model = build_model_from_checkpoint(checkpoint, device)
    settings = contract.data["analysis"]
    definition = json.loads(contract.file("scene").read_text())
    domain = contract.data.get("domain") or definition.get("domain") or {
        "x_bounds_km": [scalers["x_km"]["min"], scalers["x_km"]["max"]],
        "y_bounds_km": [scalers["y_km"]["min"], scalers["y_km"]["max"]]}
    xmin, xmax = domain["x_bounds_km"]
    ymin, ymax = domain["y_bounds_km"]
    log_min, log_max = settings.get("log_density_limits", [9.5, 12.5])
    scene = scene_from_definition(definition, contract.data["speed_km_s"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def predict(x, y, t):
        window.assert_times(t, "prediction")
        coords = normalize_coords_from_scalers(np.asarray(x), np.asarray(y), np.asarray(t), scalers)
        values = predict_log10_on_coords(model, coords, target, device, contract.data["prediction_chunk"])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite reconstruction; analysis refuses to discard failed queries")
        return values

    scores = {"checkpoint": {"path": str(checkpoint_path), "sha256": digest(checkpoint_path), "step": checkpoint["step"]},
              "role": "development; reserved observations not opened", "density_units": "dex", "observations": {}}
    for role in ("train", "development"):
        table = pd.read_csv(contract.file(role))
        prediction = predict(table.x_km.to_numpy(), table.y_km.to_numpy(), table.t_sec.to_numpy())
        scores["observations"][role] = grouped_metrics(prediction, table.log10_Ne.to_numpy(), table.beam_id.to_numpy())
        table["prediction_log10"] = prediction
        table["error_dex"] = prediction - table.log10_Ne
        table.to_csv(output / f"{role}_predictions.csv", index=False)
    queries = pd.read_csv(contract.file("queries"))
    truth = scene.evaluate(queries.x_km.to_numpy(), queries.y_km.to_numpy(), queries.t_sec.to_numpy())
    prediction = predict(queries.x_km.to_numpy(), queries.y_km.to_numpy(), queries.t_sec.to_numpy())
    queries["truth_log10"], queries["prediction_log10"] = truth["log10_Ne"], prediction
    queries["error_dex"] = prediction - truth["log10_Ne"]
    patch = truth["patch_enhancement_dex"] >= settings["patch_threshold_dex"]
    fraction = truth["high_region_fraction"]
    lo, hi = settings["front_fraction_bounds"]
    masks = {"all": np.ones(len(queries), dtype=bool), "inside_hull": queries.inside_training_hull.to_numpy(),
             "outside_hull": ~queries.inside_training_hull.to_numpy(), "patch": patch,
             "transition": (fraction >= lo) & (fraction <= hi),
             "low_interior": (fraction < lo) & ~patch, "high_interior": (fraction > hi) & ~patch}
    if "inside_common_radar_hull" in queries:
        masks["common_radar_interior"] = queries.inside_common_radar_hull.to_numpy()
    distance = queries.nearest_training_km.to_numpy()
    for lower, upper in ((0, 25), (25, 50), (50, 100), (100, np.inf)):
        masks[f"distance_{lower}_{upper}_km"] = (distance >= lower) & (distance < upper)
    regional = []
    for name, mask in masks.items():
        queries[f"mask_{name}"] = mask
        regional.append({"region": name, **error_metrics(queries.error_dex.to_numpy()[mask])})
    queries.to_csv(output / "field_queries_predictions.csv", index=False)
    pd.DataFrame(regional).to_csv(output / "regional_errors_dex.csv", index=False)
    scores["regions"] = regional
    primary = "common_radar_interior" if "common_radar_interior" in masks else "inside_hull"
    scores["primary_development_field_score"] = next(row for row in regional if row["region"] == primary)
    scores["density_predictions_outside_plot_limits"] = int(((prediction < log_min) | (prediction > log_max)).sum())
    n = settings["derivative_query_count"]
    q = queries.iloc[:n].copy()
    coordinates = normalize_coords_from_scalers(q.x_km.to_numpy(), q.y_km.to_numpy(), q.t_sec.to_numpy(), scalers)
    derivative = physical_derivatives(model, coordinates, scalers, target, device, settings.get("derivative_chunk", 512))
    names = {"x": "true_dlog10Ne_dx_km", "y": "true_dlog10Ne_dy_km", "t": "true_dlog10Ne_dt_sec",
             "xx": "true_d2log10Ne_dxx_km2", "xy": "true_d2log10Ne_dxy_km2",
             "yy": "true_d2log10Ne_dyy_km2", "tt": "true_d2log10Ne_dtt_sec2"}
    scores["derivatives"] = {}
    for name, key in names.items():
        reference = truth[key][:n]
        q[f"truth_{name}"], q[f"prediction_{name}"] = reference, derivative[name]
        q[f"error_{name}"] = derivative[name] - reference
        unit = "dex/s" if name == "t" else "dex/s²" if name == "tt" else "dex/km²" if len(name) == 2 else "dex/km"
        error = derivative[name] - reference
        inside = q.inside_training_hull.to_numpy()
        scores["derivatives"][name] = {"units": unit, **error_metrics(error),
            "inside_hull": error_metrics(error[inside]), "outside_hull": error_metrics(error[~inside])}
    q.to_csv(output / "derivative_queries_predictions.csv", index=False)
    map_values = []
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, settings["grid_nx"]), np.linspace(ymin, ymax, settings["grid_ny"]))
    for t in settings["map_times_sec"]:
        true = scene.evaluate(xx, yy, t, derivatives=False)["log10_Ne"]
        pred = predict(xx.ravel(), yy.ravel(), np.full(xx.size, t)).reshape(xx.shape)
        map_values.append((t, true, pred))
    # One common signed-error range for all four maps; actual errors are never clipped numerically.
    error_bound = max(.01, max(float(np.max(abs(p - u))) for _, u, p in map_values))
    geometry = pd.read_csv(contract.file("roles"))
    for t, true, pred in map_values:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for ax, z, title, cmap, lower, upper in zip(axes, (true, pred, pred - true),
                ("truth", "reconstruction", "error (dex)"), ("viridis", "viridis", "RdBu_r"),
                (log_min, log_min, -error_bound), (log_max, log_max, error_bound)):
            image = ax.pcolormesh(xx, yy, z, shading="auto", cmap=cmap, vmin=lower, vmax=upper)
            training = geometry[geometry.role == "train"]
            ax.scatter(training.x_km, training.y_km, c="white", edgecolors="black", s=9, linewidths=.4)
            ax.set(title=title, xlabel="x (km)", ylabel="y (km)", aspect="equal")
            fig.colorbar(image, ax=ax, label="log10 density" if title != "error (dex)" else "dex")
        fig.suptitle(f"{contract.data['case']} at {t / 3600:.4f} h; white dots = training locations")
        fig.savefig(output / f"map_{t:05d}s.png", dpi=140)
        plt.close(fig)
        np.savez_compressed(output / f"map_{t:05d}s.npz", x_km=xx, y_km=yy, truth_log10=true, prediction_log10=pred)
    x, t = np.meshgrid(np.linspace(xmin, xmax, settings.get("cut_x_samples", 101)),
                       window.sample_times(settings.get("cut_time_step_sec", 30), settings.get("query_time_offset_sec", 5)))
    y = np.full_like(x, settings["cut_y_km"])
    true = scene.evaluate(x, y, t, derivatives=False)["log10_Ne"]
    pred = predict(x.ravel(), y.ravel(), t.ravel()).reshape(x.shape)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, z, title in zip(axes, (true, pred, pred - true), ("truth", "reconstruction", "error (dex)")):
        error = title == "error (dex)"
        bound = max(.01, float(np.max(abs(pred - true))))
        im = ax.pcolormesh(x, t / 3600, z, shading="auto", cmap="RdBu_r" if error else "viridis",
                           vmin=-bound if error else log_min, vmax=bound if error else log_max)
        ax.set(title=title, xlabel="x (km)", ylabel="time (hours)")
        fig.colorbar(im, ax=ax, label="dex" if error else "log10 density")
    fig.suptitle(f"Fixed cut y={settings['cut_y_km']} km")
    fig.savefig(output / "space_time_cut.png", dpi=140)
    plt.close(fig)
    np.savez_compressed(output / "space_time_cut.npz", x_km=x, t_sec=t, truth_log10=true, prediction_log10=pred)
    scores["features"] = feature_analysis(scene, predict, settings, output, geometry[geometry.role == "train"], window=window, domain=domain)
    scores["time_window_sec"] = [window.start, window.end]
    history = pd.read_csv(Path(run) / "history.csv")
    development = pd.read_csv(Path(run) / "development_history.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    axes[0].plot(history.step, history.data_loss, label="training minibatch MSE (normalized)")
    axes[0].set_yscale("log")
    axes[1].plot(development.step, development.macro_rmse_dex, label="development macro RMSE (dex)")
    for column in ("fxx_rms", "fxy_rms", "fyy_rms", "ftt_rms"):
        axes[2].plot(history.step, history[column], label=column)
    for ax in axes:
        ax.set_xlabel("completed updates; minibatch loss is pre-update")
        ax.legend(fontsize=7)
    axes[2].set_title("Normalized diagnostic derivatives; components kept separate")
    fig.savefig(output / "training_and_diagnostics.png", dpi=140)
    plt.close(fig)
    (output / "summary.json").write_text(json.dumps(scores, indent=2, allow_nan=False) + "\n")
    (output / "execution_device.json").write_text(json.dumps(device_record("cpu" if cpu else "cuda", device, model=model), indent=2) + "\n")
    print(f"Analysis complete: {output}; selected completed update {checkpoint['step']}")
