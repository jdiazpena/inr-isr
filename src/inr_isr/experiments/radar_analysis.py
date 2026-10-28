"""Real x/y/time reconstruction analysis; no plasma truth is manufactured.

Measurement residuals describe fitting. Dense fields and derivatives describe
the reconstructed function and its support; neither establishes its accuracy
between observations.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree

from inr_isr.checkpoints import torch_load_checkpoint, build_model_from_checkpoint
from inr_isr.device import resolve_device, device_record
from inr_isr.prediction import predict_log10_on_coords
from inr_isr.transforms import normalize_coords_from_scalers
from .analysis import physical_derivatives, error_metrics
from .contract import digest, grouped_metrics
from .window import ExperimentWindow


def support(table, measurements, origin):
    """Recompute support from retained native observations, including missing gates."""
    result = table.copy()
    xy = measurements[["x_km","y_km"]].drop_duplicates().to_numpy()
    points = result[["x_km","y_km"]].to_numpy()
    result["nearest_training_km"] = cKDTree(xy).query(points)[0]
    result["inside_training_hull"] = Delaunay(xy).find_simplex(points) >= 0
    times = np.sort(measurements.t_sec.unique())
    t = result.t_sec.to_numpy()
    i = np.clip(np.searchsorted(times,t,side="right")-1,0,len(times)-2)
    result["bracketing_training_gap_sec"] = times[i+1]-times[i]
    result["nearest_training_time_sec"] = np.minimum(abs(t-times[i]),abs(t-times[i+1]))
    result["inside_observed_time_support"] = (t >= times[0]) & (t <= times[-1])
    result["unix_mid"] = t + origin
    return result


def final_checkpoint(contract, run, device):
    path = Path(run)/"model_reconstruction.pt"
    checkpoint = torch_load_checkpoint(path, device)
    contract.validate_checkpoint_loss(checkpoint)
    selection = json.loads((Path(run)/"selection.json").read_text())
    effective = json.loads((Path(run)/"run_config.json").read_text())
    if (checkpoint.get("experiment_contract_sha256") != digest(contract.path)
            or checkpoint.get("experiment_id") != contract.data["id"]
            or checkpoint.get("checkpoint_type") != "final_reconstruction_after_update"
            or selection.get("rule") != "final_update"
            or selection.get("score_based_selection") is not False
            or selection.get("withheld_beams") != 0 or selection.get("withheld_times") != 0
            or checkpoint.get("step") != selection.get("completed_updates")
            or checkpoint.get("step") != effective.get("num_steps")
            or effective["num_steps"] not in (3,contract.data["training"]["num_steps"])):
        raise ValueError("Real analysis requires this contract's final completed update, never score selection")
    for name,value in contract.data["training"].items():
        if name not in ("num_steps","output_dir","cpu") and effective.get(name) != value:
            raise ValueError(f"Effective training changed: {name}")
    window = ExperimentWindow.from_contract(contract)
    window.assert_scalers(checkpoint["coord_scalers"],True)
    if contract.data["preprocessing"] != {"coord_scalers":checkpoint["coord_scalers"],"target_scaler":checkpoint["target_scaler"]}:
        raise ValueError("Saved normalization differs from this native source/window")
    return path,checkpoint


def plot_histories(history, output):
    """Keep optimization, controller values, derivative components and gradients."""
    history.to_csv(output/"training_history.csv",index=False)
    groups = [("Loss before update",["total_loss","data_loss","curv_xy_weighted","curv_t_weighted"]),
              ("Fixed-probe derivative RMS after update",["fxx_rms","fxy_rms","fyy_rms","ftt_rms"]),
              ("Parameter gradient norms",["grad_norm_total","grad_norm_data","grad_norm_xy_weighted","grad_norm_t_weighted"]),
              ("Fraction of derivatives near zero",["fxx_frac_near_zero","fxy_frac_near_zero","fyy_frac_near_zero","ftt_frac_near_zero"])]
    fig,axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for ax,(title,columns) in zip(axes.ravel(),groups):
        for col in columns:
            if col in history:
                finite = np.isfinite(history[col])
                ax.plot(history.step[finite],history[col][finite],label=col)
        ax.set(title=title,xlabel="completed update")
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
        if title != groups[-1][0]:ax.set_yscale("symlog",linthresh=1e-12)
    fig.savefig(output/"training_and_diagnostics.png",dpi=140)
    plt.close(fig)


def analyze_radar(contract, run, output, *, cpu=False):
    window = ExperimentWindow.from_contract(contract).validate_inputs(contract)
    device = resolve_device("cpu" if cpu else "cuda")
    checkpoint_path,checkpoint = final_checkpoint(contract,run,device)
    scalers,target = checkpoint["coord_scalers"],checkpoint["target_scaler"]
    model = build_model_from_checkpoint(checkpoint,device)
    settings = contract.data["analysis"]
    origin = contract.data["observation_summary"]["unix_origin_sec"]
    obs = pd.read_csv(contract.file("train"),float_precision="round_trip")
    output = Path(output); output.mkdir(parents=True,exist_ok=False)

    def prediction(table):
        window.assert_times(table.t_sec,"real reconstruction query")
        coordinates = normalize_coords_from_scalers(*[table[c].to_numpy() for c in ("x_km","y_km","t_sec")],scalers)
        values = predict_log10_on_coords(model,coordinates,target,device,contract.data["prediction_chunk"])
        if not np.isfinite(values).all():raise ValueError("Nonfinite reconstructed density")
        return values

    obs["prediction_log10"] = prediction(obs)
    obs["residual_dex"] = obs.prediction_log10-obs.log10_Ne
    if contract.weighting_definition()['data_weighting'] == 'reported_dNe':
        obs['sigma_log10'] = obs.dNe / (obs.Ne * np.log(10.))
        obs['standardized_log_residual'] = obs.residual_dex / obs.sigma_log10
        with np.errstate(over='raise', invalid='raise'):
            obs['standardized_linear_residual'] = (np.power(10., obs.prediction_log10) - obs.Ne) / obs.dNe
    obs.to_csv(output/"measurement_residuals.csv",index=False)
    by_beam = grouped_metrics(obs.prediction_log10.to_numpy(),obs.log10_Ne.to_numpy(),obs.beam_index.to_numpy())
    pd.DataFrame(by_beam["per_beam"]).rename(columns={"beam_id":"beam_index"}).to_csv(output/"residuals_by_beam.csv",index=False)
    by_time = [{"time_index":int(i),"unix_mid":float(g.unix_mid.iloc[0]),
                **error_metrics(g.residual_dex.to_numpy())} for i,g in obs.groupby("time_index")]
    pd.DataFrame(by_time).to_csv(output/"residuals_by_time.csv",index=False)
    queries = support(pd.read_csv(contract.file("queries"),float_precision="round_trip"),obs,origin)
    queries["prediction_log10"] = prediction(queries)
    queries.to_csv(output/"field_queries_predictions.csv",index=False)
    q = queries.iloc[:settings["derivative_query_count"]].copy()
    coordinates = normalize_coords_from_scalers(*[q[c].to_numpy() for c in ("x_km","y_km","t_sec")],scalers)
    derivatives = physical_derivatives(model,coordinates,scalers,target,device,settings["derivative_chunk"])
    derivative_summary = {}
    for name,values in derivatives.items():
        if not np.isfinite(values).all():raise ValueError(f"Nonfinite physical derivative {name}")
        q[f"prediction_{name}"] = values
        unit = "dex/s" if name=="t" else "dex/s²" if name=="tt" else "dex/km²" if len(name)==2 else "dex/km"
        derivative_summary[name] = {"units":unit,"count":len(values),"rms":float(np.sqrt(np.mean(values**2))),
                                    "mean_abs":float(np.mean(abs(values))),"max_abs":float(np.max(abs(values)))}
    q.to_csv(output/"derivative_queries_predictions.csv",index=False)
    xmin,xmax = contract.data["domain"]["x_bounds_km"]
    ymin,ymax = contract.data["domain"]["y_bounds_km"]
    xx,yy = np.meshgrid(np.linspace(xmin,xmax,settings["grid_nx"]),np.linspace(ymin,ymax,settings["grid_ny"]))
    limits = settings["log_density_limits"]
    geometry = obs[["beam_index","x_km","y_km"]].drop_duplicates()
    for index,t in enumerate(settings["map_times_sec"]):
        frame = support(pd.DataFrame({"x_km":xx.ravel(),"y_km":yy.ravel(),"t_sec":t}),obs,origin)
        values = prediction(frame).reshape(xx.shape)
        inside = frame.inside_training_hull.to_numpy().reshape(xx.shape)
        fig,ax = plt.subplots(figsize=(7,6),layout="constrained")
        im = ax.pcolormesh(xx,yy,values,shading="auto",vmin=limits[0],vmax=limits[1])
        ax.contour(xx,yy,inside.astype(float),levels=[.5],colors="white",linewidths=.8)
        ax.scatter(geometry.x_km,geometry.y_km,s=15,c="white",edgecolors="black",linewidths=.6)
        stamp = pd.to_datetime(t+origin,unit="s",utc=True).strftime("%H:%M:%S UTC")
        ax.set(xlabel="x east (km)",ylabel="y north (km)",aspect="equal",
               title=f"{contract.data['case']}: {stamp}\nWhite outline: beam hull; outside = spatial extrapolation")
        fig.colorbar(im,ax=ax,label="log10 electron density (m⁻³)")
        fig.savefig(output/f"map_{index:02d}.png",dpi=140);plt.close(fig)
        np.savez_compressed(output/f"map_{index:02d}.npz",x_km=xx,y_km=yy,t_sec=t,unix_mid=t+origin,
                            prediction_log10=values,inside_training_hull=inside)
    common = settings["comparison_unix_bounds"]
    times = np.linspace(common[0]-origin,common[1]-origin,settings["cut_time_samples"])
    x,t = np.meshgrid(np.linspace(xmin,xmax,settings["cut_x_samples"]),times)
    cut = support(pd.DataFrame({"x_km":x.ravel(),"y_km":settings["cut_y_km"],"t_sec":t.ravel()}),obs,origin)
    cut["prediction_log10"] = prediction(cut)
    cut.to_csv(output/"space_time_cut.csv",index=False)
    fig,ax = plt.subplots(figsize=(10,5),layout="constrained")
    im = ax.pcolormesh(x,(t+origin-common[0])/60,cut.prediction_log10.to_numpy().reshape(x.shape),
                       shading="auto",vmin=limits[0],vmax=limits[1])
    ax.set(xlabel="x east (km)",ylabel="minutes since common support start",
           title=f"Reconstruction at y={settings['cut_y_km']} km; no plasma truth available")
    fig.colorbar(im,ax=ax,label="log10 electron density (m⁻³)")
    fig.savefig(output/"space_time_cut.png",dpi=140);plt.close(fig)
    np.savez_compressed(output/"space_time_cut.npz",x_km=x,t_sec=t,unix_mid=t+origin,
                        prediction_log10=cut.prediction_log10.to_numpy().reshape(x.shape),
                        inside_training_hull=cut.inside_training_hull.to_numpy().reshape(x.shape))
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout="constrained")
    axes[0].scatter(obs.log10_Ne,obs.prediction_log10,s=5,alpha=.5)
    axes[0].plot(limits,limits,color="black",linewidth=.7)
    axes[0].set(xlabel="observed log10 density",ylabel="predicted log10 density",title="Measurement fit, not truth error")
    axes[1].plot((obs.unix_mid-common[0])/60,obs.residual_dex,".",markersize=2)
    axes[1].set(xlabel="minutes since common support start",ylabel="residual (dex)")
    fig.savefig(output/"measurement_fit.png",dpi=140);plt.close(fig)
    history = pd.read_csv(Path(run)/"history.csv")
    required = ["fxx_rms","fxy_rms","fyy_rms","ftt_rms","grad_norm_total","grad_norm_data"]
    if any(c not in history or not np.isfinite(history[c]).any() for c in required):
        raise ValueError("Required derivative or parameter-gradient diagnostics are missing")
    plot_histories(history,output)
    summary = {"role":"all-beam real-data reconstruction; measurement residuals are not plasma truth errors",
        "checkpoint":{"path":str(checkpoint_path),"sha256":digest(checkpoint_path),"step":checkpoint["step"]},
        "execution_check_only":checkpoint["step"] != contract.data["training"]["num_steps"],
        "experiment_contract_sha256":digest(contract.path),"score_based_selection":False,
        "source":contract.data["inputs"]["hdf5"],"measurement_policy":contract.data["measurement_policy"],
        "data_loss_definition":contract.loss_definition(),
        "data_weighting_definition":contract.weighting_definition(),
        "observations":contract.data["observation_summary"],"withheld_beams":0,"withheld_times":0,
        "measurement_fit":by_beam,"time_window_sec":[window.start,window.end],
        "comparison_unix_bounds":common,"derivative_magnitudes":derivative_summary,
        "field_queries":len(queries),"inside_hull_queries":int(queries.inside_training_hull.sum()),
        "density_predictions_outside_plot_limits":int((~queries.prediction_log10.between(*limits)).sum()),
        "not_available_without_truth":["field error","derivative error","front/patch position or width error"],
        "limitations":["Residuals use fitted measurements, not independent validation.",
            "The selected native gates span an altitude band, not an exact constant-altitude plane.",
            "The hull is geometric support; sparse beams and local invalid samples still leave gaps.",
            "Two integration products of one event are not independent experiments."]}
    if contract.weighting_definition()['data_weighting'] == 'reported_dNe':
        summary['standardized_measurement_residuals'] = {
            'log_mean_square':float(np.mean(obs.standardized_log_residual ** 2)),
            'linear_mean_square':float(np.mean(obs.standardized_linear_residual ** 2)),
            'relative_dNe_max':float((obs.dNe / obs.Ne).max()),
            'meaning':'Descriptive compatibility with reported errors; not validated chi-square, coverage or plasma truth.'}
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    (output/"execution_device.json").write_text(json.dumps(device_record("cpu" if cpu else "cuda",device,model=model),indent=2)+"\n")
    print(f"Real analysis complete: {output}; final completed update {checkpoint['step']}")
