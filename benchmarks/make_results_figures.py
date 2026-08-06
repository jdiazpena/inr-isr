from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/jdiaz/postdoc/codex-inr-radar/inf_fakedata_3d/outputs/velocity_integration_benchmark")
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
SUMMARY = pd.read_csv(ROOT / "benchmark_event_summary.csv")
PAIRS = pd.read_csv(ROOT / "benchmark_regularization_comparison.csv")

COLORS = {
    "data_only": "#5B6B7A",
    "xy030_t030": "#D95F45",
    "density": "#2878B5",
    "gradient": "#E59E2F",
    "spatial": "#1B9E77",
    "temporal": "#9B4D96",
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def label(row: pd.Series) -> str:
    if row["motion"] == "flow_reversal":
        return "Flow reversal\n23 beams | 1.0 km/s | 2 min"
    return (
        f"{int(row['beam_count'])} beams | {row['speed_km_s']:.2f} km/s | "
        f"{int(row['integration_time_sec'] / 60)} min"
    )


def ordered_cases() -> list[str]:
    ordinary = SUMMARY[SUMMARY["motion"] == "left_right"].drop_duplicates("case_id")
    ordinary = ordinary.sort_values(["beam_count", "speed_km_s", "integration_time_sec"], ascending=[False, True, True])
    shear = SUMMARY[SUMMARY["motion"] == "flow_reversal"].drop_duplicates("case_id")
    return ordinary["case_id"].tolist() + shear["case_id"].tolist()


def plot_active_nrmse() -> None:
    cases = ordered_cases()
    y = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    height = 0.36
    for offset, reg in [(-height / 2, "data_only"), (height / 2, "xy030_t030")]:
        vals = []
        for case in cases:
            row = SUMMARY[(SUMMARY.case_id == case) & (SUMMARY.regularization == reg)].iloc[0]
            vals.append(100.0 * row.observed_active_normalized_rmse)
        ax.barh(y + offset, vals, height=height, color=COLORS[reg], label=reg.replace("_", " "))
    case_labels = [label(SUMMARY[SUMMARY.case_id == case].iloc[0]) for case in cases]
    ax.set_yticks(y, case_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Active-region normalized RMSE against radar-integrated truth [%]")
    ax.set_title("Regularization improves the measured-field reconstruction in every pilot case")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "active_nrmse_by_case.png", bbox_inches="tight")
    plt.close(fig)


def plot_regularization_improvement() -> None:
    cases = ordered_cases()
    p = PAIRS.set_index("case_id").loc[cases]
    y = np.arange(len(p))
    density = p["regularization_improvement_pct_observed_active_rmse_Ne"].to_numpy()
    gradient = p["regularization_improvement_pct_gradient_active_rmse"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    height = 0.36
    ax.barh(y - height / 2, density, height=height, color=COLORS["density"], label="density RMSE")
    ax.barh(y + height / 2, gradient, height=height, color=COLORS["gradient"], label="combined-gradient RMSE")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(y, [label(SUMMARY[SUMMARY.case_id == case].iloc[0]) for case in cases])
    ax.invert_yaxis()
    ax.set_xlabel("Improvement from xy030_t030 relative to data-only [%]")
    ax.set_title("Density improves uniformly; one fast case exposes a gradient tradeoff")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "regularization_improvement.png", bbox_inches="tight")
    plt.close(fig)


def plot_beam_effect() -> None:
    d = SUMMARY[(SUMMARY.motion == "left_right") & (SUMMARY.regularization == "xy030_t030")].copy()
    scenarios = sorted(d[["speed_km_s", "integration_time_sec"]].drop_duplicates().itertuples(index=False, name=None))
    x = np.arange(len(scenarios))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for offset, beams, color in [(-width / 2, 23, "#6A8CAF"), (width / 2, 42, "#D95F45")]:
        vals = []
        for speed, integration in scenarios:
            row = d[(d.beam_count == beams) & (d.speed_km_s == speed) & (d.integration_time_sec == integration)].iloc[0]
            vals.append(100.0 * row.observed_active_normalized_rmse)
        ax.bar(x + offset, vals, width=width, color=color, label=f"{beams} beams")
    ax.set_xticks(x, [f"{s:.2f} km/s\n{int(t/60)} min" for s, t in scenarios])
    ax.set_ylabel("Active-region normalized RMSE [%]")
    ax.set_title("Beam support is the strongest experimental factor in the ordinary cases")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "beam_count_effect.png", bbox_inches="tight")
    plt.close(fig)


def plot_loss_ratios() -> None:
    d = SUMMARY[SUMMARY.regularization == "xy030_t030"].copy()
    d["label"] = d.apply(label, axis=1)
    cases = ordered_cases()
    d = d.set_index("case_id").loc[cases]
    y = np.arange(len(d))
    height = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.barh(y - height / 2, d.tail_median_xy_over_data_ref, height=height, color=COLORS["spatial"], label="spatial ratio")
    ax.barh(y + height / 2, d.tail_median_t_over_data_ref, height=height, color=COLORS["temporal"], label="temporal ratio")
    ax.axvline(0.30, color="black", linestyle="--", linewidth=1.0, label="target = 0.30")
    ax.set_yticks(y, d["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Tail-median weighted loss / data reference")
    ax.set_title("Temporal controller reaches its target for 1-2 min data but saturates for 10 min data")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "achieved_loss_ratios.png", bbox_inches="tight")
    plt.close(fig)


def plot_training_vs_dense() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for reg, marker in [("data_only", "o"), ("xy030_t030", "s")]:
        d = SUMMARY[SUMMARY.regularization == reg]
        ax.scatter(
            d.final_training_rmse_log10,
            100.0 * d.observed_active_normalized_rmse,
            s=48,
            marker=marker,
            color=COLORS[reg],
            edgecolor="white",
            linewidth=0.5,
            label=reg.replace("_", " "),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Final RMSE at measured samples [dex, log scale]")
    ax.set_ylabel("Dense active-region normalized RMSE [%]")
    ax.set_title("Near-zero training error does not identify the best interpolation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "training_fit_vs_dense_error.png", bbox_inches="tight")
    plt.close(fig)


def dense_path(case_id: str, regularization: str, time_index: int) -> Path:
    return ROOT / "runs" / case_id / regularization / "error_analysis" / f"dense_reconstruction_time_{time_index:04d}.csv"


def plot_reconstruction_case(case_id: str, output_name: str, title: str) -> None:
    event_index = int(SUMMARY[SUMMARY.case_id == case_id].event_time_index.iloc[0])
    data = pd.read_csv(dense_path(case_id, "data_only", event_index))
    reg = pd.read_csv(dense_path(case_id, "xy030_t030", event_index))
    x = np.sort(data.x_km.unique())
    y = np.sort(data.y_km.unique())
    X, Y = np.meshgrid(x, y)
    truth = data.true_log10_Ne.to_numpy().reshape(len(y), len(x))
    pred_data = data.pred_log10_Ne.to_numpy().reshape(len(y), len(x))
    pred_reg = reg.pred_log10_Ne.to_numpy().reshape(len(y), len(x))
    error_reg = pred_reg - truth
    vmin = min(truth.min(), pred_data.min(), pred_reg.min())
    vmax = max(truth.max(), pred_data.max(), pred_reg.max())
    err_lim = max(abs(np.nanpercentile(error_reg, 1)), abs(np.nanpercentile(error_reg, 99)), 1e-3)
    obs = pd.read_csv(ROOT / "data" / case_id / "synthetic_observations.csv")
    obs = obs[obs.time_index == event_index]

    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.35), constrained_layout=True)
    panels = [
        (truth, "Radar-integrated truth", "plasma", vmin, vmax),
        (pred_data, "Data-only SIREN", "plasma", vmin, vmax),
        (pred_reg, "Regularized SIREN", "plasma", vmin, vmax),
        (error_reg, "Regularized error [dex]", "RdBu_r", -err_lim, err_lim),
    ]
    for ax, (values, panel_title, cmap, lo, hi) in zip(axes, panels):
        im = ax.pcolormesh(X, Y, values, shading="auto", cmap=cmap, vmin=lo, vmax=hi)
        ax.scatter(obs.x_km, obs.y_km, facecolors="none", edgecolors="black", s=13, linewidth=0.45)
        ax.set_title(panel_title)
        ax.set_xlabel("x [km]")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.78)
    axes[0].set_ylabel("y [km]")
    fig.suptitle(title, fontsize=11)
    fig.savefig(FIG / output_name, bbox_inches="tight")
    plt.close(fig)


def plot_shear_gradient() -> None:
    case_id = "flow_reversal_b23_v1p00_int02m_s0"
    event_index = 5
    data = pd.read_csv(dense_path(case_id, "data_only", event_index))
    reg = pd.read_csv(dense_path(case_id, "xy030_t030", event_index))
    x = np.sort(data.x_km.unique())
    y = np.sort(data.y_km.unique())
    X, Y = np.meshgrid(x, y)

    def mag(frame: pd.DataFrame, prefix: str) -> np.ndarray:
        return np.sqrt(
            frame[f"{prefix}_dlog10Ne_dx_km"].to_numpy() ** 2
            + frame[f"{prefix}_dlog10Ne_dy_km"].to_numpy() ** 2
        ).reshape(len(y), len(x))

    truth = mag(data, "true")
    pred_data = mag(data, "pred")
    pred_reg = mag(reg, "pred")
    vmax = max(truth.max(), pred_data.max(), pred_reg.max())
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.4), constrained_layout=True)
    for ax, values, title in zip(
        axes,
        [truth, pred_data, pred_reg],
        ["Truth gradient magnitude", "Data-only gradient magnitude", "Regularized gradient magnitude"],
    ):
        im = ax.pcolormesh(X, Y, values, shading="auto", cmap="inferno", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x [km]")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[0].set_ylabel("y [km]")
    fig.suptitle("Flow-reversal case: horizontal log-density gradient structure")
    fig.savefig(FIG / "flow_reversal_gradient_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    plot_active_nrmse()
    plot_regularization_improvement()
    plot_beam_effect()
    plot_loss_ratios()
    plot_training_vs_dense()
    plot_reconstruction_case(
        "single_b42_v0p36_dir000_int01m_s0",
        "reconstruction_42_slow_1m.png",
        "42 beams, 0.36 km/s, 1 min integration: well-supported reference case",
    )
    plot_reconstruction_case(
        "single_b23_v0p36_dir000_int01m_s0",
        "reconstruction_23_slow_1m.png",
        "23 beams, 0.36 km/s, 1 min integration: support-limited counterpart",
    )
    plot_reconstruction_case(
        "single_b42_v2p00_dir000_int01m_s0",
        "reconstruction_42_fast_1m.png",
        "42 beams, 2.00 km/s, 1 min integration: fast-motion case",
    )
    plot_reconstruction_case(
        "single_b42_v2p00_dir000_int10m_s0",
        "reconstruction_42_fast_10m.png",
        "42 beams, 2.00 km/s, 10 min integration: reconstruction of the available radar product",
    )
    plot_reconstruction_case(
        "flow_reversal_b23_v1p00_int02m_s0",
        "reconstruction_flow_reversal.png",
        "23 beams, opposing 1.00 km/s flows, 2 min integration: rare shear stress case",
    )
    plot_shear_gradient()


if __name__ == "__main__":
    main()
