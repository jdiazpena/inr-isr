"""Matched truth views; stream movie frames to the Docker-system FFmpeg encoder."""
from pathlib import Path
import json
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

from .field import StructuredScene
from .records import file_identity


def _log_density_limits(definition):
    settings = definition["render"]
    low, high = settings["vmin_ne_m3"], settings["vmax_ne_m3"]
    if not np.isfinite([low, high]).all() or not 0 < low < high:
        raise ValueError("Density colour limits must be finite, positive and ordered")
    return float(np.log10(low)), float(np.log10(high))


def scene_view(definition, geometry, grid_size=192):
    """Matched log10-density views with fixed axes and shared colour limits."""
    if grid_size < 2:
        raise ValueError("Rendering needs at least two grid nodes per dimension")
    d = definition["domain"]
    x = np.linspace(*d["x_bounds_km"], grid_size)
    y = np.linspace(*d["y_bounds_km"], grid_size)
    X, Y = np.meshgrid(x, y)
    cases = definition["cases"]
    scenes = [StructuredScene(definition, c["speed_km_s"]) for c in cases]
    fig, axes = plt.subplots(1, len(cases), figsize=(12.8, 6.4), dpi=100, squeeze=False)
    axes = axes.ravel()
    images, fronts = [], []
    vmin, vmax = _log_density_limits(definition)
    for ax, case in zip(axes, cases):
        im = ax.imshow(np.zeros_like(X), origin="lower", extent=[x[0], x[-1], y[0], y[-1]],
                       vmin=vmin, vmax=vmax,
                       cmap="viridis", interpolation="nearest", aspect="equal")
        images.append(im)
        fronts.append(ax.plot([], [], color="white", lw=1.1, label="Front midpoint")[0])
        ax.scatter(geometry.x_km, geometry.y_km, s=17, facecolors="none",
                   edgecolors="black", linewidths=0.65, label=f"{len(geometry)} sampling locations")
        ax.set(xlabel="x (km)", ylabel="y (km)", title=f"{case['id']} | {case['speed_km_s'] * 1000:g} m/s")
        ax.legend(loc="upper right", fontsize=7, facecolor="white", framealpha=0.9)
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.16, top=0.87, wspace=0.23)
    color_ax = fig.add_axes([0.92, 0.2, 0.014, 0.62])
    fig.colorbar(images[0], cax=color_ax, label=r"$\log_{10}(N_e / \mathrm{m}^{-3})$")
    title = fig.suptitle("", fontsize=13)
    caption = fig.text(0.5, 0.065, "", ha="center", fontsize=9)

    def update(t, mapping):
        hh, rem = divmod(int(round(t)), 3600)
        mm, ss = divmod(rem, 60)
        title.set_text(f"Synthetic instantaneous truth | elapsed {hh:02d}:{mm:02d}:{ss:02d}")
        caption.set_text(mapping)
        arrays = []
        for scene, im, line in zip(scenes, images, fronts):
            value = scene.evaluate(X, Y, t, derivatives=False, render_tail_sigma=10)["Ne"]
            arrays.append(value)
            im.set_data(np.log10(value))
            line.set_data(x, scene.boundary_y(x, t))
        return arrays
    return fig, X, Y, update


def write_stills(definition, geometry, output, times, *, grid_size=192):
    fig, X, Y, update = scene_view(definition, geometry, grid_size)
    files = []
    try:
        for t in times:
            arrays = update(t, "Analytic truth; point observations; no temporal averaging")
            name = f"truth_{int(round(t)):05d}s"
            fig.savefig(Path(output) / f"{name}.png", dpi=100)
            np.savez_compressed(Path(output) / f"{name}.npz", x_km=X, y_km=Y,
                                t_sec=t, **{c["id"]: a for c, a in zip(definition["cases"], arrays)})
            files.append({"density_display": "log10(Ne / m^-3)",
                          "colour_limits_log10_Ne": list(_log_density_limits(definition)),
                          "image": file_identity(Path(output) / f"{name}.png"),
                          "plot_input": file_identity(Path(output) / f"{name}.npz"), "t_sec": t})
    finally:
        plt.close(fig)
    return files


def write_movie(definition, geometry, output, *, start_sec, end_sec, frame_step_sec,
                fps=30, grid_size=192, bitrate_kbps=1800):
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            raise RuntimeError(f"Required Docker-system tool {name} is unavailable; see setup documentation")
    if not (0 <= start_sec < end_sec <= definition["duration_sec"] and
            frame_step_sec > 0 and fps > 0 and bitrate_kbps > 0):
        raise ValueError("Invalid movie time/playback settings")
    times = np.arange(start_sec, end_sec, frame_step_sec, dtype=float)
    times = np.append(times, end_sec)
    movie = Path(output) / "truth.mp4"
    if movie.exists():
        raise FileExistsError(movie)
    fig, _, _, update = scene_view(definition, geometry, grid_size)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=bitrate_kbps,
                         extra_args=["-pix_fmt", "yuv420p", "-maxrate", "2000k", "-bufsize", "4000k",
                                     "-movflags", "+faststart", "-threads", "2"])
    mapping = f"Playback: 1 video second = {fps * frame_step_sec:g} simulated seconds | fixed log10 density scale"
    try:
        with writer.saving(fig, str(movie), dpi=100):
            for i, t in enumerate(times):
                update(t, mapping)
                writer.grab_frame()
                if i % 300 == 0 or i == len(times) - 1:
                    print(f"Movie frame {i + 1}/{len(times)} | elapsed {t / 3600:.2f} h", flush=True)
    finally:
        plt.close(fig)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-count_frames", "-show_entries", "stream=codec_name,width,height,nb_read_frames,r_frame_rate,duration",
                        "-of", "json", str(movie)], text=True))
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(movie), "-f", "null", "-"], check=True,
                   stdout=subprocess.DEVNULL)
    if int(probe["streams"][0]["nb_read_frames"]) != len(times):
        raise RuntimeError("Encoded movie frame count differs from requested times")
    if movie.stat().st_size >= 200_000_000:
        raise RuntimeError("Movie exceeds the entire delivered-bundle budget")
    return {"file": file_identity(movie), "density_display": "log10(Ne / m^-3)",
            "colour_limits_log10_Ne": list(_log_density_limits(definition)),
            "probe": probe, "simulation_start_sec": start_sec,
            "simulation_end_sec": end_sec, "frame_step_sec": frame_step_sec, "fps": fps,
            "frame_count": len(times), "simulation_seconds_per_video_second": fps * frame_step_sec,
            "full_decode_passed": True, "render_tail_sigma": 10,
            "tool_versions": {n: subprocess.check_output([n, "-version"], text=True).splitlines()[0]
                              for n in ("ffmpeg", "ffprobe")}}
