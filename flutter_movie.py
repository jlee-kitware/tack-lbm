"""Frames of vorticity around the fluttering wing, encoded to MP4 with ffmpeg."""
import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flutter import FlutterSolver

parser = argparse.ArgumentParser()
parser.add_argument("--speed", type=float, default=3.0)
parser.add_argument("--settle", type=float, default=12.0, help="pitch periods before recording")
parser.add_argument("--periods", type=float, default=3.0)
parser.add_argument("--frames", type=int, default=180)
parser.add_argument("--out", default="/tmp/flutter.mp4")
parser.add_argument("--ffmpeg", required=True)
args = parser.parse_args()

s = FlutterSolver(hardening=40.0)
s.set_reduced_velocity(args.speed)
for _ in range(6000):
    s.step()
s.release()
period = 2 * np.pi / s.section.omega_alpha
history = []
for _ in range(int(args.settle * period)):
    s.step()
    history.append(s.motion()[1])

nx, ny = s.nx, s.ny
x0, x1, y0, y1 = 62, 282, 50, 142
yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
frames = Path("/tmp/flutter_frames")
frames.mkdir(exist_ok=True)
every = max(1, int(args.periods * period / args.frames))
for i in range(args.frames):
    for _ in range(every - 1):
        s.step()
        history.append(s.motion()[1])
    s.step(report=True)
    h, alpha = s.motion()
    history.append(alpha)
    ux = s.fields["ux"].numpy.reshape(s.nz, ny, nx)[0].astype(np.float64)
    uy = s.fields["uy"].numpy.reshape(s.nz, ny, nx)[0].astype(np.float64)
    vort = np.gradient(uy, axis=1) - np.gradient(ux, axis=0)
    st = s.section
    a = st.q[1]
    dx, dy = xx - s.geo.x_ea, yy - (s.geo.y_ea + st.q[0])
    xc = (dx * np.cos(a) - dy * np.sin(a)) / s.chord + s.geo.ea
    t = dx * np.sin(a) + dy * np.cos(a)
    c = np.clip(xc, 0, 1)
    half = 0.6 * s.chord * (0.2969 * np.sqrt(c) - 0.126 * c - 0.3516 * c ** 2 + 0.2843 * c ** 3 - 0.1036 * c ** 4)
    wing = (xc >= 0) & (xc <= 1) & (np.abs(t) < half)
    vort[wing] = np.nan

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#0b1020")
    ax = fig.add_axes([0.0, 0.25, 1.0, 0.75])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "vort", ["#7fd3ff", "#1f6fb2", "#0b1020", "#c2571a", "#ffc58a"])
    cmap.set_bad("#c9d1dc")
    ax.imshow(vort[y0:y1, x0:x1], origin="lower", cmap=cmap, vmin=-0.016, vmax=0.016,
              interpolation="bilinear", aspect="equal")
    ax.set_axis_off()
    ax.text(0.015, 0.95, f"Pitch-plunge flutter, NACA 0012, U* = {args.speed:.1f}", color="white",
            transform=ax.transAxes, fontsize=15, va="top")
    ax.text(0.015, 0.88, f"pitch {np.degrees(a):+5.1f} deg   plunge {st.q[0] / st.b:+.3f} b", color="#cfd6e3",
            transform=ax.transAxes, fontsize=12, va="top", family="monospace")
    tr = fig.add_axes([0.07, 0.07, 0.9, 0.14], facecolor="#0b1020")
    hist = np.array(history)
    tt = np.arange(len(hist)) / period
    tr.plot(tt, hist, color="#4fd1c5", lw=1.2)
    tr.set_xlim(0, (args.settle + args.periods) * 1.0)
    lim = max(2.0, np.abs(hist).max() * 1.15)
    tr.set_ylim(-lim, lim)
    tr.tick_params(colors="#9aa4b5", labelsize=9)
    for spine in tr.spines.values():
        spine.set_color("#3a4356")
    tr.set_ylabel("pitch, deg", color="#9aa4b5", fontsize=9)
    tr.set_xlabel("pitch periods since release", color="#9aa4b5", fontsize=9)
    fig.savefig(frames / f"f{i:04d}.png", facecolor=fig.get_facecolor())
    plt.close(fig)

tmp = args.out.replace(".mp4", ".part.mp4")
subprocess.run([args.ffmpeg, "-v", "error", "-y", "-framerate", "30", "-i", str(frames / "f%04d.png"),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", tmp], check=True)
Path(tmp).rename(args.out)
print(args.out)
