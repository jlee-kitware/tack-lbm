"""Run the wing flutter case with Catalyst Live.

    python run_flutter.py [--re 500] [--reduced-velocity 1.5]

A NACA 0012 section on pitch and plunge springs in a Tack LBM flow; lengths are in
chords, time in chord transits (chord / air speed).

Channels: ``grid`` (velocity, pressure, speed and derived fields), ``wing`` (the wing's
surface where it is now) and ``monitors`` (drag and lift coefficients, pitch in deg,
plunge in semichords, reduced velocity, Reynolds number). Steerable: Reynolds number,
which here is the air speed (the wing and air are fixed, so U* scales with Re), and the
collision model. A speed change gives the wing a gust, so it shows within a few
cycles whether the new speed damps the motion or feeds it.
"""

import argparse
import os
import signal
import time

import numpy as np

from derived import DerivedFields
from flutter import FlutterSolver
from run_lbm import MONITOR_NAMES, CatalystAdaptor
from steering import COLLISION_CODES, write_proxy_xml

FLUTTER_MONITORS = MONITOR_NAMES + ("pitch", "plunge", "reduced_velocity")


class WingChannel:
    """The wing surface, extruded across the span and capped at both ends, in chord units."""

    def __init__(self, solver, points=96, span=None):
        self.solver = solver
        self.points = points
        self.span = span
        n = points
        i = np.arange(n)
        sides = np.stack([i, (i + 1) % n, (i + 1) % n + n, i + n], axis=1).ravel()
        # Side quads, then a cap at each end of the span so the section reads as a solid face-on.
        self.connectivity = np.concatenate([sides, i[::-1], i + n]).astype(np.int64)
        self.sizes = np.concatenate([np.full(n, 4), [n, n]]).astype(np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]]).astype(np.int64)

    def __call__(self, channels):
        s = self.solver
        x, y = s.outline(self.points)
        grid_span = (s.out_dims[2] - 1) * s.stride / s.chord
        xs = np.concatenate([x, x]) / s.chord
        ys = np.concatenate([y, y]) / s.chord
        # The flow is the same in every spanwise slice, so the wing can be drawn across any span; by
        # default a little past both faces of the grid, so the caps show in front of it.
        half = 0.5 * self.span if self.span else 0.5 * grid_span + 0.5 / s.chord
        zs = np.repeat([0.5 * grid_span - half, 0.5 * grid_span + half], self.points)
        ch = channels["wing"]
        ch["type"] = "mesh"
        mesh = ch["data"]
        mesh["coordsets/coords/type"] = "explicit"
        for axis, column in zip("xyz", (xs, ys, zs)):
            mesh[f"coordsets/coords/values/{axis}"].set(np.ascontiguousarray(column))
        mesh["topologies/mesh/type"] = "unstructured"
        mesh["topologies/mesh/coordset"] = "coords"
        mesh["topologies/mesh/elements/shape"] = "polygonal"
        mesh["topologies/mesh/elements/connectivity"].set(self.connectivity)
        mesh["topologies/mesh/elements/sizes"].set(self.sizes)
        mesh["topologies/mesh/elements/offsets"].set(self.offsets)
        pitch = np.full(2 * self.points, np.degrees(s.section.q[1]))
        mesh["fields/pitch/association"] = "vertex"
        mesh["fields/pitch/topology"] = "mesh"
        mesh["fields/pitch/values"].set(pitch)


def main():
    pv_build = os.environ.get("PV_BUILD", os.path.expanduser("~/gitsync/paraview_build_cuda"))
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", default=os.path.join(here, "pipeline_flutter.py"))
    parser.add_argument("--collision", default="les", choices=["les", "bgk", "trt"])
    parser.add_argument("--re", type=float, default=500.0)
    parser.add_argument("--reduced-velocity", type=float, default=1.5,
                        help="U / (b omega_alpha) at the starting speed")
    parser.add_argument("--hardening", type=float, default=40.0)
    parser.add_argument("--size", type=int, nargs=3, default=[384, 192, 4], metavar=("NX", "NY", "NZ"))
    parser.add_argument("--steps", type=int, default=0, help="0 runs until interrupted")
    parser.add_argument("--interval", type=int, default=200, help="steps between Catalyst calls")
    parser.add_argument("--sample", type=int, default=20, help="steps between monitor samples")
    parser.add_argument("--gust", type=float, default=4.0, help="pitch swing in deg given on a speed change")
    parser.add_argument("--display-span", type=float, default=1.5,
                        help="span in chords the wing and the vortices are drawn across")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--catalyst-lib", default=os.path.join(pv_build, "lib", "catalyst"))
    parser.add_argument("--steering-xml", default=os.path.join(here, "steering.xml"))
    args = parser.parse_args()

    nx, ny, nz = args.size
    s = FlutterSolver(nx, ny, nz, reynolds=args.re, collision=args.collision, stride=args.stride,
                      hardening=args.hardening)
    s.set_reduced_velocity(args.reduced_velocity)
    speed = args.reduced_velocity / args.re
    xml = write_proxy_xml(
        args.steering_xml, args.re, args.collision, derived=True,
        reynolds_doc=f"Air speed, as the Reynolds number on the wing chord. The wing's stiffness and the air "
                     f"are fixed, so the reduced velocity U* = U / (b omega_alpha) scales with it: "
                     f"Re 500 is U* {500 * speed:.2f}, Re 1000 is U* {1000 * speed:.2f}. Each change gives the "
                     f"wing a {args.gust:g} degree gust; below the flutter speed the motion dies away, above it pitch and "
                     f"plunge lock together and grow into a steady oscillation (watch pitch and plunge in the "
                     f"monitors).",
        gust_doc="A gust that sets the wing pitching: the pitch swing it gives, in degrees. Each new value "
                 "applies one gust (to repeat the same size, set it to 0 first); the wing starts at rest.")
    # A 2D flow: the vorticity that matters is its z component, signed. Only that one, since it is
    # computed at every Live update.
    derived = DerivedFields(s, {"vorticity_z": ("dot(curl(velocity), kHat)", "cells")})
    adaptor = CatalystAdaptor(os.path.abspath(args.script), s, args.catalyst_lib, xml,
                              monitor_names=FLUTTER_MONITORS, extra_channels=WingChannel(s, span=args.display_span), derived=derived)
    print(f"[flutter] {nx}x{ny}x{nz}, chord {s.chord:g} cells, Re {s.reynolds:g}, U* {s.reduced_velocity():.2f}",
          flush=True)

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for _ in range(4000):
        s.step()
    s.release(0.0)
    adaptor.steered["gust"] = 0.0
    t_report = time.perf_counter()
    calls = 0
    last_gust = 0.0
    def sample():
        cd, cl, _ = s.coefficients()
        plunge, pitch = s.motion()
        return {"drag": cd, "lift": cl, "pitch": pitch, "plunge": plunge,
                "reduced_velocity": s.reduced_velocity(), "reynolds": s.reynolds,
                "collision": float(COLLISION_CODES[s.collision_name]),
                "derived_error": float(derived.error_column), "time": s.step_count * s.u0 / s.chord}

    while running and (args.steps == 0 or s.step_count < args.steps):
        for k in range(1, args.interval):
            s.step()
            if k % args.sample == 0:
                # Samples between Catalyst calls go straight into the monitor history.
                row = sample()
                adaptor.history["iteration"].append(float(s.step_count))
                for name in FLUTTER_MONITORS:
                    adaptor.history[name].append(row.get(name, np.nan))
        s.step(report=True)
        s.update_output()
        monitors = sample()
        plunge, pitch = monitors["plunge"], monitors["pitch"]
        step = s.step_count
        adaptor.execute(step, monitors["time"], monitors)
        reynolds = s.reynolds
        adaptor.steer()
        if s.reynolds != reynolds:
            s.release(np.radians(args.gust) * s.section.omega_alpha)
            print(f"[steer] U* {s.reduced_velocity():.2f}, gust applied", flush=True)
        gust = adaptor.steered.get("gust", 0.0)
        if gust != last_gust:
            last_gust = gust
            if gust > 0:
                s.release(np.radians(gust) * s.section.omega_alpha)
                print(f"[steer] gust of {gust:g} deg", flush=True)
        calls += 1
        if calls % 10 == 0:
            now = time.perf_counter()
            print(f"[flutter] step {step}  {10 * args.interval / (now - t_report):.0f} steps/s  "
                  f"U* {s.reduced_velocity():.2f}  pitch {pitch:+.2f} deg  plunge {plunge:+.3f} b", flush=True)
            t_report = now
    adaptor.finalize()


if __name__ == "__main__":
    main()
