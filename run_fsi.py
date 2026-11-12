"""Run the elastic wing (fluid-structure interaction) case with Catalyst Live.

    python run_fsi.py [--re 500] [--reduced-velocity 3.5]

A cantilever NACA 0012 wing, modeled with MFEM finite elements, in a 3D Tack LBM flow; the
root is a symmetry plane, so the lattice holds half of a wing clamped at its centre. Lengths
are in chords, time in chord transits.

Channels: ``grid`` (velocity, pressure, speed, every ``--stride`` cells), ``wing`` (the
deformed finite element surface, with its ``deflection`` in chords) and ``monitors`` (drag
and lift coefficients, tip pitch in deg, tip plunge in semichords, reduced velocity,
Reynolds number). Steerable: Reynolds number, which here is the air speed (U* scales with
it), the collision model and a gust.
"""

import argparse
import os
import signal
import time

import numpy as np

from fsi import FSISolver
from run_lbm import MONITOR_NAMES, CatalystAdaptor
from steering import COLLISION_CODES, write_proxy_xml

FSI_MONITORS = MONITOR_NAMES + ("pitch", "plunge", "reduced_velocity")


class WingSurface:
    """The finite element wing's outer surface where it is now, in chord units."""

    def __init__(self, solver):
        self.solver = solver
        quads = solver.fem.quads
        self.connectivity = np.ascontiguousarray(quads.ravel(), dtype=np.int64)

    def __call__(self, channels):
        s = self.solver
        pts, _ = s.wing_surface()
        rest = s.fem.rest
        deflection = np.linalg.norm(pts - rest - np.array([s.geo.x_ea, s.geo.y_ea, -0.5]), axis=1) / s.chord
        ch = channels["wing"]
        ch["type"] = "mesh"
        mesh = ch["data"]
        mesh["coordsets/coords/type"] = "explicit"
        for axis, column in zip("xyz", pts.T):
            mesh[f"coordsets/coords/values/{axis}"].set(np.ascontiguousarray(column / s.chord))
        mesh["topologies/mesh/type"] = "unstructured"
        mesh["topologies/mesh/coordset"] = "coords"
        mesh["topologies/mesh/elements/shape"] = "quad"
        mesh["topologies/mesh/elements/connectivity"].set(self.connectivity)
        mesh["fields/deflection/association"] = "vertex"
        mesh["fields/deflection/topology"] = "mesh"
        mesh["fields/deflection/values"].set(np.ascontiguousarray(deflection))


def main():
    pv_build = os.environ.get("PV_BUILD", os.path.expanduser("~/gitsync/paraview_build_cuda"))
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", default=os.path.join(here, "pipeline_fsi.py"))
    parser.add_argument("--collision", default="les", choices=["les", "bgk", "trt"])
    parser.add_argument("--re", type=float, default=500.0)
    parser.add_argument("--reduced-velocity", type=float, default=3.5,
                        help="U / (b omega_alpha) of the first torsion mode at the starting speed")
    parser.add_argument("--hardening", type=float, default=40.0)
    parser.add_argument("--size", type=int, nargs=3, default=[256, 128, 104], metavar=("NX", "NY", "NZ"))
    parser.add_argument("--steps", type=int, default=0, help="0 runs until interrupted")
    parser.add_argument("--interval", type=int, default=100, help="steps between Catalyst calls")
    parser.add_argument("--sample", type=int, default=20, help="steps between monitor samples")
    parser.add_argument("--gust", type=float, default=4.0, help="tip pitch swing in deg given on a speed change")
    parser.add_argument("--start-gust", type=float, default=0.0, help="gust in deg when the wing is freed")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--catalyst-lib", default=os.path.join(pv_build, "lib", "catalyst"))
    parser.add_argument("--steering-xml", default=os.path.join(here, "steering.xml"))
    args = parser.parse_args()

    nx, ny, nz = args.size
    s = FSISolver(nx, ny, nz, reynolds=args.re, collision=args.collision, stride=args.stride,
                  reduced_velocity=args.reduced_velocity, hardening=args.hardening)
    speed = args.reduced_velocity / args.re
    xml = write_proxy_xml(
        args.steering_xml, args.re, args.collision,
        reynolds_doc=f"Air speed, as the Reynolds number on the wing chord. The wing's stiffness and the air "
                     f"are fixed, so the reduced velocity U* = U / (b omega_alpha) of its first torsion mode "
                     f"scales with it: Re 500 is U* {500 * speed:.2f}, Re 1000 is U* {1000 * speed:.2f}. Each "
                     f"change gives the wing a {args.gust:g} degree gust; below the flutter speed the motion dies "
                     f"away, above it bending and twist lock together and grow into a steady oscillation "
                     f"(watch pitch and plunge in the monitors, both at the wing tip).",
        gust_doc="A gust that twists the wing: the tip pitch swing it gives, in degrees. Each new value applies "
                 "one gust (to repeat the same size, set it to 0 first); the wing starts at rest.")
    adaptor = CatalystAdaptor(os.path.abspath(args.script), s, args.catalyst_lib, xml,
                              monitor_names=FSI_MONITORS, extra_channels=WingSurface(s))
    print(f"[fsi] {nx}x{ny}x{nz}, chord {s.chord:g} cells, span {s.span:g}, Re {s.reynolds:g}, "
          f"U* {s.reduced_velocity():.2f}, modes {np.round(s.frequencies() / s.frequencies()[s.torsion], 2)}",
          flush=True)

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for _ in range(3000):
        s.step()
    s.held = False
    if args.start_gust:
        s.gust(args.start_gust)
    adaptor.steered["gust"] = 0.0
    t_report = time.perf_counter()
    calls = 0
    last_gust = 0.0

    def sample():
        cd, cl, _ = s.coefficients()
        plunge, pitch = s.tip()
        return {"drag": cd, "lift": cl, "pitch": pitch, "plunge": plunge,
                "reduced_velocity": s.reduced_velocity(), "reynolds": s.reynolds,
                "collision": float(COLLISION_CODES[s.collision_name]), "time": s.step_count * s.u0 / s.chord}

    while running and (args.steps == 0 or s.step_count < args.steps):
        for k in range(1, args.interval):
            s.step()
            if k % args.sample == 0:
                row = sample()
                adaptor.history["iteration"].append(float(s.step_count))
                for name in FSI_MONITORS:
                    adaptor.history[name].append(row.get(name, np.nan))
        s.step(report=True)
        s.update_output()
        monitors = sample()
        step = s.step_count
        adaptor.execute(step, monitors["time"], monitors)
        reynolds = s.reynolds
        adaptor.steer()
        if s.reynolds != reynolds:
            s.gust(args.gust)
            print(f"[steer] U* {s.reduced_velocity():.2f}, gust applied", flush=True)
        gust = adaptor.steered.get("gust", 0.0)
        if gust != last_gust:
            last_gust = gust
            if gust > 0:
                s.gust(gust)
                print(f"[steer] gust of {gust:g} deg", flush=True)
        calls += 1
        if calls % 10 == 0:
            now = time.perf_counter()
            print(f"[fsi] step {step}  {10 * args.interval / (now - t_report):.0f} steps/s  "
                  f"U* {s.reduced_velocity():.2f}  tip pitch {monitors['pitch']:+.2f} deg  "
                  f"tip plunge {monitors['plunge']:+.3f} b", flush=True)
            t_report = now
    adaptor.finalize()


if __name__ == "__main__":
    main()
