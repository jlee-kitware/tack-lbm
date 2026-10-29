"""Run the Tack LBM solver with Catalyst 2 in situ, ready for Catalyst Live.

    python run_lbm.py --script pipeline_live.py [--collision les] [--obstacle cylinder]

Channels:
    grid      uniform grid with velocity, pressure, speed (zero-copy managed memory)
    monitors  residual and force history as a points mesh, the layout trame-cfd-mcp reads
"""

import argparse
import os
import signal
import time

import numpy as np

import catalyst
import conduit

from lbm import Solver

MONITOR_NAMES = ("continuity", "momentum_x", "momentum_y", "momentum_z", "energy", "drag", "lift")


class CatalystAdaptor:
    def __init__(self, script, solver, paraview_lib):
        node = conduit.Node()
        node["catalyst/scripts/script/filename"] = script
        node["catalyst_load/implementation"] = "paraview"
        node["catalyst_load/search_paths/paraview"] = paraview_lib
        catalyst.initialize(node)
        self.solver = solver
        self.history = {name: [] for name in ("iteration", *MONITOR_NAMES)}

    def execute(self, step, time_value, monitors):
        self.history["iteration"].append(float(step))
        for name in MONITOR_NAMES:
            self.history[name].append(monitors[name])
        s = self.solver
        node = conduit.Node()
        node["catalyst/state/timestep"] = step
        node["catalyst/state/time"] = time_value
        node["catalyst/state/parameters"].append().set(f"collision={s.collision_name}")
        node["catalyst/state/parameters"].append().set(f"reynolds={s.reynolds}")

        grid = node["catalyst/channels/grid"]
        grid["type"] = "mesh"
        mesh = grid["data"]
        mesh["coordsets/coords/type"] = "uniform"
        mx, my, mz = s.out_dims
        mesh["coordsets/coords/dims/i"] = mx
        mesh["coordsets/coords/dims/j"] = my
        mesh["coordsets/coords/dims/k"] = mz
        spacing = s.stride / s.diameter
        for axis in "xyz":
            mesh[f"coordsets/coords/origin/{axis}"] = 0.0
            mesh[f"coordsets/coords/spacing/d{axis}"] = spacing
        mesh["topologies/mesh/type"] = "uniform"
        mesh["topologies/mesh/coordset"] = "coords"
        f = s.output
        velocity = mesh["fields/velocity"]
        velocity["association"] = "vertex"
        velocity["topology"] = "mesh"
        velocity["values/x"].set_external(f["ux"].numpy)
        velocity["values/y"].set_external(f["uy"].numpy)
        velocity["values/z"].set_external(f["uz"].numpy)
        for name, key in (("pressure", "p"), ("speed", "vmag")):
            fld = mesh[f"fields/{name}"]
            fld["association"] = "vertex"
            fld["topology"] = "mesh"
            fld["values"].set_external(f[key].numpy)

        mon = node["catalyst/channels/monitors"]
        mon["type"] = "mesh"
        data = mon["data"]
        n = len(self.history["iteration"])
        data["coordsets/coords/type"] = "explicit"
        data["coordsets/coords/values/x"].set(np.asarray(self.history["iteration"]))
        data["coordsets/coords/values/y"].set(np.asarray(self.history["continuity"]))
        data["coordsets/coords/values/z"].set(np.zeros(n))
        data["topologies/points/type"] = "unstructured"
        data["topologies/points/coordset"] = "coords"
        data["topologies/points/elements/shape"] = "point"
        data["topologies/points/elements/connectivity"].set(np.arange(n, dtype=np.int32))
        for name, values in self.history.items():
            fld = data[f"fields/{name}"]
            fld["association"] = "vertex"
            fld["topology"] = "points"
            fld["values"].set(np.asarray(values))
        catalyst.execute(node)

    def finalize(self):
        catalyst.finalize(conduit.Node())


def main():
    pv_build = os.environ.get("PV_BUILD", os.path.expanduser("~/gitsync/paraview_build_cuda"))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", default=os.path.join(os.path.dirname(__file__), "pipeline_live.py"))
    parser.add_argument("--collision", default="les", choices=["les", "bgk", "trt"])
    parser.add_argument("--obstacle", default="cylinder", choices=["cylinder", "sphere", "box"])
    parser.add_argument("--re", type=float, default=300.0)
    parser.add_argument("--size", type=int, nargs=3, default=[384, 128, 64], metavar=("NX", "NY", "NZ"))
    parser.add_argument("--steps", type=int, default=0, help="0 runs until interrupted")
    parser.add_argument("--interval", type=int, default=100, help="steps between Catalyst calls")
    parser.add_argument("--stride", type=int, default=2, help="grid channel subsampling (1 = full resolution)")
    parser.add_argument("--max-sps", type=float, default=0, help="throttle steps per second (0 = full speed)")
    parser.add_argument("--catalyst-lib", default=os.path.join(pv_build, "lib", "catalyst"))
    args = parser.parse_args()

    nx, ny, nz = args.size
    solver = Solver(nx, ny, nz, collision=args.collision, obstacle=args.obstacle, reynolds=args.re,
                    stride=args.stride)
    adaptor = CatalystAdaptor(os.path.abspath(args.script), solver, args.catalyst_lib)
    print(f"[tack-lbm] {nx}x{ny}x{nz} D3Q19, {args.collision}, {args.obstacle}, Re={args.re}", flush=True)

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    t_report = time.perf_counter()
    t_start = t_report
    calls = 0
    while running and (args.steps == 0 or solver.step_count < args.steps):
        for _ in range(args.interval - 1):
            solver.step()
        solver.step(report=True)
        monitors = solver.monitors(args.interval)
        step = solver.step_count
        adaptor.execute(step, step * solver.u0 / solver.diameter, monitors)
        now = time.perf_counter()
        if args.max_sps > 0:
            wait = step / args.max_sps - (now - t_start)
            if wait > 0:
                time.sleep(wait)
                now = time.perf_counter()
        calls += 1
        if calls % 10 == 0:
            sps = 10 * args.interval / (now - t_report)
            t_report = now
            print(f"[tack-lbm] step {step}  {sps:.0f} steps/s  Cd={monitors['drag']:.3f}  "
                  f"Cl={monitors['lift']:+.3f}  res_u={monitors['momentum_x']:.2e}", flush=True)
    adaptor.finalize()


if __name__ == "__main__":
    main()
