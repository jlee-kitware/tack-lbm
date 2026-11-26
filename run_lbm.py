"""Run the Tack LBM solver with Catalyst 2 in situ, ready for Catalyst Live.

    python run_lbm.py --script pipeline_live.py [--collision les] [--obstacle cylinder]

Channels:
    grid      uniform grid with velocity, pressure, speed (zero-copy managed memory), and
              derived fields from field expressions: vorticity and q_criterion by default,
              more with --derive NAME=EXPRESSION[@cells]
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
from steering import COLLISION_CODES, COLLISION_NAMES, read_steering, write_proxy_xml

MONITOR_NAMES = ("continuity", "momentum_x", "momentum_y", "momentum_z", "energy", "drag", "lift",
                 "reynolds", "collision", "rotation", "derived_error", "time")


def rotation_rate(solver):
    """Dimensionless rotation rate alpha: cylinder surface speed over inflow speed."""
    return getattr(solver, "omega_rot", 0.0) / solver.u0


def set_rotation_rate(solver, alpha):
    solver.set_actuation(float(alpha) * solver.u0)


class CatalystAdaptor:
    def __init__(self, script, solver, paraview_lib, proxy_xml, monitor_names=MONITOR_NAMES,
                 extra_channels=None, derived=None, scales=None):
        node = conduit.Node()
        node["catalyst/scripts/script/filename"] = script
        node["catalyst/proxies/steering"] = proxy_xml
        node["catalyst_load/implementation"] = "paraview"
        node["catalyst_load/search_paths/paraview"] = paraview_lib
        catalyst.initialize(node)
        self.solver = solver
        self.monitor_names = monitor_names
        self.monitor_history = 0
        self.extra_channels = extra_channels
        self.derived = derived
        # Optional SI output: {"length": metres per cell, quantity: (factor, offset)}.
        self.scales = scales
        self.history = {name: [] for name in ("iteration", *monitor_names)}
        self.steered = {"reynolds": solver.reynolds, "collision": float(COLLISION_CODES[solver.collision_name]),
                        "rotation": rotation_rate(solver), "capture_stride": 0.0}

    def execute(self, step, time_value, monitors):
        self.history["iteration"].append(float(step))
        for name in self.monitor_names:
            self.history[name].append(monitors.get(name, np.nan))
        s = self.solver
        node = conduit.Node()
        node["catalyst/state/timestep"] = step
        node["catalyst/state/time"] = time_value
        node["catalyst/state/parameters"].append().set(f"collision={s.collision_name}")
        node["catalyst/state/parameters"].append().set(f"reynolds={s.reynolds}")
        node["catalyst/state/parameters"].append().set(f"rotation={rotation_rate(s)}")

        grid = node["catalyst/channels/grid"]
        grid["type"] = "mesh"
        mesh = grid["data"]
        mesh["coordsets/coords/type"] = "uniform"
        mx, my, mz = s.out_dims
        mesh["coordsets/coords/dims/i"] = mx
        mesh["coordsets/coords/dims/j"] = my
        mesh["coordsets/coords/dims/k"] = mz
        spacing = s.stride * self.scales["length"] if self.scales else s.stride / s.diameter
        for axis in "xyz":
            mesh[f"coordsets/coords/origin/{axis}"] = 0.0
            mesh[f"coordsets/coords/spacing/d{axis}"] = spacing
        mesh["topologies/mesh/type"] = "uniform"
        mesh["topologies/mesh/coordset"] = "coords"
        f = s.output
        velocity = mesh["fields/velocity"]
        velocity["association"] = "vertex"
        velocity["topology"] = "mesh"
        def put(target, key, quantity):
            if self.scales and quantity in self.scales:
                factor, offset = self.scales[quantity]
                target.set(f[key].numpy * factor + offset)
            else:
                target.set_external(f[key].numpy)

        for axis in "xyz":
            put(velocity[f"values/{axis}"], f"u{axis}", "velocity")
        for name, key, quantity in (("pressure", "p", "pressure"), ("speed", "vmag", "velocity"),
                                    ("temperature", "temp", "temperature")):
            if key not in f:
                continue
            fld = mesh[f"fields/{name}"]
            fld["association"] = "vertex"
            fld["topology"] = "mesh"
            put(fld["values"], key, quantity)
        if self.derived:
            self.derived.add_to(mesh)

        mon = node["catalyst/channels/monitors"]
        mon["type"] = "table"
        columns = mon["data/values"]
        for name, values in self.history.items():
            columns[name].set(np.asarray(values[-self.monitor_history:] if self.monitor_history else values))
        if self.derived:
            # The last steering request's outcome as text: a table of UTF-8 byte values.
            status = node["catalyst/channels/derived_status"]
            status["type"] = "table"
            codes = list(self.derived.message.encode("utf-8")) or [0]
            status["data/values/message"].set(np.asarray(codes, dtype=np.int32))
        if self.extra_channels:
            self.extra_channels(node["catalyst/channels"])
        catalyst.execute(node)

    def steer(self):
        """Apply Reynolds number and collision model changes made by a Live client."""
        results = conduit.Node()
        catalyst.results(results)
        values = read_steering(results)
        text = values.pop("derived", None)
        if text is not None and self.derived is not None:
            self.derived.steer(text)
        s = self.solver
        changed = {k: v for k, v in values.items() if abs(v - self.steered.get(k, v)) > 1e-9}
        if not changed:
            return
        self.steered.update(changed)
        if "rotation" in changed:
            set_rotation_rate(s, self.steered["rotation"])
        if not changed.keys() - {"rotation", "capture_stride"}:
            return
        collision = COLLISION_NAMES.get(int(self.steered["collision"]), s.collision_name)
        old = (s.reynolds, s.collision_name)
        s.set_physics(collision, self.steered["reynolds"])
        note = "; the next step compiles that kernel if it is new" if collision != old[1] else ""
        print(f"[steer] Re {old[0]:g} -> {s.reynolds:g}, collision {old[1]} -> {collision}{note}", flush=True)

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
    parser.add_argument("--monitor-history", type=int, default=0,
                        help="send only the last N monitor rows each call (0 = the whole history)")
    parser.add_argument("--rotation-schedule", default="",
                        help="comma list of step:alpha events that set the cylinder rotation rate")
    parser.add_argument("--stride", type=int, default=2, help="grid channel subsampling (1 = full resolution)")
    parser.add_argument("--max-sps", type=float, default=0, help="throttle steps per second (0 = full speed)")
    parser.add_argument("--catalyst-lib", default=os.path.join(pv_build, "lib", "catalyst"))
    parser.add_argument("--derive", action="append", default=[], metavar="NAME=EXPRESSION[@cells]",
                        help="a derived grid field from a VTK calculator expression over velocity, "
                             "pressure and speed; gradients (curl, qcrit, grad, div) need @cells")
    parser.add_argument("--lateral", choices=("periodic", "slip"), default="periodic",
                        help="y boundaries: periodic, or free-slip walls (needed for rotation, where a "
                             "periodic column of cylinders turns the whole flow)")
    parser.add_argument("--no-derived", action="store_true",
                        help="leave out the default derived fields (vorticity, q_criterion)")
    parser.add_argument("--steering-xml", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "steering.xml"),
                        help="where to write the steering proxy definition the Live client also loads")
    args = parser.parse_args()

    nx, ny, nz = args.size
    solver = Solver(nx, ny, nz, collision=args.collision, obstacle=args.obstacle, reynolds=args.re,
                    stride=args.stride, lateral=args.lateral)
    rotation_doc = ("Cylinder rotation rate alpha = omega R / U (surface speed over inflow speed); "
                    "positive is counterclockwise.") if args.obstacle == "cylinder" else None
    xml = write_proxy_xml(args.steering_xml, args.re, args.collision, derived=True, rotation_doc=rotation_doc)
    from derived import DEFAULTS, DerivedFields, parse

    specs = {**({} if args.no_derived else DEFAULTS), **parse(args.derive)}
    derived = DerivedFields(solver, specs)
    adaptor = CatalystAdaptor(os.path.abspath(args.script), solver, args.catalyst_lib, xml,
                              derived=derived)
    adaptor.monitor_history = args.monitor_history
    print(f"[tack-lbm] {nx}x{ny}x{nz} D3Q19, {args.collision}, {args.obstacle}, Re={args.re}", flush=True)

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    t_report = time.perf_counter()
    t_start = t_report
    step_report = solver.step_count
    calls = 0
    schedule = sorted((int(e.split(":")[0]), float(e.split(":")[1])) for e in args.rotation_schedule.split(",") if e)
    while running and (args.steps == 0 or solver.step_count < args.steps):
        for _ in range(args.interval - 1):
            solver.step()
        solver.step(report=True)
        while schedule and solver.step_count >= schedule[0][0]:
            print(f"[tack-lbm] step {solver.step_count}: rotation event alpha={schedule[0][1]:+.2f}", flush=True)
            set_rotation_rate(solver, schedule.pop(0)[1])
        stride = max(args.interval, int(adaptor.steered.get("capture_stride", args.interval)))
        if solver.step_count % stride not in (0, args.interval % stride):
            continue
        monitors = solver.monitors(args.interval)
        monitors["reynolds"] = solver.reynolds
        monitors["collision"] = float(COLLISION_CODES[solver.collision_name])
        monitors["rotation"] = rotation_rate(solver)
        monitors["derived_error"] = float(derived.error_column)
        step = solver.step_count
        monitors["time"] = step * solver.u0 / solver.diameter
        adaptor.execute(step, monitors["time"], monitors)
        adaptor.steer()
        now = time.perf_counter()
        if args.max_sps > 0:
            wait = step / args.max_sps - (now - t_start)
            if wait > 0:
                time.sleep(wait)
                now = time.perf_counter()
        calls += 1
        if calls % 10 == 0:
            sps = (step - step_report) / (now - t_report)
            t_report, step_report = now, step
            print(f"[tack-lbm] step {step}  {sps:.0f} steps/s  Cd={monitors['drag']:.3f}  "
                  f"Cl={monitors['lift']:+.3f}  rot={monitors['rotation']:+.2f}  res_u={monitors['momentum_x']:.2e}", flush=True)
    adaptor.finalize()


if __name__ == "__main__":
    main()
