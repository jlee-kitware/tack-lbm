"""Check the derived fields against tack.dataset's gradient filters and a NumPy reference.

    python derived_check.py [--steps 400] [--size 128 64 32] [--stride 1]
"""

import argparse
import time

import numpy as np
import tack
import tack.dataset as td

from derived import DEFAULTS, DerivedFields
from lbm import Solver

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--size", type=int, nargs=3, default=[128, 64, 32])
parser.add_argument("--stride", type=int, default=1)
args = parser.parse_args()

solver = Solver(*args.size, collision="les", obstacle="cylinder", reynolds=300.0, stride=args.stride)
for _ in range(args.steps):
    solver.step()
solver.update_output()
tack.sync()
specs = dict(DEFAULTS)
specs["speed_check"] = ("mag(velocity) - speed", "points")
specs["pressure_coefficient"] = ("2 * pressure / (0.08 * 0.08)", "points")
derived = DerivedFields(solver, specs)
results = derived.evaluate()
data = derived.data
failures = 0


def report(label, got, ref, tol):
    global failures
    err = float(np.max(np.abs(got - ref)) / max(1e-30, float(np.max(np.abs(ref)))))
    ok = err < tol
    failures += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {label:58} max error {err:.1e} of the largest value")


print(f"{args.size} grid, stride {args.stride}, {args.steps} steps: output grid {solver.out_dims}")
fq = td.flow_quantities(data, td.gradients(data, "velocity", at="cells"))
report("vorticity vs td.gradients + flow_quantities", results["vorticity"][0],
       np.linalg.norm(fq.vorticity.values.numpy(), axis=1), 1e-5)
report("q_criterion vs td.gradients + flow_quantities", results["q_criterion"][0],
       fq.q_criterion.values.numpy(), 1e-5)

# NumPy: a trilinear cell's gradient at its center is the mean of the four edge differences.
mx, my, mz = solver.out_dims
h = solver.stride / solver.diameter
u = np.stack([solver.output[k].numpy.reshape(mz, my, mx) for k in ("ux", "uy", "uz")])


def d(axis):
    a = u.astype(np.float64)
    if axis == 0:
        e = (a[..., 1:] - a[..., :-1]) / h
        return 0.25 * (e[:, :-1, :-1] + e[:, 1:, :-1] + e[:, :-1, 1:] + e[:, 1:, 1:]).transpose(0, 1, 2, 3)
    if axis == 1:
        e = (a[:, :, 1:, :] - a[:, :, :-1, :]) / h
        return 0.25 * (e[:, :-1, :, :-1] + e[:, 1:, :, :-1] + e[:, :-1, :, 1:] + e[:, 1:, :, 1:])
    e = (a[:, 1:, :, :] - a[:, :-1, :, :]) / h
    return 0.25 * (e[:, :, :-1, :-1] + e[:, :, 1:, :-1] + e[:, :, :-1, 1:] + e[:, :, 1:, 1:])


dx, dy, dz = d(0), d(1), d(2)      # [component, k, j, i] at cell centers
w = np.stack([dy[2] - dz[1], dz[0] - dx[2], dx[1] - dy[0]])
report("vorticity vs NumPy cell-center differences", results["vorticity"][0],
       np.sqrt((w ** 2).sum(axis=0)).ravel(), 1e-4)
q = -0.5 * (dx[0] ** 2 + dy[1] ** 2 + dz[2] ** 2) - (dy[0] * dx[1] + dz[0] * dx[2] + dz[1] * dy[2])
report("q_criterion vs NumPy cell-center differences", results["q_criterion"][0], q.ravel(), 1e-4)
speed = results["speed_check"][0]
print(f"  INFO  mag(velocity) - speed: max |difference| {float(np.max(np.abs(speed))):.1e}")

t0 = time.perf_counter()
for _ in range(10):
    derived.evaluate()
t_eval = (time.perf_counter() - t0) / 10
t0 = time.perf_counter()
for _ in range(50):
    solver.step()
tack.sync()
t_step = (time.perf_counter() - t0) / 50
print(f"  INFO  derived fields (4 expressions in 2 kernels, to host) {1e3 * t_eval:.2f} ms per Catalyst call; "
      f"one solver step {1e3 * t_step:.2f} ms")
print("all checks passed" if not failures else f"{failures} checks failed")

