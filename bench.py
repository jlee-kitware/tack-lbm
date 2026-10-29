import sys
import time

import numpy as np

import tack
from lbm import Solver

collision = sys.argv[1] if len(sys.argv) > 1 else "les"
t0 = time.perf_counter()
s = Solver(collision=collision)
tack.sync()
print(f"{collision}: setup + first compile {time.perf_counter() - t0:.1f} s")
for _ in range(5):
    s.step()
tack.sync()
steps = 200
t0 = time.perf_counter()
for _ in range(steps):
    s.step()
tack.sync()
dt = time.perf_counter() - t0
print(f"{steps / dt:.0f} steps/s, {s.lat.n * steps / dt / 1e6:.0f} MLUPS")
for k in range(4):
    for _ in range(499):
        s.step()
    s.step(report=True)
    m = s.monitors(500)
    ux = s.fields["ux"].numpy
    print(f"step {s.step_count}: " + " ".join(f"{k_}={v:.3e}" for k_, v in m.items())
          + f" ux[min,max]=({ux.min():.3f},{ux.max():.3f}) nan={np.isnan(ux).any()}")
