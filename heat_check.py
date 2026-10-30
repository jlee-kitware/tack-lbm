"""Heat-sink duct with fixed wall temperature: stability, heat flux, and a mid-height slice image."""
import sys
import time

import numpy as np

import tack
from lbm import Solver

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
re = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
s = Solver(320, 96, 128, collision="les", obstacle="heatsink", reynolds=re, heat=True,
           diameter=12.0)
print(f"Re {re}: nu {s.nu:.4f}, alpha {s.nu / s.prandtl:.4f}, cell Peclet {s.u0 * s.prandtl / s.nu:.1f}")
s.set_wall_temperature(1.0)
s.step(report=True)
tack.sync()
t0 = time.perf_counter()
for k in range(steps // 100):
    for _ in range(99):
        s.step()
    s.step(report=True)
    m = s.monitors(100)
    if (k + 1) % 10 == 0:
        temp = s.heat_fields["temp"].numpy
        print(f"step {s.step_count}: heat into fluid {m['heat_flux']:.3f}, T range "
              f"({np.nanmin(temp):.3f}, {np.nanmax(temp):.3f}), drag {m['drag']:.2f}, "
              f"{100 * (k + 1) / (time.perf_counter() - t0):.0f} steps/s", flush=True)
tack.sync()
temp = s.heat_fields["temp"].numpy.reshape(s.nz, s.ny, s.nx)
ux = s.fields["ux"].numpy.reshape(s.nz, s.ny, s.nx)
np.save("/tmp/heat_temp_slice.npy", temp[:, 30, :])
np.save("/tmp/heat_ux_slice.npy", ux[:, 30, :])
print("nan:", bool(np.isnan(temp).any()))
i = np.unravel_index(np.argmin(temp), temp.shape)
print("min T at (z, y, x) =", i, "value", float(temp[i]))
fluid_t = temp[(temp > -1)]
print("fraction below -0.01:", float((temp < -0.01).mean()), "above 1.01:", float((temp > 1.01).mean()))
