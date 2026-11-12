import time

import numpy as np

from fsi_struct import WingFEM

t0 = time.time()
w = WingFEM()
print(f"build+modes {time.time() - t0:.1f} s, dofs {w.fes.GetVSize()}, vertices {w.nv}, quads {len(w.quads)}")
x = w.coords
print("bounds", x.min(0).round(2), x.max(0).round(2))
b = 0.5 * w.chord
m, xcg, r2 = w.section_properties()
print(f"mass/span {m:.1f}  mu {m / (np.pi * b * b):.1f}  x_alpha {xcg / b:.3f}  r_alpha2 {r2 / b ** 2:.3f}")
for i, om in enumerate(w.omega0):
    ht, at = w.h[i, -1] / b, w.alpha[i, -1]
    n = max(abs(ht), abs(at))
    axis = w.ea + (w.h[i, -1] / w.alpha[i, -1]) / w.chord if abs(at) > 1e-12 else np.nan
    disp = w.node_disp(i)
    tip = np.abs(x[:, 2] - w.span) < 1e-5
    chordwise = np.abs(disp[tip, 0]).max() / np.abs(disp[tip]).max()
    print(f"mode {i}: omega {om:.4e}  /mode0 {om / w.omega0[0]:.2f}  tip h/b {ht / n:+.2f}  alpha {at / n:+.2f}"
          f"  node-line x/c {axis:+.2f}  chordwise share {chordwise:.2f}")
