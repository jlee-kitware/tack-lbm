import sys

import numpy as np

from fsi import FSISolver
from fsi_struct import WingFEM

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
cases = [float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.75, 1.0, 1.25, 1.5]
reynolds = float(sys.argv[3]) if len(sys.argv) > 3 else 500.0
fem = WingFEM(span=64.0, n_span=16)
for ustar in cases:
    s = FSISolver(reduced_velocity=ustar, reynolds=reynolds, fem=fem)
    for i in range(2000):
        s.step()
    s.release()
    window = 2000
    amps = []
    peak = 0.0
    hist = []
    for i in range(steps):
        s.step()
        h, a = s.tip()
        hist.append((h, a))
        peak = max(peak, abs(a))
        if (i + 1) % window == 0:
            amps.append(peak)
            peak = 0.0
        if not np.isfinite(a) or abs(a) > 60:
            amps.append(float("nan"))
            break
    np.save(f"/tmp/fsi_sweep_{ustar:g}_{reynolds:g}.npy", np.array(hist))
    print(f"U* {ustar:5.2f} Re {reynolds:.0f}: tip pitch peak per {window} steps: "
          + " ".join(f"{v:.2f}" for v in amps), flush=True)
    del s
