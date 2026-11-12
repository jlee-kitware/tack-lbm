"""Offline flutter sweep: U* x hardening for the elastic wing; tip pitch and plunge per window.

    python fsi_sweep.py STEPS U1,U2,... [RE] [H1,H2,...] [SPAN]
"""
import sys
import time

import numpy as np

from fsi import FSISolver
from fsi_struct import WingFEM

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
cases = [float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [3.5, 7.0]
reynolds = float(sys.argv[3]) if len(sys.argv) > 3 else 500.0
hardenings = [float(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [200.0]
span = float(sys.argv[5]) if len(sys.argv) > 5 else 64.0
structure = sys.argv[6] if len(sys.argv) > 6 else "beam"
nz = int(span) + 40
beam_args = dict((k, float(v)) for k, v in (kv.split("=") for kv in sys.argv[7].split(","))) \
    if len(sys.argv) > 7 else {}
if structure == "beam":
    from fsi_beam import WingBeam
    fem = WingBeam(span=span, **beam_args)
    print("beam", beam_args, flush=True)
else:
    fem = WingFEM(span=span, n_span=int(span // 4))
for hardening in hardenings:
    for ustar in cases:
        s = FSISolver(nz=nz, span=span, reduced_velocity=ustar, reynolds=reynolds, fem=fem, hardening=hardening)
        for i in range(2000):
            s.step()
        s.release()
        hist = []
        t0 = time.time()
        for i in range(steps):
            s.step()
            hist.append(s.tip())
            if not np.all(np.isfinite(hist[-1])) or abs(hist[-1][1]) > 60:
                break
        rate = len(hist) / (time.time() - t0)
        hist = np.array(hist)
        np.save(f"/tmp/fsi_sweep_s{span:g}_{ustar:g}_{reynolds:g}_h{hardening:g}.npy", hist)
        win = 4000
        cols = []
        for k in range(0, len(hist) - win + 1, win):
            p, h = hist[k:k + win, 1], hist[k:k + win, 0]
            cols.append(f"({p.mean():+.1f} {0.5 * np.ptp(p):.1f} | {h.mean():+.2f} {0.5 * np.ptp(h):.2f})")
        print(f"span {span:g} U* {ustar:4.2f} H {hardening:g} ({rate:.0f} steps/s), per {win}: "
              f"(pitch mean, half range | plunge mean, half range) " + " ".join(cols), flush=True)
        del s
