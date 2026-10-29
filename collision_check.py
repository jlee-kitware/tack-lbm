"""Which collision models stay stable at which Reynolds number, and the Strouhal number at Re 300."""
import sys

import numpy as np

import tack
from lbm import Solver

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
for re in (300, 1000, 3000):
    for collision in ("bgk", "trt", "les"):
        s = Solver(collision=collision, reynolds=re)
        lift, blew_up = [], None
        for k in range(steps // 50):
            for _ in range(49):
                s.step()
            s.step(report=True)
            m = s.monitors(50)
            if not np.isfinite(m["drag"]):
                blew_up = s.step_count
                break
            lift.append(m["lift"])
        tack.sync()
        if blew_up:
            print(f"Re {re:5d} {collision}: diverged at step {blew_up}", flush=True)
            continue
        cl = np.array(lift[len(lift) // 2:])
        cl -= cl.mean()
        spec = np.abs(np.fft.rfft(cl))
        freq = np.fft.rfftfreq(len(cl), d=50)[np.argmax(spec[1:]) + 1]
        st = freq * s.diameter / s.u0
        print(f"Re {re:5d} {collision}: stable, Cl amplitude {np.abs(cl).max():.2f}, Strouhal {st:.3f}", flush=True)
