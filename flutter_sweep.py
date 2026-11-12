"""Release the wing at several reduced velocities U* = U / (b omega_alpha) and measure whether the
pitch motion grows or decays: growth per pitch period from the envelope of |alpha|."""
import argparse
import time

import numpy as np

from flutter import FlutterSolver

parser = argparse.ArgumentParser()
parser.add_argument("--re", type=float, default=500.0)
parser.add_argument("--speeds", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0, 6.0])
parser.add_argument("--periods", type=float, default=8.0)
parser.add_argument("--save", default="")
parser.add_argument("--ea", type=float, default=0.4)
parser.add_argument("--x-alpha", type=float, default=0.1)
parser.add_argument("--mu", type=float, default=20.0)
parser.add_argument("--collision", default="les")
parser.add_argument("--hardening", type=float, default=0.0)
parser.add_argument("--chord", type=float, default=48.0)
parser.add_argument("--size", type=int, nargs=3, default=[384, 192, 4])
args = parser.parse_args()

histories = {}
for us in args.speeds:
    s = FlutterSolver(*args.size, chord=args.chord, reynolds=args.re, ea=args.ea, x_alpha=args.x_alpha, mu=args.mu,
                      collision=args.collision, hardening=args.hardening)
    s.set_reduced_velocity(us)
    for _ in range(6000):
        s.step()
    s.release()
    period = 2 * np.pi / s.section.omega_alpha
    steps = int(args.periods * period)
    t0 = time.perf_counter()
    rec = np.zeros((steps, 4))
    for k in range(steps):
        s.step()
        h, a = s.motion()
        rec[k] = (k / period, h, a, s.coefficients()[1])
    histories[us] = rec
    a = rec[:, 2]
    peaks = [i for i in range(1, steps - 1) if abs(a[i]) >= abs(a[i - 1]) and abs(a[i]) > abs(a[i + 1])]
    pk_t, pk_a = rec[peaks, 0], np.abs(a[peaks])
    late = pk_t > 2.0
    growth = np.exp(np.polyfit(pk_t[late], np.log(pk_a[late]), 1)[0]) if late.sum() > 3 else float("nan")
    zero = np.where(np.diff(np.sign(a)) != 0)[0]
    freq = (len(zero) - 1) / 2 / (rec[zero[-1], 0] - rec[zero[0], 0]) if len(zero) > 3 else float("nan")
    print(f"U* {us:4.1f}: growth per period {growth:6.3f}, pitch amp end {np.abs(a[-int(period):]).max():6.2f} deg, "
          f"plunge amp end {np.abs(rec[-int(period):, 1]).max():.3f} b, freq/f_alpha {freq:.2f}, "
          f"{steps / (time.perf_counter() - t0):.0f} steps/s", flush=True)
if args.save:
    np.savez(args.save, **{f"u{us}": h for us, h in histories.items()})
