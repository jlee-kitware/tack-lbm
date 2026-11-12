"""Plot pitch and plunge histories saved by flutter_sweep.py --save."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data = np.load(sys.argv[1])
out = sys.argv[2]
keys = sorted(data.files, key=lambda k: float(k[1:]))
fig, axes = plt.subplots(len(keys), 1, figsize=(10, 2.1 * len(keys)), sharex=True)
for ax, key in zip(np.atleast_1d(axes), keys):
    rec = data[key]
    ax.plot(rec[:, 0], rec[:, 2], color="#0A7C8C", lw=1.2, label="pitch (deg)")
    ax2 = ax.twinx()
    ax2.plot(rec[:, 0], rec[:, 1], color="#C2571A", lw=1.0, alpha=0.8, label="plunge (h/b)")
    ax.set_ylabel("pitch, deg", color="#0A7C8C")
    ax2.set_ylabel("plunge, h/b", color="#C2571A")
    ax.set_title(f"U* = {float(key[1:]):.2f}", loc="left", fontsize=10)
    ax.grid(alpha=0.3)
np.atleast_1d(axes)[-1].set_xlabel("time, pitch periods (2 pi / omega_alpha)")
fig.tight_layout()
fig.savefig(out, dpi=110)
print(out)
