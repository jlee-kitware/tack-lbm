"""One TRT step from Tack against a NumPy reference (TRT with Smagorinsky subgrid viscosity)."""
import numpy as np

import tack
from lbm import CX, CY, CZ, W, TRT, NoHeat, Solver, Cylinder, D3Q19, Flow, Quiet, ManagedArray, FIELDS, lbm_step, _opposite

tack.set_default_device("cuda")
nx, ny, nz = 8, 6, 4
n = nx * ny * nz
lat = D3Q19(nx, ny, nz)
rng = np.random.default_rng(1)
c = np.array([CX, CY, CZ]).T
w = np.array(W)
opp = np.array(_opposite())


def feq(rho, u):
    cu = 3.0 * (u @ c.T)
    return w * rho[:, None] * (1 + cu + 0.5 * cu ** 2 - 1.5 * (u ** 2).sum(1)[:, None])


rho0 = 1 + 0.01 * rng.standard_normal(n)
u0 = 0.05 * rng.standard_normal((n, 3))
f = feq(rho0, u0) + 1e-3 * rng.standard_normal((n, 19))
f_in = np.ascontiguousarray(f.T.reshape(-1)).astype(np.float32)

trt = TRT(0.002)
geo = Cylinder(-100.0, -100.0, 0.0)
flow = Flow(0.0, 1e9, 0.0)
fields = {k: ManagedArray(n).tack for k in FIELDS}
a_in = tack.array(f_in)
a_out = tack.empty((19 * n,))
heat = Solver.__new__(Solver)
heat.heat_enabled = False
heat.nx = nx
heat._setup_heat(n)
lbm_step(lat, trt, geo, flow, Quiet(fields), a_in, a_out, heat.heat, heat.g[0], heat.g[1])
got = a_out.numpy().reshape(19, n).T

idx = np.arange(n)
x, y, z = idx % nx, (idx // nx) % ny, idx // (nx * ny)
fs = np.empty((n, 19))
for q in range(19):
    src = (((z - c[q, 2]) % nz) * ny + (y - c[q, 1]) % ny) * nx + (x - c[q, 0]) % nx
    fs[:, q] = f_in.reshape(19, n)[q, src]
rho = fs.sum(1)
u = (fs @ c) / rho[:, None]
eq = feq(rho, u)
neq = fs - eq
pi_ab = np.einsum("qa,qb,nq->nab", c, c, neq)
pi = np.sqrt((pi_ab ** 2).sum(axis=(1, 2)))
tau = 0.5 * (trt.tau0 + np.sqrt(trt.tau0 ** 2 + 25.456 * trt.cs ** 2 * pi / rho))
omega_p = (1 / tau)[:, None]
omega_m = (1 / (trt.magic / (tau - 0.5) + 0.5))[:, None]
fp, fm = 0.5 * (fs + fs[:, opp]), 0.5 * (fs - fs[:, opp])
ep, em = 0.5 * (eq + eq[:, opp]), 0.5 * (eq - eq[:, opp])
ref = fs - omega_p * (fp - ep) - omega_m * (fm - em)
err = float(np.abs(got - ref).max())
print("max |tack - numpy|:", err)
assert err < 1e-5
