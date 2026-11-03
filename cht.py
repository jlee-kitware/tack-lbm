"""Conjugate heat transfer: Tack LBM air flow through a finned heat sink, PyMFEM conduction in the sink.

Partitioned coupling every ``interval`` lattice steps (Dirichlet-Neumann, the stable
pairing when the solid conducts far better than the fluid):

  fluid -> solid   heat flux leaving each wetted solid voxel, as a Neumann load
                   (W^T q, W evaluating the MFEM field at voxel centres)
  solid -> fluid   the solid temperature at every sink voxel (W T), which the LBM
                   holds at the wall by anti-bounce-back
"""

import time

import numpy as np

import tack
from lbm import Solver
from solid import HeatSinkSolid, spec_of


@tack.kernel
def sink_mask(lat, geo, sink, solid):
    for idx in range(lat.n):
        x = idx % lat.nx
        y = (idx // lat.nx) % lat.ny
        z = idx // (lat.nx * lat.ny)
        sink[idx] = geo.sink(x, y, z)
        solid[idx] = geo.solid(x, y, z)


class ConjugateHeat:
    def __init__(self, size=(320, 96, 128), reynolds=300.0, collision="les", power=50.0,
                 kappa=200.0, capacity=0.05, order=2, interval=100, stride=2, device="cuda"):
        nx, ny, nz = size
        self.fluid = Solver(nx, ny, nz, collision=collision, obstacle="heatsink", reynolds=reynolds,
                            heat=True, diameter=12.0, stride=stride, device=device)
        self.interval = interval
        lat = self.fluid.lat
        sink, solid = tack.zeros((lat.n,)), tack.zeros((lat.n,))
        sink_mask(lat, self.fluid.geo, sink, solid)
        self.voxels = np.flatnonzero(sink.numpy() > 0.5)
        fluid3 = (solid.numpy() < 0.5).reshape(nz, ny, nx)
        wet = sum(np.roll(fluid3, s, axis=a) for a in range(3) for s in (1, -1)).reshape(-1)
        self.wet_faces = wet[self.voxels].astype(np.float64)
        self.surface = np.flatnonzero(self.wet_faces > 0)
        x = self.voxels % nx
        y = (self.voxels // nx) % ny
        z = self.voxels // (nx * ny)
        t0 = time.perf_counter()
        self.solid = HeatSinkSolid(spec_of(self.fluid), order=order)
        self.W = self.solid.voxel_map(np.stack([x, y, z], axis=1).astype(float))
        self.Ws = self.W[self.surface]
        g = self.fluid.geo
        cx, cz = 0.5 * (g.x0 + g.x1), 0.5 * (g.z0 + g.z1)
        half = 0.15 * (g.x1 - g.x0)
        chip = (cx - half, cx + half, cz - half, cz + half, g.y_base - 0.5)
        self.kappa, self.capacity, self.chip, self.power = kappa, capacity, chip, power
        self.recouple()
        self.setup_seconds = time.perf_counter() - t0
        self.push_temperature()

    def recouple(self):
        """Set up the solid for the fluid's current viscosity (again after a Reynolds number change)."""
        f = self.fluid
        self.h = 2.0 * f.nu / f.prandtl
        self.solid.setup(kappa=self.kappa, alpha=f.nu / f.prandtl, capacity=self.capacity,
                         dt=float(self.interval), power=self.power, chip=self.chip,
                         robin=(self.Ws, self.h * self.wet_faces[self.surface]))

    def push_temperature(self):
        self.fluid.heat_fields["t_solid"].numpy[self.voxels] = self.W @ self.solid.dofs()

    def set_power(self, power):
        self.power = power
        self.solid.set_power(power)

    def advance(self):
        """``interval`` fluid steps, then one coupled exchange and solid step."""
        f = self.fluid
        for _ in range(self.interval - 1):
            f.step()
        f.step(report=True)
        monitors = f.monitors(self.interval)
        t_fluid_sum = f.heat_fields["q_solid"].numpy[self.voxels[self.surface]].astype(np.float64)
        iterations = self.solid.step(fluid_temperature=t_fluid_sum / self.wet_faces[self.surface])
        self.push_temperature()
        T = self.solid.dofs()
        t_wall = self.Ws @ T
        heat_flux = float(self.h * (self.wet_faces[self.surface] * t_wall - t_fluid_sum).sum())
        monitors.update(power=self.power, heat_flux=heat_flux, solid_t_max=float(T.max()),
                        solid_t_mean=float(T.mean()), cg_iterations=float(iterations))
        return monitors


if __name__ == "__main__":
    import sys

    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    t0 = time.perf_counter()
    cht = ConjugateHeat()
    print(f"setup {time.perf_counter() - t0:.1f} s (solid + maps {cht.setup_seconds:.1f} s): "
          f"{len(cht.voxels)} sink voxels, {cht.solid.ndofs} solid dofs, W nnz {cht.W.nnz}", flush=True)
    t0 = time.perf_counter()
    for k in range(steps // cht.interval):
        m = cht.advance()
        if (k + 1) % 20 == 0:
            print(f"step {cht.fluid.step_count}: power {m['power']:.1f}, heat into air {m['heat_flux']:.2f}, "
                  f"solid T max {m['solid_t_max']:.3f} mean {m['solid_t_mean']:.3f}, CG {m['cg_iterations']:.0f}, "
                  f"{cht.interval * (k + 1) / (time.perf_counter() - t0):.0f} steps/s", flush=True)
