"""Fluid-structure interaction of a cantilever wing: 3D Tack lattice Boltzmann flow around
an elastic NACA 0012 wing whose deflection comes from an MFEM finite element model.

The root sits on a symmetry plane (z = -1/2) and the top of the domain is free slip, so the
lattice holds one half of a wing clamped at its centre. Every step:

  fluid -> structure   lift and pitching moment per lattice layer by momentum exchange,
                       projected on the wing's vibration modes as generalized forces
  structure -> fluid   each layer's plunge and pitch (and their rates) from the modal state,
                       through moving-wall bounce-back
"""

import numpy as np

import tack
from fsi_struct import Modal, WingFEM
from lbm import COLLISIONS, CX, CY, CZ, D3Q19, FIELDS, Flow, ManagedArray, Quiet, Report, coarsen


def _reflect_z():
    return [next(p for p in range(19) if (CX[p], CY[p], CZ[p]) == (CX[q], CY[q], -CZ[q])) for q in range(19)]


@tack.device_class
class FlexWing:
    """NACA 0012 wing from the root plane to ``span``; ``kin`` holds h, alpha, dh/dt, dalpha/dt per
    layer and ``load`` collects fx, fy, mz per layer."""

    def __init__(self, chord, x_ea, y_ea, ea, span, kin, load, reflect):
        self.chord, self.x_ea, self.y_ea, self.ea, self.span = chord, x_ea, y_ea, ea, span
        self.kin, self.load, self.reflect = kin, load, reflect

    @tack.func
    def solid(self, x, y, z):
        h = self.kin[4 * z]
        a = self.kin[4 * z + 1]
        dx = float(x) - self.x_ea
        dy = float(y) - (self.y_ea + h)
        s = dx * cos(a) - dy * sin(a)
        t = dx * sin(a) + dy * cos(a)
        xc = s / self.chord + self.ea
        c = min(max(xc, 0.0), 1.0)
        half = 0.6 * self.chord * (0.2969 * sqrt(c) - 0.1260 * c - 0.3516 * c * c + 0.2843 * c * c * c
                                   - 0.1036 * c * c * c * c)
        inside = (1.0 if xc >= 0.0 else 0.0) * (1.0 if xc <= 1.0 else 0.0) * (1.0 if abs(t) < half else 0.0)
        return inside * (1.0 if float(z) + 0.5 < self.span else 0.0)

    @tack.func
    def wall_ux(self, x, y, z):
        return self.kin[4 * z + 3] * (y - self.y_ea - self.kin[4 * z])

    @tack.func
    def wall_uy(self, x, y, z):
        return self.kin[4 * z + 2] - self.kin[4 * z + 3] * (x - self.x_ea)


@tack.kernel
def fsi_init(lat, geo, flow, f):
    for idx in range(lat.n):
        x = idx % lat.nx
        y = (idx // lat.nx) % lat.ny
        z = idx // (lat.nx * lat.ny)
        ux = flow.u0 * (1.0 - geo.solid(x, y, z))
        for q in range(lat.Q):
            f[q * lat.n + idx] = lat.feq(q, 1.0, ux, 0.0, 0.0)


@tack.kernel
def fsi_step(lat, phys, geo, flow, diag, f_in, f_out):
    for idx in range(lat.n):
        x = idx % lat.nx
        y = (idx // lat.nx) % lat.ny
        z = idx // (lat.nx * lat.ny)
        here = geo.solid(x, y, z)
        fluid = 1.0 - here
        f = tack.local_array(tack.f32, lat.Q)
        post = tack.local_array(tack.f32, lat.Q)
        fx = 0.0
        fy = 0.0
        mz = 0.0
        links = 0.0
        for q in range(lat.Q):
            o = lat.opp[q]
            xs = (x - lat.cx[q] + lat.nx) % lat.nx
            ys = (y - lat.cy[q] + lat.ny) % lat.ny
            zr = z - lat.cz[q]
            flip = (1 if zr < 0 else 0) + (1 if zr > lat.nz - 1 else 0)
            zs = min(max(zr, 0), lat.nz - 1)
            qs = q + flip * (geo.reflect[q] - q)
            wall = geo.solid(xs, ys, zs)
            pulled = f_in[qs * lat.n + (zs * lat.ny + ys) * lat.nx + xs]
            bounced = f_in[o * lat.n + idx]
            px = float(x) - 0.5 * float(lat.cx[q])
            py = float(y) - 0.5 * float(lat.cy[q])
            cu = float(lat.cx[q]) * geo.wall_ux(px, py, zs) + float(lat.cy[q]) * geo.wall_uy(px, py, zs)
            back = bounced + 6.0 * lat.w[q] * cu
            f[q] = pulled + wall * (back - pulled)
            hit = wall * fluid * (bounced + back)
            dfx = hit * float(lat.cx[o])
            dfy = hit * float(lat.cy[o])
            fx += dfx
            fy += dfy
            mz += (px - geo.x_ea) * dfy - (py - geo.y_ea - geo.kin[4 * z]) * dfx
            links += wall * fluid
        if links > 0.0:
            tack.atomic_add(geo.load, 3 * z, fx)
            tack.atomic_add(geo.load, 3 * z + 1, fy)
            tack.atomic_add(geo.load, 3 * z + 2, mz)
        rho = 0.0
        jx = 0.0
        jy = 0.0
        jz = 0.0
        for q in range(lat.Q):
            rho += f[q]
            jx += f[q] * float(lat.cx[q])
            jy += f[q] * float(lat.cy[q])
            jz += f[q] * float(lat.cz[q])
        ux = fluid * jx / rho
        uy = fluid * jy / rho
        uz = fluid * jz / rho
        phys.collide(lat, f, post, rho, ux, uy, uz)
        t = max(0.0, (float(x) - flow.sponge_start) / (float(lat.nx) - flow.sponge_start))
        sigma = max(here, flow.sponge_strength * t * t)
        target_x = fluid * flow.u0 + here * geo.wall_ux(float(x), float(y), z)
        target_y = here * geo.wall_uy(float(x), float(y), z)
        rho_s = rho + sigma * (1.0 - rho)
        ux_s = ux + sigma * (target_x - ux)
        uy_s = uy + sigma * (target_y - uy)
        uz_s = (1.0 - sigma) * uz
        for q in range(lat.Q):
            neq = post[q] - lat.feq(q, rho, ux, uy, uz)
            f_out[q * lat.n + idx] = lat.feq(q, rho_s, ux_s, uy_s, uz_s) + (1.0 - sigma) * neq
        if diag.enabled == 1:
            diag.ux[idx] = ux
            diag.uy[idx] = uy
            diag.uz[idx] = uz
            diag.p[idx] = (rho - 1.0) / 3.0
            diag.vmag[idx] = sqrt(ux * ux + uy * uy + uz * uz)


class FSISolver:
    OUTPUT = ("ux", "uy", "uz", "p", "vmag")

    def __init__(self, nx=256, ny=128, nz=104, chord=32.0, span=64.0, ea=0.4, u0=0.08, reynolds=500.0,
                 collision="les", reduced_velocity=1.5, damping=0.005, hardening=40.0, device="cuda", stride=1,
                 fem=None):
        tack.set_default_device(device)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.lat = D3Q19(nx, ny, nz)
        self.chord, self.span, self.u0 = chord, span, u0
        self.diameter = chord
        self.flow = Flow(u0, 0.75 * nx, 1.0)
        self.collision_name = collision
        self.set_reynolds(reynolds)
        self.fem = fem or WingFEM(chord=chord, span=span, ea=ea, n_span=int(span // 4))
        self.torsion = self._first_torsion()
        self.mode_h, self.mode_alpha = self.fem.sections_at(np.arange(nz) + 0.5)
        self.structure = Modal(self.fem, self.fem.omega0, damping=damping, hardening=hardening)
        self.set_reduced_velocity(reduced_velocity)
        self._kin = ManagedArray(4 * nz)
        self._load = ManagedArray(3 * nz)
        self.reflect = tack.array(np.array(_reflect_z(), np.int32))
        self.geo = FlexWing(chord, 0.25 * nx, 0.5 * ny, ea, span, self._kin.tack, self._load.tack, self.reflect)
        self.held = True
        n = self.lat.n
        self.f = [tack.empty((19 * n,)), tack.empty((19 * n,))]
        self.fields = {name: ManagedArray(n) for name in FIELDS}
        views = {name: a.tack for name, a in self.fields.items()}
        self.quiet, self.report = Quiet(views), Report(views)
        self.stride = stride
        self.out_dims = tuple((d + stride - 1) // stride for d in (nx, ny, nz))
        if stride > 1:
            m = self.out_dims[0] * self.out_dims[1] * self.out_dims[2]
            self.output = {name: ManagedArray(m) for name in self.OUTPUT}
        else:
            self.output = {name: self.fields[name] for name in self.OUTPUT}
        fsi_init(self.lat, self.geo, self.flow, self.f[0])
        self.step_count = 0
        self.layer_load = np.zeros((nz, 3))
        self.drag = self.lift = self.moment = 0.0

    def _first_torsion(self):
        b = 0.5 * self.chord
        for i in range(len(self.fem.omega0)):
            if abs(self.fem.alpha[i, -1]) > abs(self.fem.h[i, -1]) / b and abs(self.fem.alpha[i, -1]) > 0.0:
                return i
        return 1

    def set_reynolds(self, reynolds):
        self.reynolds = float(reynolds)
        self.nu = self.u0 * self.chord / self.reynolds
        self.phys = COLLISIONS[self.collision_name](self.nu)

    def set_physics(self, collision, reynolds):
        """A new air speed at fixed air viscosity and wing: Re and U* both scale with it."""
        speed_ratio = float(reynolds) / self.reynolds
        self.collision_name = collision
        self.set_reynolds(reynolds)
        self.set_reduced_velocity(self.reduced_velocity() * speed_ratio)

    def reduced_velocity(self):
        return self.u0 / (0.5 * self.chord * self.structure.omega[self.torsion])

    def set_reduced_velocity(self, value):
        target = self.u0 / (0.5 * self.chord * value)
        self.structure.omega = self.fem.omega0 * (target / self.fem.omega0[self.torsion])

    def frequencies(self):
        return self.structure.omega.copy()

    def update_output(self):
        if self.stride > 1:
            mx, my, mz = self.out_dims
            for name in self.OUTPUT:
                coarsen(self.lat, self.stride, self.fields[name].tack, self.output[name].tack, mx, my, mz)

    def release(self, tip_twist_rate=None):
        """Free the wing, nudged with a twist rate in its first torsion mode (default: a 2 degree tip swing)."""
        self.held = False
        k = self.torsion
        tip = self.fem.alpha[k, -1]
        rate = np.radians(2.0) * self.structure.omega[k] if tip_twist_rate is None else tip_twist_rate
        self.structure.v[k] += rate / tip
        self._sync_kinematics()

    def gust(self, degrees):
        """A sudden tip twist rate, as from a gust: the tip swings by about ``degrees``."""
        k = self.torsion
        self.structure.v[k] += np.radians(degrees) * self.structure.omega[k] / self.fem.alpha[k, -1]

    def _sync_kinematics(self):
        s = self.structure
        kin = self._kin.numpy.reshape(self.nz, 4)
        kin[:, 0] = s.q @ self.mode_h
        kin[:, 1] = s.q @ self.mode_alpha
        kin[:, 2] = s.v @ self.mode_h
        kin[:, 3] = s.v @ self.mode_alpha

    def step(self, report=False):
        self._load.numpy[:] = 0.0
        fsi_step(self.lat, self.phys, self.geo, self.flow, self.report if report else self.quiet,
                 self.f[0], self.f[1])
        self.f.reverse()
        tack.sync()
        load = self._load.numpy.reshape(self.nz, 3).astype(np.float64)
        load[:, 2] *= -1.0
        self.layer_load = load
        self.drag, self.lift, self.moment = load.sum(0) / self.span
        if not self.held:
            force = self.mode_h @ load[:, 1] + self.mode_alpha @ load[:, 2]
            self.structure.step(force)
            self._sync_kinematics()
        self.step_count += 1

    def tip(self):
        """Tip plunge in semichords and tip pitch in degrees."""
        kin = self._kin.numpy.reshape(self.nz, 4)
        k = min(int(self.span - 0.5), self.nz - 1)
        return kin[k, 0] / (0.5 * self.chord), np.degrees(kin[k, 1])

    def motion(self):
        return self.tip()

    def coefficients(self):
        q = 0.5 * self.u0 * self.u0 * self.chord
        return self.drag / q, self.lift / q, self.moment / (q * self.chord)

    def wing_surface(self):
        """Deformed FEM wing surface in lattice coordinates: points and quads."""
        pts = self.fem.deformed(self.structure.q)
        pts = pts + np.array([self.geo.x_ea, self.geo.y_ea, -0.5])
        return pts, self.fem.quads
