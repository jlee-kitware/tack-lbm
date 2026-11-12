"""Pitch-plunge flutter of a NACA 0012 wing section in a Tack lattice Boltzmann flow.

The wing is rigid and sits on two springs: plunge h (up positive) and pitch alpha
(nose up positive) about an elastic axis at chord fraction ``ea``. The fluid and
the two structural equations exchange data every lattice step:

  fluid -> structure   lift and pitching moment by momentum exchange on every wall link
  structure -> fluid   the wing's shape and rigid-body velocity, through bounce-back
                       on a moving wall; nodes the wing uncovers start from equilibrium
                       at the wall velocity
"""

from math import pi

import numpy as np

import tack
from lbm import COLLISIONS, D3Q19, FIELDS, Flow, ManagedArray, Quiet, Report


@tack.device_class
class Wing:
    """NACA 0012 section spanning z. ``state`` holds h, alpha, dh/dt, dalpha/dt; ``load`` collects fx, fy, mz."""

    def __init__(self, chord, x_ea, y_ea, ea, state, load):
        self.chord, self.x_ea, self.y_ea, self.ea = chord, x_ea, y_ea, ea
        self.state, self.load = state, load

    @tack.func
    def solid(self, x, y, z):
        h = self.state[0]
        a = self.state[1]
        dx = float(x) - self.x_ea
        dy = float(y) - (self.y_ea + h)
        s = dx * cos(a) - dy * sin(a)
        t = dx * sin(a) + dy * cos(a)
        xc = s / self.chord + self.ea
        c = min(max(xc, 0.0), 1.0)
        half = 0.6 * self.chord * (0.2969 * sqrt(c) - 0.1260 * c - 0.3516 * c * c + 0.2843 * c * c * c
                                   - 0.1036 * c * c * c * c)
        return (1.0 if xc >= 0.0 else 0.0) * (1.0 if xc <= 1.0 else 0.0) * (1.0 if abs(t) < half else 0.0)

    @tack.func
    def wall_ux(self, x, y):
        return self.state[3] * (y - self.y_ea - self.state[0])

    @tack.func
    def wall_uy(self, x, y):
        return self.state[2] - self.state[3] * (x - self.x_ea)


@tack.kernel
def wing_init(lat, geo, flow, f):
    for idx in range(lat.n):
        x = idx % lat.nx
        y = (idx // lat.nx) % lat.ny
        z = idx // (lat.nx * lat.ny)
        ux = flow.u0 * (1.0 - geo.solid(x, y, z))
        for q in range(lat.Q):
            f[q * lat.n + idx] = lat.feq(q, 1.0, ux, 0.0, 0.0)


@tack.kernel
def wing_step(lat, phys, geo, flow, diag, f_in, f_out):
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
            zs = (z - lat.cz[q] + lat.nz) % lat.nz
            wall = geo.solid(xs, ys, zs)
            pulled = f_in[q * lat.n + (zs * lat.ny + ys) * lat.nx + xs]
            bounced = f_in[o * lat.n + idx]
            px = float(x) - 0.5 * float(lat.cx[q])
            py = float(y) - 0.5 * float(lat.cy[q])
            cu = float(lat.cx[q]) * geo.wall_ux(px, py) + float(lat.cy[q]) * geo.wall_uy(px, py)
            back = bounced + 6.0 * lat.w[q] * cu
            f[q] = pulled + wall * (back - pulled)
            hit = wall * fluid * (bounced + back)
            dfx = hit * float(lat.cx[o])
            dfy = hit * float(lat.cy[o])
            fx += dfx
            fy += dfy
            mz += (px - geo.x_ea) * dfy - (py - geo.y_ea - geo.state[0]) * dfx
            links += wall * fluid
        if links > 0.0:
            tack.atomic_add(geo.load, 0, fx)
            tack.atomic_add(geo.load, 1, fy)
            tack.atomic_add(geo.load, 2, mz)
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
        # The sponge and every solid node relax to a target: free stream in the sponge,
        # the wall's own velocity inside the wing.
        t = max(0.0, (float(x) - flow.sponge_start) / (float(lat.nx) - flow.sponge_start))
        sigma = max(here, flow.sponge_strength * t * t)
        target_x = fluid * flow.u0 + here * geo.wall_ux(float(x), float(y))
        target_y = here * geo.wall_uy(float(x), float(y))
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


class Section:
    """Typical-section structure per unit span, fluid density 1: mass ratio mu, centre of gravity
    x_alpha semichords aft of the elastic axis, squared radius of gyration r_alpha2 (semichords^2)
    and frequency ratio omega_h / omega_alpha."""

    def __init__(self, semichord, mu=20.0, x_alpha=0.1, r_alpha2=0.24, freq_ratio=0.4, omega_alpha=0.002,
                 damping=0.0, hardening=0.0):
        b = semichord
        self.b, self.mu, self.x_alpha, self.r_alpha2, self.freq_ratio = b, mu, x_alpha, r_alpha2, freq_ratio
        self.m = mu * pi * b * b
        self.damping = damping
        self.hardening = hardening
        self.q = np.zeros(2)
        self.v = np.zeros(2)
        self.set_frequency(omega_alpha)

    def set_frequency(self, omega_alpha):
        m, b = self.m, self.b
        self.omega_alpha = omega_alpha
        self.mass = np.array([[m, -m * self.x_alpha * b], [-m * self.x_alpha * b, m * self.r_alpha2 * b * b]])
        k_h = m * (self.freq_ratio * omega_alpha) ** 2
        k_a = m * self.r_alpha2 * b * b * omega_alpha ** 2
        self.stiffness = np.diag([k_h, k_a])
        self.dampers = 2.0 * self.damping * np.diag([np.sqrt(k_h * m), np.sqrt(k_a * m * self.r_alpha2 * b * b)])
        self.inverse = np.linalg.inv(self.mass)

    def step(self, lift, moment, dt=1.0):
        """Semi-implicit Euler; the pitch spring stiffens as k_alpha (alpha + hardening alpha^3)."""
        spring = self.stiffness @ self.q
        spring[1] += self.stiffness[1, 1] * self.hardening * self.q[1] ** 3
        force = np.array([lift, moment]) - spring - self.dampers @ self.v
        self.v += dt * (self.inverse @ force)
        self.q += dt * self.v


class FlutterSolver:
    def __init__(self, nx=384, ny=192, nz=4, chord=48.0, ea=0.4, u0=0.08, reynolds=500.0, collision="les",
                 device="cuda", **section):
        tack.set_default_device(device)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.lat = D3Q19(nx, ny, nz)
        self.chord, self.u0 = chord, u0
        self.flow = Flow(u0, 0.75 * nx, 1.0)
        self.collision_name = collision
        self.set_reynolds(reynolds)
        self._state = ManagedArray(4)
        self._load = ManagedArray(3)
        self.geo = Wing(chord, 0.25 * nx, 0.5 * ny, ea, self._state.tack, self._load.tack)
        self.section = Section(0.5 * chord, **section)
        self.held = True
        n = self.lat.n
        self.f = [tack.empty((19 * n,)), tack.empty((19 * n,))]
        self.fields = {name: ManagedArray(n) for name in FIELDS}
        views = {name: a.tack for name, a in self.fields.items()}
        self.quiet, self.report = Quiet(views), Report(views)
        wing_init(self.lat, self.geo, self.flow, self.f[0])
        self.step_count = 0
        self.drag = self.lift = self.moment = 0.0

    def set_reynolds(self, reynolds):
        self.reynolds = float(reynolds)
        self.nu = self.u0 * self.chord / self.reynolds
        self.phys = COLLISIONS[self.collision_name](self.nu)

    def reduced_velocity(self):
        return self.u0 / (self.section.b * self.section.omega_alpha)

    def set_reduced_velocity(self, value):
        self.section.set_frequency(self.u0 / (self.section.b * value))

    def release(self, alpha_rate=None):
        """Free the wing, nudged with a pitch rate (default: a 2 degree swing at omega_alpha)."""
        self.held = False
        rate = np.radians(2.0) * self.section.omega_alpha if alpha_rate is None else alpha_rate
        self.section.v[1] += rate
        self._sync_state()

    def _sync_state(self):
        s = self.section
        self._state.numpy[:] = (s.q[0], s.q[1], s.v[0], s.v[1])

    def step(self, report=False):
        self._load.numpy[:] = 0.0
        wing_step(self.lat, self.phys, self.geo, self.flow, self.report if report else self.quiet,
                  self.f[0], self.f[1])
        self.f.reverse()
        tack.sync()
        fx, fy, mz = (float(v) / self.nz for v in self._load.numpy)
        self.drag, self.lift, self.moment = fx, fy, -mz
        if not self.held:
            self.section.step(self.lift, self.moment)
            self._sync_state()
        self.step_count += 1

    def coefficients(self):
        q = 0.5 * self.u0 * self.u0 * self.chord
        return self.drag / q, self.lift / q, self.moment / (q * self.chord)

    def motion(self):
        s = self.section
        return s.q[0] / s.b, np.degrees(s.q[1])
