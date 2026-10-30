"""Branchless D3Q19 lattice Boltzmann solver written in Tack.

The physics is assembled at runtime from device classes. Tack compiles one
specialized ``lbm_step`` for each combination of collision model, obstacle
and diagnostics variant, so the run configuration never shows up as a
branch inside the kernel.
"""

import ctypes

import numpy as np

import tack

CX = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0]
CY = [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1]
CZ = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1]
W = [1 / 3] + [1 / 18] * 6 + [1 / 36] * 12

# D3Q7 for temperature: rest plus the six faces, c_s^2 = 1/4.
CX7 = [0, 1, -1, 0, 0, 0, 0]
CY7 = [0, 0, 0, 1, -1, 0, 0]
CZ7 = [0, 0, 0, 0, 0, 1, -1]
W7 = [1 / 4] + [1 / 8] * 6
OPP7 = [0, 2, 1, 4, 3, 6, 5]


def _opposite():
    return [next(p for p in range(19) if (CX[p], CY[p], CZ[p]) == (-CX[q], -CY[q], -CZ[q]))
            for q in range(19)]


@tack.device_class
class D3Q19:
    Q = 19

    def __init__(self, nx, ny, nz):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.n = nx * ny * nz
        self.cx = tack.array(np.array(CX, np.int32))
        self.cy = tack.array(np.array(CY, np.int32))
        self.cz = tack.array(np.array(CZ, np.int32))
        self.w = tack.array(np.array(W, np.float32))
        self.opp = tack.array(np.array(_opposite(), np.int32))

    @tack.func
    def feq(self, q, rho, ux, uy, uz):
        cu = 3.0 * (float(self.cx[q]) * ux + float(self.cy[q]) * uy + float(self.cz[q]) * uz)
        return self.w[q] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy + uz * uz))


# Collision models. Each writes post-collision populations into ``post``.

@tack.device_class
class BGK:
    def __init__(self, nu):
        self.omega = 1.0 / (3.0 * nu + 0.5)

    @tack.func
    def collide(self, lat, f, post, rho, ux, uy, uz):
        for q in range(lat.Q):
            post[q] = f[q] - self.omega * (f[q] - lat.feq(q, rho, ux, uy, uz))
        return 1.0 / self.omega


@tack.func
def smagorinsky_tau(lat, f, rho, ux, uy, uz, tau0, cs):
    """Relaxation time with a Smagorinsky eddy viscosity from the local non-equilibrium stress."""
    pxx = 0.0
    pyy = 0.0
    pzz = 0.0
    pxy = 0.0
    pxz = 0.0
    pyz = 0.0
    for q in range(lat.Q):
        neq = f[q] - lat.feq(q, rho, ux, uy, uz)
        ex = float(lat.cx[q])
        ey = float(lat.cy[q])
        ez = float(lat.cz[q])
        pxx += ex * ex * neq
        pyy += ey * ey * neq
        pzz += ez * ez * neq
        pxy += ex * ey * neq
        pxz += ex * ez * neq
        pyz += ey * ez * neq
    pi = sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz))
    return 0.5 * (tau0 + sqrt(tau0 * tau0 + 25.456 * cs * cs * pi / rho))


@tack.device_class
class TRT:
    """Two-relaxation-time collision with a Smagorinsky subgrid viscosity.

    Plain TRT puts the odd moments' relaxation rate near zero as the viscosity
    approaches its lower limit and goes unstable on this case beyond Re ~100.
    The subgrid viscosity keeps the even rate away from 2, and the magic
    parameter 3/16 places the bounce-back wall halfway between nodes for any
    viscosity.
    """
    magic = 0.1875
    cs = 0.16

    def __init__(self, nu):
        self.tau0 = 3.0 * nu + 0.5

    @tack.func
    def collide(self, lat, f, post, rho, ux, uy, uz):
        tau = smagorinsky_tau(lat, f, rho, ux, uy, uz, self.tau0, self.cs)
        omega_p = 1.0 / tau
        omega_m = 1.0 / (self.magic / (tau - 0.5) + 0.5)
        for q in range(lat.Q):
            o = lat.opp[q]
            eq = lat.feq(q, rho, ux, uy, uz)
            eo = lat.feq(o, rho, ux, uy, uz)
            post[q] = (f[q] - omega_p * (0.5 * (f[q] + f[o]) - 0.5 * (eq + eo))
                       - omega_m * (0.5 * (f[q] - f[o]) - 0.5 * (eq - eo)))
        return tau


@tack.device_class
class SmagorinskyLES:
    cs = 0.16

    def __init__(self, nu):
        self.tau0 = 3.0 * nu + 0.5

    @tack.func
    def collide(self, lat, f, post, rho, ux, uy, uz):
        tau = smagorinsky_tau(lat, f, rho, ux, uy, uz, self.tau0, self.cs)
        omega = 1.0 / tau
        for q in range(lat.Q):
            post[q] = f[q] - omega * (f[q] - lat.feq(q, rho, ux, uy, uz))
        return tau


# Obstacles as inlined signed-distance tests; solid() is 1.0 inside, 0.0 outside.

@tack.device_class
class Cylinder:
    def __init__(self, x, y, r):
        self.x, self.y, self.r = x, y, r

    @tack.func
    def solid(self, x, y, z):
        dx = float(x) - self.x
        dy = float(y) - self.y
        return 1.0 if dx * dx + dy * dy < self.r * self.r else 0.0

    @tack.func
    def sink(self, x, y, z):
        return self.solid(x, y, z)


@tack.device_class
class Sphere:
    def __init__(self, x, y, z, r):
        self.x, self.y, self.z, self.r = x, y, z, r

    @tack.func
    def solid(self, x, y, z):
        dx = float(x) - self.x
        dy = float(y) - self.y
        dz = float(z) - self.z
        return 1.0 if dx * dx + dy * dy + dz * dz < self.r * self.r else 0.0

    @tack.func
    def sink(self, x, y, z):
        return self.solid(x, y, z)


@tack.device_class
class Box:
    def __init__(self, x, y, z, hx, hy, hz):
        self.x, self.y, self.z = x, y, z
        self.hx, self.hy, self.hz = hx, hy, hz

    @tack.func
    def solid(self, x, y, z):
        d = max(abs(float(x) - self.x) - self.hx, abs(float(y) - self.y) - self.hy,
                abs(float(z) - self.z) - self.hz)
        return 1.0 if d < 0.0 else 0.0

    @tack.func
    def sink(self, x, y, z):
        return self.solid(x, y, z)


@tack.device_class
class HeatSink:
    """A finned heat sink on the floor of a duct.

    The base plate and fins (parallel to the flow, along x) are the sink; the
    duct's floor and ceiling (y = 0 and y = ny - 1) are adiabatic walls. All
    sizes are in lattice cells.
    """

    def __init__(self, ny, x0, length, z0, width, base, fin_height, fin_thickness, fin_pitch):
        self.ny = ny
        self.x0, self.x1 = float(x0), float(x0 + length)
        self.z0, self.z1 = float(z0), float(z0 + width)
        self.y_base = 1.0 + base
        self.y_top = 1.0 + base + fin_height
        self.fin_thickness = fin_thickness
        self.fin_pitch = fin_pitch

    @tack.func
    def sink(self, x, y, z):
        fx = float(x)
        fy = float(y)
        fz = float(z)
        inside = (1.0 if fx >= self.x0 else 0.0) * (1.0 if fx < self.x1 else 0.0) \
            * (1.0 if fz >= self.z0 else 0.0) * (1.0 if fz < self.z1 else 0.0) \
            * (1.0 if fy >= 1.0 else 0.0)
        base = 1.0 if fy < self.y_base else 0.0
        fin = (1.0 if fy < self.y_top else 0.0) \
            * (1.0 if (z - int(self.z0)) % self.fin_pitch < self.fin_thickness else 0.0)
        return inside * max(base, fin)

    @tack.func
    def solid(self, x, y, z):
        wall = max(1.0 if y <= 0 else 0.0, 1.0 if y >= self.ny - 1 else 0.0)
        return max(wall, self.sink(x, y, z))


# Thermal variants. ``enabled`` is a class constant, so runs without heat
# compile without the temperature distribution at all.

@tack.device_class
class NoHeat:
    enabled = 0
    Q = 7
    pr_t = 0.85

    def __init__(self, arrays):
        self.cx = arrays["cx"]
        self.cy = arrays["cy"]
        self.cz = arrays["cz"]
        self.w = arrays["w"]
        self.opp = arrays["opp"]
        self.t_solid = arrays["t_solid"]
        self.q_solid = arrays["q_solid"]
        self.temp = arrays["temp"]
        self.alpha = 0.0
        self.nu = 0.0

    @tack.func
    def geq(self, q, temp, ux, uy, uz):
        cu = float(self.cx[q]) * ux + float(self.cy[q]) * uy + float(self.cz[q]) * uz
        return self.w[q] * temp * (1.0 + 4.0 * cu)

    @tack.func
    def diffusivity(self, tau):
        """Molecular plus subgrid diffusivity, with turbulent Prandtl number pr_t."""
        return self.alpha + max(0.0, (tau - 0.5) / 3.0 - self.nu) / self.pr_t


@tack.device_class
class Heat(NoHeat):
    enabled = 1
    pr_t = 0.85

    def __init__(self, arrays, alpha, nu):
        super().__init__(arrays)
        self.alpha = alpha
        self.nu = nu


# Diagnostics variants. The flag is a class constant, so the step that skips
# moments, residuals and forces compiles without that code.

@tack.device_class
class Quiet:
    enabled = 0

    def __init__(self, fields):
        self.ux, self.uy, self.uz = fields["ux"], fields["uy"], fields["uz"]
        self.p, self.vmag = fields["p"], fields["vmag"]
        self.r_rho, self.r_ux, self.r_uy, self.r_uz, self.r_ke = (
            fields["r_rho"], fields["r_ux"], fields["r_uy"], fields["r_uz"], fields["r_ke"])
        self.fx, self.fy, self.fz = fields["fx"], fields["fy"], fields["fz"]


@tack.device_class
class Report(Quiet):
    enabled = 1


@tack.device_class
class Flow:
    def __init__(self, u0, sponge_start, sponge_strength):
        self.u0 = u0
        self.sponge_start = sponge_start
        self.sponge_strength = sponge_strength


@tack.kernel
def lbm_init(lat, geo, flow, f, seed_amp):
    for idx in range(lat.n):
        x = idx % lat.nx
        y = (idx // lat.nx) % lat.ny
        z = idx // (lat.nx * lat.ny)
        fluid = 1.0 - geo.solid(x, y, z)
        ux = flow.u0 * fluid
        uy = seed_amp * flow.u0 * fluid * sin(6.2831853 * float(x) / float(lat.nx)) \
            * cos(6.2831853 * float(z) / float(lat.nz))
        for q in range(lat.Q):
            f[q * lat.n + idx] = lat.feq(q, 1.0, ux, uy, 0.0)


@tack.kernel
def lbm_step(lat, phys, geo, flow, diag, f_in, f_out, heat, g_in, g_out):
    for idx in range(lat.n):
        x = idx % lat.nx
        y = (idx // lat.nx) % lat.ny
        z = idx // (lat.nx * lat.ny)
        here = geo.solid(x, y, z)
        fluid = 1.0 - here
        f = tack.local_array(tack.f32, lat.Q)
        post = tack.local_array(tack.f32, lat.Q)
        mx = 0.0
        my = 0.0
        mz = 0.0
        for q in range(lat.Q):
            o = lat.opp[q]
            xs = (x - lat.cx[q] + lat.nx) % lat.nx
            ys = (y - lat.cy[q] + lat.ny) % lat.ny
            zs = (z - lat.cz[q] + lat.nz) % lat.nz
            wall = geo.solid(xs, ys, zs)
            pulled = f_in[q * lat.n + (zs * lat.ny + ys) * lat.nx + xs]
            bounced = f_in[o * lat.n + idx]
            f[q] = pulled + wall * (bounced - pulled)
            hit = 2.0 * wall * fluid * bounced
            mx += hit * float(lat.cx[o])
            my += hit * float(lat.cy[o])
            mz += hit * float(lat.cz[o])
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
        tau = phys.collide(lat, f, post, rho, ux, uy, uz)
        # The sponge moves density and velocity toward the inflow state and damps the
        # non-equilibrium part, so the result stays consistent with an equilibrium.
        t = max(0.0, (float(x) - flow.sponge_start) / (float(lat.nx) - flow.sponge_start))
        sigma = max(here, flow.sponge_strength * t * t)
        rho_s = rho + sigma * (1.0 - rho)
        ux_s = ux + sigma * (flow.u0 * fluid - ux)
        uy_s = (1.0 - sigma) * uy
        uz_s = (1.0 - sigma) * uz
        for q in range(lat.Q):
            neq = post[q] - lat.feq(q, rho, ux, uy, uz)
            f_out[q * lat.n + idx] = lat.feq(q, rho_s, ux_s, uy_s, uz_s) + (1.0 - sigma) * neq
        if heat.enabled == 1:
            # Temperature: D3Q7 advection-diffusion. Sink walls hold the solid's
            # temperature (anti-bounce-back); duct walls are adiabatic (bounce-back).
            g = tack.local_array(tack.f32, heat.Q)
            temp = 0.0
            for q in range(heat.Q):
                o = heat.opp[q]
                xs = (x - heat.cx[q] + lat.nx) % lat.nx
                ys = (y - heat.cy[q] + lat.ny) % lat.ny
                zs = (z - heat.cz[q] + lat.nz) % lat.nz
                nidx = (zs * lat.ny + ys) * lat.nx + xs
                wall = geo.solid(xs, ys, zs)
                hot = geo.sink(xs, ys, zs)
                pulled = g_in[q * lat.n + nidx]
                bounced = g_in[o * lat.n + idx]
                at_wall = bounced * (1.0 - 2.0 * hot) + hot * 2.0 * heat.w[q] * heat.t_solid[nidx]
                g[q] = pulled + wall * (at_wall - pulled)
                temp += g[q]
            alpha = heat.diffusivity(tau)
            omega_g = 1.0 / (4.0 * alpha + 0.5)
            temp_s = (1.0 - sigma) * temp
            for q in range(heat.Q):
                eq = heat.geq(q, temp, ux, uy, uz)
                post_g = g[q] - omega_g * (g[q] - eq)
                g_out[q * lat.n + idx] = heat.geq(q, temp_s, ux_s, uy_s, uz_s) + (1.0 - sigma) * (post_g - eq)
            if diag.enabled == 1:
                heat.temp[idx] = fluid * temp + here * heat.t_solid[idx]
                # Each wetted sink face gets this fluid cell's temperature (summed per solid voxel);
                # the solid takes the exchange h (T_s - T_f) implicitly.
                for q in range(1, heat.Q):
                    xs = (x - heat.cx[q] + lat.nx) % lat.nx
                    ys = (y - heat.cy[q] + lat.ny) % lat.ny
                    zs = (z - heat.cz[q] + lat.nz) % lat.nz
                    nidx = (zs * lat.ny + ys) * lat.nx + xs
                    hot = geo.sink(xs, ys, zs)
                    tack.atomic_add(heat.q_solid, nidx, fluid * hot * temp)
        if diag.enabled == 1:
            p = (rho - 1.0) / 3.0
            ke = 0.5 * rho * (ux * ux + uy * uy + uz * uz)
            ox = diag.ux[idx]
            oy = diag.uy[idx]
            oz = diag.uz[idx]
            op = diag.p[idx]
            ok = 0.5 * (ox * ox + oy * oy + oz * oz)
            diag.r_rho[idx] = (3.0 * (p - op)) * (3.0 * (p - op))
            diag.r_ux[idx] = (ux - ox) * (ux - ox)
            diag.r_uy[idx] = (uy - oy) * (uy - oy)
            diag.r_uz[idx] = (uz - oz) * (uz - oz)
            diag.r_ke[idx] = (ke - ok) * (ke - ok)
            diag.ux[idx] = ux
            diag.uy[idx] = uy
            diag.uz[idx] = uz
            diag.p[idx] = p
            diag.vmag[idx] = sqrt(ux * ux + uy * uy + uz * uz)
            diag.fx[idx] = mx
            diag.fy[idx] = my
            diag.fz[idx] = mz


@tack.kernel
def coarsen(lat, s, src, dst, mx, my, mz):
    for idx in range(mx * my * mz):
        i = idx % mx
        j = (idx // mx) % my
        k = idx // (mx * my)
        dst[idx] = src[((k * s) * lat.ny + j * s) * lat.nx + i * s]


class ManagedArray:
    """A CUDA managed allocation seen both as a Tack array and as a NumPy array."""

    def __init__(self, n):
        from cuda.bindings import driver as cu

        self._cu = cu
        err, self.ptr = cu.cuMemAllocManaged(n * 4, cu.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL.value)
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemAllocManaged failed: {err}")
        address = int(self.ptr)
        self.numpy = np.ctypeslib.as_array((ctypes.c_float * n).from_address(address))
        self.numpy[:] = 0.0
        self.tack = tack.array_from_ptr(address, (n,), tack.f32, writable=True)

    def free(self):
        self._cu.cuMemFree(self.ptr)


COLLISIONS = {"bgk": BGK, "trt": TRT, "les": SmagorinskyLES}
FIELDS = ("ux", "uy", "uz", "p", "vmag", "r_rho", "r_ux", "r_uy", "r_uz", "r_ke", "fx", "fy", "fz")


class Solver:
    OUTPUT = ("ux", "uy", "uz", "p", "vmag")

    def __init__(self, nx=384, ny=128, nz=64, collision="les", obstacle="cylinder",
                 reynolds=300.0, u0=0.08, diameter=24.0, device="cuda", stride=1,
                 heat=False, prandtl=0.7):
        tack.set_default_device(device)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.lat = D3Q19(nx, ny, nz)
        self.u0, self.diameter = u0, diameter
        self.step_count = 0
        self.flow = Flow(u0, 0.75 * nx, 1.0)
        self.set_obstacle(obstacle)
        self.set_physics(collision, reynolds)
        n = self.lat.n
        self.f = [tack.empty((19 * n,)), tack.empty((19 * n,))]
        self.fields = {name: ManagedArray(n) for name in FIELDS}
        views = {name: a.tack for name, a in self.fields.items()}
        self.quiet = Quiet(views)
        self.report = Report(views)
        self.heat_enabled = heat
        self.prandtl = prandtl
        self._setup_heat(n)
        self.stride = stride
        self.out_dims = tuple((d + stride - 1) // stride for d in (nx, ny, nz))
        outputs = self.OUTPUT + (("temp",) if heat else ())
        if stride > 1:
            m = self.out_dims[0] * self.out_dims[1] * self.out_dims[2]
            self.output = {name: ManagedArray(m) for name in outputs}
        else:
            self.output = {name: self.heat_fields.get(name) or self.fields[name] for name in outputs}
        lbm_init(self.lat, self.geo, self.flow, self.f[0], 0.05)
        self.step(report=True)
        self.step_count = 0

    def _setup_heat(self, n):
        small = 1 if not self.heat_enabled else n
        self.heat_fields = {name: ManagedArray(small) for name in ("t_solid", "q_solid", "temp")}
        arrays = {
            "cx": tack.array(np.array(CX7, np.int32)), "cy": tack.array(np.array(CY7, np.int32)),
            "cz": tack.array(np.array(CZ7, np.int32)), "w": tack.array(np.array(W7, np.float32)),
            "opp": tack.array(np.array(OPP7, np.int32)),
            **{name: a.tack for name, a in self.heat_fields.items()},
        }
        self.g = [tack.zeros((7 * small,)), tack.zeros((7 * small,))]
        if self.heat_enabled:
            self.heat = Heat(arrays, self.nu / self.prandtl, self.nu)
        else:
            self.heat = NoHeat(arrays)
        self._heat_arrays = arrays

    def set_wall_temperature(self, value):
        """Fill every solid voxel's temperature (until the solid solver provides one)."""
        self.heat_fields["t_solid"].numpy[:] = value

    def update_output(self):
        if self.stride > 1:
            mx, my, mz = self.out_dims
            for name in self.output:
                src = self.heat_fields[name] if name == "temp" else self.fields[name]
                coarsen(self.lat, self.stride, src.tack, self.output[name].tack, mx, my, mz)

    def set_physics(self, collision, reynolds):
        self.collision_name, self.reynolds = collision, float(reynolds)
        self.nu = self.u0 * self.diameter / self.reynolds
        self.phys = COLLISIONS[collision](self.nu)
        if getattr(self, "heat_enabled", False):
            self.heat = Heat(self._heat_arrays, self.nu / self.prandtl, self.nu)

    def set_obstacle(self, kind):
        nx, ny, nz, r = self.nx, self.ny, self.nz, self.diameter / 2
        x0, y0 = 0.22 * nx, 0.5 * ny + 1.7
        self.obstacle_name = kind
        self.geo = {"cylinder": lambda: Cylinder(x0, y0, r),
                    "sphere": lambda: Sphere(x0, y0, 0.5 * nz, r),
                    "box": lambda: Box(x0, y0, 0.5 * nz, r, r, 0.3 * nz),
                    "heatsink": lambda: HeatSink(ny, int(0.2 * nx), int(0.3 * nx), int(0.125 * nz),
                                                 int(0.75 * nz), 6, int(0.45 * ny), 4, 12)}[kind]()

    def step(self, report=False):
        diag = self.report if report else self.quiet
        if report and self.heat_enabled:
            self.heat_fields["q_solid"].tack.fill(0.0)
        lbm_step(self.lat, self.phys, self.geo, self.flow, diag, self.f[0], self.f[1],
                 self.heat, self.g[0], self.g[1])
        self.f.reverse()
        self.g.reverse()
        self.step_count += 1

    def monitors(self, interval):
        self.update_output()
        tack.sync()
        n = float(self.lat.n)
        flds = self.fields
        total = {key: float(flds[key].numpy.sum(dtype=np.float64)) for key in flds}
        res = {name: float(np.sqrt(total[key] / n)) / max(interval, 1)
               for name, key in (("continuity", "r_rho"), ("momentum_x", "r_ux"),
                                 ("momentum_y", "r_uy"), ("momentum_z", "r_uz"), ("energy", "r_ke"))}
        scale = 0.5 * self.u0 * self.u0 * self.diameter * self.nz
        res["drag"] = total["fx"] / scale
        res["lift"] = total["fy"] / scale
        return res
