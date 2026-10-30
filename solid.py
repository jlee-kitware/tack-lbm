"""Heat conduction in the heat sink with PyMFEM, coupled to the LBM through voxel maps.

Lattice units throughout: the fluid has (rho c_p) = 1 and diffusivity alpha, so a
solid conductivity ratio kappa gives k_s = kappa * alpha, and the solid's heat
capacity is a free parameter that sets how fast it warms up (demo time scale).

The mesh covers exactly the voxels the LBM treats as heat-sink solid. With
halfway bounce-back the wall sits half a cell from the last solid node, so a
voxel row i0..i1-1 becomes the interval [i0 - 0.5, i1 - 0.5].
"""

import bisect
import os

import numpy as np

import mfem.ser as mfem


def _segments(lo, hi, size):
    """Split [lo, hi] into pieces no longer than ``size``."""
    n = max(1, int(np.ceil((hi - lo) / size - 1e-9)))
    return list(np.linspace(lo, hi, n + 1))


class HeatSinkSolid:
    def __init__(self, spec, order=3, element_size=4.0, device=None):
        """``spec`` is the HeatSink layout in lattice cells (see ``spec_of``); ``device``
        defaults to $MFEM_DEVICE, else "cpu"."""
        self.spec = spec
        self.device = mfem.Device(device or os.environ.get("MFEM_DEVICE", "cpu"))
        self._build_mesh(element_size)
        self.fec = mfem.H1_FECollection(order, 3)
        self.fes = mfem.FiniteElementSpace(self.mesh, self.fec)
        self.T = mfem.GridFunction(self.fes)
        self.T.Assign(0.0)
        self.ndofs = self.fes.GetVSize()

    # -- geometry -----------------------------------------------------------
    def _build_mesh(self, h):
        s = self.spec
        xs = _segments(s["x0"] - 0.5, s["x1"] - 0.5, h)
        ys_base = _segments(0.5, s["y_base"] - 0.5, h / 2)
        ys_fin = _segments(s["y_base"] - 0.5, s["y_top"] - 0.5, h)
        zs, fin_of = [s["z0"] - 0.5], []
        z = s["z0"]
        while z < s["z1"]:
            fin_end = min(z + s["thickness"], s["z1"])
            gap_end = min(z + s["pitch"], s["z1"])
            for a, b, is_fin in ((z, fin_end, True), (fin_end, gap_end, False)):
                if b > a:
                    pts = _segments(a - 0.5, b - 0.5, h / 2)
                    zs.extend(pts[1:])
                    fin_of.extend([is_fin] * (len(pts) - 1))
            z += s["pitch"]
        self.lines = (np.array(xs), np.array(ys_base + ys_fin[1:]), np.array(zs))
        ny_base = len(ys_base) - 1
        lx, ly, lz = self.lines
        vid = {}
        verts = []

        def vertex(i, j, k):
            key = (i, j, k)
            if key not in vid:
                vid[key] = len(verts)
                verts.append((lx[i], ly[j], lz[k]))
            return vid[key]

        hexes = []
        for k in range(len(lz) - 1):
            for j in range(len(ly) - 1):
                if j >= ny_base and not fin_of[k]:
                    continue
                for i in range(len(lx) - 1):
                    hexes.append([vertex(i, j, k), vertex(i + 1, j, k), vertex(i + 1, j + 1, k),
                                  vertex(i, j + 1, k), vertex(i, j, k + 1), vertex(i + 1, j, k + 1),
                                  vertex(i + 1, j + 1, k + 1), vertex(i, j + 1, k + 1)])
        mesh = mfem.Mesh(3, len(verts), len(hexes), 0)
        for v in verts:
            mesh.AddVertex(list(map(float, v)))
        for hx in hexes:
            mesh.AddHex(hx, 1)
        mesh.FinalizeTopology()
        mesh.Finalize(True, True)
        # boundary attribute 1: base bottom on the duct floor (adiabatic); 2: wetted surface
        y_floor = 0.5
        for b in range(mesh.GetNBE()):
            v = np.array([mesh.GetVertexArray(i) for i in mesh.GetBdrElementVertices(b)])
            mesh.GetBdrElement(b).SetAttribute(1 if np.allclose(v[:, 1], y_floor) else 2)
        mesh.SetAttributes()
        self.mesh = mesh
        self._hex_index = {}
        e = 0
        for k in range(len(lz) - 1):
            for j in range(len(ly) - 1):
                if j >= ny_base and not fin_of[k]:
                    continue
                for i in range(len(lx) - 1):
                    self._hex_index[(i, j, k)] = e
                    e += 1

    # -- lattice coupling -----------------------------------------------------
    def voxel_map(self, voxels):
        """Sparse matrix W with T_voxel = W @ T_dofs for voxel centres (integer lattice nodes)."""
        from scipy.sparse import csr_matrix

        lx, ly, lz = self.lines
        rows, cols, vals = [], [], []
        ip = mfem.IntegrationPoint()
        for r, (x, y, z) in enumerate(voxels):
            i = min(bisect.bisect_right(lx, x) - 1, len(lx) - 2)
            j = min(bisect.bisect_right(ly, y) - 1, len(ly) - 2)
            k = min(bisect.bisect_right(lz, z) - 1, len(lz) - 2)
            # A point on a face shared with empty space (a fin's side) belongs to the cell below.
            for di, dj, dk in ((0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1), (-1, -1, 0),
                               (-1, 0, -1), (0, -1, -1), (-1, -1, -1)):
                if (i + di, j + dj, k + dk) in self._hex_index:
                    i, j, k = i + di, j + dj, k + dk
                    break
            e = self._hex_index[(i, j, k)]
            ip.Set3((x - lx[i]) / (lx[i + 1] - lx[i]), (y - ly[j]) / (ly[j + 1] - ly[j]),
                    (z - lz[k]) / (lz[k + 1] - lz[k]))
            fe = self.fes.GetFE(e)
            shape = mfem.Vector(fe.GetDof())
            fe.CalcShape(ip, shape)
            dofs = self.fes.GetElementDofs(e)
            rows.extend([r] * len(dofs))
            cols.extend(dofs)
            vals.extend(shape.GetDataArray())
        return csr_matrix((vals, (rows, cols)), shape=(len(voxels), self.ndofs))

    # -- physics --------------------------------------------------------------
    def setup(self, kappa, alpha, capacity, dt, power, chip, solver="direct", robin=None):
        """Implicit Euler for C dT/dt = div(k grad T) + Q, with the wall flux as a load.

        ``solver="direct"`` assembles the (constant) system once and factors it with SciPy,
        so a step is two triangular solves; ``"pcg"`` uses partial assembly and CG, which
        runs on the GPU when MFEM is built with CUDA.
        """
        self.k = kappa * alpha
        self.capacity, self.dt = capacity, dt
        # robin = (W_surface, h per surface voxel): heat leaves as h (T_voxel - mean adjacent T_fluid)
        self.robin = robin
        self.chip = chip
        self.power = power
        self.solver = solver
        if solver == "direct":
            self._setup_direct()
            return
        self.form = mfem.BilinearForm(self.fes)
        self.form.SetAssemblyLevel(mfem.AssemblyLevel_PARTIAL)
        self._c_dt = mfem.ConstantCoefficient(capacity / dt)
        self._k = mfem.ConstantCoefficient(self.k)
        self.form.AddDomainIntegrator(mfem.MassIntegrator(self._c_dt))
        self.form.AddDomainIntegrator(mfem.DiffusionIntegrator(self._k))
        self.form.Assemble()
        self.mass = mfem.BilinearForm(self.fes)
        self.mass.SetAssemblyLevel(mfem.AssemblyLevel_PARTIAL)
        self.mass.AddDomainIntegrator(mfem.MassIntegrator(self._c_dt))
        self.mass.Assemble()
        self.source = self._source_vector(power)
        self.ess = mfem.intArray()
        self.A = mfem.OperatorPtr()
        self.form.FormSystemMatrix(self.ess, self.A)
        self.prec = mfem.OperatorJacobiSmoother(self.form, self.ess)
        self.cg = mfem.CGSolver()
        self.cg.SetRelTol(1e-8)
        self.cg.SetMaxIter(500)
        self.cg.SetPrintLevel(0)
        self.cg.SetPreconditioner(self.prec)
        self.cg.SetOperator(self.A.Ptr())
        self.cg.iterative_mode = True
        self.rhs = mfem.Vector(self.ndofs)
        self.flux_load = mfem.Vector(self.ndofs)
        self.flux_load.Assign(0.0)

    def _assembled(self, *integrators):
        from scipy.sparse import csr_matrix

        form = mfem.BilinearForm(self.fes)
        for integ in integrators:
            form.AddDomainIntegrator(integ)
        form.Assemble()
        form.Finalize()
        a = form.SpMat()
        n = a.Height()
        return csr_matrix((a.GetDataArray().copy(), a.GetJArray().copy(), a.GetIArray().copy()), shape=(n, n))

    def _setup_direct(self):
        from scipy.sparse.linalg import splu

        self._c_dt = mfem.ConstantCoefficient(self.capacity / self.dt)
        self._k = mfem.ConstantCoefficient(self.k)
        from scipy.sparse import diags

        self.M = self._assembled(mfem.MassIntegrator(self._c_dt))
        K = self._assembled(mfem.DiffusionIntegrator(self._k))
        A = self.M + K
        if self.robin is not None:
            Ws, h = self.robin
            self.WsT_h = (Ws.T @ diags(h)).tocsr()
            A = A + self.WsT_h @ Ws
        self.lu = splu(A.tocsc())
        self.source = self._source_vector(self.power)

    def _source_vector(self, power):
        x0, x1, z0, z1, y1 = self.chip
        volume = (x1 - x0) * (z1 - z0) * (y1 - 0.5)

        class Chip(mfem.PyCoefficient):
            def EvalValue(self, p):
                inside = x0 <= p[0] <= x1 and z0 <= p[2] <= z1 and p[1] <= y1
                return power / volume if inside else 0.0

        self._chip = Chip()
        lf = mfem.LinearForm(self.fes)
        lf.AddDomainIntegrator(mfem.DomainLFIntegrator(self._chip))
        lf.Assemble()
        # The chip's edges cut through elements, where quadrature misjudges the step; the
        # basis sums to one, so scaling the load by power / sum makes it release exactly ``power``.
        total = lf.GetDataArray().sum()
        self.source_scale = power / total if total > 0 else 0.0
        lf *= self.source_scale
        return lf

    def set_power(self, power):
        self.power = power
        self.source = self._source_vector(power)

    def step(self, wall_flux_load=None, fluid_temperature=None):
        """Advance one coupling interval; ``wall_flux_load`` is heat leaving through the wall, per DOF."""
        if self.solver == "direct":
            T = self.T.GetDataArray()
            rhs = self.M @ T + self.source.GetDataArray()
            if wall_flux_load is not None:
                rhs -= wall_flux_load
            if self.robin is not None and fluid_temperature is not None:
                rhs += self.WsT_h @ fluid_temperature
            T[:] = self.lu.solve(rhs)
            return 0
        self.mass.Mult(self.T, self.rhs)
        self.rhs += self.source
        if wall_flux_load is not None:
            self.flux_load.Assign(np.ascontiguousarray(wall_flux_load, dtype=np.float64))
            self.rhs -= self.flux_load
        self.cg.Mult(self.rhs, self.T)
        return self.cg.GetNumIterations()

    def dofs(self):
        return self.T.GetDataArray()


def spec_of(solver):
    """The heat sink's layout from a Solver using the ``heatsink`` obstacle."""
    g = solver.geo
    return {"x0": int(g.x0), "x1": int(g.x1), "z0": int(g.z0), "z1": int(g.z1),
            "y_base": int(g.y_base), "y_top": int(g.y_top),
            "thickness": int(g.fin_thickness), "pitch": int(g.fin_pitch)}


if __name__ == "__main__":
    import time

    spec = {"x0": 64, "x1": 160, "z0": 16, "z1": 112, "y_base": 7, "y_top": 50,
            "thickness": 4, "pitch": 12}
    import sys

    order = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    t0 = time.perf_counter()
    solid = HeatSinkSolid(spec, order=order)
    print(f"mesh: {solid.mesh.GetNE()} hexes, {solid.ndofs} dofs, built in {time.perf_counter() - t0:.1f} s")
    attrs = [solid.mesh.GetBdrAttribute(b) for b in range(solid.mesh.GetNBE())]
    print("boundary faces: floor", attrs.count(1), "wetted", attrs.count(2))
    # map check: a linear field is reproduced exactly at voxel centres
    vox = [(x, y, z) for x in range(64, 160, 7) for y in range(1, 50, 5) for z in range(16, 112, 3)
           if (y < 7 or (z - 16) % 12 < 4)]
    t0 = time.perf_counter()
    W = solid.voxel_map(vox)
    nodes = mfem.GridFunction(solid.fes)

    class Linear(mfem.PyCoefficient):
        def EvalValue(self, p):
            return 0.3 * p[0] - 0.2 * p[1] + 0.1 * p[2]

    nodes.ProjectCoefficient(Linear())
    exact = np.array([0.3 * x - 0.2 * y + 0.1 * z for x, y, z in vox])
    err = np.abs(W @ nodes.GetDataArray() - exact).max()
    print(f"voxel map: {len(vox)} voxels in {time.perf_counter() - t0:.1f} s, linear-field error {err:.2e}")
    # steady-ish heating with no cooling: temperature must rise by power*dt/C per step overall
    t0 = time.perf_counter()
    solid.setup(kappa=200.0, alpha=0.005, capacity=1.0, dt=100.0, power=50.0,
                chip=(96.0, 128.0, 48.0, 80.0, 3.5), solver=sys.argv[2] if len(sys.argv) > 2 else "direct")
    print(f"setup ({solid.solver}) {time.perf_counter() - t0:.1f} s")
    for n in range(8):
        t0 = time.perf_counter()
        it = solid.step()
        T = solid.dofs()
        print(f"step {n}: {time.perf_counter() - t0:.2f} s, CG iterations {it}, T range ({T.min():.3f}, {T.max():.3f})")
