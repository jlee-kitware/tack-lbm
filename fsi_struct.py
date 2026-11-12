"""Cantilever NACA 0012 wing as a 3D linear elastic finite element model (MFEM).

The wing is clamped at its root (z = 0) and meshed with hexahedra mapped from a box in
(chord, thickness, span) parameters. Its stiffness and mass give the vibration modes, and
the wing's dynamics run in those modal coordinates. Each mode is reduced to a rigid
section motion per span station, plunge h(z) and pitch alpha(z) about the reference
axis, which is what the fluid sees. Lattice units throughout: fluid density 1.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import mfem.ser as mfem

ROOT = 1


def naca_half(xc):
    xc = np.clip(xc, 0.0, 1.0)
    return 0.6 * (0.2969 * np.sqrt(xc) - 0.126 * xc - 0.3516 * xc ** 2 + 0.2843 * xc ** 3 - 0.1036 * xc ** 4)


def _csr(a):
    m = a.SpMat()
    return sp.csr_matrix((m.GetDataArray().copy(), m.GetJArray().copy(), m.GetIArray().copy()),
                         shape=(m.Height(), m.Width()))


class WingFEM:
    """``regions`` maps (chord_from, chord_to) to (stiffness factor, density factor) over the
    base material E0, rho0; the default is a spar near the quarter chord and trailing-edge ballast."""

    def __init__(self, chord=32.0, span=96.0, ea=0.4, order=2, n_chord=24, n_thick=2, n_span=24,
                 rho0=190.0, E0=1.0e3, poisson=0.3, regions=None, caps=None, edge=0.01, modes=6):
        """``caps`` (chord_from, chord_to, stiffness factor) makes the outer half of the thickness in that
        chord band a spar's caps: stiff in bending, adding little torsional stiffness."""
        self.chord, self.span, self.ea = chord, span, ea
        if regions is None:
            regions = {(0.0, 0.05): (1.0, 1.0), (0.05, 0.3): (12.0, 1.0),
                       (0.3, 0.7): (1.0, 1.0), (0.7, 1.0): (1.0, 3.0)}
        self.regions = regions
        mesh = mfem.Mesh.MakeCartesian3D(n_chord, n_thick, n_span, mfem.Element.HEXAHEDRON, 1.0, 2.0, span)
        v = np.array(mesh.GetVertexArray(), float)
        u = edge + (1.0 - 2.0 * edge) * 0.5 * (1.0 - np.cos(np.pi * v[:, 0]))
        t = v[:, 1] - 1.0
        names = list(regions)
        for i in range(mesh.GetNE()):
            ids = mesh.GetElementVertices(i)
            uc, tc = u[ids].mean(), t[ids].mean()
            attr = 1 + next(k for k, (a, b) in enumerate(names) if a <= uc <= b)
            if caps and caps[0] <= uc <= caps[1] and abs(tc) > 0.5:
                attr = len(names) + 1
            mesh.SetAttribute(i, attr)
        mesh.SetAttributes()
        mesh.SetCurvature(order, False, 3, mfem.Ordering.byVDIM)
        nodes = mesh.GetNodes()
        x = nodes.GetDataArray().reshape(-1, 3)
        u = edge + (1.0 - 2.0 * edge) * 0.5 * (1.0 - np.cos(np.pi * x[:, 0]))
        t = x[:, 1] - 1.0
        x[:, 0] = (u - ea) * chord
        x[:, 1] = t * np.maximum(naca_half(u), 0.004) * chord
        self.mesh = mesh
        self.nv = mesh.GetNV()
        self.coords = x.copy()
        self.rest = x[: self.nv].copy()
        self.fes = fes = nodes.FESpace()
        stiff = np.array([regions[k][0] for k in names] + ([caps[2]] if caps else [])) * E0
        dens = np.array([regions[k][1] for k in names] + ([1.0] if caps else [])) * rho0
        lam = mfem.PWConstCoefficient(mfem.Vector(stiff * poisson / ((1 + poisson) * (1 - 2 * poisson))))
        mu = mfem.PWConstCoefficient(mfem.Vector(stiff / (2 * (1 + poisson))))
        rho = mfem.PWConstCoefficient(mfem.Vector(dens))
        k = mfem.BilinearForm(fes)
        k.AddDomainIntegrator(mfem.ElasticityIntegrator(lam, mu))
        k.Assemble()
        k.Finalize()
        m = mfem.BilinearForm(fes)
        m.AddDomainIntegrator(mfem.VectorMassIntegrator(rho))
        m.Assemble()
        m.Finalize()
        K, M = _csr(k), _csr(m)
        self.mass_total = M.sum() / 3.0

        bdr = mfem.intArray([1 if a == ROOT else 0 for a in range(1, mesh.bdr_attributes.Max() + 1)])
        ess = mfem.intArray()
        fes.GetEssentialTrueDofs(bdr, ess)
        fixed = np.zeros(K.shape[0], bool)
        fixed[np.array(ess.ToList(), int)] = True
        free = np.flatnonzero(~fixed)
        Kf, Mf = K[free][:, free].tocsc(), M[free][:, free].tocsc()
        w2, phi = sla.eigsh(Kf, k=modes, M=Mf, sigma=0.0, which="LM")
        order_ = np.argsort(w2)
        self.omega0 = np.sqrt(np.maximum(w2[order_], 0.0))
        self.shapes = np.zeros((modes, K.shape[0]))
        self.shapes[:, free] = phi[:, order_].T
        self._M = M
        self._Kf, self._free = Kf, free
        self._fit_sections()
        self._surface()

    def node_disp(self, i):
        return self.shapes[i].reshape(-1, 3)

    def _fit_sections(self):
        """Rigid section motion of each mode at every node layer along the span."""
        z = self.coords[:, 2]
        layers = np.unique(np.round(z, 6))
        self.layer_z = layers
        nm = len(self.omega0)
        self.h = np.zeros((nm, len(layers)))
        self.alpha = np.zeros((nm, len(layers)))
        for j, zl in enumerate(layers):
            sel = np.abs(z - zl) < 1e-5
            s, t = self.coords[sel, 0], self.coords[sel, 1]
            # uy = h - alpha s, ux = g + alpha t; the chordwise shift g is invisible to the fluid
            n = sel.sum()
            A = np.zeros((2 * n, 3))
            A[:n, 0] = 1.0
            A[:n, 1] = -s
            A[n:, 1] = t
            A[n:, 2] = 1.0
            for i in range(nm):
                d = self.node_disp(i)[sel]
                (self.h[i, j], self.alpha[i, j], _), *_ = np.linalg.lstsq(
                    A, np.concatenate([d[:, 1], d[:, 0]]), rcond=None)

    def _surface(self):
        quads = []
        for i in range(self.mesh.GetNBE()):
            if self.mesh.GetBdrAttribute(i) != ROOT:
                quads.append(self.mesh.GetBdrElementVertices(i))
        self.quads = np.array(quads, np.int64)

    def sections_at(self, zs):
        """Mode shapes as (h, alpha) arrays sampled at span positions ``zs`` (zero past the tip)."""
        zs = np.asarray(zs, float)
        inside = zs <= self.span
        h = np.array([np.interp(zs, self.layer_z, hi) * inside for hi in self.h])
        a = np.array([np.interp(zs, self.layer_z, ai) * inside for ai in self.alpha])
        return h, a

    def deformed(self, q):
        return self.rest + np.tensordot(q, self.shapes[:, : 3 * self.nv].reshape(-1, self.nv, 3), 1)

    def elastic_axis(self):
        """Chord fraction where a vertical tip load bends the wing without twisting its tip."""
        tip = sel = np.abs(self.coords[:, 2] - self.span) < 1e-5
        s_tip, t_tip = self.coords[sel, 0], self.coords[sel, 1]
        solve = sla.splu(self._Kf)
        twists = []
        stations = (-0.2 * self.chord, 0.3 * self.chord)
        for s0 in stations:
            band = tip & (np.abs(self.coords[:, 0] - s0) <= np.sort(np.abs(self.coords[tip, 0] - s0))[3])
            f = np.zeros(3 * len(self.coords))
            f[3 * np.flatnonzero(band) + 1] = 1.0 / band.sum()
            u = np.zeros_like(f)
            u[self._free] = solve.solve(f[self._free])
            d = u.reshape(-1, 3)[sel]
            n = sel.sum()
            A = np.zeros((2 * n, 3))
            A[:n, 0], A[:n, 1], A[n:, 1], A[n:, 2] = 1.0, -s_tip, t_tip, 1.0
            (_, alpha, _), *_ = np.linalg.lstsq(A, np.concatenate([d[:, 1], d[:, 0]]), rcond=None)
            twists.append(alpha)
        s_zero = stations[0] - twists[0] * (stations[1] - stations[0]) / (twists[1] - twists[0])
        return self.ea + s_zero / self.chord

    def section_properties(self):
        """Mass per span, CG and gyration about the reference axis, from the mass matrix by layer."""
        lumped = np.asarray(self._M.sum(axis=1)).ravel()[0::3]
        m = lumped.sum() / self.span
        xcg = (lumped * self.coords[:, 0]).sum() / lumped.sum()
        r2 = (lumped * (self.coords[:, 0] ** 2 + self.coords[:, 1] ** 2)).sum() / lumped.sum()
        return m, xcg, r2


class Modal:
    """Modal dynamics q'' + 2 zeta w q' + w^2 q (1 + H a^2) = Q, mass normalized modes. The wing
    stiffens in torsion: ``a`` is the tip twist of the whole motion, in radians."""

    def __init__(self, fem, omega, damping=0.0, hardening=40.0):
        self.fem = fem
        self.omega = np.array(omega, float)
        self.damping, self.hardening = damping, hardening
        self.tip_twist = np.asarray(fem.alpha[:, -1], float)
        self.q = np.zeros(len(omega))
        self.v = np.zeros(len(omega))

    def step(self, force, dt=1.0):
        a = float(self.tip_twist @ self.q)
        spring = self.omega ** 2 * self.q * (1.0 + self.hardening * a * a)
        self.v += dt * (force - spring - 2.0 * self.damping * self.omega * self.v)
        self.q += dt * self.v
