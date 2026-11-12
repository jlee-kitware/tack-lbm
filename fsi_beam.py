"""Cantilever wing as an aeroelastic beam: MFEM finite elements along the span.

The classic flutter model (as in the Goland wing): bending (Timoshenko) and twist (St. Venant
torsion) of a beam on the elastic axis, coupled through the section's mass, whose centre of
gravity sits behind that axis. The section is a NACA 0012, rigid in its own plane, so each
span station moves by plunge and pitch, which is what the fluid sees.

Per unit span, fluid density 1, semichord b: mass m = mu pi b^2, elastic axis at chord fraction
``elastic_axis``, centre of gravity x_alpha semichords behind it, inertia about the elastic axis
m r_alpha2 b^2. The bending and torsional stiffnesses set the uncoupled frequency ratio
omega_h / omega_alpha of the first modes; the overall level is scaled later to a reduced velocity.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import mfem.ser as mfem

from fsi_struct import naca_half


def _csr(form):
    m = form.SpMat()
    return sp.csr_matrix((m.GetDataArray().copy(), m.GetJArray().copy(), m.GetIArray().copy()),
                         shape=(m.Height(), m.Width()))


class WingBeam:
    def __init__(self, chord=32.0, span=64.0, ea=0.4, elastic_axis=0.15, mu=8.0, x_alpha=0.4, r_alpha2=0.5,
                 freq_ratio=0.6, elements=32, order=2, modes=6, sections=33, contour=40):
        self.chord, self.span, self.ea = chord, span, ea
        self.elastic_axis, self.mu, self.x_alpha, self.r_alpha2 = elastic_axis, mu, x_alpha, r_alpha2
        b = 0.5 * chord
        m = mu * np.pi * b * b
        s_alpha = m * x_alpha * b
        i_alpha = m * r_alpha2 * b * b
        gj = 1.0
        # Uncoupled cantilever frequencies: bending 1.875^2 sqrt(EI / m L^4), torsion pi / 2 sqrt(GJ / I L^2).
        ei = m * (gj / i_alpha) * (freq_ratio * np.pi * span / (2.0 * 1.875 ** 2)) ** 2
        kga = 1.0e3 * ei / (span / elements) ** 2

        mesh = mfem.Mesh.MakeCartesian1D(elements, span)
        fes = mfem.FiniteElementSpace(mesh, mfem.H1_FECollection(order, 1))
        n = fes.GetVSize()

        def form(integrator):
            a = mfem.BilinearForm(fes)
            a.AddDomainIntegrator(integrator)
            a.Assemble()
            a.Finalize()
            return _csr(a)

        k1 = form(mfem.DiffusionIntegrator(mfem.ConstantCoefficient(1.0)))
        m1 = form(mfem.MassIntegrator(mfem.ConstantCoefficient(1.0)))
        mixed = mfem.MixedBilinearForm(fes, fes)
        mixed.AddDomainIntegrator(mfem.MixedScalarDerivativeIntegrator(mfem.ConstantCoefficient(1.0)))
        mixed.Assemble()
        mixed.Finalize()
        d = _csr(mixed)  # d[i, j] = int dN_j/dz N_i
        # Unknowns [w, theta, phi]: deflection and bending rotation on the elastic axis, twist.
        z = None
        K = sp.bmat([[kga * k1, -kga * d.T, z],
                     [-kga * d, ei * k1 + kga * m1, z],
                     [z, z, gj * k1]]).tocsr()
        rotary = m * (0.05 * chord) ** 2
        M = sp.bmat([[m * m1, z, -s_alpha * m1],
                     [z, rotary * m1, z],
                     [-s_alpha * m1, z, i_alpha * m1]]).tocsr()
        bdr = mfem.intArray([1] + [0] * (mesh.bdr_attributes.Max() - 1))
        ess = mfem.intArray()
        fes.GetEssentialTrueDofs(bdr, ess)
        root = np.array(ess.ToList(), int)
        fixed = np.zeros(3 * n, bool)
        for blk in range(3):
            fixed[blk * n + root] = True
        free = np.flatnonzero(~fixed)
        w2, phi = sla.eigsh(K[free][:, free].tocsc(), k=modes, M=M[free][:, free].tocsc(), sigma=0.0, which="LM")
        order_ = np.argsort(w2)
        self.omega0 = np.sqrt(np.maximum(w2[order_], 0.0))
        shapes = np.zeros((modes, 3 * n))
        shapes[:, free] = phi[:, order_].T

        # Nodal span positions of the H1 dofs, to sample the shapes.
        coords = mfem.GridFunction(fes)
        coords.ProjectCoefficient(_XCoefficient())
        zdof = coords.GetDataArray().copy()
        order_z = np.argsort(zdof)
        self.layer_z = zdof[order_z]
        w = shapes[:, :n][:, order_z]
        twist = shapes[:, 2 * n:][:, order_z]
        # Plunge of the reference axis (chord fraction ea) from the elastic axis's deflection and the twist.
        s_e = (elastic_axis - ea) * chord
        self.h = w + twist * s_e
        self.alpha = twist

        self._surface(sections, contour)

    def sections_at(self, zs):
        zs = np.asarray(zs, float)
        inside = zs <= self.span
        h = np.array([np.interp(zs, self.layer_z, hi) * inside for hi in self.h])
        a = np.array([np.interp(zs, self.layer_z, ai) * inside for ai in self.alpha])
        return h, a

    def _surface(self, sections, contour):
        beta = np.linspace(0.0, np.pi, contour)
        xc = 0.5 * (1.0 - np.cos(beta))
        half = naca_half(xc)
        ring_s = np.concatenate([xc, xc]) - self.ea
        ring_t = np.concatenate([half, -half])
        self._ring = (ring_s * self.chord, ring_t * self.chord)
        self._stations = np.linspace(0.0, self.span, sections)
        m = contour
        quads = []
        for k in range(sections - 1):
            a, c = k * 2 * m, (k + 1) * 2 * m
            for j in range(m - 1):
                quads.append([a + j, a + j + 1, c + j + 1, c + j])
                quads.append([a + m + j + 1, a + m + j, c + m + j, c + m + j + 1])
            quads.append([a + m - 1, a + 2 * m - 1, c + 2 * m - 1, c + m - 1])
        tip = (sections - 1) * 2 * m
        for j in range(m - 1):
            quads.append([tip + j, tip + j + 1, tip + m + j + 1, tip + m + j])
        # Wound so the normals point out of the wing.
        self.quads = np.array(quads, np.int64)[:, ::-1].copy()
        self.rest = self.deformed(np.zeros(len(self.omega0)))

    def deformed(self, q):
        h_modes, a_modes = self.sections_at(self._stations)
        h, a = q @ h_modes, q @ a_modes
        s, t = self._ring
        pts = []
        for z, hk, ak in zip(self._stations, h, a):
            x = s * np.cos(ak) + t * np.sin(ak)
            y = hk - s * np.sin(ak) + t * np.cos(ak)
            pts.append(np.column_stack([x, y, np.full_like(x, z)]))
        return np.concatenate(pts)


class _XCoefficient(mfem.PyCoefficient):
    def EvalValue(self, x):
        return x[0]
