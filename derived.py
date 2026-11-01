"""Derived fields for the grid channel: field expressions over the solver's output grid.

Each derived field is a VTK calculator expression (tack-expr) evaluated on the uniform
grid the grid channel describes. Pointwise expressions are vertex fields; expressions
with gradients (curl, qcrit, grad, div) are evaluated at cell centers and sent as
element fields. Fields: velocity (vector), pressure, speed, and temperature when the
heat model is on.
"""

import os
import sys

import numpy as np
import tack
import tack.dataset as td

sys.path.insert(0, os.environ.get("TACK_EXPR", os.path.expanduser("~/gitsync/tack-expr")))
import tack_expr as fx  # noqa: E402

DEFAULTS = {
    "vorticity": ("mag(curl(velocity))", "cells"),
    "q_criterion": ("qcrit(velocity)", "cells"),
}


def parse(specs):
    """'name=expression' or 'name=expression@cells' (default @points) into {name: (expr, at)}."""
    out = {}
    for spec in specs:
        name, _, expr = spec.partition("=")
        if not name or not expr:
            raise SystemExit(f"--derive takes NAME=EXPRESSION[@cells|@points], not {spec!r}")
        at = "points"
        body, sep, where = expr.rpartition("@")
        if sep and where in ("cells", "points"):
            expr, at = body, where
        out[name.strip()] = (expr.strip(), at)
    return out


@tack.kernel
def _pack(ux, uy, uz, out, n):
    for i in range(n):
        out[i] = tack.vec3(ux[i], uy[i], uz[i])


class DerivedFields:
    """Compiles the expressions once, then evaluates them on each Catalyst call."""

    def __init__(self, solver, specs):
        self.solver = solver
        mx, my, mz = solver.out_dims
        h = solver.stride / solver.diameter
        self.data = td.uniform_grid((mx, my, mz), spacing=(h, h, h))
        self.n = mx * my * mz
        self.velocity = tack.empty((self.n,), tack.vec3())
        self.data.fields["velocity"] = td.Field(td.H1(self.data), self.velocity)
        for name, key in (("pressure", "p"), ("speed", "vmag"), ("temperature", "temp")):
            if key in solver.output:
                self.data.fields[name] = td.Field(td.H1(self.data), solver.output[key].tack)
        self.compiled = {}
        for name, (expr, at) in specs.items():
            try:
                self.compiled[name] = fx.compile_expression(expr, self.data, at=at)
            except fx.ExprError as e:
                raise SystemExit(f"--derive {name}: {e}") from None

    def evaluate(self):
        """{name: (values, 'vertex' or 'element')} for the current output arrays."""
        out = self.solver.output
        _pack(out["ux"].tack, out["uy"].tack, out["uz"].tack, self.velocity, self.n)
        results = {}
        for name, compiled in self.compiled.items():
            values = compiled(self.data).values.numpy()
            results[name] = (values, "element" if compiled.mode == "cells" else "vertex")
        return results

    def add_to(self, mesh):
        """Add the derived fields to a Conduit Blueprint mesh node."""
        for name, (values, association) in self.evaluate().items():
            fld = mesh[f"fields/{name}"]
            fld["association"] = association
            fld["topology"] = "mesh"
            if values.ndim == 2:
                for k, axis in enumerate("xyz"):
                    fld[f"values/{axis}"].set(np.ascontiguousarray(values[:, k]))
            else:
                fld["values"].set(values)

