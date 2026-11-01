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
    """Compiles the expressions once, then evaluates them on each Catalyst call. A Live
    client can replace the steered ones (steer) while the solver runs."""

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
        self.base = {}
        for name, (expr, at) in specs.items():
            try:
                self.base[name] = fx.compile_expression(expr, self.data, at=at)
            except fx.ExprError as e:
                raise SystemExit(f"--derive {name}: {e}") from None
        self.steered = {}
        self.request = ""
        self.error_column = 0

    @property
    def compiled(self):
        return {**self.base, **self.steered}

    def steer(self, text):
        """Replace the steered fields with text's NAME=EXPRESSION[@cells] entries, separated
        by ';'. On an error the previous ones stay, and error_column is the 1-based column of
        the error in text (-1 for an error with no column)."""
        if text == self.request:
            return
        self.request = text
        compiled, offset = {}, 0
        try:
            for entry in text.split(";"):
                start = offset + len(entry) - len(entry.lstrip())
                offset += len(entry) + 1
                if not entry.strip():
                    continue
                name, sep, rest = entry.partition("=")
                name = name.strip()
                if not sep or not name.isidentifier():
                    raise ValueError(f"'{entry.strip()}' is not NAME=EXPRESSION")
                if name in self.data.fields or name in self.base:
                    raise ValueError(f"'{name}' is already a grid field")
                body, at_sign, where = rest.rpartition("@")
                expr, at = (body, where.strip()) if at_sign and where.strip() in ("cells", "points") else (rest, "points")
                expr_start = start + entry.lstrip().index("=") + 1 + (len(expr) - len(expr.lstrip()))
                try:
                    compiled[name] = fx.compile_expression(expr.strip(), self.data, at=at)
                except fx.ExprError as e:
                    self.error_column = expr_start + e.pos + 1
                    print(f"[derived] {name}: {e}", flush=True)
                    return
        except ValueError as e:
            self.error_column = -1
            print(f"[derived] {e}", flush=True)
            return
        self.steered, self.error_column = compiled, 0
        names = ", ".join(compiled) or "none"
        print(f"[derived] steered fields: {names}", flush=True)

    def evaluate(self):
        """{name: (values, 'vertex' or 'element')} for the current output arrays."""
        results = {}
        if not self.compiled:
            return results
        out = self.solver.output
        _pack(out["ux"].tack, out["uy"].tack, out["uz"].tack, self.velocity, self.n)
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

