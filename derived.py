"""Derived fields for the grid channel: field expressions over the solver's output grid.

Each derived field is a VTK calculator expression (tack-expr) evaluated on the uniform
grid the grid channel describes. Pointwise expressions are vertex fields; expressions
with gradients (curl, qcrit, grad, div) are evaluated at cell centers and sent as
element fields. The expressions at each location are compiled into one kernel, so the
fields they read and the gradients they take are computed once per Catalyst call.
Fields: velocity (vector), pressure, speed, and temperature when the heat model is on.
"""

import os
import re
import sys

import numpy as np
import tack
import tack.dataset as td

# tack-expr lives in Tack's tree, packages/tack-expr/src; TACK_EXPR points elsewhere.
sys.path.insert(0, os.environ.get("TACK_EXPR", os.path.expanduser("~/gitsync/tack/packages/tack-expr/src")))
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
    """Compiles the expressions, one kernel per location, and evaluates them on each
    Catalyst call. A Live client can replace the steered ones (steer) while the solver runs."""

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
        for name, (expr, at) in specs.items():
            try:
                fx.compile_expression(expr, self.data, at=at)
            except fx.ExprError as e:
                raise SystemExit(f"--derive {name}: {e}") from None
        self.base = dict(specs)
        self.steered = {}
        self.request = ""
        self.error_column = 0
        self.message = "ok: no steered fields"
        self._group()

    def _group(self):
        """One compiled kernel per output location, for the base and steered fields."""
        specs = {**self.base, **self.steered}
        self.groups = {}
        for at in ("points", "cells"):
            exprs = {name: expr for name, (expr, where) in specs.items() if where == at}
            if exprs:
                self.groups[at] = fx.compile_many(exprs, self.data, at=at)

    @property
    def names(self):
        return [*self.base, *self.steered]

    def steer(self, text):
        """Replace the steered fields with text's NAME=EXPRESSION[@cells] entries, separated
        by ';'. On an error the previous ones stay, and error_column is the 1-based column of
        the error in text (-1 for an error with no column)."""
        if text == self.request:
            return
        self.request = text
        steered, offset = {}, 0
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
                # Field names in any case ("Velocity" reads velocity); same length, so columns hold.
                for field in self.data.fields:
                    expr = re.sub(rf"\b{field}\b", field, expr, flags=re.IGNORECASE)
                expr_start = start + entry.lstrip().index("=") + 1 + (len(expr) - len(expr.lstrip()))
                try:
                    fx.compile_expression(expr.strip(), self.data, at=at)
                except fx.ExprError as e:
                    self.error_column = expr_start + e.pos + 1
                    self.message = f"error in {name} at column {self.error_column} of the text: {e}"
                    print(f"[derived] {name}: {e}", flush=True)
                    return
                steered[name] = (expr.strip(), at)
        except ValueError as e:
            self.error_column = -1
            self.message = f"error: {e}"
            print(f"[derived] {e}", flush=True)
            return
        self.steered, self.error_column = steered, 0
        self.message = f"ok: steered fields {', '.join(steered) or 'none'}"
        self._group()
        print(f"[derived] steered fields: {', '.join(steered) or 'none'}", flush=True)

    def evaluate(self):
        """{name: (values, 'vertex' or 'element')} for the current output arrays."""
        results = {}
        if not self.groups:
            return results
        out = self.solver.output
        _pack(out["ux"].tack, out["uy"].tack, out["uz"].tack, self.velocity, self.n)
        for at, compiled in self.groups.items():
            association = "element" if at == "cells" else "vertex"
            for name, field in compiled(self.data).items():
                results[name] = (field.values.numpy(), association)
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

