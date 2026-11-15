"""A Catalyst channel of streamlines traced by Tack on the solver's output grid.

The paths are computed in situ, on the device that holds the flow: td.streamlines from a
rake of seeds through the velocity, then td.probe for speed (and temperature with the heat
model) along them. The channel is an unstructured mesh of line segments, positions in the
grid channel's units.
"""

import numpy as np

import tack
import tack.dataset as td
from tack import arrays

from derived import _pack


def rake(x, y_range, z_range, ny=6, nz=12):
    """Seeds on a plane of constant x, in lattice units: ``ny`` by ``nz`` points."""
    y = np.linspace(*y_range, ny)
    z = np.linspace(*z_range, nz)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    return np.stack([np.full(yy.size, x), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)


class StreamlineChannel:
    def __init__(self, solver, seeds, scales=None, name="streamlines", steps=800, step=0.5):
        """``seeds`` in lattice units; ``scales`` a function giving the adaptor's SI scales
        (``None`` keeps lattice units); ``step`` an arc length in lattice cells."""
        self.solver, self.name, self.scales = solver, name, scales
        self.steps, self.step = steps, float(step)
        mx, my, mz = solver.out_dims
        h = float(solver.stride)
        self.data = td.uniform_grid((mx, my, mz), spacing=(h, h, h))
        self.n = mx * my * mz
        self.velocity = tack.empty((self.n,), tack.vec3())
        self.data.fields["velocity"] = td.Field(td.H1(self.data), self.velocity)
        self.sampled = ["velocity"]
        if "temp" in solver.output:
            self.data.fields["temperature"] = td.Field(td.H1(self.data), solver.output["temp"].tack)
            self.sampled.append("temperature")
        self.seeds = np.asarray(seeds, np.float32)

    def trace(self):
        """``(points, segments, fields)``: positions and the line segments' point pairs as
        host arrays, and each sampled field's values at the points, in lattice units."""
        out = self.solver.output
        _pack(out["ux"].tack, out["uy"].tack, out["uz"].tack, self.velocity, self.n)
        lines = td.streamlines(self.data, "velocity", self.seeds, self.step, self.steps,
                               step_unit="length")
        sampled = td.probe(self.data, lines, fields=self.sampled)
        points = np.asarray(lines.positions(), np.float64)
        offsets = arrays.to_host(lines.topology.line_offsets).astype(np.int64)
        chain = arrays.to_host(lines.topology.line_points).astype(np.int64)
        firsts = np.ones(len(chain), bool)
        firsts[offsets[1:-1]] = False
        firsts[0] = False
        ends = np.flatnonzero(firsts)
        segments = np.stack([chain[ends - 1], chain[ends]], axis=1).reshape(-1)
        fields = {name: arrays.to_host(sampled.fields[name].values) for name in self.sampled}
        return points, segments, fields

    def __call__(self, channels):
        points, segments, fields = self.trace()
        scales = self.scales() if self.scales else None
        length = scales["length"] if scales else 1.0
        ch = channels[self.name]
        ch["type"] = "mesh"
        mesh = ch["data"]
        mesh["coordsets/coords/type"] = "explicit"
        for axis, column in zip("xyz", points.T):
            mesh[f"coordsets/coords/values/{axis}"].set(np.ascontiguousarray(column * length))
        mesh["topologies/mesh/type"] = "unstructured"
        mesh["topologies/mesh/coordset"] = "coords"
        mesh["topologies/mesh/elements/shape"] = "line"
        mesh["topologies/mesh/elements/connectivity"].set(segments)
        velocity = np.asarray(fields["velocity"], np.float64).reshape(-1, 3)
        factor = scales["velocity"][0] if scales else 1.0
        speed = mesh["fields/speed"]
        speed["association"] = "vertex"
        speed["topology"] = "mesh"
        speed["values"].set(np.linalg.norm(velocity, axis=1) * factor)
        if "temperature" in fields:
            factor, offset = scales["temperature"] if scales else (1.0, 0.0)
            temperature = mesh["fields/temperature"]
            temperature["association"] = "vertex"
            temperature["topology"] = "mesh"
            temperature["values"].set(np.asarray(fields["temperature"], np.float64) * factor + offset)
