"""Run the conjugate heat transfer case with Catalyst Live.

    python run_cht.py [--re 300] [--power 50] [--kappa 200]

Channels: ``grid`` (velocity, pressure, speed, temperature), ``solid`` (the MFEM
temperature in the heat sink, order-3 elements subdivided by Tack into linear
hexahedra) and ``monitors`` (residuals, forces, heater power, heat into the air,
peak solid temperature). Steerable: Reynolds number, collision model, heater power.
"""

import argparse
import os
import signal
import time

import numpy as np

import tack
import tack.dataset as td
from tack.interop.mfem import mfem_to_dataset

from cht import ConjugateHeat
from run_lbm import MONITOR_NAMES, CatalystAdaptor
from steering import COLLISION_CODES, COLLISION_NAMES, write_proxy_xml

CHT_MONITORS = MONITOR_NAMES + ("power", "heat_flux", "solid_t_max", "solid_t_mean")


class SolidChannel:
    """The solid's high-order temperature as linear hexahedra, through Tack's MFEM interop."""

    def __init__(self, cht, level=3):
        self.cht = cht
        self.level = level
        self.points = None

    def _linear(self):
        from tack.interop.vtk import dataset_to_vtk
        from vtkmodules.util.numpy_support import vtk_to_numpy

        solid = self.cht.solid
        data = mfem_to_dataset(solid.mesh, {"temperature": solid.T})
        grid = dataset_to_vtk(td.subdivide(data, self.level))
        points = vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float64)
        cells = vtk_to_numpy(grid.GetCells().GetConnectivityArray()).astype(np.int64)
        temperature = vtk_to_numpy(grid.GetPointData().GetArray("temperature")).astype(np.float64)
        return points, cells, temperature

    def __call__(self, channels):
        if self.points is None:
            # The mesh is fixed: subdivide once with Tack, then map MFEM DOFs to the
            # sub-cell points so later calls are one sparse product.
            self.points, self.cells, _ = self._linear()
            self.map = self.cht.solid.voxel_map(self.points)
        self.temperature = self.map @ self.cht.solid.dofs()
        # Lattice units to the grid channel's units (cylinder diameters / characteristic length).
        scale = 1.0 / self.cht.fluid.diameter
        ch = channels["solid"]
        ch["type"] = "mesh"
        mesh = ch["data"]
        mesh["coordsets/coords/type"] = "explicit"
        for axis, column in zip("xyz", self.points.T):
            mesh[f"coordsets/coords/values/{axis}"].set(np.ascontiguousarray(column * scale))
        mesh["topologies/mesh/type"] = "unstructured"
        mesh["topologies/mesh/coordset"] = "coords"
        mesh["topologies/mesh/elements/shape"] = "hex"
        mesh["topologies/mesh/elements/connectivity"].set(self.cells)
        mesh["fields/temperature/association"] = "vertex"
        mesh["fields/temperature/topology"] = "mesh"
        mesh["fields/temperature/values"].set(self.temperature)


def main():
    pv_build = os.environ.get("PV_BUILD", os.path.expanduser("~/gitsync/paraview_build_cuda"))
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", default=os.path.join(here, "pipeline_cht.py"))
    parser.add_argument("--collision", default="les", choices=["les", "bgk", "trt"])
    parser.add_argument("--re", type=float, default=300.0)
    parser.add_argument("--power", type=float, default=50.0)
    parser.add_argument("--kappa", type=float, default=200.0, help="solid to air conductivity ratio")
    parser.add_argument("--capacity", type=float, default=0.05, help="solid heat capacity (sets warm-up speed)")
    parser.add_argument("--size", type=int, nargs=3, default=[320, 96, 128], metavar=("NX", "NY", "NZ"))
    parser.add_argument("--steps", type=int, default=0, help="0 runs until interrupted")
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--catalyst-lib", default=os.path.join(pv_build, "lib", "catalyst"))
    parser.add_argument("--steering-xml", default=os.path.join(here, "steering.xml"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    cht = ConjugateHeat(tuple(args.size), reynolds=args.re, collision=args.collision, power=args.power,
                        kappa=args.kappa, capacity=args.capacity, interval=args.interval, stride=args.stride)
    fluid = cht.fluid
    print(f"[cht] {len(cht.voxels)} sink voxels, {cht.solid.ndofs} solid dofs, setup "
          f"{time.perf_counter() - t0:.1f} s", flush=True)
    xml = write_proxy_xml(args.steering_xml, args.re, args.collision, power=args.power)
    adaptor = CatalystAdaptor(os.path.abspath(args.script), fluid, args.catalyst_lib, xml,
                              monitor_names=CHT_MONITORS, extra_channels=SolidChannel(cht))
    adaptor.steered["power"] = args.power

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    t_report = time.perf_counter()
    calls = 0
    while running and (args.steps == 0 or fluid.step_count < args.steps):
        monitors = cht.advance()
        monitors["reynolds"] = fluid.reynolds
        monitors["collision"] = float(COLLISION_CODES[fluid.collision_name])
        step = fluid.step_count
        adaptor.execute(step, step * fluid.u0 / fluid.diameter, monitors)
        power = adaptor.steered.get("power", cht.power)
        adaptor.steer()
        if adaptor.steered.get("power", power) != cht.power:
            cht.set_power(adaptor.steered["power"])
            print(f"[steer] power -> {cht.power:g}", flush=True)
        calls += 1
        if calls % 10 == 0:
            now = time.perf_counter()
            print(f"[cht] step {step}  {10 * args.interval / (now - t_report):.0f} steps/s  power {cht.power:g}  "
                  f"heat into air {monitors['heat_flux']:.2f}  solid T max {monitors['solid_t_max']:.3f}",
                  flush=True)
            t_report = now
    adaptor.finalize()


if __name__ == "__main__":
    main()
