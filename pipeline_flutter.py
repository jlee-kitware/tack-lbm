"""Catalyst pipeline for the wing flutter case: vortex geometry computed in situ.

The grid stays in the solver unless a client asks for it. Each Live update publishes small
polydata instead: ``vortex_cores`` (filled regions where the Liutex magnitude is above a
fraction of its peak, with the signed ``liutex_z``: positive counterclockwise) and
``vorticity_lines`` (isolines of the z vorticity), both on the mid-span section, and
``vortices``: those two with the solver's ``wing`` surface in one dataset, so a single display
colored by its ``spin`` shows them all: +1 on a counterclockwise vortex, -1 on a clockwise one,
0 on the wing, and +-0.3 on the lines by the sign of their vorticity.

The flow is two-dimensional, the same in every spanwise slice, so its 3D Liutex isosurfaces are
the section's cores carried straight along the span. ``vortices`` draws them that way: the cores
extruded into solid tubes across the span the wing is drawn over, and the vorticity lines on the
back plane behind them.
"""
import numpy as np

from paraview import simple
from paraview.catalyst import Options
from paraview.simple import CreateSteerableParameters
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkAppendPolyData, vtkContourFilter, vtkPolyDataNormals
from vtkmodules.vtkFiltersModeling import vtkLinearExtrusionFilter
from vtkmodules.vtkFiltersFlowPaths import vtkLiutex
from vtkmodules.vtkFiltersGeometry import vtkImageDataGeometryFilter
from vtkmodules.vtkFiltersGeneral import vtkClipDataSet
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkImagingCore import vtkExtractVOI

options = Options()
options.EnableCatalystLive = 1
options.CatalystLiveURL = "localhost:22222"
# Hold the run at its first step until the assistant connects.
options.CatalystLiveWaitForClient = 1
options.CatalystLiveTrigger = "TimeStep"
options.CatalystLiveTrigger.UseStartTimeStep = 0
# Every Catalyst call is a Live update, so the solver's --interval sets how often the app sees the
# wing: a fluttering wing needs well over ten updates per cycle to read as motion.
options.CatalystLiveTrigger.Frequency = 1

CORE_FRACTION = 0.05
LINE_FRACTIONS = (0.04, 0.1, 0.25)

channels = [simple.TrivialProducer(registrationName=name) for name in ("grid", "wing", "monitors", "derived_status")]
grid = channels[0]
cores = simple.TrivialProducer(registrationName="vortex_cores")
lines = simple.TrivialProducer(registrationName="vorticity_lines")
combined = simple.TrivialProducer(registrationName="vortices")
wing = channels[1]
steering = CreateSteerableParameters("LBMSteering")
derived = CreateSteerableParameters("LBMDerived", "DerivedParameters", "derived")

_voi, _liutex, _layer, _surface = vtkExtractVOI(), vtkLiutex(), vtkExtractVOI(), vtkImageDataGeometryFilter()
_liutex.SetInputConnection(_voi.GetOutputPort())
_liutex.SetInputArrayToProcess(0, 0, 0, 0, "velocity")
_layer.SetInputConnection(_liutex.GetOutputPort())
_surface.SetInputConnection(_layer.GetOutputPort())
_clip, _faces, _contour = vtkClipDataSet(), vtkGeometryFilter(), vtkContourFilter()
_clip.SetInputConnection(_surface.GetOutputPort())
_faces.SetInputConnection(_clip.GetOutputPort())
_contour.SetInputConnection(_surface.GetOutputPort())
_wing_faces, _append = vtkGeometryFilter(), vtkAppendPolyData()
_extrude, _normals = vtkLinearExtrusionFilter(), vtkPolyDataNormals()
_extrude.SetExtrusionTypeToVectorExtrusion()
_extrude.SetVector(0.0, 0.0, 1.0)
_extrude.CappingOn()
_normals.SetInputConnection(_extrude.GetOutputPort())
_normals.SetFeatureAngle(60.0)
_normals.SplittingOn()


def _image(data):
    if isinstance(data, vtkCompositeDataSet):
        it = data.NewIterator()
        it.InitTraversal()
        return it.GetCurrentDataObject() if not it.IsDoneWithTraversal() else None
    return data


def _with_array(data, name, values):
    out = vtkPolyData()
    out.ShallowCopy(data)
    array = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
    array.SetName(name)
    out.GetPointData().AddArray(array)
    return out


def _at_z(data, z):
    """A copy with every point moved to the plane z."""
    out = vtkPolyData()
    out.DeepCopy(data)
    if out.GetNumberOfPoints():
        xyz = dsa.WrapDataObject(out).Points
        xyz[:, 2] = z
        out.GetPoints().Modified()
    return out


def _tubes(cores_2d, z0, z1):
    """The section's cores carried straight from z0 to z1, as closed solids with smooth sides."""
    if not cores_2d.GetNumberOfCells():
        return cores_2d
    # Only plain arrays go on: vtkLiutex's outputs can be read-only accelerator arrays, which the
    # normals filter cannot copy into when it splits points.
    flat = _at_z(cores_2d, z0)
    spin = np.array(dsa.WrapDataObject(flat).PointData["liutex_z"], dtype=np.float32)
    flat.GetPointData().Initialize()
    flat.GetCellData().Initialize()
    flat = _with_array(flat, "spin", np.sign(spin))
    _extrude.SetInputData(flat)
    _extrude.SetScaleFactor(z1 - z0)
    _normals.Update()
    out = vtkPolyData()
    out.ShallowCopy(_normals.GetOutput())
    return out


def _publish(producer, data, time):
    producer.GetClientSideObject().SetOutput(data)
    producer.SMProxy.MarkModified(producer.SMProxy)
    producer.UpdatePipeline(time)


def catalyst_execute(info):
    # Catalyst 2 Live does not update channel producers itself.
    for producer in channels:
        producer.UpdatePipeline(info.time)
    image = _image(grid.GetClientSideObject().GetOutputDataObject(0))
    if image is None or image.GetPointData().GetArray("velocity") is None:
        return
    i0, i1, j0, j1, k0, k1 = image.GetExtent()
    k = (k0 + k1) // 2
    _voi.SetInputData(image)
    # The solver refills the same arrays in place, so the image's modified time does not move.
    _voi.Modified()
    _voi.SetVOI(i0, i1, j0, j1, max(k0, k - 1), min(k1, k + 1))
    _layer.SetVOI(i0, i1, j0, j1, k, k)
    _surface.Update()
    section = _surface.GetOutput()
    pd = dsa.WrapDataObject(section).PointData
    liutex = np.asarray(pd["Liutex"])
    pd_vtk = section.GetPointData()
    signed = numpy_to_vtk(np.ascontiguousarray(liutex[:, 2]), deep=True)
    signed.SetName("liutex_z")
    pd_vtk.AddArray(signed)
    u = np.asarray(pd["velocity"]).reshape(j1 - j0 + 1, i1 - i0 + 1, 3)
    h = image.GetSpacing()
    wz = np.gradient(u[..., 1], h[0], axis=1) - np.gradient(u[..., 0], h[1], axis=0)
    vort = numpy_to_vtk(np.ascontiguousarray(wz.ravel()), deep=True)
    vort.SetName("vorticity_z")
    pd_vtk.AddArray(vort)

    peak = float(np.abs(liutex[:, 2]).max())
    pd_vtk.SetActiveScalars("LiutexMagnitude")
    _clip.SetValue(CORE_FRACTION * peak if peak > 0 else 1e30)
    _faces.Update()
    out_cores = vtkPolyData()
    out_cores.ShallowCopy(_faces.GetOutput())
    out_cores.GetPointData().SetActiveScalars("liutex_z")

    w = float(np.abs(wz).max())
    _contour.SetInputArrayToProcess(0, 0, 0, 0, "vorticity_z")
    levels = [s * f * w for f in LINE_FRACTIONS for s in (-1.0, 1.0)]
    _contour.SetNumberOfContours(len(levels))
    for n, value in enumerate(levels):
        _contour.SetValue(n, value)
    _contour.Update()
    out_lines = vtkPolyData()
    out_lines.ShallowCopy(_contour.GetOutput())

    _publish(cores, out_cores, info.time)
    _publish(lines, out_lines, info.time)

    wing_data = _image(wing.GetClientSideObject().GetOutputDataObject(0))
    z0, z1 = image.GetBounds()[4:6]
    parts = []
    if wing_data is not None:
        _wing_faces.SetInputData(wing_data)
        _wing_faces.Modified()
        _wing_faces.Update()
        wing_faces = _wing_faces.GetOutput()
        z0, z1 = wing_faces.GetBounds()[4:6]
        parts.append(_with_array(wing_faces, "spin", np.zeros(wing_faces.GetNumberOfPoints())))
    parts.append(_tubes(out_cores, z0, z1))
    if out_lines.GetNumberOfPoints():
        sign = np.sign(np.asarray(dsa.WrapDataObject(out_lines).PointData["vorticity_z"]))
        parts.append(_with_array(_at_z(out_lines, z0), "spin", 0.3 * sign))
    _append.RemoveAllInputs()
    for part in parts:
        _append.AddInputData(part)
    _append.Update()
    out_all = vtkPolyData()
    out_all.ShallowCopy(_append.GetOutput())
    out_all.GetPointData().SetActiveScalars("spin")
    _publish(combined, out_all, info.time)
