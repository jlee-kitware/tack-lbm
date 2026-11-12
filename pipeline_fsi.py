"""Catalyst pipeline for the elastic wing case: 3D vortex structures computed in situ.

The grid stays in the solver unless a client asks for it. Each Live update publishes:

``vortex_cores``  isosurfaces of the Liutex magnitude (CORE_LEVEL inflow speeds per chord), with ``spin``:
                  the spanwise share of the rotation, +1 counterclockwise about +z (the shed
                  vortices of a nose-down swing), -1 clockwise, near 0 for the streamwise tip vortex
``vortices``      the cores and the deformed finite element ``wing`` (spin 0) in one dataset

The lattice holds half of the wing, clamped at the symmetry plane: a semi-span model on a wall,
as flutter models are tested in wind tunnels. MIRROR shows the whole wing instead.
"""
import numpy as np

from paraview import simple
from paraview.catalyst import Options
from paraview.simple import CreateSteerableParameters
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.util import numpy_support
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkCompositeDataSet, vtkImageData, vtkPolyData
from vtkmodules.vtkFiltersCore import (vtkAppendPolyData, vtkClipPolyData, vtkFlyingEdges3D,
                                       vtkImplicitPolyDataDistance, vtkPolyDataNormals)
from vtkmodules.vtkFiltersFlowPaths import vtkLiutex
from vtkmodules.vtkFiltersGeneral import vtkReflectionFilter
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter

options = Options()
options.EnableCatalystLive = 1
options.CatalystLiveURL = "localhost:22222"
options.CatalystLiveWaitForClient = 1
options.CatalystLiveTrigger = "TimeStep"
options.CatalystLiveTrigger.UseStartTimeStep = 0
options.CatalystLiveTrigger.Frequency = 1

# Core surfaces at a Liutex magnitude of CORE_LEVEL inflow speeds per chord, trimmed WALL_BAND
# chords from the wing, where the boundary layer's shear reads as rotation on this grid.
CORE_LEVEL = 0.2
WALL_BAND = 0.15
MIRROR = False

channels = [simple.TrivialProducer(registrationName=name) for name in ("grid", "wing", "monitors")]
grid, wing = channels[0], channels[1]
cores = simple.TrivialProducer(registrationName="vortex_cores")
combined = simple.TrivialProducer(registrationName="vortices")
steering = CreateSteerableParameters("LBMSteering")

_liutex = vtkLiutex()
_liutex.SetInputArrayToProcess(0, 0, 0, 0, "velocity")
_iso = vtkFlyingEdges3D()
_iso.ComputeNormalsOn()
_iso.ComputeScalarsOff()
_iso.InterpolateAttributesOn()
_iso.SetInputArrayToProcess(0, 0, 0, 0, "magnitude")
_wing_faces, _wing_normals = vtkGeometryFilter(), vtkPolyDataNormals()
_wing_normals.SetInputConnection(_wing_faces.GetOutputPort())
_wing_normals.SetFeatureAngle(70.0)
_distance, _trim = vtkImplicitPolyDataDistance(), vtkClipPolyData()
_trim.SetClipFunction(_distance)
_append = vtkAppendPolyData()
_mirror, _mirror_faces = vtkReflectionFilter(), vtkGeometryFilter()
_mirror.SetPlaneToZ()
_mirror.CopyInputOn()
_mirror_faces.SetInputConnection(_mirror.GetOutputPort())


def _first(data):
    if isinstance(data, vtkCompositeDataSet):
        it = data.NewIterator()
        it.InitTraversal()
        return it.GetCurrentDataObject() if not it.IsDoneWithTraversal() else None
    return data


def _plain(name, values):
    array = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
    array.SetName(name)
    return array


def _plain_poly(data):
    """A copy built from plain arrays. Accelerated filters can hand back accelerator arrays that
    other threaded filters must not read once their owner re-executes."""
    out = vtkPolyData()
    if not data.GetNumberOfPoints():
        return out
    wrapped = dsa.WrapDataObject(data)
    points = vtkPoints()
    points.SetData(_plain("Points", np.array(wrapped.Points)))
    out.SetPoints(points)
    polys = vtkCellArray()
    polys.DeepCopy(data.GetPolys())
    out.SetPolys(polys)
    pd = data.GetPointData()
    for i in range(pd.GetNumberOfArrays()):
        name = pd.GetArrayName(i)
        if name:
            out.GetPointData().AddArray(_plain(name, np.array(wrapped.PointData[name])))
    if out.GetPointData().GetArray("Normals") is not None:
        out.GetPointData().SetActiveNormals("Normals")
    return out


def _with_array(data, name, values):
    out = vtkPolyData()
    out.ShallowCopy(data)
    out.GetPointData().AddArray(_plain(name, values))
    return out


def _publish(producer, data, time):
    producer.GetClientSideObject().SetOutput(data)
    producer.SMProxy.MarkModified(producer.SMProxy)
    producer.UpdatePipeline(time)


def _core_surface(image, wing_faces):
    # Liutex works on a plain copy of the velocity: the solver's channel arrays can arrive as
    # accelerator arrays, which threaded VTK filters must not read.
    # A deep copy into an interleaved array reads any layout the channel arrives in (with async
    # Catalyst it is a structure-of-arrays copy, which numpy cannot wrap).
    interleaved = vtkFloatArray()
    interleaved.DeepCopy(image.GetPointData().GetArray("velocity"))
    velocity = numpy_support.vtk_to_numpy(interleaved).reshape(-1, 3).copy()
    source = vtkImageData()
    source.CopyStructure(image)
    source.GetPointData().AddArray(_plain("velocity", velocity))
    _liutex.SetInputData(source)
    _liutex.Modified()
    _liutex.Update()
    pd = dsa.WrapDataObject(_liutex.GetOutput()).PointData
    liutex = np.array(pd["Liutex"])
    mag = np.linalg.norm(liutex, axis=1)
    nx, ny, nz = image.GetDimensions()
    inflow = float(velocity[:, 0].reshape(nz, ny, nx)[:, :, 0].mean())
    if inflow <= 0.0 or mag.max() < CORE_LEVEL * inflow:
        return vtkPolyData()
    # A plain image with plain arrays: vtkLiutex's outputs can be read-only accelerator arrays.
    plain = vtkImageData()
    plain.CopyStructure(image)
    plain.GetPointData().AddArray(_plain("magnitude", mag))
    plain.GetPointData().AddArray(_plain("spin", np.clip(liutex[:, 2] / np.maximum(mag, 1e-30), -1, 1)))
    _iso.SetInputData(plain)
    _iso.SetValue(0, CORE_LEVEL * inflow)
    _iso.Update()
    out = _plain_poly(_iso.GetOutput())
    if wing_faces is not None and out.GetNumberOfPoints():
        _distance.SetInput(wing_faces)
        _trim.SetInputData(out)
        _trim.SetValue(WALL_BAND)
        _trim.Update()
        out = _plain_poly(_trim.GetOutput())
    return out


def _mirrored(data, plane_z):
    if not MIRROR or not data.GetNumberOfPoints():
        return data
    _mirror.SetCenter(plane_z)
    _mirror.SetPlane(_mirror.USE_Z)
    _mirror.SetInputData(data)
    _mirror.Modified()
    _mirror_faces.Update()
    return _plain_poly(_mirror_faces.GetOutput())


def catalyst_execute(info):
    for producer in channels:
        producer.UpdatePipeline(info.time)
    image = _first(grid.GetClientSideObject().GetOutputDataObject(0))
    if image is None or image.GetPointData().GetArray("velocity") is None:
        return
    wing_data = _first(wing.GetClientSideObject().GetOutputDataObject(0))
    faces = None
    if wing_data is not None:
        _wing_faces.SetInputData(wing_data)
        _wing_faces.Modified()
        _wing_normals.Update()
        faces = _plain_poly(_wing_normals.GetOutput())
    # The wing's root lies on the symmetry plane, half a lattice cell below the first layer.
    plane_z = faces.GetBounds()[4] if faces is not None else image.GetOrigin()[2]
    surface = _core_surface(image, faces)
    out_cores = _mirrored(surface, plane_z)
    out_cores.GetPointData().SetActiveScalars("spin")
    _publish(cores, out_cores, info.time)

    parts = [surface] if surface.GetNumberOfPoints() else []
    if faces is not None:
        parts.append(_with_array(faces, "spin", np.zeros(faces.GetNumberOfPoints())))
    _append.RemoveAllInputs()
    for part in parts:
        _append.AddInputData(part)
    _append.Update()
    both = vtkPolyData()
    both.ShallowCopy(_append.GetOutput())
    out_all = _mirrored(both, plane_z)
    out_all.GetPointData().SetActiveScalars("spin")
    _publish(combined, out_all, info.time)
