import os

from paraview import simple
from paraview.catalyst import Options
from paraview.simple import CreateSteerableParameters

options = Options()
options.EnableCatalystLive = 1
options.CatalystLiveURL = "localhost:22222"
# Hold the run at its first step until the assistant connects.
options.CatalystLiveWaitForClient = 1
options.CatalystLiveTrigger = "TimeStep"
options.CatalystLiveTrigger.UseStartTimeStep = 0
options.CatalystLiveTrigger.Frequency = 300

# Registering the channels makes them extractable from a Live client.
grid = simple.TrivialProducer(registrationName="grid")
monitors = simple.TrivialProducer(registrationName="monitors")
derived_status = simple.TrivialProducer(registrationName="derived_status")

# Steerable parameters a Live client can edit; catalyst_results returns them
# under catalyst/steerable.
steering = CreateSteerableParameters("LBMSteering")
# The derived-field expressions, as text, come back under catalyst/derived.
derived = CreateSteerableParameters("LBMDerived", "DerivedParameters", "derived")

# The lean channel: the grid with only speed and the solver-computed cell fields (q_criterion
# and the steered derived fields), so a Live client can contour vortices without the whole grid.
lean = simple.PassArrays(registrationName="lean", Input=grid)
lean.PointDataArrays = ["speed"]
lean.CellDataArrays = ["q_criterion"]
LEAN_SKIP = {"vorticity"}

# A solver-side isosurface of q_criterion, on request (TACK_LBM_VORTICES=1): the smallest
# transfer, but the contour runs on the CPU in the solver loop, so it refreshes less often.
vortices = None
VORTEX_EVERY = 5
if os.environ.get("TACK_LBM_VORTICES") == "1":
    _q_points = simple.CellDatatoPointData(registrationName="_q_points", Input=grid)
    _q_points.PassCellData = 0
    vortices = simple.Contour(registrationName="vortices", Input=_q_points)
    vortices.ContourBy = ["POINTS", "q_criterion"]
    vortices.Isosurfaces = [0.005]
    vortices.ComputeScalars = 1


def catalyst_execute(info):
    # Catalyst 2 Live does not update channel producers itself, so a Live
    # extract would otherwise see empty data.
    grid.UpdatePipeline(info.time)
    monitors.UpdatePipeline(info.time)
    derived_status.UpdatePipeline(info.time)
    cells = [name for name in grid.CellData.keys() if name not in LEAN_SKIP]
    if cells != list(lean.CellDataArrays):
        lean.CellDataArrays = cells
    lean.UpdatePipeline(info.time)
    if vortices is not None and info.cycle // 100 % VORTEX_EVERY == 0:
        _q_points.UpdatePipeline(info.time)
        q_max = _q_points.PointData["q_criterion"].GetRange()[1]
        if q_max > 0:
            vortices.Isosurfaces = [0.03 * q_max]
        vortices.UpdatePipeline(info.time)
    if info.cycle % 1000 == 0:
        print(f"[catalyst] cycle {info.cycle} time {info.time:.3f}", flush=True)
