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

# A solver-side extract: the Q-criterion isosurface computed in situ from the solver's
# q_criterion field, so a Live client receives the surface instead of the whole grid.
_q_points = simple.CellDatatoPointData(registrationName="_q_points", Input=grid)
_q_points.PassCellData = 0
vortices = simple.Contour(registrationName="vortices", Input=_q_points)
vortices.ContourBy = ["POINTS", "q_criterion"]
vortices.Isosurfaces = [0.005]
vortices.ComputeScalars = 1
VORTEX_EVERY = 5


def catalyst_execute(info):
    # Catalyst 2 Live does not update channel producers itself, so a Live
    # extract would otherwise see empty data.
    grid.UpdatePipeline(info.time)
    monitors.UpdatePipeline(info.time)
    derived_status.UpdatePipeline(info.time)
    if info.cycle // 100 % VORTEX_EVERY == 0:
        _q_points.UpdatePipeline(info.time)
        q_max = _q_points.PointData["q_criterion"].GetRange()[1]
        if q_max > 0:
            vortices.Isosurfaces = [0.03 * q_max]
        vortices.UpdatePipeline(info.time)
    if info.cycle % 1000 == 0:
        print(f"[catalyst] cycle {info.cycle} time {info.time:.3f}", flush=True)
