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

# Steerable parameters a Live client can edit; catalyst_results returns them
# under catalyst/steerable.
steering = CreateSteerableParameters("LBMSteering")
# The derived-field expressions, as text, come back under catalyst/derived.
derived = CreateSteerableParameters("LBMDerived", "DerivedParameters", "derived")


def catalyst_execute(info):
    # Catalyst 2 Live does not update channel producers itself, so a Live
    # extract would otherwise see empty data.
    grid.UpdatePipeline(info.time)
    monitors.UpdatePipeline(info.time)
    if info.cycle % 1000 == 0:
        print(f"[catalyst] cycle {info.cycle} time {info.time:.3f}", flush=True)
