from paraview import simple
from paraview.catalyst import Options
from paraview.simple import CreateSteerableParameters

options = Options()
options.EnableCatalystLive = 1
options.CatalystLiveURL = "localhost:22222"
options.CatalystLiveTrigger = "TimeStep"
options.CatalystLiveTrigger.UseStartTimeStep = 0
options.CatalystLiveTrigger.Frequency = 300

# Registering the channels makes them extractable from a Live client.
channels = [simple.TrivialProducer(registrationName=name) for name in ("grid", "solid", "monitors")]
steering = CreateSteerableParameters("LBMSteering")


def catalyst_execute(info):
    # Catalyst 2 Live does not update channel producers itself.
    for producer in channels:
        producer.UpdatePipeline(info.time)
    if info.cycle % 1000 == 0:
        print(f"[catalyst] cycle {info.cycle} time {info.time:.3f}", flush=True)
