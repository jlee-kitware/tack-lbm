"""A Catalyst pipeline that prints the grid channel's arrays: checks the derived fields
arrive without a Live client.

    python run_lbm.py --script pipeline_check.py --steps 300 --interval 100
"""
from paraview import servermanager, simple

grid = simple.TrivialProducer(registrationName="grid")
monitors = simple.TrivialProducer(registrationName="monitors")


def catalyst_execute(info):
    grid.UpdatePipeline(info.time)
    data = servermanager.Fetch(grid)
    if hasattr(data, "GetPartition"):
        data = data.GetPartition(0)
    parts = []
    for kind, attrs in (("point", data.GetPointData()), ("cell", data.GetCellData())):
        for i in range(attrs.GetNumberOfArrays()):
            a = attrs.GetArray(i)
            lo, hi = a.GetRange(-1) if a.GetNumberOfComponents() > 1 else a.GetRange()
            parts.append(f"{kind}:{a.GetName()}[{a.GetNumberOfTuples()}] {lo:.3g}..{hi:.3g}")
    print(f"[check] cycle {info.cycle}: " + "; ".join(parts), flush=True)

