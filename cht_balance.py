"""Energy balance of the coupled case: at steady state the heat into the air equals the heater power."""
import sys

from cht import ConjugateHeat

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
c = ConjugateHeat()
print(f"source scale {c.solid.source_scale:.3f}, source total {c.solid.source.GetDataArray().sum():.2f}")
for k in range(steps // c.interval):
    m = c.advance()
    if (k + 1) % 40 == 0:
        print(f"step {c.fluid.step_count}: power {m['power']:.1f}, heat into air {m['heat_flux']:.2f} "
              f"({100 * m['heat_flux'] / m['power']:.0f}%), solid T max {m['solid_t_max']:.3f}", flush=True)
