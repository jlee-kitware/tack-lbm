"""SI units for the heat sink case: the lattice's units, written as scaled SI units for Tack.

The solvers stay in lattice units. A lattice unit is a scaled SI unit ("4.17e-04*m"), so
``tack.dataset.units`` converts published values and tack-expr can combine them with SI
quantities. The scales follow from three choices:

  length       the sink's length in metres over its length in cells
  time         the air's kinematic viscosity: dt = nu_lattice dx^2 / nu_air
  temperature  the heater: ``power_lattice`` lattice units of heat per step are ``power_w`` watts

Steering the Reynolds number changes the lattice viscosity and so dt; ``update()`` recomputes it.
"""

from tack.dataset import units

AIR = {"nu": 1.57e-5, "k": 0.0263, "rho": 1.184, "cp": 1007.0}  # 25 degC, 1 atm
ALUMINUM = {"k": 237.0}


class HeatSinkUnits:
    def __init__(self, fluid, sink_length=0.040, power_w=15.0, power_lattice=50.0, ambient_c=25.0):
        g = fluid.geo
        self.fluid = fluid
        self.dx = sink_length / (g.x1 - g.x0)
        self.ambient_c = ambient_c
        self.rho_cp = AIR["rho"] * AIR["cp"]
        self.update()
        self.theta = power_w * self.dt / (power_lattice * self.rho_cp * self.dx ** 3)

    def update(self):
        self.dt = self.fluid.nu * self.dx ** 2 / AIR["nu"]

    def lattice(self, quantity):
        dx, dt = self.dx, self.dt
        scale = {"length": (dx, "m"), "velocity": (dx / dt, "m/s"), "time": (dt, "s"),
                 "pressure": (AIR["rho"] * (dx / dt) ** 2, "Pa"), "temperature": (self.theta, "K"),
                 "power": (self.rho_cp * self.theta * dx ** 3 / dt, "W")}[quantity]
        return f"{scale[0]!r}*{scale[1]}"

    def factor(self, quantity, unit):
        return units.conversion(self.lattice(quantity), unit)[0]

    def celsius(self, rise):
        """A lattice temperature rise above the inflow air, in degrees Celsius."""
        return self.ambient_c + self.factor("temperature", "K") * rise

    def inflow_speed(self):
        return self.factor("velocity", "m/s") * self.fluid.u0

    @staticmethod
    def conductivity_ratio():
        return ALUMINUM["k"] / AIR["k"]
