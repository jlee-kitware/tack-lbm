"""Catalyst steering definition for the LBM solver.

The proxy XML is generated so its defaults match the run's starting values;
the Live client loads the same file to know the proxy's properties.
"""

COLLISION_CODES = {"bgk": 0, "trt": 1, "les": 2}
COLLISION_NAMES = {v: k for k, v in COLLISION_CODES.items()}

PROXY_XML = """<ServerManagerConfiguration>
  <ProxyGroup name="sources">
    <SourceProxy name="LBMSteering" class="vtkSteeringDataGenerator" label="LBM steering">
      <IntVectorProperty name="PartitionType" command="SetPartitionType"
                         number_of_elements="1" default_values="0" panel_visibility="never" />
      <IntVectorProperty name="FieldAssociation" command="SetFieldAssociation"
                         number_of_elements="1" default_values="2" panel_visibility="never" />
      <DoubleVectorProperty name="Reynolds" label="Reynolds number" command="SetTuple1Double"
                            use_index="1" clean_command="Clear" initial_string="reynolds"
                            number_of_elements_per_command="1" repeat_command="1"
                            number_of_elements="1" default_values="{reynolds}">
        <DoubleRangeDomain name="range" min="20" max="3000" />
        <Documentation>Reynolds number based on cylinder diameter and inflow speed.</Documentation>
      </DoubleVectorProperty>
      <IntVectorProperty name="Collision" label="Collision model" command="SetTuple1Int"
                         use_index="1" clean_command="Clear" initial_string="collision"
                         number_of_elements_per_command="1" repeat_command="1"
                         number_of_elements="1" default_values="{collision}">
        <EnumerationDomain name="enum">
          <Entry value="0" text="BGK" />
          <Entry value="1" text="TRT with LES" />
          <Entry value="2" text="Smagorinsky LES" />
        </EnumerationDomain>
        <Documentation>Collision operator; each choice is a separately JIT-compiled Tack kernel.</Documentation>
      </IntVectorProperty>
{extra}
    </SourceProxy>
{proxies}
  </ProxyGroup>
</ServerManagerConfiguration>
"""

# Text lives in a steering proxy of its own: vtkSteeringDataGenerator needs every array of a
# proxy to have the same number of tuples, and the text's length changes.
DERIVED_XML = """    <SourceProxy name="LBMDerived" class="vtkSteeringDataGenerator" label="LBM derived fields">
      <IntVectorProperty name="PartitionType" command="SetPartitionType"
                         number_of_elements="1" default_values="0" panel_visibility="never" />
      <IntVectorProperty name="FieldAssociation" command="SetFieldAssociation"
                         number_of_elements="1" default_values="2" panel_visibility="never" />
      <IntVectorProperty name="DerivedFields" label="Derived fields" command="SetTuple1Int"
                         use_index="1" clean_command="Clear" initial_string="derived"
                         number_of_elements_per_command="1" repeat_command="1"
                         number_of_elements="1" default_values="0">
        <Hints><Expression encoding="utf-8" /></Hints>
        <Documentation>Extra fields on the grid channel, computed from VTK calculator expressions over
        velocity, pressure and speed: NAME=EXPRESSION, separated by ';', with @cells after an
        expression that takes derivatives (curl, qcrit, grad, div). Example:
        helicity=dot(velocity, curl(velocity))@cells; cp=2 * pressure / 0.0064.
        Empty text removes them. The derived_error monitor is 0 when the last text compiled,
        otherwise the column of the error in the text.</Documentation>
      </IntVectorProperty>
    </SourceProxy>"""

POWER_XML = """      <DoubleVectorProperty name="Power" label="Heater power" command="SetTuple1Double"
                            use_index="1" clean_command="Clear" initial_string="power"
                            number_of_elements_per_command="1" repeat_command="1"
                            number_of_elements="1" default_values="{power}">
        <DoubleRangeDomain name="range" min="0" max="500" />
        <Documentation>Heat released by the chip under the heat sink's base, in lattice units.</Documentation>
      </DoubleVectorProperty>"""


def write_proxy_xml(path, reynolds, collision, power=None, derived=False):
    extra = POWER_XML.format(power=power) if power is not None else ""
    proxies = DERIVED_XML if derived else ""
    with open(path, "w") as f:
        f.write(PROXY_XML.format(reynolds=reynolds, collision=COLLISION_CODES[collision], extra=extra,
                                 proxies=proxies))
    return path


def decode_text(node):
    """Text a steering client sent as an array of UTF-8 byte values (0 alone: empty)."""
    v = node.value() if hasattr(node, "value") else node
    codes = [int(c) for c in (v if hasattr(v, "__len__") else [v])]
    return bytes(c for c in codes if 0 < c < 256).decode("utf-8", "replace").strip()


def read_steering(results):
    """Pull steered values out of a catalyst_results node; missing values come back as None."""
    values = {}
    for mesh in ("catalyst/derived/data", "catalyst/steerable/data"):
        if not results.has_path(mesh):
            continue
        data = results[mesh]
        for i in range(data.number_of_children()):
            part = data.child(i)
            for path in ("state/fields/derived", "fields/derived/values"):
                if part.has_path(path):
                    values["derived"] = decode_text(part[path])
    if not results.has_path("catalyst/steerable/data"):
        return values
    data = results["catalyst/steerable/data"]
    for i in range(data.number_of_children()):
        part = data.child(i)
        for key in ("reynolds", "collision", "power"):
            for path in (f"state/fields/{key}", f"fields/{key}/values"):
                if part.has_path(path):
                    v = part[path]
                    v = v.value() if hasattr(v, "value") else v
                    values[key] = float(v[0] if hasattr(v, "__len__") else v)
    return values
