# tack-lbm

A branchless D3Q19 lattice Boltzmann solver written in [Tack](https://berkgeveci.github.io/tack/latest/),
instrumented with Catalyst 2 and Catalyst Live so the CFD assistant can monitor and analyze it while it runs.

The physics is composed at runtime from Tack device classes. Each combination of collision model
(`bgk`, `trt` with a Smagorinsky subgrid model, `les`), obstacle (`cylinder`, `sphere`, `box`) and diagnostics variant compiles to its
own specialized kernel, so run options never become branches in the kernel. Solid walls, bounce-back,
momentum-exchange forces and the outflow sponge are arithmetic on the obstacle's signed-distance test.

Fields live in CUDA managed memory: the kernels write them and Conduit points at the same buffers,
which ParaView imports zero-copy.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install "numpy==1.26.4" "cuda-python>=13.2.0" -e ~/gitsync/tack/packages/tack-core
# add Conduit and Catalyst Python modules to the venv with a .pth file (see env.sh for paths)
./run_live.sh --collision les --obstacle cylinder --re 300
```

`run_live.sh` starts the solver (Catalyst Live on port 22222) and `cfd_chat.py --live` on port 8080.
Click Connect, then ask the assistant for the flow field or the residuals.

## Channels

| Channel | Contents |
|---|---|
| `grid` | Uniform grid (subsampled by `--stride`, default 2) with `velocity`, `pressure`, `speed` |
| `monitors` | Residual and force history: `iteration`, `continuity`, `momentum_x/y/z`, `energy`, `drag`, `lift` |

Catalyst Live sends extracts every 300 steps (`pipeline_live.py`).

## Derived fields

The grid channel also carries fields computed from VTK calculator expressions with
[tack-expr](../tack-expr), each compiled to one Tack kernel: `vorticity` (`mag(curl(velocity))`)
and `q_criterion` (`qcrit(velocity)`) by default, at cell centers. Add more with `--derive`:

```bash
python run_lbm.py --derive "cp=2 * pressure / (0.08 * 0.08)" --derive "omega_z=curl(velocity)[2]@cells"
```

A pointwise expression is a vertex field; one with gradients needs `@cells` and is an element
field. `--no-derived` leaves out the defaults. `derived_check.py` compares the derived fields with
`tack.dataset`'s gradient filters and a NumPy reference, and `pipeline_check.py` prints what the
grid channel carries, without a Live client.

## Steering

`steering.py` defines a Catalyst steering proxy (`LBMSteering`) with the Reynolds number and the
collision model. The pipeline script registers it with `CreateSteerableParameters`; after each
`catalyst_execute` the solver calls `catalyst_results` and applies changes made by a Live client.
A Reynolds change only updates a runtime kernel argument. A collision change selects another
kernel specialization, which Tack compiles on first use and caches on disk.

The CFD assistant loads the same proxy definition (`CATALYST_LIVE_PROXIES`, set by `run_live.sh`)
and exposes the parameters as a Re field and BGK/TRT/LES buttons in its header, and to the chat
through `get_simulation_parameters` and `set_simulation_parameter`.

## Checks

- `trt_ref.py` compares one TRT step against a NumPy reference.
- `collision_check.py` runs each collision model at Re 300, 1000 and 3000 and reports stability and
  the Strouhal number. BGK diverges at Re 3000; TRT and LES carry a Smagorinsky subgrid viscosity and
  stay stable. All three give a Strouhal number of 0.200 at Re 300.
