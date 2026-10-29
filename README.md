# tack-lbm

A branchless D3Q19 lattice Boltzmann solver written in [Tack](https://berkgeveci.github.io/tack/latest/),
instrumented with Catalyst 2 and Catalyst Live so the CFD assistant can monitor and analyze it while it runs.

The physics is composed at runtime from Tack device classes. Each combination of collision model
(`bgk`, `trt`, `les`), obstacle (`cylinder`, `sphere`, `box`) and diagnostics variant compiles to its
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
