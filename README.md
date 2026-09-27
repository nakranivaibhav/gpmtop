# gpmtop

**See what your NVIDIA GPU is bound by, live.**

`nvidia-smi` and `nvitop` tell you the GPU is "98% utilized". That number only
means *some* kernel was running, not that the GPU was doing useful work.
`gpmtop` reads the GPU's hardware activity counters (GPM), shows where you sit
on a speed-of-light map, and tells you what to fix.

![gpmtop cycling through compute-bound, memory-bound, stop-start, launch-bound and FP32 phases](https://raw.githubusercontent.com/nakranivaibhav/gpmtop/main/assets/demo.gif)

<sub>Recorded with `gpmtop --demo`, played at 2× speed.</sub>

## Install

```bash
pipx install gpmtop        # or: uv tool install gpmtop
gpmtop
gpmtop --demo              # no GPU? replay synthetic data
```

You need Linux, `nvidia-smi` on your `PATH`, Python 3.9+, and an NVIDIA GPU
from the **Hopper generation or newer** (see [Requirements](#requirements)).

## Reading the map

The map plots two numbers, both as a percentage of the hardware's peak:

- **x: DRAM bandwidth.** How close you are to the memory limit.
- **y: compute.** Whichever of the tensor-core pipe or the FP32 pipe is busier.

The bright dot is the current second. The fading trail is the last 30 seconds,
so a dot jumping between regions means the job alternates between two states.

| Region | What it means | What usually helps |
|---|---|---|
| **COMPUTE-BOUND** | Tensor cores are busy. This is the good case. | Lower precision (FP8/FP4), or doing less work |
| **MEMORY-BOUND** | DRAM is saturated and the math pipes wait on it. Typical of element-wise ops, norms and unfused attention. | Fuse ops: `torch.compile`, fused or custom kernels |
| **UNDER-UTILISED** | Neither compute nor memory is busy. | See the verdict line below |
| **NEAR PEAK** | Compute and memory are both busy. | You're close to the hardware limits |

Under the map, a verdict line gives the diagnosis. It uses the last 5 seconds:

| Verdict | Signal | What usually helps |
|---|---|---|
| Launch- or input-bound | SM Activity < 50% | Bigger batches, CUDA graphs, `torch.compile`, more dataloader workers, removing `.item()` / `.cpu()` syncs |
| Memory-bound | SM busy, DRAM ≥ 60% | Fusion |
| Math on FP32 CUDA cores | SM busy, FP32 ≥ 40%, tensor low | bf16 autocast, or `torch.set_float32_matmul_precision("high")` for TF32 |
| Latency-bound | SM busy, but pipes and DRAM are quiet | Small grids, syncs or atomics. Profile the top kernels in Nsight Compute |
| Compute-bound | Tensor ≥ 50% | Lower precision or less work |
| **⚠ Stalled N of the last 30s** | Busy bursts separated by idle gaps | Dataloader, host syncs, checkpointing |

The thresholds are rules of thumb. They're constants at the top of
`src/gpmtop/__init__.py` if you want to tune them.

## Honest caveats

- **This is not a roofline.** GPM reports how busy each pipe is, not FLOPs or
  bytes, so arithmetic intensity can't be computed. The map shows how close
  you are to each limit, like the Speed of Light section in Nsight Compute.
- **"Tensor 80%" means the tensor pipe was busy 80% of the time.** It does not
  mean 80% of peak FLOPs. Precision and tile efficiency also matter.
- **Samples are 1-second averages over many kernels.** Use gpmtop to find
  *which* problem you have, then use Nsight Systems or Nsight Compute to find
  *where* it is.
- **SM Occupancy isn't used in the verdicts.** Efficient GEMMs often run at
  low occupancy by design, so a low value isn't a problem on its own.

## Options

```
gpmtop --metrics 2,3,5,10,12   # GPM metric IDs, see `nvidia-smi dmon -h`
gpmtop --history 60            # window for avg/max, in samples (default 300)
```

The map needs metric 10 (DRAM) and at least one of 5 (tensor) or 12 (FP32).

## Requirements

gpmtop reads GPM (GPU Performance Monitoring) counters. NVIDIA supports GPM
only on [Hopper or newer](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlGpmFunctions.html) GPUs.

| Works | Doesn't work |
|---|---|
| **Hopper:** H100, H200, GH200 | **Ada:** RTX 4090/4080, L4, L40, L40S, RTX 6000 Ada |
| **Blackwell:** B200, GB200, RTX 50-series | **Ampere:** A100, A10, RTX 30-series |
| | **Older:** V100, T4, and so on |

Tested on an RTX 5090 with driver 580. On an unsupported GPU, gpmtop doesn't
crash; after about 10 seconds it shows "No GPM metrics from this GPU".

Ada launched alongside Hopper, but it comes earlier in NVIDIA's architecture
order and doesn't get GPM. On Ada and Ampere datacenter cards, NVIDIA's
[DCGM](https://github.com/NVIDIA/DCGM) exposes similar profiling metrics, but
gpmtop doesn't read DCGM yet.

Reports from other GPUs are welcome in an issue.

## License

MIT
