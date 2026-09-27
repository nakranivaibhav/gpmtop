"""gpmtop - see what your NVIDIA GPU is bound by, live.

Reads GPM metrics from `nvidia-smi dmon`, plots them on a speed-of-light map
and says what is limiting the GPU: launch/input, memory, compute or latency.

Usage: gpmtop [--metrics 2,3,5,10,12] [--history 300] [--demo]
Ctrl-C to quit.
"""
import argparse
import math
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque

__version__ = "0.1.0"

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

LABELS = {
    "gract": "Graphics", "smutil": "SM Activity", "smocc": "SM Occupancy",
    "intact": "INT Pipe", "mmaact": "Tensor (MMA)", "dfmaact": "Tensor DFMA",
    "hmmaact": "Tensor HMMA", "immaact": "Tensor IMMA", "dram": "DRAM BW",
    "fp64": "FP64 Pipe", "fp32": "FP32 Pipe", "fp16": "FP16 Pipe",
    "pcitx": "PCIe TX", "pcirx": "PCIe RX", "nvltx": "NVLink TX", "nvlrx": "NVLink RX",
}
BLOCKS = " ▏▎▍▌▋▊▉█"
TRAIL = 30  # seconds of history drawn on the speed-of-light map
FADE = ["grey23", "grey30", "grey42", "grey58", "grey74"]  # oldest -> newest

# Diagnosis thresholds (% over the last SMOOTH samples).
SMOOTH = 5
IDLE, SM_LOW, MMA_HIGH, DRAM_HIGH, FP32_HIGH, PIPE_LOW = 5, 50, 50, 60, 40, 30
GPM_GRACE = 10  # samples per GPU before we decide GPM isn't supported

LEFT_W, BAR_W = 50, 16


def color(pct):
    return "green" if pct < 50 else "yellow" if pct < 80 else "red"


def bar(pct, width):
    pct = max(0.0, min(100.0, pct))
    cells = pct / 100 * width
    full = int(cells)
    part = BLOCKS[int((cells - full) * 8)] if full < width else ""
    t = Text("█" * full + part, style=color(pct))
    t.append("·" * (width - full - len(part)), style="grey30")
    return t


def sol_map(points, style, w, h):
    """Speed-of-light map. x = DRAM % of peak bandwidth, y = busiest compute pipe %.

    points: [(dram, compute)] oldest -> newest; the last one is drawn as the live dot.
    The dashed lines are the DRAM_HIGH / MMA_HIGH thresholds used by diagnose().
    """
    col = lambda x: round(max(0.0, min(x, 100.0)) / 100 * (w - 1))
    row = lambda y: round((1 - max(0.0, min(y, 100.0)) / 100) * (h - 1))
    cx, ry = col(DRAM_HIGH), row(MMA_HIGH)
    grid = [[(" ", None)] * w for _ in range(h)]
    is_label = [[False] * w for _ in range(h)]
    for r in range(h):
        grid[r][cx] = ("┊", "grey27")
    for c in range(w):
        grid[ry][c] = ("┼" if c == cx else "┈", "grey27")

    # Quadrant labels; the one holding the live dot is lit in the verdict colour.
    live = (col(points[-1][0]) > cx, row(points[-1][1]) < ry) if points else None
    for (right, top), label in {(False, True): "COMPUTE-BOUND", (True, True): "NEAR PEAK",
                                (False, False): "UNDER-UTILISED", (True, False): "MEMORY-BOUND"}.items():
        r = 0 if top else h - 1
        c = cx + 2 if right else 1
        lstyle = f"bold {style}" if live == (right, top) else "grey35"
        for i, ch in enumerate(label[:max(0, (w - cx - 3) if right else (cx - 2))]):
            grid[r][c + i] = (ch, lstyle)
            is_label[r][c + i] = True

    for i, (x, y) in enumerate(points[:-1]):
        age = i / max(1, len(points) - 1)  # 0 = oldest
        if is_label[row(y)][col(x)]:
            continue
        grid[row(y)][col(x)] = ("•" if age > 0.7 else "·", FADE[min(len(FADE) - 1, int(age * len(FADE)))])
    if points:
        grid[row(points[-1][1])][col(points[-1][0])] = ("●", f"bold {style}")

    lines = [Text("↑ compute % (tensor/FP32)   → DRAM BW %", style="grey50")]
    for r in range(h):
        label = "100" if r == 0 else f"{MMA_HIGH:>3}" if r == ry else "  0" if r == h - 1 else "   "
        t = Text(f"{label}┤", style="grey50")
        for ch, st in grid[r]:
            t.append(ch, style=st)
        lines.append(t)
    axis = list("─" * w)
    for pos, s in ((0, "0"), (cx - 1, str(DRAM_HIGH)), (w - 3, "100")):
        axis[pos:pos + len(s)] = s
    lines.append(Text("   └" + "".join(axis), style="grey50"))
    return Group(*lines)


def diagnose(m):
    """Map smoothed GPM metrics to (style, verdict, nudge)."""
    sm, mma, dram, fp32 = (m.get(k) for k in ("smutil", "mmaact", "dram", "fp32"))
    if sm is None:
        return None
    mma, dram, fp32 = mma or 0, dram or 0, fp32 or 0
    if sm < IDLE:
        return "grey50", "Idle", "nothing is running on this GPU"
    if mma >= MMA_HIGH:
        return ("green", "Compute-bound on tensor cores (the good case)",
                "only lower precision (FP8/FP4) or less work will speed this up")
    if sm < SM_LOW:
        return ("red", "Launch- or input-bound: GPU is waiting on the CPU",
                "bigger batches, CUDA graphs, torch.compile, more dataloader workers, drop .item()/host syncs")
    if dram >= DRAM_HIGH:
        return ("yellow", "Memory-bound on element-wise ops",
                "fuse ops (torch.compile / custom kernels) to cut DRAM round-trips")
    if fp32 >= FP32_HIGH:
        return ("yellow", "Math on FP32 CUDA cores, tensor cores mostly idle",
                "bf16 autocast, or torch.set_float32_matmul_precision('high') for TF32")
    if mma < PIPE_LOW and fp32 < PIPE_LOW:
        return ("yellow", "SMs busy but pipes and DRAM quiet: latency-bound",
                "small grids, syncs or atomics; profile the top kernels in Nsight Compute")
    return ("cyan", "Mixed: no single limiter",
            "profile one step with torch.profiler / Nsight Systems")


def to_num(s):
    try:
        return float(s)
    except ValueError:
        return None


class State:
    def __init__(self, history):
        self.cols, self.units = [], []
        self.latest = {}  # gpu -> {col: value}
        self.rows = defaultdict(int)  # gpu -> samples seen
        self.gpm_seen = set()  # gpus that reported at least one GPM value
        self.hist = defaultdict(lambda: defaultdict(lambda: deque(maxlen=history)))
        self.stamp = ""
        self.other = deque(maxlen=5)  # unparsed lines, e.g. nvidia-smi error text
        self.lock = threading.Lock()

    def feed(self, line):
        parts = line.split()
        if not parts:
            return
        if line.startswith("#"):
            parts[0] = parts[0].lstrip("#")
            if not self.cols:
                self.cols = parts
            elif not self.units:
                self.units = parts
            return
        if len(parts) != len(self.cols):
            self.other.append(line.strip())
            return
        row = dict(zip(self.cols, parts))
        gpu = row.get("gpu", "0")
        with self.lock:
            self.stamp = row.get("Time", "")
            self.latest[gpu] = {k: to_num(v) for k, v in row.items()}
            self.rows[gpu] += 1
            if any(self.latest[gpu].get(c) is not None
                   for c, u in zip(self.cols, self.units) if u.startswith("GPM")):
                self.gpm_seen.add(gpu)
            for k, v in self.latest[gpu].items():
                self.hist[gpu][k].append(v)


def gpu_info():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,power.limit,memory.used,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    info = {}
    for line in out.strip().splitlines():
        idx, name, plim, mused, mtot = [x.strip() for x in line.split(",")]
        info[idx] = dict(name=name, plim=to_num(plim), mused=to_num(mused), mtot=to_num(mtot))
    return info


def render(state, info, width):
    with state.lock:
        gpus = sorted(state.latest, key=int)
        gpm_cols = [c for c, u in zip(state.cols, state.units) if u.startswith("GPM")]
        chart_w = max(10, width - LEFT_W - 4 - 2 - 4 - 1)  # borders/padding, gap, y-axis, margin
        chart_h = max(7, len(gpm_cols) + 2)  # title + map + axis ≈ left column height
        panels = []
        for g in gpus:
            cur, hist, gi = state.latest[g], state.hist[g], info.get(g, {})

            pwr, plim = cur.get("pwr") or 0, gi.get("plim") or 600
            gtemp, mtemp = cur.get("gtemp") or 0, cur.get("mtemp")
            head = Group(
                Text("PWR  ", style="bold") + bar(pwr / plim * 100, BAR_W) + Text(f" {pwr:.0f}/{plim:.0f} W"),
                Text.assemble(("GPU ", "bold"), (f"{gtemp:.0f}°C", color(gtemp * 1.1)),
                              ("   MEM ", "bold"), (f"{mtemp:.0f}°C" if mtemp is not None else "--", "grey50"),
                              ("   VRAM ", "bold"),
                              (f"{gi.get('mused', 0) / 1024:.1f}/{gi.get('mtot', 0) / 1024:.1f} GiB", "cyan")),
            )

            t = Table(box=None, padding=(0, 1), show_edge=False, header_style="grey50")
            t.add_column("metric", width=15, no_wrap=True)
            t.add_column("now", width=BAR_W, no_wrap=True)
            t.add_column("%", justify="right", width=3)
            t.add_column("avg/max", justify="right", width=7, style="grey50")
            smooth = {}
            for c in gpm_cols:
                v = cur.get(c)
                vals = [x for x in hist[c] if x is not None]
                recent = [x for x in list(hist[c])[-SMOOTH:] if x is not None]
                smooth[c] = sum(recent) / len(recent) if recent else None
                t.add_row(Text(LABELS.get(c, c), style="bold"),
                          bar(v, BAR_W) if v is not None else Text("warming up…", style="grey50"),
                          f"{v:.0f}" if v is not None else "-",
                          f"{sum(vals) / len(vals):.0f}/{max(vals):.0f}" if vals else "--")

            verdict = diagnose(smooth)
            if "dram" in gpm_cols and ({"mmaact", "fp32"} & set(gpm_cols)):
                n = len(hist["dram"])
                series = [list(hist[k]) if k in gpm_cols else [None] * n for k in ("dram", "mmaact", "fp32")]
                points = []
                for d, m, f in list(zip(*series))[-TRAIL:]:
                    if d is not None and (m is not None or f is not None):
                        points.append((d, max(m or 0, f or 0)))
                right = sol_map(points, verdict[0] if verdict else "white", chart_w, chart_h)
            else:
                right = Text("speed-of-light map needs --metrics with 10 (DRAM) and 5 (tensor) or 12 (FP32)",
                             style="grey50")

            body = Table.grid(padding=(0, 1))
            body.add_column(width=LEFT_W)
            body.add_column()
            body.add_row(Group(head, Text(""), t), right)

            parts = [body]
            if g not in state.gpm_seen and state.rows[g] >= GPM_GRACE:
                parts += [Rule(style="grey23"), Text.assemble(
                    ("✗ ", "red"), ("No GPM metrics from this GPU", "bold red"), ("  →  ", "grey50"),
                    ("GPM needs a recent NVIDIA GPU and driver (e.g. Hopper, Blackwell); "
                     "try `nvidia-smi dmon --gpm-metrics 2` to check", "italic"))]
            if verdict:
                vstyle, title, nudge = verdict
                parts += [Rule(style="grey23"),
                          Text.assemble(("▶ ", vstyle), (title, "bold " + vstyle),
                                        ("  →  ", "grey50"), (nudge, "italic"))]
                # A 5s average hides stop-start jobs; flag them from the trail instead.
                sm = [v for v in list(hist["smutil"])[-TRAIL:] if v is not None]
                stalled = sum(v < SM_LOW for v in sm)
                gaps = sum(a >= SM_LOW > b for a, b in zip(sm, sm[1:]))  # busy -> idle drops
                if vstyle not in ("red", "grey50") and stalled >= 3 and gaps >= 2:
                    parts.append(Text.assemble(
                        ("⚠ ", "red"), (f"GPU stalled {stalled} of the last {len(sm)}s", "bold red"),
                        ("  →  ", "grey50"),
                        ("gaps between busy bursts: dataloader, host syncs or checkpointing", "italic")))

            panels.append(Panel(Group(*parts),
                                title=f"[bold]GPU {g}[/] · {gi.get('name', '')}",
                                title_align="left", border_style="bright_blue"))
        footer = Text(f" {state.stamp}  ·  gpmtop {__version__}  ·  verdict on last {SMOOTH}s  ·  Ctrl-C to quit",
                      style="grey50")
    return Group(*panels, footer) if panels else Text("waiting for nvidia-smi dmon…", style="grey50")


DEMO_PHASES = [  # (name, seconds, smutil, smocc, mmaact, dram, fp32, watts)
    ("compute", 20, 96, 25, 75, 35, 15, 520),
    ("memory", 20, 88, 35, 10, 75, 20, 380),
    ("stop-start", 25, None, None, None, None, None, None),
    ("launch", 20, 28, 10, 12, 15, 5, 180),
    ("fp32", 15, 92, 40, 3, 30, 65, 400),
]


def demo_feed(state):
    """Replay synthetic dmon output that walks through every verdict."""
    rnd = random.Random(0)
    state.feed("#Time gpu pwr gtemp mtemp smutil smocc mmaact dram fp32")
    state.feed("#HH:MM:SS Idx W C C GPM:% GPM:% GPM:% GPM:% GPM:%")
    t = 0
    while True:
        for name, secs, *vals in DEMO_PHASES:
            for i in range(secs):
                if name == "stop-start":  # 4s of compute, 2s waiting on data
                    vals = DEMO_PHASES[0][2:] if i % 6 < 4 else (8, 3, 2, 4, 2, 150)
                jit = lambda v, a: max(0.0, min(100.0, v + rnd.uniform(-a, a)))
                sm, occ, mma, dram, fp32, w = vals
                temp = 45 + w / 20 + 2 * math.sin(t / 30)
                state.feed(f" {time.strftime('%H:%M:%S')} 0 {w + rnd.uniform(-15, 15):.0f} {temp:.0f} - "
                           f"{jit(sm, 3):.0f} {jit(occ, 3):.0f} {jit(mma, 6):.0f} {jit(dram, 6):.0f} {jit(fp32, 3):.0f}")
                t += 1
                time.sleep(1)


def main():
    ap = argparse.ArgumentParser(prog="gpmtop", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", default="2,3,5,10,12", help="GPM metric ids (see `nvidia-smi dmon -h`)")
    ap.add_argument("--history", type=int, default=300, help="samples of history to keep (default 300)")
    ap.add_argument("--demo", action="store_true", help="replay synthetic data; no GPU needed")
    ap.add_argument("--version", action="version", version=f"gpmtop {__version__}")
    args = ap.parse_args()

    state = State(args.history)
    if args.demo:
        proc = None
        get_info = lambda: {"0": dict(name="Demo GPU", plim=600.0, mused=21504.0, mtot=32768.0)}
        threading.Thread(target=demo_feed, args=(state,), daemon=True).start()
    else:
        if not shutil.which("nvidia-smi"):
            sys.exit("gpmtop: nvidia-smi not found. Install the NVIDIA driver, or try `gpmtop --demo`.")
        proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "p", "--gpm-metrics", args.metrics, "-o", "T"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        get_info = gpu_info
        threading.Thread(target=lambda: [state.feed(l) for l in proc.stdout], daemon=True).start()

    info, info_t = get_info(), time.time()
    try:
        with Live(auto_refresh=False, screen=True) as live:
            while proc is None or proc.poll() is None:
                if time.time() - info_t > 2:
                    info, info_t = get_info(), time.time()
                live.update(render(state, info, live.console.width), refresh=True)
                time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        if proc:
            proc.terminate()
    if proc and proc.returncode not in (None, 0, -2, -15):
        err = (proc.stderr.read() or " / ".join(state.other)).strip()
        sys.exit(f"gpmtop: nvidia-smi dmon exited with {proc.returncode}: {err or 'no output'}")


if __name__ == "__main__":
    main()
