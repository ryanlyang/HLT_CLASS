"""Linux allocation measurements; local mocks cannot become remote evidence."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import threading
import time

from .contracts import artifact


def allocation(*, cpus: int):
    env = os.environ
    if (platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}
            or not env.get("SLURM_JOB_ID", "").isdigit()
            or env.get("SLURM_JOB_PARTITION") != "debug"
            or int(env.get("SLURM_CPUS_PER_TASK", "0")) != cpus
            or int(env.get("SLURM_JOB_NUM_NODES", "0")) != 1
            or env.get("SLURM_JOB_ACCOUNT") != "reu-aisocial"
            or env.get("CONDA_PREFIX") != "/home/ryreu/miniconda3/envs/atlas_kd_sporc"
            or env.get("PYTHONNOUSERSITE") != "1"
            or env.get("PYTHONDONTWRITEBYTECODE") != "1"):
        raise PermissionError("A real source-pinned SPORC CPU allocation is required")
    if any(env.get(key, "") not in {"", "0", "NoDevFiles"}
           for key in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS")):
        raise PermissionError("Response execution is CPU-only")
    if not env.get("LD_LIBRARY_PATH", "").startswith(env["CONDA_PREFIX"]+"/lib"):
        raise PermissionError("Scientific environment library precedence differs")
    return artifact("ALLOCATION", job_id=env["SLURM_JOB_ID"], host=platform.node(),
                    cpus=cpus, partition=env["SLURM_JOB_PARTITION"], account=env["SLURM_JOB_ACCOUNT"],
                    cluster=env.get("SLURM_CLUSTER_NAME"), process_id=os.getpid(), gpus=0)


class Measurement:
    """Sample summed live process-tree RSS/CPU without reading other users' data."""

    def __init__(self):
        if platform.system() != "Linux":
            raise PermissionError("Production RSS evidence requires Linux /proc")
        self.stop = threading.Event()
        self.peak = 0
        self.cpu = {}
        self.samples = 0
        self.error = None

    def sample(self):
        pending, seen, rss = [os.getpid()], set(), 0
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                directory = Path(f"/proc/{pid}")
                fields = (directory/"stat").read_text().rsplit(") ", 1)[1].split()
                # Process birth tick disambiguates possible PID reuse.
                key = (pid, fields[19])
                self.cpu[key] = max(self.cpu.get(key, 0.),
                                    (int(fields[11])+int(fields[12]))/os.sysconf("SC_CLK_TCK"))
                rss += int(fields[21])*os.sysconf("SC_PAGE_SIZE")
                for task in (directory/"task").iterdir():
                    pending.extend(map(int, (task/"children").read_text().split()))
            except FileNotFoundError:
                continue  # A child completed during this sample.
        self.peak = max(self.peak, rss)
        self.samples += 1

    def _loop(self):
        while not self.stop.wait(.25):
            try:
                self.sample()
            except Exception as exc:
                self.error = exc
                return

    def __enter__(self):
        self.started = time.monotonic()
        self.sample()
        self.initial_cpu = sum(self.cpu.values())
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set(); self.thread.join(); self.sample()
        self.elapsed = time.monotonic()-self.started
        if kind is None and self.error is not None:
            raise self.error

    def report(self):
        return dict(wall_seconds=self.elapsed, sampled_peak_tree_rss_bytes=self.peak,
                    sampled_cpu_seconds=sum(self.cpu.values())-self.initial_cpu,
                    samples=self.samples, sample_period_seconds=.25,
                    memory_method="sum_live_process_tree_RSS_including_shared_pages_conservative",
                    short_lived_child_cpu_may_be_undercounted=True)
