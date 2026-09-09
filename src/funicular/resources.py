"""Resource accounting, cancellation and process safety.

* ``JobContext`` — one per ingest job. Carries a cancel flag and the PIDs of every external
  process the job spawned, so a cancel (or the memory guard) can kill the whole tree.
* ``run`` — the only way the code base spawns subprocesses. Applies per-child rlimits (a fork
  bomb inside Ghostscript or tesseract cannot take the machine down), registers the PID with
  the current job, and honours cancellation while the child runs.
* ``SystemMonitor`` — cheap psutil snapshot for the UI / doctor, on Apple Silicon and Linux.
* ``MemoryGuard`` — background thread: kills any job whose process tree exceeds the memory or
  process-count cap, and reports system pressure so the job manager can hold admission.
"""

from __future__ import annotations

import contextvars
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

log = logging.getLogger(__name__)


class Cancelled(RuntimeError):
    """Raised inside a job when it has been cancelled (by a user or the memory guard)."""


@dataclass
class JobContext:
    id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
    reason: str = ""
    pids: set[int] = field(default_factory=set)
    started: float = field(default_factory=time.monotonic)
    peak_rss_mb: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            if not self.cancel_event.is_set():
                self.reason = reason
            self.cancel_event.set()
        self.kill_children()

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check(self) -> None:
        if self.cancelled:
            raise Cancelled(self.reason or "cancelled")

    def register(self, pid: int) -> None:
        with self._lock:
            self.pids.add(pid)

    def unregister(self, pid: int) -> None:
        with self._lock:
            self.pids.discard(pid)

    def live_pids(self) -> list[int]:
        with self._lock:
            return [p for p in self.pids if psutil.pid_exists(p)]

    def tree(self) -> list[psutil.Process]:
        procs: list[psutil.Process] = []
        for pid in self.live_pids():
            try:
                p = psutil.Process(pid)
                procs.append(p)
                procs.extend(p.children(recursive=True))
            except psutil.Error:
                continue
        return procs

    def kill_children(self) -> None:
        for pid in self.live_pids():
            kill_tree(pid)


current_job: contextvars.ContextVar[JobContext | None] = contextvars.ContextVar(
    "funicular_current_job", default=None
)


def check_cancel() -> None:
    job = current_job.get()
    if job is not None:
        job.check()


# --------------------------------------------------------------------------------------------
# subprocesses
# --------------------------------------------------------------------------------------------
CHILD_MAX_PROCS = int(os.environ.get("FUNICULAR_CHILD_MAX_PROCS", "64"))
CHILD_MAX_MB = int(os.environ.get("FUNICULAR_CHILD_MAX_MB", "6144"))


def _child_limits() -> None:  # pragma: no cover - runs in the child between fork and exec
    try:
        import resource

        # Fork-bomb guard: a child may not have more than N processes in its tree.
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        cap = CHILD_MAX_PROCS if hard == resource.RLIM_INFINITY else min(CHILD_MAX_PROCS, hard)
        resource.setrlimit(resource.RLIMIT_NPROC, (cap, hard))
        if sys.platform == "linux":
            # Address-space cap works on Linux; macOS ignores RLIMIT_AS, the MemoryGuard
            # covers it there.
            bytes_cap = CHILD_MAX_MB * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (bytes_cap, bytes_cap))
        # Core dumps off: a crashing Ghostscript should not write gigabytes to disk.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:  # noqa: BLE001, S110 - never block the exec because a limit failed
        pass
    try:
        os.setpgid(0, 0)  # own process group so kill_tree can take the whole group
    except OSError:
        pass


def run(
    argv: list[str],
    *,
    timeout: float,
    input_bytes: bytes | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """subprocess.run with limits, job registration and cancellation. argv only, no shell."""
    job = current_job.get()
    if job is not None:
        job.check()
    proc = subprocess.Popen(  # noqa: S603 - argv list, no shell
        argv,
        stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        preexec_fn=_child_limits if os.name == "posix" else None,  # noqa: PLW1509
    )
    if job is not None:
        job.register(proc.pid)
    deadline = time.monotonic() + timeout
    out = err = b""
    try:
        if input_bytes is not None:
            # Feed stdin in a thread so a cancel can still kill a blocked child.
            def _feed() -> None:
                try:
                    assert proc.stdin is not None
                    proc.stdin.write(input_bytes)
                    proc.stdin.close()
                except OSError:
                    pass

            threading.Thread(target=_feed, daemon=True).start()
        while True:
            try:
                out, err = proc.communicate(timeout=0.25)
                if job is not None and job.cancelled:
                    # The cancel killed the child before we noticed; report it as a cancel,
                    # never as a "successful" run with a partial result.
                    raise Cancelled(job.reason or "cancelled")
                break
            except subprocess.TimeoutExpired:
                if job is not None and job.cancelled:
                    kill_tree(proc.pid)
                    out, err = proc.communicate(timeout=5)
                    raise Cancelled(job.reason or "cancelled") from None
                if time.monotonic() > deadline:
                    kill_tree(proc.pid)
                    out, err = proc.communicate(timeout=5)
                    raise subprocess.TimeoutExpired(argv, timeout, out, err) from None
    finally:
        if job is not None:
            job.unregister(proc.pid)
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def kill_tree(pid: int, grace: float = 2.0) -> int:
    """Terminate a process and all its descendants. Returns how many were signalled."""
    try:
        root = psutil.Process(pid)
    except psutil.Error:
        return 0
    procs = [root]
    try:
        procs += root.children(recursive=True)
    except psutil.Error:
        pass
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _gone, alive = psutil.wait_procs(procs, timeout=grace)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    return len(procs)


# --------------------------------------------------------------------------------------------
# system snapshot
# --------------------------------------------------------------------------------------------
_GPU_CACHE: dict[str, Any] | None = None


def gpu_info() -> dict[str, Any]:
    """What acceleration is available. Cached; torch import is lazy and optional."""
    global _GPU_CACHE
    if _GPU_CACHE is not None:
        return _GPU_CACHE
    info: dict[str, Any] = {
        "platform": sys.platform,
        "machine": platform.machine(),
        "apple_silicon": sys.platform == "darwin" and platform.machine() == "arm64",
        "chip": None,
        "torch": None,
        "torch_device": "cpu",
        "mlx": False,
        "ane_note": None,
    }
    if sys.platform == "darwin":
        sysctl = shutil.which("sysctl")
        if sysctl:
            try:
                out = subprocess.run(  # noqa: S603
                    [sysctl, "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                info["chip"] = out.stdout.strip() or None
            except OSError, subprocess.TimeoutExpired:
                pass
        info["ane_note"] = (
            "Apple Neural Engine is used by Vision OCR and Core ML models automatically; "
            "torch uses the GPU via Metal (MPS), MLX uses GPU + unified memory."
        )
    try:
        import torch  # type: ignore[import-not-found]

        info["torch"] = torch.__version__
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["torch_device"] = "mps"
        elif torch.cuda.is_available():
            info["torch_device"] = "cuda"
    except Exception as exc:  # noqa: BLE001 - torch is optional
        log.debug("torch unavailable: %s", exc)
    try:
        import mlx.core  # type: ignore[import-not-found]  # noqa: F401

        info["mlx"] = True
    except Exception as exc:  # noqa: BLE001
        log.debug("mlx unavailable: %s", exc)
    _GPU_CACHE = info
    return info


def preferred_torch_device() -> str:
    return gpu_info()["torch_device"]


class SystemMonitor:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._proc = psutil.Process(os.getpid())
        psutil.cpu_percent(interval=None)  # prime

    def snapshot(self, jobs: Iterable[JobContext] = ()) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        try:
            du = shutil.disk_usage(self.data_dir if self.data_dir.exists() else Path.home())
            disk = {"total_gb": round(du.total / 2**30, 1), "free_gb": round(du.free / 2**30, 1)}
        except OSError:
            disk = {"total_gb": None, "free_gb": None}
        try:
            load = os.getloadavg()
        except AttributeError, OSError:
            load = (0.0, 0.0, 0.0)
        job_rows = []
        for j in jobs:
            rss = 0
            n = 0
            for p in j.tree():
                try:
                    rss += p.memory_info().rss
                    n += 1
                except psutil.Error:
                    continue
            job_rows.append(
                {
                    "id": j.id,
                    "seconds": round(time.monotonic() - j.started, 1),
                    "child_procs": n,
                    "child_rss_mb": round(rss / 2**20, 1),
                    "peak_rss_mb": round(j.peak_rss_mb, 1),
                    "cancelled": j.cancelled,
                }
            )
        return {
            "time": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_count": psutil.cpu_count(logical=True),
            "load": [round(x, 2) for x in load],
            "mem_total_gb": round(vm.total / 2**30, 2),
            "mem_available_gb": round(vm.available / 2**30, 2),
            "mem_percent": vm.percent,
            "swap_used_gb": round(sw.used / 2**30, 2),
            "process_rss_mb": round(self._proc.memory_info().rss / 2**20, 1),
            "process_threads": self._proc.num_threads(),
            "disk": disk,
            "gpu": gpu_info(),
            "jobs": job_rows,
        }


# --------------------------------------------------------------------------------------------
# memory guard
# --------------------------------------------------------------------------------------------
class MemoryGuard(threading.Thread):
    """Watches job process trees. Kills a job whose tree exceeds the caps; flags pressure."""

    def __init__(
        self,
        jobs_provider,
        *,
        job_cap_mb: int = 8192,
        job_max_procs: int = 96,
        min_free_mb: int = 1024,
        interval: float = 1.0,
    ) -> None:
        super().__init__(name="memory-guard", daemon=True)
        self.jobs_provider = jobs_provider  # () -> Iterable[JobContext]
        self.job_cap_mb = job_cap_mb
        self.job_max_procs = job_max_procs
        self.min_free_mb = min_free_mb
        self.interval = interval
        self.pressure = False
        self.last_kill: str | None = None
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - the guard must never die
                log.warning("memory guard tick failed: %s", exc)

    def tick(self) -> None:
        vm = psutil.virtual_memory()
        self.pressure = vm.available < self.min_free_mb * 2**20
        for job in list(self.jobs_provider()):
            tree = job.tree()
            rss = 0
            for p in tree:
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    continue
            rss_mb = rss / 2**20
            job.peak_rss_mb = max(job.peak_rss_mb, rss_mb)
            if rss_mb > self.job_cap_mb:
                self.last_kill = f"{job.id}: {rss_mb:.0f} MB > cap {self.job_cap_mb} MB"
                log.error("memory guard killing %s", self.last_kill)
                job.cancel(f"killed: process tree used {rss_mb:.0f} MB (cap {self.job_cap_mb} MB)")
            elif len(tree) > self.job_max_procs:
                self.last_kill = f"{job.id}: {len(tree)} processes > cap {self.job_max_procs}"
                log.error("memory guard killing %s", self.last_kill)
                job.cancel(f"killed: {len(tree)} processes (cap {self.job_max_procs})")

    def admission_ok(self, needed_mb: float) -> bool:
        vm = psutil.virtual_memory()
        return vm.available >= (needed_mb + self.min_free_mb) * 2**20
