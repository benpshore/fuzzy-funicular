import os
import subprocess
import sys
import threading
import time

import psutil
import pytest

from funicular import resources
from funicular.resources import (
    Cancelled,
    JobContext,
    MemoryGuard,
    SystemMonitor,
    current_job,
    kill_tree,
    run,
)


def test_run_captures_output_and_registers_pid():
    ctx = JobContext(id="j1")
    token = current_job.set(ctx)
    try:
        proc = run(
            [sys.executable, "-c", "import os,sys; print(os.getpid()); sys.exit(3)"], timeout=10
        )
    finally:
        current_job.reset(token)
    assert proc.returncode == 3
    assert proc.stdout.strip().isdigit()
    assert ctx.pids == set()  # unregistered after exit


def test_run_times_out_and_kills():
    t = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
    assert time.monotonic() - t < 10


CHILD_WITH_GRANDCHILD = (
    "import subprocess,sys,time; "
    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); time.sleep(60)"
)


def test_cancel_kills_running_child_tree():
    ctx = JobContext(id="j2")
    started = threading.Event()
    result = {}

    def worker():
        # contextvars are per thread: the worker binds its own job, as the job pool does.
        token = current_job.set(ctx)
        started.set()
        try:
            run([sys.executable, "-c", CHILD_WITH_GRANDCHILD], timeout=120)
        except Cancelled as exc:
            result["cancelled"] = str(exc)
        finally:
            current_job.reset(token)

    th = threading.Thread(target=worker)
    th.start()
    started.wait(5)
    deadline = time.monotonic() + 10
    while not ctx.live_pids() and time.monotonic() < deadline:
        time.sleep(0.05)
    pids = ctx.live_pids()
    assert pids
    # wait until the grandchild exists too
    deadline = time.monotonic() + 10
    while len(ctx.tree()) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    tree = [p.pid for p in ctx.tree()]
    assert len(tree) >= 2
    ctx.cancel("test cancel")
    th.join(15)
    assert result["cancelled"] == "test cancel"
    time.sleep(0.5)
    assert not any(
        psutil.pid_exists(p) and psutil.Process(p).status() != psutil.STATUS_ZOMBIE for p in tree
    )


def test_check_cancel_raises_only_when_cancelled():
    ctx = JobContext(id="j3")
    token = current_job.set(ctx)
    try:
        resources.check_cancel()
        ctx.cancel("stop")
        with pytest.raises(Cancelled, match="stop"):
            resources.check_cancel()
    finally:
        current_job.reset(token)
    current_job.set(None)
    resources.check_cancel()  # no job: no-op


def test_kill_tree_on_missing_pid():
    assert kill_tree(99999999) == 0


def test_child_process_limits_are_applied():
    if sys.platform == "win32":
        pytest.skip("posix only")
    # The child reports its own limits. (Root bypasses RLIMIT_NPROC enforcement on Linux, so
    # we assert the limit is set rather than fork-bombing the test box.)
    code = (
        "import resource, os\n"
        "print(resource.getrlimit(resource.RLIMIT_NPROC)[0])\n"
        "print(resource.getrlimit(resource.RLIMIT_CORE)[0])\n"
        "print(os.getpgid(0) == os.getpid())\n"
    )
    proc = run([sys.executable, "-c", code], timeout=60)
    nproc, core, own_group = proc.stdout.decode().split()
    assert int(nproc) <= resources.CHILD_MAX_PROCS
    assert int(core) == 0
    assert own_group == "True"
    if os.geteuid() != 0:
        # Non-root (the normal case on a Mac): a fork burst beyond the cap must be refused.
        burst = (
            "import os\n"
            "n=0; kids=[]\n"
            "for i in range(300):\n"
            "    try: pid=os.fork()\n"
            "    except BlockingIOError: break\n"
            "    if pid==0:\n"
            "        import time; time.sleep(5); os._exit(0)\n"
            "    kids.append(pid); n+=1\n"
            "print(n)\n"
            "for k in kids: os.kill(k, 9)\n"
        )
        forks = int(run([sys.executable, "-c", burst], timeout=60).stdout.strip() or 0)
        assert forks < 300


def test_system_monitor_snapshot(tmp_path):
    mon = SystemMonitor(tmp_path)
    ctx = JobContext(id="x")
    snap = mon.snapshot([ctx])
    assert snap["mem_total_gb"] > 0 and snap["cpu_count"] >= 1
    assert snap["disk"]["total_gb"] is not None
    assert snap["jobs"][0]["id"] == "x"
    assert "torch_device" in snap["gpu"]


def test_memory_guard_kills_over_cap():
    ctx = JobContext(id="big")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    ctx.register(proc.pid)
    guard = MemoryGuard(lambda: [ctx], job_cap_mb=0.001, job_max_procs=10)
    guard.tick()
    proc.wait(10)
    assert ctx.cancelled and "cap" in ctx.reason
    assert guard.last_kill and "big" in guard.last_kill


def test_memory_guard_process_count_cap():
    ctx = JobContext(id="many")
    procs = [
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"]) for _ in range(3)
    ]
    for p in procs:
        ctx.register(p.pid)
    guard = MemoryGuard(lambda: [ctx], job_cap_mb=100000, job_max_procs=2)
    guard.tick()
    for p in procs:
        p.wait(10)
    assert ctx.cancelled and "processes" in ctx.reason


def test_admission_uses_free_memory():
    guard = MemoryGuard(lambda: [], min_free_mb=0)
    assert guard.admission_ok(1)
    assert not guard.admission_ok(10**9)
