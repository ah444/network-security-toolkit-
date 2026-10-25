"""Peak memory and time of ``load_dataset`` on one file (v5.2 step 2.3a), to show that a
refactor of the loader costs no more than the code before it.

Each run is a fresh process. In it, after the imports: B = the working set now, P0 = the peak
so far; then ``load_dataset``; then P = the peak. The peak only grows and nothing but the load
ran between the readings, so a run counts only if P > P0 (the load set a new peak), and its
memory is P - B. A run with P == P0 is unreliable (the file is too small to show) and is not
a pass: use a bigger file. The first run only warms the operating system's file cache and is
not counted. Then 3 runs (5 if their times vary by more than 10 % of the median), compared
by their medians.

Memory readings: Windows, the process working set (GetProcessMemoryInfo, through ctypes);
Linux, resident pages (/proc/self/statm) and ru_maxrss (KiB); macOS, ru_maxrss (bytes) for
both B and P0, so P - B counts from the peak after the imports (no current-size reading in
the standard library there).

Usage (from the project root):
  python scripts/measure_load.py FILE [--json OUT]           measure
  python scripts/measure_load.py FILE --compare BASE.json   measure, then check against BASE:
      P - B at most +5 %, time at most +10 % (exit code 1 if not, 2 if unreliable)
"""

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEMORY_LIMIT = 1.05  # P - B, relative to the baseline
TIME_LIMIT = 1.10
TIME_SPREAD = 0.10  # (slowest - fastest) / median above this: two more runs


def _windows_memory() -> tuple[int, int]:
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    current_process = ctypes.windll.kernel32.GetCurrentProcess
    current_process.restype = wintypes.HANDLE  # a 64-bit pseudo-handle, not a C int
    info = ctypes.windll.psapi.GetProcessMemoryInfo
    info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not info(current_process(), ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return counters.WorkingSetSize, counters.PeakWorkingSetSize


def memory() -> tuple[int, int]:
    """(current, peak) bytes of this process."""
    system = platform.system()
    if system == "Windows":
        return _windows_memory()
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if system == "Darwin" else 1024  # bytes on macOS, KiB on Linux
    if system == "Linux":
        with open("/proc/self/statm", encoding="ascii") as handle:
            current = int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        return current, peak
    return peak, peak


def child(path: str) -> None:
    sys.path.insert(0, str(ROOT))
    try:
        from toolkit.core.services.ids.datasets import load_dataset
    except ModuleNotFoundError:
        from toolkit.services.ids.datasets import load_dataset
    gc.collect()
    current, peak_before = memory()
    start = time.perf_counter()
    loaded = load_dataset(path)
    seconds = time.perf_counter() - start
    _, peak_after = memory()
    print(json.dumps({"B": current, "P0": peak_before, "P": peak_after, "seconds": seconds,
                      "rows": len(loaded.frame), "columns": loaded.frame.shape[1]}))


def run_once(path: Path) -> dict:
    done = subprocess.run([sys.executable, __file__, "--child", str(path)], capture_output=True,
                          text=True, encoding="utf-8", check=True)
    return json.loads(done.stdout)


def measure(path: Path) -> dict:
    run_once(path)  # warm-up: fills the file cache, not counted
    runs = [run_once(path) for _ in range(3)]
    times = [r["seconds"] for r in runs]
    if (max(times) - min(times)) / statistics.median(times) > TIME_SPREAD:
        runs += [run_once(path) for _ in range(2)]
    valid = [r for r in runs if r["P"] > r["P0"]]
    times = [r["seconds"] for r in runs]
    return {
        "file": path.name, "bytes": path.stat().st_size, "rows": runs[0]["rows"],
        "columns": runs[0]["columns"], "system": platform.system(),
        "python": platform.python_version(), "runs": runs,
        "reliable": len(valid) == len(runs),
        "memory_median": statistics.median(r["P"] - r["B"] for r in valid) if valid else None,
        "seconds_median": statistics.median(times),
        "seconds_spread": (max(times) - min(times)) / statistics.median(times),
    }


def compare(result: dict, base: dict) -> int:
    if not (result["reliable"] and base["reliable"]):
        print("UNRELIABLE: a run did not set a new peak (P == P0); use a bigger file")
        return 2
    memory_ratio = result["memory_median"] / base["memory_median"]
    time_ratio = result["seconds_median"] / base["seconds_median"]
    ok = memory_ratio <= MEMORY_LIMIT and time_ratio <= TIME_LIMIT
    print(f"P - B: {memory_ratio:.3f} x the baseline (limit {MEMORY_LIMIT}); time: "
          f"{time_ratio:.3f} x (limit {TIME_LIMIT}); {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("file")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json", metavar="OUT", help="also write the result to OUT")
    parser.add_argument("--compare", metavar="BASE", help="a result written with --json")
    args = parser.parse_args()
    if args.child:
        child(args.file)
        return 0
    result = measure(Path(args.file))
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    if args.compare:
        return compare(result, json.loads(Path(args.compare).read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
