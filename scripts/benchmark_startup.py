"""Simple startup time benchmark for AgentsMCP lazy-loading.

Measures import time and memory footprint for key entry modules.
"""
from __future__ import annotations

import importlib
import sys
import time


def bench_import(module_name: str) -> float:
    start = time.perf_counter()
    importlib.import_module(module_name)
    return time.perf_counter() - start


def main() -> None:
    targets = [
        "agentsmcp",            # package
        "agentsmcp.cli",        # CLI entry
        "agentsmcp.ui",         # UI package (lazy)
        "agentsmcp.tools",      # Tools package (lazy)
    ]

    print("Startup benchmark (lower is better):\n")
    for mod in targets:
        # reset any previous imports of submodules for fairer timing
        to_del = [k for k in list(sys.modules.keys()) if k == mod or k.startswith(mod + ".")]
        for k in to_del:
            del sys.modules[k]
        t = bench_import(mod)
        print(f"  import {mod:<20s} -> {t*1000:.1f} ms")


if __name__ == "__main__":
    main()

