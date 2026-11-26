"""The 26 experiments of the thesis (Chapter 7): systems, row settings and the runner.

    from hydyreg.benchmarks import systems, rows
    result = rows.run_row(rows.get_row("incr/isolette"))
"""
from . import systems, rows  # noqa: F401  (public submodules)

__all__ = ["systems", "rows"]
