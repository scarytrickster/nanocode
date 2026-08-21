"""Run the benchmark: python -m benchmark"""

from __future__ import annotations

import sys
import tempfile

from benchmark.report import render_report
from benchmark.runner import run_benchmark
from benchmark.tasks import TASKS, TASKS_BY_ID


def main() -> None:
    selected = (
        tuple(TASKS_BY_ID[name] for name in sys.argv[1:])
        if len(sys.argv) > 1
        else TASKS
    )

    with tempfile.TemporaryDirectory() as root:
        results = run_benchmark(selected, root)

    print(render_report(results))


if __name__ == "__main__":
    main()
