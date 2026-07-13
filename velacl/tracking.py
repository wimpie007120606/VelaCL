from __future__ import annotations

import json
import platform
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch

from velacl.utils import atomic_json, git_commit


class RunTracker(AbstractContextManager):
    """JSON-first tracker with transparent optional MLflow mirroring."""

    def __init__(self, run_dir: Path, name: str, parameters: dict[str, Any]):
        self.run_dir = run_dir
        self.name = name
        self.parameters = parameters
        self.started = time.perf_counter()
        self.metrics: dict[str, Any] = {}
        self.mlflow = None

    def __enter__(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            import mlflow

            mlflow.set_tracking_uri(str(self.run_dir.parent.parent / "mlruns"))
            mlflow.set_experiment("velacl")
            mlflow.start_run(run_name=self.name)
            mlflow.log_params({k: str(v) for k, v in self.parameters.items()})
            self.mlflow = mlflow
        except ImportError:
            pass
        return self

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics.update(metrics)
        if self.mlflow:
            self.mlflow.log_metrics(metrics, step=step)

    def __exit__(self, exc_type, exc_value, traceback):
        record = {
            "run_name": self.name,
            "status": "failed" if exc_type else "complete",
            "git_commit": git_commit(),
            "parameters": self.parameters,
            "metrics": self.metrics,
            "hardware": {
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.cuda.is_available(),
            },
            "training_time_seconds": time.perf_counter() - self.started,
        }
        atomic_json(self.run_dir / "run.json", record)
        if self.mlflow:
            self.mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
        return False


def update_registry(path: str | Path, candidates: list[dict]) -> dict:
    path = Path(path)
    registry = (
        json.loads(path.read_text())
        if path.exists()
        else {"models": [], "aliases": {}, "history": []}
    )
    registry["models"] = candidates
    eligible = [m for m in candidates if m["metrics"]["average_accuracy"] >= 0.25]
    if eligible:
        champion = max(
            eligible,
            key=lambda m: (
                m["metrics"]["average_accuracy"] - m["metrics"]["average_forgetting"],
                m["method"],
            ),
        )
        previous = registry.get("aliases", {}).get("champion")
        rollback = registry.get("aliases", {}).get("rollback")
        if rollback == champion["method"]:
            rollback = None
        registry["aliases"] = {
            "champion": champion["method"],
            "production": champion["method"],
            "candidate": champion["method"],
            "rollback": previous if previous != champion["method"] else rollback,
        }
        registry["history"] = [
            item for item in registry["history"] if item.get("from") != item.get("to")
        ]
        if previous != champion["method"]:
            registry["history"].append(
                {
                    "action": "promotion",
                    "from": previous,
                    "to": champion["method"],
                    "gate": "accuracy>=0.25",
                }
            )
    atomic_json(path, registry)
    return registry
