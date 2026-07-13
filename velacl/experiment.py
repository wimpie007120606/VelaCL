from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from velacl.config import Config, load_config
from velacl.data import Event, load_events, write_events
from velacl.metrics import continual_metrics
from velacl.model import ByteTransformer
from velacl.replay import ReplayBuffer, active_balanced_select
from velacl.tracking import RunTracker, update_registry
from velacl.trainer import evaluate, load_checkpoint, save_checkpoint, train_stage
from velacl.utils import atomic_json, seed_everything, sha256

CORE_METHODS = {"static", "naive", "random_replay", "active_balanced_replay"}
ABLATION_METHODS = {"active_uncertainty_replay", "active_diversity_replay"}
VALID_METHODS = CORE_METHODS | ABLATION_METHODS


def _selection_record(selection) -> dict:
    return {
        "event_id": selection.event.id,
        "text": selection.event.text,
        "language": selection.event.language,
        "intent": selection.event.intent,
        "domain": selection.event.domain,
        "privacy_risk": selection.event.privacy_risk,
        "score": selection.score,
        "uncertainty": selection.uncertainty,
        "diversity": selection.diversity,
        "language_scarcity": selection.language_scarcity,
        "intent_novelty": selection.intent_novelty,
        "forgetting_risk": selection.forgetting_risk,
        "safety_importance": selection.safety_importance,
        "selection_reason": selection.reason,
        "annotation_status": "pending",
    }


def run_method(
    method: str,
    config: Config,
    events: list[Event],
    device: torch.device,
    resume: str | None = None,
) -> dict:
    if method not in VALID_METHODS:
        raise ValueError(f"unknown method: {method}")
    labels = sorted({event.intent for event in events})
    label_to_id = {label: index for index, label in enumerate(labels)}
    stages = sorted({event.stage for event in events})
    run_dir = Path(config.artifact_dir) / method
    seed_everything(config.seed)
    start_stage = 0
    if resume:
        model, checkpoint = load_checkpoint(resume, device)
        if checkpoint["method"] != method:
            raise ValueError("checkpoint method does not match requested method")
        start_stage = int(checkpoint["stage"]) + 1
    else:
        model = ByteTransformer(config.model, len(labels)).to(device)
    buffer = ReplayBuffer(config.training.replay_capacity, config.seed)
    if resume:
        buffer.events = [Event(**value) for value in checkpoint.get("replay_events", [])]
    matrix: list[list[float]] = [[] for _ in stages]
    stage_results = []
    annotation_queue = []
    parameters = {
        "method": method,
        "seed": config.seed,
        **asdict(config.training),
        **asdict(config.model),
    }
    with RunTracker(run_dir, method, parameters) as tracker:
        for stage in stages:
            incoming = [
                event for event in events if event.stage == stage and event.split == "train"
            ]
            if stage < start_stage:
                continue
            selections = []
            if method == "static" and stage > 0:
                train_events = []
            elif method == "naive" or stage == 0:
                train_events = incoming
            elif method == "random_replay":
                train_events = incoming + buffer.events
            else:
                selections = active_balanced_select(
                    model,
                    incoming,
                    buffer.events,
                    label_to_id,
                    len(incoming),
                    device,
                )
                if method == "active_uncertainty_replay":
                    selections.sort(key=lambda value: (-value.uncertainty, value.event.id))
                elif method == "active_diversity_replay":
                    selections.sort(key=lambda value: (-value.diversity, value.event.id))
                selections = selections[: config.training.annotation_budget]
                selected_events = [selection.event for selection in selections]
                train_events = selected_events + buffer.events
                if method == "active_balanced_replay":
                    annotation_queue.extend(
                        _selection_record(selection) for selection in selections
                    )
            training = train_stage(
                model, train_events, label_to_id, config, device, config.seed + stage
            )
            if method == "random_replay":
                buffer.random_update(incoming)
            elif method in {"active_balanced_replay", *ABLATION_METHODS}:
                buffer.balanced_update(
                    [selection.event for selection in selections] if selections else incoming
                )
            task_metrics = []
            for task in stages:
                test = [event for event in events if event.stage == task and event.split == "test"]
                task_eval = evaluate(model, test, label_to_id, device)
                matrix[task].append(task_eval["macro_f1"])
                task_metrics.append(task_eval)
            seen_test = [
                event for event in events if event.split == "test" and event.stage <= stage
            ]
            aggregate = evaluate(model, seen_test, label_to_id, device)
            result = {
                "stage": stage,
                "training": training,
                "aggregate": aggregate,
                "tasks": task_metrics,
            }
            stage_results.append(result)
            save_checkpoint(
                run_dir / f"stage-{stage}.pt",
                model,
                labels,
                config,
                stage,
                method,
                buffer.events,
            )
            tracker.log_metrics({f"stage_{stage}_macro_f1": aggregate["macro_f1"]}, step=stage)
        continual = continual_metrics(matrix) if start_stage == 0 else {}
        if continual:
            tracker.log_metrics(continual)
    final_checkpoint = run_dir / f"stage-{stages[-1]}.pt"
    artifact = {
        "schema_version": 1,
        "method": method,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": sha256(config.data_path),
        "device": str(device),
        "platform": platform.platform(),
        "labels": labels,
        "stages": stage_results,
        "forgetting_matrix": matrix,
        "continual": continual,
        "annotation_queue": annotation_queue,
        "checkpoint": str(final_checkpoint),
    }
    atomic_json(run_dir / "metrics.json", artifact)
    return artifact


def run_experiments(
    config: Config, methods: list[str] | None = None, resume: str | None = None
) -> dict:
    data_path = Path(config.data_path)
    if not data_path.exists():
        write_events(data_path, config.seed)
    events = load_events(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = methods or config.methods
    if resume and len(selected) != 1:
        raise ValueError("resume requires exactly one method")
    artifacts = [run_method(method, config, events, device, resume) for method in selected]
    candidates = [
        {
            "method": item["method"],
            "checkpoint": item["checkpoint"],
            "metrics": item["continual"],
            "dataset_sha256": item["dataset_sha256"],
        }
        for item in artifacts
        if item["continual"] and item["method"] in CORE_METHODS
    ]
    ablations = [
        {
            "method": item["method"],
            "checkpoint": item["checkpoint"],
            "metrics": item["continual"],
            "dataset_sha256": item["dataset_sha256"],
        }
        for item in artifacts
        if item["continual"] and item["method"] in ABLATION_METHODS
    ]
    registry = update_registry(config.registry_path, candidates) if candidates else {}
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "methods": candidates,
        "registry": registry,
        "ablations": ablations,
    }
    atomic_json(Path(config.artifact_dir) / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible VelaCL experiments")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--method", action="append", choices=sorted(VALID_METHODS))
    parser.add_argument("--resume", help="resume a single method after the checkpoint stage")
    args = parser.parse_args()
    result = run_experiments(load_config(args.config), args.method, args.resume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
