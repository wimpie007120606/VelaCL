from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    max_length: int = 96
    d_model: int = 48
    nhead: int = 4
    num_layers: int = 1
    dim_feedforward: int = 96
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    replay_capacity: int = 64
    annotation_budget: int = 8
    gradient_accumulation: int = 1
    mixed_precision: bool = False


@dataclass
class Config:
    seed: int = 42
    data_path: str = "data/streaming/events.jsonl"
    artifact_dir: str = "experiments/runs"
    registry_path: str = "experiments/registry.json"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    methods: list[str] = field(
        default_factory=lambda: [
            "static",
            "naive",
            "random_replay",
            "active_balanced_replay",
            "active_uncertainty_replay",
            "active_diversity_replay",
        ]
    )


def load_config(path: str | Path) -> Config:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config(
        seed=int(raw.get("seed", 42)),
        data_path=raw.get("data_path", "data/streaming/events.jsonl"),
        artifact_dir=raw.get("artifact_dir", "experiments/runs"),
        registry_path=raw.get("registry_path", "experiments/registry.json"),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        methods=raw.get("methods", Config().methods),
    )
