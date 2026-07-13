from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from velacl.config import Config
from velacl.data import Event
from velacl.metrics import classification_metrics, expected_calibration_error, grouped_accuracy
from velacl.model import ByteTransformer, batch_encode, encode_text


class EventDataset(Dataset):
    def __init__(self, events: list[Event], label_to_id: dict[str, int], max_length: int):
        self.events = events
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int):
        event = self.events[index]
        return (
            torch.tensor(encode_text(event.text, self.max_length), dtype=torch.long),
            torch.tensor(self.label_to_id[event.intent], dtype=torch.long),
        )


def train_stage(
    model: ByteTransformer,
    events: list[Event],
    label_to_id: dict[str, int],
    config: Config,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    if not events:
        return {"loss": 0.0, "examples": 0, "throughput": 0.0}
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        EventDataset(events, label_to_id, config.model.max_length),
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.training.mixed_precision and device.type == "cuda"
    )
    model.train()
    losses = []
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(config.training.epochs):
        for batch_index, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss = criterion(model(inputs), labels) / config.training.gradient_accumulation
            scaler.scale(loss).backward()
            if (
                batch_index + 1
            ) % config.training.gradient_accumulation == 0 or batch_index + 1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item() * config.training.gradient_accumulation))
    duration = max(time.perf_counter() - started, 1e-9)
    return {
        "loss": sum(losses) / len(losses),
        "examples": len(events),
        "throughput": len(events) * config.training.epochs / duration,
    }


@torch.no_grad()
def evaluate(
    model: ByteTransformer, events: list[Event], label_to_id: dict[str, int], device: torch.device
) -> dict:
    model.eval()
    predictions: list[int] = []
    truths: list[int] = []
    confidences: list[float] = []
    records: list[dict] = []
    for start in range(0, len(events), 64):
        batch = events[start : start + 64]
        logits = model(
            batch_encode([event.text for event in batch], model.config.max_length, device)
        )
        probabilities = logits.softmax(dim=-1)
        confidence, prediction = probabilities.max(dim=-1)
        for event, pred, conf in zip(batch, prediction.tolist(), confidence.tolist(), strict=True):
            truth = label_to_id[event.intent]
            predictions.append(pred)
            truths.append(truth)
            confidences.append(conf)
            records.append(
                {"language": event.language, "domain": event.domain, "correct": pred == truth}
            )
    metrics = classification_metrics(truths, predictions, sorted(label_to_id.values()))
    metrics["calibration_error"] = expected_calibration_error(
        confidences, [p == t for p, t in zip(predictions, truths, strict=True)]
    )
    metrics["by_language"] = grouped_accuracy(records, "language")
    metrics["by_domain"] = grouped_accuracy(records, "domain")
    return metrics


def save_checkpoint(
    path: str | Path,
    model: ByteTransformer,
    labels: list[str],
    config: Config,
    stage: int,
    method: str,
    replay_events: list[Event] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "model_config": asdict(config.model),
            "stage": stage,
            "method": method,
            "replay_events": [asdict(event) for event in replay_events or []],
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[ByteTransformer, dict]:
    from velacl.config import ModelConfig

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ByteTransformer(ModelConfig(**checkpoint["model_config"]), len(checkpoint["labels"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, checkpoint


def predict(
    model: ByteTransformer, texts: list[str], labels: list[str], device: torch.device
) -> list[dict]:
    model.eval()
    with torch.no_grad():
        probabilities = model(batch_encode(texts, model.config.max_length, device)).softmax(dim=-1)
    results = []
    for text, row in zip(texts, probabilities, strict=True):
        confidence, index = row.max(dim=-1)
        results.append(
            {
                "text": text,
                "intent": labels[int(index)],
                "confidence": float(confidence),
                "scores": {label: float(score) for label, score in zip(labels, row, strict=True)},
            }
        )
    return results
