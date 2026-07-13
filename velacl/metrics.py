from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]
) -> dict[str, float]:
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    scores = []
    for label in labels:
        tp = int(np.sum((true == label) & (pred == label)))
        fp = int(np.sum((true != label) & (pred == label)))
        fn = int(np.sum((true == label) & (pred != label)))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    accuracy = float(np.mean(true == pred))
    return {"accuracy": accuracy, "macro_f1": float(np.mean(scores)), "micro_f1": accuracy}


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> float:
    if not confidences:
        return 0.0
    conf = np.asarray(confidences)
    corr = np.asarray(correct, dtype=float)
    ece = 0.0
    for low in np.linspace(0, 1, bins, endpoint=False):
        mask = (conf > low) & (conf <= low + 1 / bins)
        if mask.any():
            ece += float(mask.mean() * abs(corr[mask].mean() - conf[mask].mean()))
    return ece


def continual_metrics(matrix: list[list[float]]) -> dict[str, float]:
    values = np.asarray(matrix, dtype=float)
    if values.size == 0:
        return {"average_accuracy": 0.0, "average_forgetting": 0.0, "backward_transfer": 0.0}
    final = values[:, -1]
    learned = []
    forgetting = []
    backward = []
    for task in range(values.shape[0]):
        start = min(task, values.shape[1] - 1)
        learned.append(final[task])
        history = values[task, start:]
        forgetting.append(max(0.0, float(history.max() - final[task])))
        backward.append(float(final[task] - values[task, start]))
    return {
        "average_accuracy": float(np.mean(learned)),
        "average_forgetting": float(np.mean(forgetting)),
        "backward_transfer": float(np.mean(backward)),
    }


def grouped_accuracy(records: list[dict], key: str) -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record["correct"])
    return {name: sum(values) / len(values) for name, values in sorted(grouped.items())}
