from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from velacl.data import Event
from velacl.model import ByteTransformer, batch_encode


@dataclass
class Selection:
    event: Event
    score: float
    uncertainty: float
    diversity: float
    language_scarcity: float
    intent_novelty: float
    forgetting_risk: float
    safety_importance: float
    reason: str


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.events: list[Event] = []

    def random_update(self, incoming: list[Event]) -> None:
        population = self.events + incoming
        self.events = self.rng.sample(population, min(self.capacity, len(population)))

    def balanced_update(self, incoming: list[Event]) -> None:
        population = self.events + incoming
        groups: dict[tuple[str, str], list[Event]] = defaultdict(list)
        for event in population:
            groups[(event.language, event.intent)].append(event)
        for values in groups.values():
            self.rng.shuffle(values)
        result: list[Event] = []
        while len(result) < min(self.capacity, len(population)) and groups:
            for key in sorted(list(groups)):
                if groups[key] and len(result) < self.capacity:
                    result.append(groups[key].pop())
                if not groups[key]:
                    del groups[key]
        self.events = result


@torch.no_grad()
def active_balanced_select(
    model: ByteTransformer,
    events: list[Event],
    memory: list[Event],
    label_to_id: dict[str, int],
    budget: int,
    device: torch.device,
) -> list[Selection]:
    if not events:
        return []
    model.eval()
    ids = batch_encode([event.text for event in events], model.config.max_length, device)
    logits, embeddings = model(ids, return_embedding=True)
    probabilities = logits.softmax(dim=-1)
    uncertainty = 1.0 - probabilities.max(dim=-1).values.cpu().numpy()
    embeddings_np = embeddings.cpu().numpy()
    if memory:
        memory_ids = batch_encode([event.text for event in memory], model.config.max_length, device)
        _, memory_embeddings = model(memory_ids, return_embedding=True)
        distances = np.linalg.norm(
            embeddings_np[:, None] - memory_embeddings.cpu().numpy()[None, :], axis=2
        )
        diversity = distances.min(axis=1)
        diversity /= max(float(diversity.max()), 1e-8)
    else:
        diversity = np.ones(len(events))
    language_counts = Counter(event.language for event in memory)
    intent_counts = Counter(event.intent for event in memory)
    max_language = max(language_counts.values(), default=1)
    max_intent = max(intent_counts.values(), default=1)
    selections = []
    for index, event in enumerate(events):
        language_scarcity = 1 - language_counts[event.language] / max_language
        intent_novelty = 1 - intent_counts[event.intent] / max_intent
        prediction = int(probabilities[index].argmax().item())
        forgetting_risk = float(prediction != label_to_id[event.intent])
        safety = float(event.safety_risk or event.privacy_risk)
        parts = {
            "uncertainty": float(uncertainty[index]),
            "diversity": float(diversity[index]),
            "language scarcity": language_scarcity,
            "intent novelty": intent_novelty,
            "forgetting risk": forgetting_risk,
            "safety importance": safety,
        }
        score = (
            0.30 * parts["uncertainty"]
            + 0.20 * parts["diversity"]
            + 0.15 * parts["language scarcity"]
            + 0.15 * parts["intent novelty"]
            + 0.10 * parts["forgetting risk"]
            + 0.10 * parts["safety importance"]
        )
        reason = max(parts, key=parts.get)
        selections.append(Selection(event, score, *parts.values(), reason))
    return sorted(selections, key=lambda value: (-value.score, value.event.id))[:budget]
