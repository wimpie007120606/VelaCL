from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class Event:
    id: str
    timestamp: str
    stage: int
    split: str
    text: str
    language: str
    domain: str
    intent: str
    source: str
    difficulty: str
    annotation_status: str
    privacy_risk: bool
    safety_risk: bool


# Small, openly redistributable project-owned fixture. It is deliberately a systems
# benchmark, not a claim to represent natural language diversity or production traffic.
PHRASES = {
    "en": {
        "greeting": ["hello, I need help", "good morning support"],
        "balance": ["what is my account balance", "show the money in my account"],
        "payment_failed": ["my card payment failed", "the payment did not go through"],
        "fraud": ["I see a transaction I did not make", "someone used my bank card"],
        "reset_pin": ["I forgot my pin", "help me reset my account pin"],
        "close_account": ["please close my account", "I want to cancel this account"],
        "policy": ["what is the new transfer limit", "explain the updated fee policy"],
        "human_agent": ["let me speak to a person", "connect me to an agent"],
    },
    "af": {
        "greeting": ["hallo, ek het hulp nodig", "goeie more ondersteuning"],
        "balance": ["wat is my rekeningsaldo", "wys die geld in my rekening"],
        "payment_failed": ["my kaartbetaling het misluk", "die betaling het nie deurgegaan nie"],
        "fraud": [
            "ek sien 'n transaksie wat ek nie gemaak het nie",
            "iemand het my bankkaart gebruik",
        ],
        "reset_pin": ["ek het my pin vergeet", "help my om my pin te herstel"],
        "close_account": ["maak asseblief my rekening toe", "ek wil hierdie rekening kanselleer"],
        "policy": ["wat is die nuwe oordraglimiet", "verduidelik die nuwe fooibeleid"],
        "human_agent": ["laat my met 'n persoon praat", "verbind my met 'n agent"],
    },
    "zu": {
        "greeting": ["sawubona, ngidinga usizo", "sawubona bosekelo"],
        "balance": ["ithini ibhalansi ye-akhawunti yami", "ngibonise imali ku-akhawunti"],
        "payment_failed": ["inkokhelo yekhadi yehlulekile", "inkokhelo ayiphumelelanga"],
        "fraud": ["ngibona umsebenzi engingawenzanga", "umuntu usebenzise ikhadi lami"],
        "reset_pin": ["ngikhohlwe iphinikhodi", "ngisize ngishintshe iphinikhodi"],
        "close_account": ["ngicela uvale i-akhawunti", "ngifuna ukukhansela i-akhawunti"],
        "policy": ["uyini umkhawulo omusha wokudlulisa", "chaza inqubomgomo entsha"],
        "human_agent": ["ngifuna ukukhuluma nomuntu", "ngixhumanise nomenzeli"],
    },
    "xh": {
        "greeting": ["molo, ndifuna uncedo", "molweni baxhasi"],
        "balance": ["ithini imali eseakhawuntini yam", "ndibonise imali yam"],
        "payment_failed": ["intlawulo yekhadi ayiphumelelanga", "intlawulo ayidlulanga"],
        "fraud": ["ndibona intlawulo endingayenzanga", "umntu usebenzise ikhadi lam"],
        "reset_pin": ["ndilibele iphini yam", "ndincede nditshintshe iphini"],
        "close_account": ["nceda uvale iakhawunti yam", "ndifuna ukurhoxisa iakhawunti"],
        "policy": ["uthini umda omtsha wodluliselo", "chaza umgaqo omtsha"],
        "human_agent": ["ndifuna ukuthetha nomntu", "ndidibanise nomncedisi"],
    },
    "st": {
        "greeting": ["dumela, ke hloka thuso", "dumela basebetsi"],
        "balance": ["balanse ya akhaonto ke bokae", "mpontshe tjhelete ya ka"],
        "payment_failed": ["tefo ya karete e hlolehile", "tefo ha e a feta"],
        "fraud": ["ke bona tefo eo ke sa e etsang", "motho o sebedisitse karete ya ka"],
        "reset_pin": ["ke lebetse pin", "nthuse ho fetola pin"],
        "close_account": ["ka kopo kwala akhaonto", "ke batla ho hlakola akhaonto"],
        "policy": ["moedi o motjha wa phetiso ke ofe", "hlalosa molao o motjha"],
        "human_agent": ["ke batla ho bua le motho", "nkopanye le moemedi"],
    },
}

STAGES = {
    0: (["en", "af"], ["greeting", "balance", "human_agent"], "support"),
    1: (["en", "af"], ["payment_failed"], "payments"),
    2: (["zu"], ["greeting", "balance", "payment_failed", "human_agent"], "support"),
    3: (["en", "af", "zu"], ["fraud", "reset_pin"], "security"),
    4: (["xh", "st"], ["greeting", "balance", "payment_failed", "fraud", "human_agent"], "support"),
    5: (["en", "af", "zu", "xh", "st"], ["policy", "close_account"], "policy"),
}


def build_events(seed: int = 42) -> list[Event]:
    rng = random.Random(seed)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    events: list[Event] = []
    counter = 0
    for stage, (languages, intents, domain) in STAGES.items():
        for language in languages:
            for intent in intents:
                for phrase in PHRASES[language][intent]:
                    for split in ("train", "test"):
                        # Train/test differ without hiding which base phrase they came from.
                        suffix = rng.choice(["", " please", " now"]) if language == "en" else ""
                        counter += 1
                        events.append(
                            Event(
                                id=f"evt-{counter:04d}",
                                timestamp=(start + timedelta(days=counter)).isoformat(),
                                stage=stage,
                                split=split,
                                text=phrase + (suffix if split == "train" else ""),
                                language=language,
                                domain=domain,
                                intent=intent,
                                source="velacl-curated-v1",
                                difficulty="medium" if intent in {"fraud", "policy"} else "easy",
                                annotation_status="gold",
                                privacy_risk=intent in {"balance", "reset_pin", "fraud"},
                                safety_risk=intent == "fraud",
                            )
                        )
    return events


def write_events(path: str | Path, seed: int = 42) -> list[Event]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = build_events(seed)
    path.write_text(
        "".join(json.dumps(asdict(e), ensure_ascii=False) + "\n" for e in events), encoding="utf-8"
    )
    return events


def load_events(path: str | Path) -> list[Event]:
    return [
        Event(**json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/streaming/events.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"wrote {len(write_events(args.output, args.seed))} events to {args.output}")


if __name__ == "__main__":
    main()
