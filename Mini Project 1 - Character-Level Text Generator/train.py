"""Train and sample a smoothed character-level bigram model."""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "names.txt"
MODEL_PATH = ROOT / "model" / "bigram_model.json"
RESULTS = ROOT / "results"
START = "<START>"
END = "\n"
SEED = 42


def pairs_for(names: list[str]):
    for name in names:
        previous = START
        for char in name.strip().lower():
            yield previous, char
            previous = char
        yield previous, END


def nll(pairs, probabilities):
    pairs = list(pairs)
    return sum(-math.log(probabilities[a][b]) for a, b in pairs) / len(pairs)


def sample(probabilities, rng, limit=16, starter=""):
    chars = list(starter.lower())
    previous = chars[-1] if chars else START
    for _ in range(limit):
        row = probabilities[previous]
        options = list(row)
        weights = list(row.values())
        token = rng.choices(options, weights=weights, k=1)[0]
        if token == END:
            break
        chars.append(token)
        previous = token
    return "".join(chars)


def main():
    names = [line.strip().lower() for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(SEED)
    shuffled = names[:]
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * 0.8))
    train_names, val_names = shuffled[:cut], shuffled[cut:]
    vocab = sorted(set("".join(names)) | {END})
    tokens = [START, *vocab]
    train_pairs, val_pairs = list(pairs_for(train_names)), list(pairs_for(val_names))

    # Before fitting, every possible next token has equal probability.
    uniform = {previous: {token: 1 / len(tokens[1:]) for token in tokens[1:]} for previous in tokens}
    before_train = nll(train_pairs, uniform)
    before_val = nll(val_pairs, uniform)

    counts = defaultdict(Counter)
    for previous, token in train_pairs:
        counts[previous][token] += 1
    # Laplace/add-one smoothing keeps every transition possible.
    probabilities = {}
    for previous in tokens:
        row = counts[previous]
        total = sum(row.values()) + len(vocab)
        probabilities[previous] = {token: (row[token] + 1) / total for token in vocab}

    losses = {
        "metric": "mean negative log-likelihood (nats per next character)",
        "seed": SEED,
        "examples_total": len(names),
        "examples_train": len(train_names),
        "examples_validation": len(val_names),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "untrained_uniform": {"train": before_train, "validation": before_val},
        "trained_smoothed_bigram": {"train": nll(train_pairs, probabilities), "validation": nll(val_pairs, probabilities)},
        "split_note": "Validation names were held out before transition counts were fitted.",
    }
    model = {"kind": "character bigram", "vocabulary": vocab, "start_token": START, "end_token": END,
             "smoothing": "add-one", "probabilities": probabilities}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "losses.json").write_text(json.dumps(losses, indent=2), encoding="utf-8")
    (RESULTS / "losses.csv").write_text(
        "stage,split,mean_nll_nats_per_character\n"
        f"untrained_uniform,train,{before_train:.6f}\n"
        f"untrained_uniform,validation,{before_val:.6f}\n"
        f"trained_smoothed_bigram,train,{losses['trained_smoothed_bigram']['train']:.6f}\n"
        f"trained_smoothed_bigram,validation,{losses['trained_smoothed_bigram']['validation']:.6f}\n", encoding="utf-8")

    before_samples = [sample(uniform, random.Random(SEED + i)) for i in range(10)]
    after_samples = [sample(probabilities, random.Random(SEED + 100 + i)) for i in range(10)]
    starter_samples = [sample(probabilities, random.Random(SEED + 200 + i), starter="a") for i in range(10)]
    out = ["Ten generated samples (after training):"]
    out.extend(f"{i+1:02}. {s or '[empty]'}" for i, s in enumerate(after_samples))
    out += ["", "Before-training uniform baseline (comparison):"]
    out.extend(f"{i+1:02}. {s or '[empty]'}" for i, s in enumerate(before_samples))
    out += ["", "Ten trained samples continued from starter 'a':"]
    out.extend(f"{i+1:02}. {s}" for i, s in enumerate(starter_samples))
    (RESULTS / "samples.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(json.dumps(losses, indent=2))
    print(f"\nWrote {MODEL_PATH.relative_to(ROOT)}, results/losses.json, results/losses.csv, results/samples.txt")


if __name__ == "__main__":
    main()
