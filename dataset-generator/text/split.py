"""Разделение примеров на выборки без пересечения групп."""

import json
import random


def assign_splits(samples: list[dict], ratios: list[float], seed: int) -> list[dict]:
    groups = {}
    unique = []
    seen = set()
    for sample in samples:
        base = {key: sample[key] for key in ("action_id", "args", "context")}
        group_key = json.dumps(base, ensure_ascii=False, sort_keys=True)
        sample_key = json.dumps({**base, "text": sample["text"]}, ensure_ascii=False, sort_keys=True)
        if sample_key in seen:
            continue
        seen.add(sample_key)
        if group_key not in groups:
            groups[group_key] = f"group_{len(groups) + 1:06d}"
        unique.append({
            **sample,
            "sample_id": f"sample_{len(unique) + 1:06d}",
            "group_id": groups[group_key],
        })

    groups_by_action = {}
    for sample in unique:
        groups_by_action.setdefault(sample["action_id"], set()).add(sample["group_id"])
    rng = random.Random(seed)
    assignments = {}
    for action_id in sorted(groups_by_action):
        group_ids = sorted(groups_by_action[action_id])
        rng.shuffle(group_ids)
        active = [index for index, ratio in enumerate(ratios) if ratio > 0]
        sizes = [len(group_ids) * ratio for ratio in ratios]
        counts = [int(size) for size in sizes]
        remainder = len(group_ids) - sum(counts)
        order = sorted(active, key=lambda i: sizes[i] - counts[i], reverse=True)
        for index in order[:remainder]:
            counts[index] += 1
        for index in active:
            if counts[index] == 0 and len(group_ids) >= len(active):
                donor = max(active, key=lambda i: counts[i])
                counts[donor] -= 1
                counts[index] = 1
        start = 0
        for name, count in zip(("train", "validation", "test"), counts):
            for group_id in group_ids[start:start + count]:
                assignments[group_id] = name
            start += count
    return [{**sample, "split": assignments[sample["group_id"]]} for sample in unique]
