"""Deterministic demo data for a learning-content marketplace."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np


TOPICS = {
    "recsys": ("Recommender Systems", "ranking retrieval embeddings personalization slate"),
    "mlops": ("MLOps", "deployment registry monitoring rollback reproducibility"),
    "vision": ("Computer Vision", "detection segmentation camera edge inference"),
    "data": ("Data Engineering", "streaming warehouse quality lineage pipelines"),
    "nlp": ("Language AI", "transformers retrieval agents evaluation prompting"),
    "product": ("Product Analytics", "experiments metrics retention causal funnels"),
    "backend": ("Backend Systems", "apis databases caching queues reliability"),
    "responsible": ("Responsible AI", "fairness privacy safety explainability governance"),
}

FORMATS = ("Lab", "Case study", "Deep dive", "Field guide")
DIFFICULTIES = ("Foundation", "Applied", "Advanced")
TITLE_PATTERNS = (
    "From Signals to Systems",
    "The Production Playbook",
    "Failure Modes in Practice",
    "A Decision-First Blueprint",
    "Metrics That Change the Answer",
    "Build, Measure, Repair",
    "Under the Hood",
    "The No-Leakage Workshop",
    "Operating at Real Constraints",
    "A Senior Engineer's Review",
)
ACTION_WEIGHT = {"complete": 3.0, "save": 2.0, "open": 0.55, "dismiss": -1.5}


def build_demo_dataset(seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    """Create reproducible users, catalog items, and time-ordered implicit events."""

    rng = np.random.default_rng(seed)
    reference = datetime(2026, 8, 1, 12, tzinfo=UTC)
    topic_keys = tuple(TOPICS)
    creators = tuple(f"Studio {chr(65 + index)}" for index in range(20))

    items: list[dict] = []
    for topic_index, topic in enumerate(topic_keys):
        topic_name, keywords = TOPICS[topic]
        for local_index, pattern in enumerate(TITLE_PATTERNS):
            item_index = topic_index * len(TITLE_PATTERNS) + local_index
            item_format = FORMATS[(topic_index + local_index) % len(FORMATS)]
            difficulty_index = (topic_index * 2 + local_index) % len(DIFFICULTIES)
            age_days = int((item_index * 11 + topic_index * 7) % 180)
            quality = round(0.72 + 0.25 * float(rng.beta(5, 2)), 3)
            duration = int(18 + ((item_index * 17) % 88))
            title = f"{topic_name}: {pattern}"
            items.append(
                {
                    "item_id": f"L{item_index + 1:03d}",
                    "title": title,
                    "topic": topic,
                    "topic_name": topic_name,
                    "format": item_format,
                    "difficulty": difficulty_index + 1,
                    "difficulty_name": DIFFICULTIES[difficulty_index],
                    "duration_min": duration,
                    "creator": creators[(item_index * 7 + topic_index) % len(creators)],
                    "quality": quality,
                    "age_days": age_days,
                    "published_at": (reference - timedelta(days=age_days)).isoformat(),
                    "description": (
                        f"A {item_format.lower()} on {topic_name.lower()} that connects "
                        f"{keywords.replace(' ', ', ')} to an operational decision."
                    ),
                    "text": f"{title} {topic_name} {item_format} {keywords}",
                }
            )

    first_names = (
        "Avery", "Mina", "Noah", "Yuna", "Eli", "Iris", "Leo", "Nora",
        "Kai", "Rina", "Theo", "Sora", "Milo", "Lena", "Owen", "Aya",
    )
    roles = ("ML engineer", "Data analyst", "Backend engineer", "Product scientist")
    users: list[dict] = []
    for index in range(64):
        primary = topic_keys[index % len(topic_keys)]
        secondary = topic_keys[(index * 3 + 2) % len(topic_keys)]
        if secondary == primary:
            secondary = topic_keys[(index + 3) % len(topic_keys)]
        users.append(
            {
                "user_id": f"U{index + 1:03d}",
                "name": f"{first_names[index % len(first_names)]} {index + 1:02d}",
                "role": roles[index % len(roles)],
                "primary_topic": primary,
                "secondary_topic": secondary,
                "preferred_format": FORMATS[(index * 5) % len(FORMATS)],
                "level": index % 3 + 1,
                "time_budget_min": (25, 45, 70, 100)[index % 4],
            }
        )

    events: list[dict] = []
    for user_index, user in enumerate(users):
        remaining = list(range(len(items)))
        start = reference - timedelta(days=145 - user_index % 9)
        for step in range(34):
            utilities = []
            for item_index in remaining:
                item = items[item_index]
                topic_match = 1.0 if item["topic"] == user["primary_topic"] else 0.55 if item["topic"] == user["secondary_topic"] else 0.0
                format_match = float(item["format"] == user["preferred_format"])
                difficulty_match = 1.0 - abs(item["difficulty"] - user["level"]) / 2
                duration_match = max(0.0, 1.0 - abs(item["duration_min"] - user["time_budget_min"]) / 100)
                utilities.append(2.6 * topic_match + 0.55 * format_match + 0.6 * difficulty_match + 0.35 * duration_match + 0.8 * item["quality"])
            logits = np.asarray(utilities) / 0.9
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            chosen_position = int(rng.choice(len(remaining), p=probabilities))
            chosen_index = remaining.pop(chosen_position)
            item = items[chosen_index]

            topic_match = 1.0 if item["topic"] == user["primary_topic"] else 0.6 if item["topic"] == user["secondary_topic"] else 0.0
            affinity = 0.24 + 0.42 * topic_match + 0.14 * float(item["format"] == user["preferred_format"]) + 0.2 * item["quality"]
            draw = float(rng.random())
            if draw < affinity * 0.42:
                action = "complete"
            elif draw < affinity * 0.78:
                action = "save"
            elif draw < min(0.94, affinity + 0.16):
                action = "open"
            else:
                action = "dismiss"

            occurred_at = start + timedelta(days=step * 4, hours=(user_index * 3 + step * 5) % 24)
            events.append(
                {
                    "event_id": f"E{len(events) + 1:05d}",
                    "user_id": user["user_id"],
                    "item_id": item["item_id"],
                    "action": action,
                    "weight": ACTION_WEIGHT[action],
                    "occurred_at": occurred_at.isoformat(),
                }
            )

    events.sort(key=lambda event: (event["occurred_at"], event["event_id"]))
    return items, users, events


def temporal_split(events: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split each user's event stream into 60% history, 20% ranker, 20% evaluation."""

    grouped: dict[str, list[dict]] = {}
    for event in events:
        grouped.setdefault(event["user_id"], []).append(event)

    history: list[dict] = []
    ranker: list[dict] = []
    evaluation: list[dict] = []
    for user_events in grouped.values():
        ordered = sorted(user_events, key=lambda event: event["occurred_at"])
        history_end = int(len(ordered) * 0.60)
        ranker_end = int(len(ordered) * 0.80)
        history.extend(ordered[:history_end])
        ranker.extend(ordered[history_end:ranker_end])
        evaluation.extend(ordered[ranker_end:])
    return history, ranker, evaluation

