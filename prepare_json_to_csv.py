import json
import os
from typing import List, Dict, Tuple
import pandas as pd


def load_json(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        return json.load(f)


def to_sessions(
    records: List[Dict],
    window_size: int = 50,
    step_size: int = 50,
    drop_label_neg1: bool = False,
    map_neg1_to: int = 0,
) -> List[Tuple[List[str], int]]:
    """
    Convert flat message records into fixed-size sessions.

    - records: list of {"query": str, "label": int}
    - window_size: number of messages per session
    - step_size: slide step (set == window_size for non-overlap)
    - drop_label_neg1: if True, drop messages with label == -1
    - map_neg1_to: if not dropping, map -1 -> map_neg1_to (0 or 1)

    Returns list of (messages_list, session_label) where session_label is 1 if any message in the window is 1 else 0.
    """
    msgs: List[str] = []
    labels: List[int] = []

    for r in records:
        q = r.get("query", "").strip()
        l = r.get("label", 0)
        if l == -1:
            if drop_label_neg1:
                continue
            l = map_neg1_to
        # normalize to 0/1
        l = 1 if l == 1 else 0
        if q:
            msgs.append(q)
            labels.append(l)

    sessions: List[Tuple[List[str], int]] = []
    n = len(msgs)
    for start in range(0, n, step_size):
        end = min(start + window_size, n)
        if start >= end:
            break
        chunk_msgs = msgs[start:end]
        chunk_labels = labels[start:end]
        sess_label = 1 if any(x == 1 for x in chunk_labels) else 0
        sessions.append((chunk_msgs, sess_label))
    return sessions


def write_csv(sessions: List[Tuple[List[str], int]], out_csv: str) -> None:
    df = pd.DataFrame({
        "Content": [" ;-; ".join(s[0]) for s in sessions],
        "Label": [s[1] for s in sessions],
    })
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)


def main():
    in_path = os.environ.get("INPUT_JSON", "dataset/input_data.json")
    out_dir = os.environ.get("OUTPUT_DIR", "dataset")
    window_size = int(os.environ.get("WINDOW_SIZE", 50))
    step_size = int(os.environ.get("STEP_SIZE", window_size))
    train_ratio = float(os.environ.get("TRAIN_RATIO", 0.8))
    drop_label_neg1 = os.environ.get("DROP_LABEL_NEG1", "false").lower() == "true"
    map_neg1_to = int(os.environ.get("MAP_NEG1_TO", 0))

    records = load_json(in_path)
    sessions = to_sessions(
        records,
        window_size=window_size,
        step_size=step_size,
        drop_label_neg1=drop_label_neg1,
        map_neg1_to=map_neg1_to,
    )

    # split train/test
    split = int(len(sessions) * train_ratio)
    train_sessions = sessions[:split]
    test_sessions = sessions[split:]

    write_csv(train_sessions, os.path.join(out_dir, "train.csv"))
    write_csv(test_sessions, os.path.join(out_dir, "test.csv"))

    print(f"Wrote {len(train_sessions)} train sessions → {os.path.join(out_dir, 'train.csv')}")
    print(f"Wrote {len(test_sessions)} test sessions → {os.path.join(out_dir, 'test.csv')}")


if __name__ == "__main__":
    main()

