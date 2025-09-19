import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import pandas as pd


TS_RE = re.compile(
    r"^(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"(?P<day>\d{1,2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+"
    r"(?P<rest>.*)$"
)

LINE_RE = re.compile(r'^\s*"(?P<msg>.*)"\s*,\s*(?P<label>-?\d+)\s*$')


MONTHS = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
}


@dataclass
class LogRecord:
    ts: datetime
    host: str
    msg: str
    label: int  # -1, 0, 1


def parse_line(line: str, default_year: int) -> Optional[Tuple[str, int, Optional[datetime], Optional[str]]]:
    m = LINE_RE.match(line)
    if not m:
        return None
    msg = m.group('msg')
    label = int(m.group('label'))
    tm = TS_RE.match(msg)
    if not tm:
        return msg, label, None, None
    mon = tm.group('mon')
    day = int(tm.group('day'))
    time_s = tm.group('time')
    host = tm.group('host')
    try:
        ts = datetime.strptime(f"{default_year}-{MONTHS[mon]:02d}-{day:02d} {time_s}", "%Y-%m-%d %H:%M:%S")
    except Exception:
        ts = None
    return msg, label, ts, host


def read_labeled_log(path: str, default_year: int) -> List[LogRecord]:
    out: List[LogRecord] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            parsed = parse_line(line, default_year)
            if not parsed:
                continue
            msg, label, ts, host = parsed
            if ts is None:
                # skip lines without parsable timestamp
                continue
            out.append(LogRecord(ts=ts, host=host or "", msg=msg, label=label))
    out.sort(key=lambda r: (r.host, r.ts))
    return out


def sessions_by_gap(
    records: List[LogRecord],
    gap_seconds: int = 300,
    group_by_host: bool = True,
    drop_label_neg1: bool = False,
    map_neg1_to: int = 0,
    keep_unknown_sessions: bool = True,
) -> List[Tuple[List[str], Optional[int], Optional[str]]]:
    sessions: List[Tuple[List[str], Optional[int], Optional[str]]] = []
    current_msgs: List[str] = []
    current_labels_raw: List[int] = []  # keep original labels {-1,0,1}
    prev_ts: Optional[datetime] = None
    prev_host: Optional[str] = None
    gap = timedelta(seconds=gap_seconds)

    def flush():
        nonlocal current_msgs, current_labels_raw
        if not current_msgs:
            return
        has_pos = any(l == 1 for l in current_labels_raw)
        has_zero = any(l == 0 for l in current_labels_raw)
        has_unknown = any(l == -1 for l in current_labels_raw)
        if has_pos:
            sess_label: Optional[int] = 1
        elif has_zero:
            sess_label = 0
        elif has_unknown and keep_unknown_sessions:
            sess_label = None
        else:
            # default to normal if no signal
            sess_label = 0
        sessions.append((current_msgs, sess_label, prev_host))
        current_msgs, current_labels_raw = [], []

    for r in records:
        l_raw = r.label
        if l_raw == -1 and drop_label_neg1:
            # skip completely
            continue

        if prev_ts is None:
            prev_ts, prev_host = r.ts, r.host
            current_msgs.append(r.msg)
            current_labels_raw.append(l_raw if l_raw in (-1, 0, 1) else 0)
            continue

        host_changed = (r.host != prev_host) if group_by_host else False
        time_gap = (r.ts - prev_ts) > gap

        if host_changed or time_gap:
            flush()
            prev_ts, prev_host = r.ts, r.host
        else:
            prev_ts = r.ts
            if group_by_host:
                prev_host = r.host
        current_msgs.append(r.msg)
        current_labels_raw.append(l_raw if l_raw in (-1, 0, 1) else 0)

    flush()
    return sessions


def sessions_by_fixed_window(
    records: List[LogRecord],
    window_seconds: int = 300,
    step_seconds: Optional[int] = None,
    group_by_host: bool = True,
    drop_label_neg1: bool = False,
    map_neg1_to: int = 0,
    keep_unknown_sessions: bool = True,
) -> List[Tuple[List[str], Optional[int], Optional[str]]]:
    if step_seconds is None:
        step_seconds = window_seconds

    # Bucket records by host if requested
    buckets = {}
    for r in records:
        buckets.setdefault(r.host if group_by_host else "__all__", []).append(r)

    sessions: List[Tuple[List[str], Optional[int], Optional[str]]] = []
    for host_key, recs in buckets.items():
        recs.sort(key=lambda x: x.ts)
        if not recs:
            continue
        start_time = recs[0].ts
        end_time = recs[-1].ts
        cur = start_time
        w = timedelta(seconds=window_seconds)
        s = timedelta(seconds=step_seconds)
        while cur <= end_time:
            w_end = cur + w
            msgs: List[str] = []
            labels_raw: List[int] = []
            for r in recs:
                if cur <= r.ts < w_end:
                    l = r.label
                    if l == -1 and drop_label_neg1:
                        continue
                    labels_raw.append(l)
                    msgs.append(r.msg)
            if msgs:
                has_pos = any(l == 1 for l in labels_raw)
                has_zero = any(l == 0 for l in labels_raw)
                has_unknown = any(l == -1 for l in labels_raw)
                if has_pos:
                    sess_label: Optional[int] = 1
                elif has_zero:
                    sess_label = 0
                elif has_unknown and keep_unknown_sessions:
                    sess_label = None
                else:
                    sess_label = 0
                sessions.append((msgs, sess_label, host_key if group_by_host else None))
            cur += s
    return sessions


def write_csv(sessions: List[Tuple[List[str], int]], out_csv: str) -> None:
    df = pd.DataFrame({
        "Content": [" ;-; ".join(s[0]) for s in sessions],
        "Label": [s[1] for s in sessions],
    })
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)


def main():
    # Inputs and options via env vars for convenience
    in_path = os.environ.get("INPUT_LOG", "dataset/raw_log_1_labeled.log")
    out_dir = os.environ.get("OUTPUT_DIR", "dataset")
    # Default to 2025 if YEAR not provided, but allow env override
    year = int(os.environ.get("YEAR", 2025))
    mode = os.environ.get("MODE", "gap")  # gap | fixed
    group_by_host = os.environ.get("GROUP_BY_HOST", "true").lower() == "true"
    drop_label_neg1 = os.environ.get("DROP_LABEL_NEG1", "false").lower() == "true"
    map_neg1_to = int(os.environ.get("MAP_NEG1_TO", 0))
    gap_seconds = int(os.environ.get("GAP_SECONDS", 300))
    window_seconds = int(os.environ.get("WINDOW_SECONDS", 300))
    step_seconds = os.environ.get("STEP_SECONDS")
    step_seconds = int(step_seconds) if step_seconds is not None else None
    train_ratio = float(os.environ.get("TRAIN_RATIO", 0.8))
    stratified = os.environ.get("STRATIFIED_SPLIT", "false").lower() == "true"

    records = read_labeled_log(in_path, default_year=year)
    if mode == "gap":
        sessions = sessions_by_gap(
            records,
            gap_seconds=gap_seconds,
            group_by_host=group_by_host,
            drop_label_neg1=drop_label_neg1,
            map_neg1_to=map_neg1_to,
            keep_unknown_sessions=True,
        )
    else:
        sessions = sessions_by_fixed_window(
            records,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            group_by_host=group_by_host,
            drop_label_neg1=drop_label_neg1,
            map_neg1_to=map_neg1_to,
            keep_unknown_sessions=True,
        )

    # Separate labeled vs unknown sessions
    labeled = [(msgs, int(lbl), host) for msgs, lbl, host in sessions if lbl is not None]
    unknown = [(msgs, host) for msgs, lbl, host in sessions if lbl is None]

    # Split train/test (only labeled ones)
    if stratified:
        # Keep label ratios similar in train and test while preserving chronological order within each class
        labeled_with_idx = [(i, s[0], s[1], s[2]) for i, s in enumerate(labeled)]  # (orig_idx, msgs, label, host)
        pos = [x for x in labeled_with_idx if x[2] == 1]
        neg = [x for x in labeled_with_idx if x[2] == 0]

        n_train_pos = int(len(pos) * train_ratio)
        n_train_neg = int(len(neg) * train_ratio)

        train_idx = set([i for (i, _, _, _) in pos[:n_train_pos]] + [i for (i, _, _, _) in neg[:n_train_neg]])

        train_sessions = [(msgs, int(lbl), host) for (i, msgs, lbl, host) in labeled_with_idx if i in train_idx]
        test_sessions = [(msgs, int(lbl), host) for (i, msgs, lbl, host) in labeled_with_idx if i not in train_idx]
    else:
        # Chronological split by session order
        split = int(len(labeled) * train_ratio)
        train_sessions = labeled[:split]
        test_sessions = labeled[split:]

    # Write labeled with Host column
    df_train = pd.DataFrame({
        'Content': [" ;-; ".join(x[0]) for x in train_sessions],
        'Label': [x[1] for x in train_sessions],
        'Host': [x[2] for x in train_sessions],
    })
    df_test = pd.DataFrame({
        'Content': [" ;-; ".join(x[0]) for x in test_sessions],
        'Label': [x[1] for x in test_sessions],
        'Host': [x[2] for x in test_sessions],
    })
    os.makedirs(out_dir, exist_ok=True)
    df_train.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    df_test.to_csv(os.path.join(out_dir, 'test.csv'), index=False)

    # Write unknown sessions for later pseudo-labeling/inference
    if unknown:
        df_unknown = pd.DataFrame({
            "Content": [" ;-; ".join(s) for s, _ in unknown],
            "Host": [h for _, h in unknown]
        })
        df_unknown.to_csv(os.path.join(out_dir, "unlabeled_sessions.csv"), index=False)

    print(f"Read {len(records)} labeled messages from {in_path}")
    print(f"Built {len(sessions)} sessions (mode={mode}, group_by_host={group_by_host})")
    if stratified:
        print("Using STRATIFIED_SPLIT=true: preserved label ratios in train/test.")
    print(f"Labeled sessions: {len(labeled)}, Unknown sessions: {len(unknown)}")
    print(f"Wrote {len(train_sessions)} train sessions → {os.path.join(out_dir, 'train.csv')}")
    print(f"Wrote {len(test_sessions)} test sessions → {os.path.join(out_dir, 'test.csv')}")
    if unknown:
        print(f"Wrote {len(unknown)} unlabeled sessions → {os.path.join(out_dir, 'unlabeled_sessions.csv')}")


if __name__ == "__main__":
    main()
