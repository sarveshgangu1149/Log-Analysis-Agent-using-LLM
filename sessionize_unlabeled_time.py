import os
import re
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

MONTHS = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
}


def parse_line(line: str, default_year: int) -> Optional[Tuple[datetime, str, str]]:
    line = line.strip().strip('"')
    m = TS_RE.match(line)
    if not m:
        return None
    mon = m.group('mon')
    day = int(m.group('day'))
    time_s = m.group('time')
    host = m.group('host')
    try:
        ts = datetime.strptime(f"{default_year}-{MONTHS[mon]:02d}-{day:02d} {time_s}", "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None
    return ts, host, line


def read_log(path: str, default_year: int) -> List[Tuple[datetime, str, str]]:
    out: List[Tuple[datetime, str, str]] = []
    with open(path, 'r') as f:
        for line in f:
            parsed = parse_line(line, default_year)
            if parsed:
                out.append(parsed)
    out.sort(key=lambda x: (x[1], x[0]))  # by host, then ts
    return out


def sessions_by_gap(records: List[Tuple[datetime, str, str]], gap_seconds: int = 300, group_by_host: bool = True) -> List[List[str]]:
    sessions: List[List[str]] = []
    current: List[str] = []
    prev_ts: Optional[datetime] = None
    prev_host: Optional[str] = None
    gap = timedelta(seconds=gap_seconds)

    def flush():
        nonlocal current
        if current:
            sessions.append(current)
            current = []

    for ts, host, msg in records:
        if prev_ts is None:
            prev_ts, prev_host = ts, host
            current.append(msg)
            continue
        host_changed = (host != prev_host) if group_by_host else False
        time_gap = (ts - prev_ts) > gap
        if host_changed or time_gap:
            flush()
            prev_ts, prev_host = ts, host
        else:
            prev_ts = ts
            if group_by_host:
                prev_host = host
        current.append(msg)

    flush()
    return sessions


def sessions_by_fixed_window(records: List[Tuple[datetime, str, str]], window_seconds: int = 300, step_seconds: Optional[int] = None, group_by_host: bool = True) -> List[List[str]]:
    if step_seconds is None:
        step_seconds = window_seconds
    # bucket per host
    buckets = {}
    for ts, host, msg in records:
        buckets.setdefault(host if group_by_host else "__all__", []).append((ts, msg))

    sessions: List[List[str]] = []
    for _, recs in buckets.items():
        recs.sort(key=lambda x: x[0])
        if not recs:
            continue
        start_time = recs[0][0]
        end_time = recs[-1][0]
        w = timedelta(seconds=window_seconds)
        s = timedelta(seconds=step_seconds)
        cur = start_time
        while cur <= end_time:
            w_end = cur + w
            msgs = [msg for ts, msg in recs if cur <= ts < w_end]
            if msgs:
                sessions.append(msgs)
            cur += s
    return sessions


def main():
    in_path = os.environ.get("INPUT_LOG", "dataset/recent_unlabeled.log")
    out_csv = os.environ.get("OUTPUT_CSV", "dataset/unlabeled_sessions.csv")
    year = int(os.environ.get("YEAR", datetime.now().year))
    mode = os.environ.get("MODE", "gap")
    group_by_host = os.environ.get("GROUP_BY_HOST", "true").lower() == "true"
    gap_seconds = int(os.environ.get("GAP_SECONDS", 300))
    window_seconds = int(os.environ.get("WINDOW_SECONDS", 300))
    step_seconds = os.environ.get("STEP_SECONDS")
    step_seconds = int(step_seconds) if step_seconds is not None else None

    records = read_log(in_path, year)
    if mode == "gap":
        sessions = sessions_by_gap(records, gap_seconds=gap_seconds, group_by_host=group_by_host)
    else:
        sessions = sessions_by_fixed_window(records, window_seconds=window_seconds, step_seconds=step_seconds, group_by_host=group_by_host)

    df = pd.DataFrame({
        "Content": [" ;-; ".join(s) for s in sessions]
    })
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Read {len(records)} log lines from {in_path}")
    print(f"Built {len(sessions)} unlabeled sessions → {out_csv}")


if __name__ == "__main__":
    main()

