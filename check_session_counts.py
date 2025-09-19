import os
import csv
from pathlib import Path
import pandas as pd

SEP = ' ;-; '


def count_nonempty_lines(path: Path) -> int:
    n = 0
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def sum_session_lines(csv_path: Path) -> tuple[int, int]:
    if not csv_path.exists():
        return 0, 0
    df = pd.read_csv(csv_path)
    if 'Content' not in df.columns:
        return 0, 0
    sessions = len(df)
    lines = 0
    for content in df['Content'].fillna(''):
        if content:
            lines += sum(1 for s in str(content).split(SEP) if str(s).strip())
    return sessions, lines


def main():
    base = Path('dataset')
    raw_path = base / 'raw_log_1_labeled.log'

    raw_total = count_nonempty_lines(raw_path) if raw_path.exists() else None

    train_sess, train_lines = sum_session_lines(base / 'train.csv')
    test_sess, test_lines = sum_session_lines(base / 'test.csv')
    unl_sess, unl_lines = sum_session_lines(base / 'unlabeled_sessions.csv')

    print(f'Raw labeled lines: {raw_total}')
    print(f'train.csv -> sessions: {train_sess:5d} | lines: {train_lines:5d}')
    print(f'test.csv  -> sessions: {test_sess:5d} | lines: {test_lines:5d}')
    print(f'unlabeled -> sessions: {unl_sess:5d} | lines: {unl_lines:5d}')
    total = train_lines + test_lines + unl_lines
    print(f'TOTAL sessionized lines: {total}')

    if raw_total is not None:
        if total == raw_total:
            print('OK: Sessionized line count matches raw labeled lines.')
        else:
            print('Mismatch: Sessionized lines != raw lines.')
            print('Common reasons:')
            print(' - Lines without a parsable timestamp were skipped during sessionization.')
            print(' - You used DROP_LABEL_NEG1=true (dropped unknown lines).')
            print(' - File encoding or blank lines differences.')


if __name__ == '__main__':
    main()

