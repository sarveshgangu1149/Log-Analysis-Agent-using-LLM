**Slide 1: From Labeled Logs → Sessions**
- **Input Source:** `dataset/raw_log_1_labeled.log` with lines like `"<message>", <label>` where messages begin with `<Mon> <DD> <HH:MM:SS> <host> ...` and label in `{-1,0,1}`.
- **Parsing:** Extracts timestamp + host; drops lines without parsable timestamps; sorts by `(host, ts)` for stable chronology. Code: `prepare_time_sessions.py:34`, `prepare_time_sessions.py:54`, `prepare_time_sessions.py:67`.
- **Two Sessionization Modes:**
  - **Time Gap Sessions:** New session when host changes or time gap `> GAP_SECONDS` (default 300s). Code: `prepare_time_sessions.py:71`, `prepare_time_sessions.py:117`, `prepare_time_sessions.py:118`.
  - **Fixed Window Sessions:** Sliding windows of `WINDOW_SECONDS` with step `STEP_SECONDS` (default = window; overlap if step < window) over each host’s timeline. Code: `prepare_time_sessions.py:134`, `prepare_time_sessions.py:161`.
- **Host Grouping:** By default, sessions don’t cross hosts (`GROUP_BY_HOST=true`). Code: `prepare_time_sessions.py:205`, `prepare_time_sessions.py:149`.
- **Unknown Lines Handling:** Can drop `label == -1` lines via `DROP_LABEL_NEG1=true`. Code: `prepare_time_sessions.py:206`, `prepare_time_sessions.py:107`, `prepare_time_sessions.py:168`.
- **Config Via Env Vars:** `MODE={gap|fixed}`, `GAP_SECONDS`, `WINDOW_SECONDS`, `STEP_SECONDS`, `GROUP_BY_HOST`, `TRAIN_RATIO`, `STRATIFIED_SPLIT`, `YEAR`. Code: `prepare_time_sessions.py:198`–`prepare_time_sessions.py:213`.

Example commands:
- Gap-based: `MODE=gap GAP_SECONDS=300 GROUP_BY_HOST=true YEAR=2025 python prepare_time_sessions.py`
- Fixed window: `MODE=fixed WINDOW_SECONDS=300 STEP_SECONDS=300 GROUP_BY_HOST=true YEAR=2025 python prepare_time_sessions.py`

**Slide 2: Session Labels, Split, and Outputs**
- **Session Label Aggregation:** Priority: if any line has label `1` → session label `1`; else if any `0` → `0`; else if any `-1` → session label `None` (unknown); else default `0`. Gap mode code: `prepare_time_sessions.py:90`–`prepare_time_sessions.py:101`; Fixed-window code: `prepare_time_sessions.py:173`–`prepare_time_sessions.py:184`.
- **Train/Test Only Use Labeled Sessions:** Unknown sessions (label `None`) are excluded from train/test and saved separately. Code: `prepare_time_sessions.py:236`–`prepare_time_sessions.py:268`.
- **Split Strategy:**
  - **Chronological split** by session order (default) using `TRAIN_RATIO` (default 0.8). Code: `prepare_time_sessions.py:255`–`prepare_time_sessions.py:258`.
  - Optional **stratified split** to preserve class ratios while keeping per-class chronology. Toggle with `STRATIFIED_SPLIT=true`. Code: `prepare_time_sessions.py:241`–`prepare_time_sessions.py:253`.
- **CSV Schema:** `train.csv` / `test.csv` contain:
  - `Content`: session lines joined by separator `" ;-; "`.
  - `Label`: session label in `{0,1}`.
  Code: `prepare_time_sessions.py:189`–`prepare_time_sessions.py:195`.
- **Artifacts:**
  - Labeled: `dataset/train.csv`, `dataset/test.csv`.
  - Unlabeled sessions (for inference/pseudo-labeling): `dataset/unlabeled_sessions.csv`. Code: `prepare_time_sessions.py:264`–`prepare_time_sessions.py:268`.
- **Downstream Use:** `train.py` reads `dataset/train.csv` to fine-tune the model; `predict_sessions.py` and `summarize_anomalies.py` operate on session `Content` built with the same `" ;-; "` separator.

Tip: Set `STEP_SECONDS < WINDOW_SECONDS` in fixed mode to create overlapping windows for higher recall; increase `GAP_SECONDS` in gap mode to merge sparser events into longer sessions.

