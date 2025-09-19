import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import LogLLM
from customDataset import CustomCollator
from prepare_time_sessions import read_labeled_log, sessions_by_gap, sessions_by_fixed_window


class InferenceDataset(Dataset):
    def __init__(self, sessions_csv: str):
        df = pd.read_csv(sessions_csv)
        if 'Content' not in df.columns:
            raise ValueError("Input CSV must have a 'Content' column")
        self.contents = df['Content'].fillna("").tolist()

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        seq = str(self.contents[idx]).split(' ;-; ')
        return seq, 0


def predict_sessions_csv(input_csv: str, output_csv: str, max_content_len: int, max_seq_len: int, batch_size: int):
    root = Path(__file__).parent
    bert_path = str(root / 'bert-base-uncased')
    _llm_env = os.environ.get('LLM_PATH')
    if _llm_env:
        llama_path = _llm_env
    else:
        tiny = root / 'TinyLlama-1.1B-Chat-v1.0'
        llama3 = root / 'Meta-Llama-3-8B'
        llama_path = str(tiny if tiny.exists() else llama3)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model = LogLLM(bert_path, llama_path, ft_path=str(root / 'ft_model_LOCAL'), is_train_mode=False,
                   device=device, max_content_len=max_content_len, max_seq_len=max_seq_len)
    collator = CustomCollator(model.Bert_tokenizer, max_seq_len=max_seq_len, max_content_len=max_content_len)
    ds = InferenceDataset(input_csv)
    num_workers = int(os.environ.get('NUM_WORKERS', 2))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=collator)

    preds_txt = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dl, desc='Predict'):
            inputs = batch['inputs'].to(device)
            seq_positions = batch['seq_positions']
            outputs_ids = model(inputs, seq_positions)
            out_txt = model.Llama_tokenizer.batch_decode(outputs_ids)
            for t in out_txt:
                t_low = t.lower()
                if 'anomalous' in t_low:
                    preds_txt.append('anomalous')
                elif 'normal' in t_low:
                    preds_txt.append('normal')
                else:
                    preds_txt.append('unknown')

    label_map = {'anomalous': 1, 'normal': 0, 'unknown': -1}
    preds_label = [label_map.get(t, -1) for t in preds_txt]
    df = pd.read_csv(input_csv)
    out = df.copy()
    out['PredictedText'] = preds_txt
    out['PredictedLabel'] = preds_label
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Wrote predictions → {output_csv}")


def main():
    # Build sessions from a raw log and predict labels
    in_log = os.environ.get('INPUT_LOG', 'dataset/test_dataset.log')
    out_dir = os.environ.get('OUTPUT_DIR', 'dataset')
    mode = os.environ.get('MODE', 'gap')  # gap|fixed
    year = int(os.environ.get('YEAR', 2025))
    group_by_host = os.environ.get('GROUP_BY_HOST', 'true').lower() == 'true'
    gap_seconds = int(os.environ.get('GAP_SECONDS', 20))
    window_seconds = int(os.environ.get('WINDOW_SECONDS', 20))
    step_seconds_env = os.environ.get('STEP_SECONDS')
    step_seconds = int(step_seconds_env) if step_seconds_env is not None else None

    # Tokenization & batching params
    max_content_len = int(os.environ.get('MAX_CONTENT_LEN', 96))
    max_seq_len = int(os.environ.get('MAX_SEQ_LEN', 96))
    batch_size = int(os.environ.get('BATCH_SIZE', 16))

    # Sessionize
    records = read_labeled_log(in_log, default_year=year)
    if mode == 'gap':
        sessions = sessions_by_gap(records, gap_seconds=gap_seconds, group_by_host=group_by_host,
                                   drop_label_neg1=False, map_neg1_to=0, keep_unknown_sessions=True)
    else:
        sessions = sessions_by_fixed_window(records, window_seconds=window_seconds, step_seconds=step_seconds,
                                            group_by_host=group_by_host, drop_label_neg1=False, map_neg1_to=0,
                                            keep_unknown_sessions=True)
    unlabeled = [(msgs, lbl) for msgs, lbl in sessions]
    # Build a CSV of sessions to feed into predictor (ignore labels if present)
    sessions_csv = str(Path(out_dir) / 'test_sessions_for_pred.csv')
    pd.DataFrame({'Content': [' ;-; '.join(msgs) for msgs, _ in unlabeled]}).to_csv(sessions_csv, index=False)

    # Predict labels
    output_csv = os.environ.get('OUTPUT_CSV', str(Path(out_dir) / 'test_predictions.csv'))
    predict_sessions_csv(sessions_csv, output_csv, max_content_len, max_seq_len, batch_size)


if __name__ == '__main__':
    main()

