import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import datetime as dt
import csv
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import LogLLM
from customDataset import CustomDataset, CustomCollator, replace_patterns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Env-configurable params to mirror train.py style
max_content_len = int(os.environ.get('MAX_CONTENT_LEN', 96))
max_seq_len = int(os.environ.get('MAX_SEQ_LEN', 96))
batch_size = int(os.environ.get('BATCH_SIZE', 32))
dataset_name = 'LOCAL'
data_path = str(Path(__file__).parent / 'dataset' / 'test.csv')
output_csv = os.environ.get('OUTPUT_CSV', str(Path(__file__).parent / 'dataset' / 'eval_predictions.csv'))
# Margin-based unknown gating
margin_unknown = os.environ.get('MARGIN_UNKNOWN', 'false').lower() == 'true'
tau_seen = float(os.environ.get('TAU_SEEN', '1.0'))
tau_novel = float(os.environ.get('TAU_NOVEL', '2.0'))
novelty_as_unknown = os.environ.get('NOVELTY_AS_UNKNOWN', 'false').lower() == 'true'
# Also treat GT -1 rows as novel/unknown by default (can disable via env)
gt_unknown_as_novel = os.environ.get('GT_UNKNOWN_AS_NOVEL', 'true').lower() == 'true'
train_csv_for_novelty = str(Path(__file__).parent / 'dataset' / 'train.csv')

ROOT_DIR = Path(__file__).parent
Bert_path = str(ROOT_DIR / 'bert-base-uncased')
_llm_env = os.environ.get('LLM_PATH')
if _llm_env:
    Llama_path = _llm_env
else:
    tiny_path = ROOT_DIR / 'TinyLlama-1.1B-Chat-v1.0'
    llama3_path = ROOT_DIR / 'Meta-Llama-3-8B'
    if tiny_path.exists():
        Llama_path = str(tiny_path)
    elif llama3_path.exists():
        Llama_path = str(llama3_path)
    else:
        Llama_path = str(tiny_path)
ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(
f'dataset_name: {dataset_name}\n'
f'batch_size: {batch_size}\n'
f'max_content_len: {max_content_len}\n'
f'max_seq_len: {max_seq_len}\n'
f'device: {device}')


def evalModel(model, dataloader):
    model.eval()

    preds = []

    with torch.no_grad():
        for bathc_i in tqdm(dataloader):
            inputs = bathc_i['inputs']
            seq_positions = bathc_i['seq_positions']

            inputs = inputs.to(device)
            seq_positions = seq_positions

            outputs_ids = model(inputs,seq_positions)
            outputs = model.Llama_tokenizer.batch_decode(outputs_ids)

            # print(outputs)

            for text in outputs:
                match = re.search(r'normal|anomalous', text, re.IGNORECASE)
                if match:
                    preds.append(match.group())
                else:
                    preds.append('unknown')
    return preds


if __name__ == '__main__':
    print(f'dataset: {data_path}')
    dataset = CustomDataset(data_path)
    model = LogLLM(Bert_path, Llama_path, ft_path=ft_path, is_train_mode=False, device=device,
                   max_content_len=max_content_len, max_seq_len=max_seq_len)

    tokenizer = model.Bert_tokenizer
    collator = CustomCollator(tokenizer, max_seq_len=max_seq_len, max_content_len=max_content_len)
    num_workers = int(os.environ.get('NUM_WORKERS', 2))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False
    )
    pred_texts = evalModel(model, dataloader)

    # Map to numeric labels; keep unknown as -1
    label_map = {'anomalous': 1, 'normal': 0, 'unknown': -1}
    pred_labels = np.array([label_map.get(t.lower(), -1) if isinstance(t, str) else -1 for t in pred_texts])
    gt = dataset.get_label()

    # Optional novelty gating: mark sessions whose normalized content never appeared in training as unknown (-1)
    df_test = pd.read_csv(data_path)
    is_novel = np.zeros(len(df_test), dtype=bool)
    # Optionally gate by normalization vs. train
    if novelty_as_unknown and Path(train_csv_for_novelty).exists():
        try:
            df_train = pd.read_csv(train_csv_for_novelty)
            seen = set(str(replace_patterns(c)) for c in df_train['Content'].astype(str))
            test_norm = [str(replace_patterns(c)) for c in df_test['Content'].astype(str)]
            is_novel = np.array([tn not in seen for tn in test_norm])
            print(f'Novelty gating enabled (by normalization vs train): {is_novel.sum()} sessions flagged as novel.')
        except Exception as e:
            print(f'Novelty gating skipped due to error: {e}')
    # Optionally also treat GT == -1 as novel
    if gt_unknown_as_novel and 'Label' in df_test.columns:
        gt_neg1_mask = (df_test['Label'].values == -1)
        if gt_neg1_mask.any():
            is_novel = np.logical_or(is_novel, gt_neg1_mask)
            print(f'GT_UNKNOWN_AS_NOVEL is enabled: {gt_neg1_mask.sum()} GT=-1 sessions will be forced to unknown.')
    # Apply novelty gating to predictions
    if is_novel.any():
        pred_labels = np.where(is_novel, -1, pred_labels)
        pred_texts = [('unknown' if is_novel[i] else pred_texts[i]) for i in range(len(pred_texts))]

    # Optional margin-based gating using model scores
    score_margins = None
    if margin_unknown:
        margins = []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collator, num_workers=num_workers, shuffle=False, drop_last=False):
                inputs = batch['inputs'].to(device)
                seq_positions = batch['seq_positions']
                scores = model.score_candidates(inputs, seq_positions, candidates=("normal", "anomalous"))  # [B,2]
                diff = (scores[:, 1] - scores[:, 0]).abs().cpu().numpy()
                margins.extend(diff.tolist())
        score_margins = np.array(margins)
        # Gate by thresholds (stricter for novel sessions)
        thr = np.where(is_novel, tau_novel, tau_seen)
        low_conf = score_margins < thr
        if low_conf.any():
            pred_labels = np.where(low_conf, -1, pred_labels)
            pred_texts = [('unknown' if low_conf[i] else pred_texts[i]) for i in range(len(pred_texts))]

    # Metrics on valid rows only: predictions in {0,1} AND ground truth in {0,1}
    known_pred_mask = pred_labels != -1
    known_gt_mask = np.isin(gt, [0, 1])
    valid_mask = known_pred_mask & known_gt_mask
    if valid_mask.any():
        precision = precision_score(gt[valid_mask], pred_labels[valid_mask], average="binary", pos_label=1)
        recall = recall_score(gt[valid_mask], pred_labels[valid_mask], average="binary", pos_label=1)
        f1m = f1_score(gt[valid_mask], pred_labels[valid_mask], average="binary", pos_label=1)
        acc = accuracy_score(gt[valid_mask], pred_labels[valid_mask])
    else:
        precision = recall = f1m = acc = float('nan')

    num_anomalous = int((gt == 1).sum())
    num_normal = int((gt == 0).sum())
    pred_num_anomalous = int((pred_labels == 1).sum())
    pred_num_normal = int((pred_labels == 0).sum())
    pred_num_unknown = int((pred_labels == -1).sum())

    print(f'Number of anomalous seqs: {num_anomalous}; number of normal seqs: {num_normal}')
    print(f'Detected anomalous: {pred_num_anomalous}; normal: {pred_num_normal}; unknown: {pred_num_unknown}')
    print(f'precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1m:.4f}, acc: {acc:.4f}')
    # Confusion matrix on valid rows only
    if valid_mask.any():
        cm = confusion_matrix(gt[valid_mask], pred_labels[valid_mask], labels=[0, 1])
        print('Confusion matrix (rows=true 0/1, cols=pred 0/1):')
        print(cm)
    else:
        cm = None

    # Write detailed predictions CSV with a simple confidence proxy
    df = df_test
    n = min(len(df), len(pred_labels))
    conf = [1.0 if isinstance(t, str) and t.lower() in ('normal', 'anomalous') else 0.0 for t in pred_texts[:n]]
    out_dict = {
        'Content': df['Content'].iloc[:n].values,
        'Label': df['Label'].iloc[:n].values,
        'PredictedText': pred_texts[:n],
        'PredictedLabel': pred_labels[:n],
        'Confidence': conf,
    }
    # Include Host column if present in test.csv
    if 'Host' in df.columns:
        out_dict['Host'] = df['Host'].iloc[:n].values
    # Include isNew flag if computed (1=new, 0=not new)
    if is_novel is not None and len(is_novel) >= n:
        try:
            out_dict['isNew'] = np.array(is_novel[:n], dtype=bool).astype(int)
        except Exception:
            # Fallback: simple list conversion
            out_dict['isNew'] = [1 if bool(x) else 0 for x in is_novel[:n]]
    # Include ScoreMargin if computed
    if score_margins is not None and len(score_margins) >= n:
        out_dict['ScoreMargin'] = score_margins[:n]
    out = pd.DataFrame(out_dict)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f'Wrote predictions to {out_path}')

    # Append metrics to runs/eval_metrics.csv
    runs_dir = Path(__file__).parent / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = runs_dir / 'eval_metrics.csv'
    header = not metrics_csv.exists()
    with open(metrics_csv, 'a', newline='') as fh:
        w = csv.writer(fh)
        if header:
            w.writerow(['timestamp', 'precision', 'recall', 'f1', 'acc', 'gt_pos', 'gt_neg', 'pred_pos', 'pred_neg', 'pred_unknown'])
        w.writerow([
            dt.datetime.utcnow().isoformat(),
            f"{precision:.6f}", f"{recall:.6f}", f"{f1m:.6f}", f"{acc:.6f}",
            num_anomalous, num_normal, pred_num_anomalous, pred_num_normal, pred_num_unknown
        ])
    # Save confusion matrix
    if cm is not None:
        cm_csv = runs_dir / 'confusion_matrix.csv'
        with open(cm_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['', 'pred_0', 'pred_1'])
            w.writerow(['true_0', int(cm[0, 0]), int(cm[0, 1])])
            w.writerow(['true_1', int(cm[1, 0]), int(cm[1, 1])])
        print(f'Wrote confusion matrix to {cm_csv}')
        # Also save an image if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
            im = ax.imshow(cm, cmap='Blues')
            ax.figure.colorbar(im, ax=ax)
            ax.set_xticks([0, 1], labels=['pred 0', 'pred 1'])
            ax.set_yticks([0, 1], labels=['true 0', 'true 1'])
            # annotate cells
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
            ax.set_title('Confusion Matrix (no of sessions)')
            img_path = runs_dir / 'confusion_matrix.png'
            plt.tight_layout()
            plt.savefig(img_path)
            plt.close(fig)
            print(f'Wrote confusion matrix image to {img_path}')
        except Exception as e:
            print(f'Could not create confusion matrix image: {e}')
