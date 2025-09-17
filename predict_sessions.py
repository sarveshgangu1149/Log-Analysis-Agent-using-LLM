import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import LogLLM
from customDataset import CustomCollator


class InferenceDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        if 'Content' not in df.columns:
            raise ValueError("Input CSV must have a 'Content' column")
        self.contents = df['Content'].fillna("").tolist()

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        content = self.contents[idx]
        seq = content.split(' ;-; ')
        # return sequence with a dummy label (unused by inference collator path)
        return seq, 0


def predict(model: LogLLM, dataloader: DataLoader, device: torch.device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predict"):
            inputs = batch['inputs'].to(device)
            seq_positions = batch['seq_positions']
            outputs_ids = model(inputs, seq_positions)
            texts = model.Llama_tokenizer.batch_decode(outputs_ids)
            for text in texts:
                m = re.search(r'normal|anomalous', text, re.IGNORECASE)
                if not m:
                    preds.append('unknown')
                else:
                    preds.append('anomalous' if m.group().lower() == 'anomalous' else 'normal')
    return preds


def main():
    input_csv = os.environ.get('INPUT_CSV', 'dataset/unlabeled_sessions.csv')
    output_csv = os.environ.get('OUTPUT_CSV', 'dataset/predictions.csv')
    max_content_len = int(os.environ.get('MAX_CONTENT_LEN', 100))
    max_seq_len = int(os.environ.get('MAX_SEQ_LEN', 128))
    batch_size = int(os.environ.get('BATCH_SIZE', 16))

    root = Path(__file__).parent
    bert_path = str(root / 'bert-base-uncased')
    llama_path = str(root / 'Meta-Llama-3-8B')
    dataset_name = os.environ.get('DATASET_NAME', 'LOCAL')
    ft_path = os.path.join(root, f"ft_model_{dataset_name}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ds = InferenceDataset(input_csv)
    # Build a collator with the same tokenizer as training will use
    # We need model first to get tokenizer, but collator needs tokenizer to build inputs.
    # Workaround: initialize model to get tokenizer, then dataloader.
    model = LogLLM(bert_path, llama_path, ft_path=ft_path, is_train_mode=False, device=device,
                   max_content_len=max_content_len, max_seq_len=max_seq_len)
    collator = CustomCollator(model.Bert_tokenizer, max_seq_len=max_seq_len, max_content_len=max_content_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collator, drop_last=False)

    pred_text = predict(model, dl, device)
    label_map = {'anomalous': 1, 'normal': 0, 'unknown': -1}
    pred_label = [label_map.get(t, -1) for t in pred_text]

    inp_df = pd.read_csv(input_csv)
    out_df = inp_df.copy()
    out_df['PredictedText'] = pred_text
    out_df['PredictedLabel'] = pred_label
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote predictions for {len(out_df)} sessions → {output_csv}")


if __name__ == '__main__':
    main()
