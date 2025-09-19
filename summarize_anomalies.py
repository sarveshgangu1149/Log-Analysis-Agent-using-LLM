import os
import re
from typing import List, Tuple
import pandas as pd


SEP = ' ;-; '


def split_session(content: str) -> List[str]:
    return [s for s in content.split(SEP) if s.strip()]


def extract_key_phrases(lines: List[str], max_phrases: int = 2) -> str:
    phrases = []
    for ln in lines:
        l = ln.strip()
        # Common error/failed patterns first
        m = re.search(r'(error:.*)$', l, re.IGNORECASE)
        if m:
            phrases.append(m.group(1))
            continue
        m = re.search(r'(failed(?: to)? .*)$', l, re.IGNORECASE)
        if m:
            phrases.append(m.group(1))
            continue
        # After common components like systemd/kernel/service
        m = re.search(r'\b(?:systemd\[[^\]]+\]|kernel|sshd|multipathd|kdumpctl|rngd|dbus-daemon)[^:]*:\s*(.*)$', l, re.IGNORECASE)
        if m:
            phrases.append(m.group(1))
            continue
        # Fallback: take right-most clause after colon
        if ':' in l:
            phrases.append(l.split(':')[-1].strip())
        else:
            phrases.append(l)

    # Normalize whitespace and de-duplicate while preserving order
    norm = []
    seen = set()
    for p in phrases:
        p2 = re.sub(r'\s+', ' ', p)
        if p2 and p2.lower() not in seen:
            seen.add(p2.lower())
            norm.append(p2)
    return ' | '.join(norm[:max_phrases]) if norm else ''


def llm_summarize(lines: List[str], model_path: str) -> str:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    text = '\n'.join(lines[:20])
    text = text[:2000]
    prompt = (
        "Summarize the main issue in the following system log lines in one short sentence.\n"
        "Be concise and avoid timestamps or hostnames.\n\n"
        f"Logs:\n{text}\n\nSummary:"
    )

    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    device = (
        torch.device('cuda:0') if torch.cuda.is_available()
        else (torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else torch.device('cpu'))
    )
    mdl.to(device)

    inputs = tok(prompt, return_tensors='pt').to(device)
    out = mdl.generate(**inputs, max_new_tokens=48, do_sample=False)
    txt = tok.decode(out[0], skip_special_tokens=True)
    # Return the tail after 'Summary:'
    m = re.search(r'Summary:\s*(.*)$', txt, re.DOTALL)
    return m.group(1).strip() if m else txt[-200:].strip()


def main():
    input_csv = os.environ.get('INPUT_CSV', 'dataset/predictions.csv')
    output_csv = os.environ.get('OUTPUT_CSV', 'dataset/anomaly_summaries.csv')
    use_llm = os.environ.get('USE_LLM', 'false').lower() == 'true'
    model_path = os.environ.get('LLM_PATH')
    if use_llm and not model_path:
        # Try local defaults
        root = os.path.dirname(__file__)
        cand = [os.path.join(root, 'TinyLlama-1.1B-Chat-v1.0'), os.path.join(root, 'Meta-Llama-3-8B')]
        for c in cand:
            if os.path.exists(c):
                model_path = c
                break

    df = pd.read_csv(input_csv)
    # Determine which label column to use
    if 'PredictedLabel' in df.columns:
        mask = df['PredictedLabel'] == 1
    elif 'Label' in df.columns:
        mask = df['Label'] == 1
    else:
        raise ValueError("Input CSV must contain 'PredictedLabel' or 'Label' column")

    out_rows = []
    for idx, row in df[mask].iterrows():
        content = row['Content']
        lines = split_session(content)
        summary = ''
        if use_llm and model_path:
            try:
                summary = llm_summarize(lines, model_path)
            except Exception:
                summary = extract_key_phrases(lines)
        else:
            summary = extract_key_phrases(lines)
        out_rows.append({
            'Index': idx,
            'Summary': summary,
            'Content': content,
        })

    out_df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote {len(out_df)} anomaly summaries → {output_csv}")


if __name__ == '__main__':
    main()

