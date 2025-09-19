import os
import re
import pandas as pd
from pathlib import Path


def build_normalizer():
    patterns = [
        r'True', r'true', r'False', r'false',
        r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b',
        r'\b(Mon|Monday|Tue|Tuesday|Wed|Wednesday|Thu|Thursday|Fri|Friday|Sat|Saturday|Sun|Sunday)\b',
        r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+\b',
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?',
        r'([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}',
        r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}',
        r'[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*',
        r'\b[0-9a-fA-F]{8}\b',
        r'\b[0-9a-fA-F]{10}\b',
        r'(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)',
        r'(\w+[\w\.]*)@(\w+[\w\.]*)',
        r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*',
    ]
    combined = re.compile('|'.join(patterns))

    def norm_text(t: str) -> str:
        t = re.sub(r'[\.]{3,}', '.. ', t)
        return re.sub(combined, '<*>', t)

    return norm_text


def main():
    root = Path(__file__).parent
    test_csv = root / 'dataset' / 'test.csv'
    pred_csv = root / 'dataset' / 'eval_predictions.csv'
    out_dir = root / 'runs'
    out_dir.mkdir(exist_ok=True)
    out_conflicts = out_dir / 'mismatch_groups.csv'

    df_t = pd.read_csv(test_csv)
    df_p = pd.read_csv(pred_csv)

    # Merge on content (1:1 expected)
    df = df_t.merge(df_p[['Content', 'PredictedLabel', 'PredictedText']], on='Content', how='inner')

    # Confusion counts
    tp = ((df.Label == 1) & (df.PredictedLabel == 1)).sum()
    tn = ((df.Label == 0) & (df.PredictedLabel == 0)).sum()
    fp = ((df.Label == 0) & (df.PredictedLabel == 1)).sum()
    fn = ((df.Label == 1) & (df.PredictedLabel == 0)).sum()
    print(f'Confusion: TP={tp}, TN={tn}, FP={fp}, FN={fn}')

    # Group by normalized session content to find inconsistent predictions for the same template/session
    norm = build_normalizer()
    df['NormContent'] = df['Content'].astype(str).apply(norm)
    grp = df.groupby('NormContent')

    rows = []
    for key, g in grp:
        pred_set = set(g['PredictedLabel'].tolist())
        label_set = set(g['Label'].tolist())
        if len(g) > 1 and (len(pred_set) > 1 or len(label_set) > 1):
            # collect a sample row and counts
            rows.append({
                'NormContent': key,
                'Count': len(g),
                'GT_Labels': sorted(list(label_set)),
                'Pred_Labels': sorted(list(pred_set)),
                'ExampleContent': g['Content'].iloc[0][:300],
            })

    if rows:
        out_df = pd.DataFrame(rows).sort_values(by='Count', ascending=False)
        out_df.to_csv(out_conflicts, index=False)
        print(f'Wrote conflict groups → {out_conflicts} (groups where similar sessions have mixed labels/predictions)')
    else:
        print('No conflicting groups found for normalized content.')

    # Top false positives/negatives samples for manual review
    df[df.Label.eq(0) & df.PredictedLabel.eq(1)].head(20).to_csv(out_dir / 'sample_false_positives.csv', index=False)
    df[df.Label.eq(1) & df.PredictedLabel.eq(0)].head(20).to_csv(out_dir / 'sample_false_negatives.csv', index=False)
    print('Wrote sample_false_positives.csv and sample_false_negatives.csv to runs/.')


if __name__ == '__main__':
    main()

