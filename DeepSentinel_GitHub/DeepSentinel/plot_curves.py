#!/usr/bin/env python3
"""
plot_curves.py — Plot training curves from training_log.csv
Usage:
    python plot_curves.py
    python plot_curves.py --log training_log.csv --out curves.png
"""
import argparse, os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='training_log.csv')
    parser.add_argument('--out', default='training_curves.png')
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"ERROR: {args.log} not found. Run train.py first."); sys.exit(1)

    rows = []
    with open(args.log) as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if k not in ('phase',) else v)
                         for k, v in row.items()})

    wu = [r for r in rows if r['phase'] == 'warmup']
    ft = [r for r in rows if r['phase'] == 'finetune']
    offset = max([r['epoch'] for r in wu], default=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('DeepSentinel — Training Curves', fontsize=14, fontweight='bold')

    for ax, key, title in [(ax1, 'loss', 'Loss'), (ax2, 'auc', 'AUC-ROC')]:
        if wu:
            xs = [r['epoch'] for r in wu]
            ax.plot(xs, [r[f'tr_{key}'] for r in wu], 'b-o', ms=4, label='WU Train')
            ax.plot(xs, [r[f'va_{key}'] for r in wu], 'b--s', ms=4, label='WU Val')
        if ft:
            xs = [r['epoch'] + offset for r in ft]
            ax.plot(xs, [r[f'tr_{key}'] for r in ft], 'r-o', ms=4, label='FT Train')
            ax.plot(xs, [r[f'va_{key}'] for r in ft], 'r--s', ms=4, label='FT Val')
        ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        ax.set_title(f'Training & Validation {title}')
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Curves saved → {args.out}")

if __name__ == '__main__':
    main()
