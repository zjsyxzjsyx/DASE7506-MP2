"""Report figures from stored records only (runs/experiments/results.json, runs/diagnostics/diagnostics.json).

Static light-mode figures for print. Colours are the first three slots of the reference categorical palette
(blue, orange, aqua), which validate all-pairs for colour-vision deficiency in light mode; aqua is below 3:1
contrast, so bars that use it carry value labels.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'axes.edgecolor': INK2,
    'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'axes.grid.axis': 'y',
    'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True, 'legend.frameon': False,
})


PAPER = False  # --paper: no in-figure titles; the report's captions carry them


def title(ax, text, pad=None):
    if not PAPER:
        ax.set_title(text, loc='left', **({'pad': pad} if pad else {}))


def bars(ax, x, heights, width, color, label=None, labels=False):
    """Thin bars with a surface-coloured gap between neighbours; value labels above in text ink,
    or inside the bar top when labels='inside' (used where a line runs along the bar tops)."""
    b = ax.bar(x, heights, width, color=color, edgecolor=SURFACE, linewidth=1.5, label=label)
    if labels:
        for rect, h in zip(b, heights):
            inside = labels == 'inside'
            ax.text(rect.get_x() + rect.get_width()/2, h - 3 if inside else h + 1, f'{h:.0f}', ha='center',
                    va='top' if inside else 'bottom', fontsize=7.5, color=SURFACE if inside else INK2)
    return b


def reliability(diag_rows, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    conf = [r['mean_conf']*100 for r in diag_rows]
    acc = [r['acc'] for r in diag_rows]
    x = range(len(conf))
    bars(ax, x, acc, 0.6, BLUE, 'accuracy', labels='inside')
    ax.plot(x, conf, color=INK2, linewidth=1.5, linestyle='--', marker='o', markersize=4,
            label='mean confidence')
    ax.set_xticks(list(x), [f'Q{k+1}\n{c:.0f}%' for k, c in enumerate(conf)])
    ax.set_xlabel('confidence quintile (mean confidence)')
    ax.set_ylabel('accuracy on val-hard (%)')
    ax.set_ylim(0, 108)
    title(ax, 'Calibration of the chosen answer', pad=20)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7.5)
    fig.tight_layout(); save(fig, out)


def patterns(gap, out):
    table = gap['per_pattern']
    keys = sorted(table, key=lambda k: -table[k]['val_n'])
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    x = range(len(keys))
    w = 0.38
    maj = [100*table[k]['majority_val'] for k in keys]
    fin = [100*table[k]['model_val'] for k in keys]
    bars(ax, [i - w/2 for i in x], maj, w, ORANGE, 'majority vote', labels=True)
    bars(ax, [i + w/2 for i in x], fin, w, BLUE, 'final model', labels=True)
    ax.set_xticks(list(x), [f"{k.replace(' ', '')}\nn={table[k]['val_n']}" for k in keys])
    ax.set_ylabel('accuracy on val-hard (%)')
    ax.set_ylim(0, 108)
    title(ax, 'Accuracy by vote pattern (sorted cluster sizes)')
    ax.legend(loc='upper right', ncol=2, fontsize=7.5)
    fig.tight_layout(); save(fig, out)


COST_POINTS = {  # seed-check name -> short label
    'single probe, no cross': 'single probe',
    'ensemble: centered + raw, no cross': '2 probes',
    'full (default)': 'final',
    'ensemble: centered + MLP': '+MLP probe',
    'ensemble: all four': '4 probes',
    'default + n-gram text probe': '+n-gram',
    'default + vote-pattern interactions': '+interactions',
    'default + boosted trees d2': '+trees d2',
}


def cost(results, rows_by_name, out):
    seeds = results['diagnostics']['seed_check']
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.grid(True, axis='both')
    for name, label in COST_POINTS.items():
        s = seeds[name]
        params = rows_by_name[name]['n_params']
        final = name == 'full (default)'
        ax.errorbar(params, s['cv_mean'], yerr=s['cv_std'], fmt='o', markersize=6 if final else 5,
                    color=ORANGE if final else BLUE, ecolor=INK2, elinewidth=1, capsize=2,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        # labels left of the point for the crowded ~2k-parameter column, right otherwise
        dx, dy, ha = {'+interactions': (1.12, 0.10, 'left'), '+trees d2': (1/1.12, 0.0, 'right'),
                      'final': (1.12, -0.06, 'left'), '2 probes': (1.12, 0.0, 'left'),
                      '+MLP probe': (1/1.12, -0.12, 'right'), '4 probes': (1.12, 0.12, 'left')}.get(label, (1.12, 0.08, 'left'))
        ax.text(params*dx, s['cv_mean'] + dy, label, fontsize=7.5, color=ORANGE if final else INK, ha=ha,
                va='center', fontweight='bold' if final else 'normal')
    ax.set_xscale('log')
    ax.set_xlabel('learned parameters (log scale)')
    ax.set_ylabel('train-hard CV accuracy (%)\nmean ± sd over 5 seeds')
    title(ax, 'More parameters, no more accuracy')
    ax.set_xlim(300, 1.2e6)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    fig.tight_layout(); save(fig, out)


def length_pairs(diag, out):
    subsets = [('all', 'all pairs'), ('length_matched_|dlog|<0.1', 'length-matched'),
               ('correct_longer_dlog>0.1', 'correct is longer'), ('correct_shorter_dlog<-0.1', 'correct is shorter')]
    series = [('final', 'final model', BLUE), ('handcrafted_only', 'handcrafted only', ORANGE),
              ('longest', 'longest wins', AQUA)]
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    w = 0.26
    for j, (key, label, color) in enumerate(series):
        hs = [100*diag[s][key] for s, _ in subsets]
        bars(ax, [i + (j - 1)*w for i in range(len(subsets))], hs, w, color, label, labels=True)
    ax.set_xticks(range(len(subsets)), [f"{lab}\n{diag[s]['n_pairs']} pairs" for s, lab in subsets])
    ax.set_ylabel('pairwise accuracy (%)\ncorrect scored above wrong')
    ax.set_ylim(0, 112)
    title(ax, 'Correct-vs-wrong candidate pairs by length difference (val-hard)', pad=20)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7.5)
    fig.tight_layout(); save(fig, out)


def save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.png'), dpi=220)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('runs/experiments/results.json'))
    parser.add_argument('--diagnostics', type=Path, default=Path('runs/diagnostics/diagnostics.json'))
    parser.add_argument('--output', type=Path, default=Path('../report/figures'))
    parser.add_argument('--paper', action='store_true', help='omit in-figure titles (captions carry them)')
    args = parser.parse_args()
    global PAPER
    PAPER = args.paper
    if PAPER:  # printed page is white: no tinted chart surface
        plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
    results = json.loads(args.results.read_text(encoding='utf-8'))
    diag = json.loads(args.diagnostics.read_text(encoding='utf-8'))
    rows = {r['method']: r for r in results['rows']}
    reliability(results['diagnostics']['calibration_val_hard'], args.output/'fig_calibration')
    patterns(diag['gap'], args.output/'fig_vote_patterns')
    cost(results, rows, args.output/'fig_cost_accuracy')
    length_pairs(diag['length_pairs_val_hard'], args.output/'fig_length_pairs')
    print('written:', sorted(p.name for p in args.output.glob('fig_*')))


if __name__ == '__main__':
    main()
