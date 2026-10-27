# Figures and tables from the saved results.

import glob
import json
import os
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image

# IEEE column and page width (inches)
COL_W = 3.5
PAGE_W = 7.16

# colour-blind safe colours
MODEL_COLORS = {"cnn": "#2a78d6", "mobilenet": "#eb6834", "vgg16": "#1baf7a",
                "vgg16_ft": "#8c5a00"}
MODEL_MARKERS = {"cnn": "o", "mobilenet": "s", "vgg16": "D", "vgg16_ft": "^"}
MODEL_LABELS = {"cnn": "CNN (scratch)", "mobilenet": "MobileNet",
                "vgg16": "VGG16 (frozen)", "vgg16_ft": "VGG16 (block 5 tuned)"}
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]


FONT_DIRS = ["/usr/share/fonts/truetype/msttcorefonts",
             "/usr/share/fonts/truetype/stix", "/usr/share/fonts/opentype/stix"]


def register_fonts():
    # fonts installed after matplotlib started (Colab) are not in its cache
    for d in FONT_DIRS:
        for path in glob.glob(os.path.join(d, "*.[ot]tf")):
            font_manager.fontManager.addfont(path)
    names = {f.name for f in font_manager.fontManager.ttflist}
    if "Times New Roman" not in names:
        print("Times New Roman not found; falling back to STIX for text")


def set_style():
    register_fonts()
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": TEXT_2,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, name + ".pdf"))
    fig.savefig(os.path.join(fig_dir, name + ".png"))
    plt.close(fig)


def load_results(results_dir, models, seeds):
    # results[model] holds all seeds in 'runs' and the first seed at top level
    out = {}
    for m in models:
        runs = {}
        for s in seeds:
            pre = os.path.join(results_dir, f"{m}_s{s}")
            if not (os.path.exists(pre + "_log.csv")
                    and os.path.exists(pre + "_eval.json")):
                continue
            with open(pre + "_eval.json") as f:
                ev = json.load(f)
            runs[s] = {"log": pd.read_csv(pre + "_log.csv"), "eval": ev,
                       "per_class": pd.read_csv(pre + "_per_class.csv"),
                       "confusion": pre + "_confusion.npy"}
        if not runs:
            print(f"skipping {m}: results not found")
            continue
        first = next(iter(runs))
        out[m] = {"runs": runs, "seed": first, **runs[first]}
    return out


def seed_stats(result, key):
    # mean, min, max, sd over seeds
    v = np.array([r["eval"][key] for r in result["runs"].values()], dtype=float)
    return v.mean(), v.min(), v.max(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def fig_samples(train_items, class_names, fig_dir, n=8, seed=0):
    rng = np.random.default_rng(seed)
    by_class = {}
    for path, label in train_items:
        by_class.setdefault(label, []).append(path)
    labels = rng.choice(sorted(by_class), size=n, replace=False)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(COL_W, COL_W * rows / cols * 1.18))
    for ax, lab in zip(axes.flat, labels):
        path = rng.choice(by_class[lab])
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(wrap(class_names[lab]), fontsize=6, pad=2, linespacing=1.0)
        ax.axis("off")
    for ax in list(axes.flat)[len(labels):]:
        ax.axis("off")
    fig.subplots_adjust(wspace=0.05, hspace=0.25)
    save(fig, fig_dir, "samples")


def fig_class_counts(counts, fig_dir):
    counts = np.asarray(counts)
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    bins = np.arange(counts.min() // 10 * 10, counts.max() + 10, 10)
    ax.hist(counts, bins=bins, color=MODEL_COLORS["cnn"], edgecolor="white",
            linewidth=0.6)
    med = np.median(counts)
    ax.axvline(med, color=TEXT_2, linewidth=0.8, linestyle=":")
    ax.text(med, ax.get_ylim()[1] * 0.95, f" median {med:.0f}", color=TEXT_2,
            fontsize=7, va="top")
    ax.set_xlabel("Training images per species")
    ax.set_ylabel("Number of species")
    save(fig, fig_dir, "class_counts")


def fig_curves(results, fig_dir):
    # first seed only
    models = list(results)
    fig, axes = plt.subplots(2, len(models), figsize=(PAGE_W, 3.1),
                             sharey="row", squeeze=False)
    for j, m in enumerate(models):
        log = results[m]["log"]
        ep = log["epoch"].to_numpy() + 1
        c = MODEL_COLORS[m]
        best = int(np.argmin(log["val_loss"].to_numpy()))
        lr = log["lr"].to_numpy() if "lr" in log else None
        drops = ep[1:][np.diff(lr) < 0] if lr is not None else []
        for i, (key, ylabel) in enumerate([("accuracy", "Accuracy"),
                                           ("loss", "Loss")]):
            ax = axes[i, j]
            for d in drops:
                ax.axvline(d - 0.5, color=GRID, linewidth=0.8, zorder=0)
            ax.plot(ep, log[key], color=c, linestyle="--", linewidth=1.0,
                    alpha=0.8, label="Training")
            ax.plot(ep, log["val_" + key], color=c, linewidth=1.4,
                    label="Validation")
            ax.plot(ep[best], log["val_" + key].iloc[best], marker="o",
                    markersize=4, color=c, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=3)
            ax.set_xlim(0.5, ep[-1] + 0.5)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(MODEL_LABELS[m], color=TEXT)
                ax.set_ylim(top=1.0)
            else:
                ax.set_xlabel("Epoch")
                ax.set_ylim(bottom=0)
    axes[0, 0].legend(loc="lower right", handlelength=2.2)
    fig.tight_layout(w_pad=1.0, h_pad=0.6)
    save(fig, fig_dir, "training_curves")


def fig_metrics(results, fig_dir):
    # dot = mean over seeds, line = min to max
    metrics = [("accuracy", "Accuracy"), ("macro_precision", "Macro precision"),
               ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1")]
    models = list(results)
    fig, ax = plt.subplots(figsize=(COL_W, 0.45 * len(models) + 0.9))
    ys = np.arange(len(metrics))[::-1]
    offs = np.linspace(0.27, -0.27, len(models))
    lo = 1.0
    for k, m in enumerate(models):
        stats = [seed_stats(results[m], key) for key, _ in metrics]
        lo = min(lo, min(st[1] for st in stats))
        y = ys + offs[k]
        ax.hlines(y, [st[1] for st in stats], [st[2] for st in stats],
                  color=MODEL_COLORS[m], linewidth=1.0, zorder=2)
        ax.scatter([st[0] for st in stats], y, s=20, color=MODEL_COLORS[m],
                   marker=MODEL_MARKERS[m], edgecolor="white", linewidth=0.6,
                   label=MODEL_LABELS[m], zorder=3)
        for st, yi in zip(stats, y):
            ax.text(st[2], yi, f"  {st[0]:.3f}", va="center", fontsize=6,
                    color=TEXT_2)
    ax.set_yticks(ys, [lab for _, lab in metrics])
    ax.set_xlim(np.floor(lo * 50) / 50 - 0.005, 1.012)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Score on test set")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handletextpad=0.2, columnspacing=1.0)
    save(fig, fig_dir, "test_metrics")


def fig_per_class_f1(results, fig_dir):
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    for m in results:
        f1 = np.sort(np.mean([r["per_class"]["f1"].to_numpy()
                              for r in results[m]["runs"].values()], axis=0))
        x = np.arange(1, len(f1) + 1)
        ax.plot(x, f1, color=MODEL_COLORS[m], label=MODEL_LABELS[m])
    n = len(f1)
    ax.set_xlim(1, n)
    ax.set_ylim(top=1.01)
    ax.set_xlabel("Species, ranked by F1 score")
    ax.set_ylabel("Per-class F1 (mean of seeds)")
    ax.legend(loc="lower right")
    save(fig, fig_dir, "per_class_f1")


def fig_hardest(results, model, fig_dir, k=15):
    pc = results[model]["per_class"].sort_values("f1").head(k)[::-1]
    fig, ax = plt.subplots(figsize=(COL_W, 0.16 * k + 0.5))
    y = np.arange(len(pc))
    ax.barh(y, pc["f1"], height=0.7, color=MODEL_COLORS[model])
    for yi, v in zip(y, pc["f1"]):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=6, color=TEXT_2)
    ax.set_yticks(y, [pretty(s) for s in pc["species"]], fontsize=6)
    ax.set_xlim(0, 1.08)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"F1 score ({MODEL_LABELS[model]})")
    save(fig, fig_dir, f"hardest_{model}")


def fig_confusion(cm, model, fig_dir):
    # with 525 classes an imshow is too fine to see, so plot only the errors
    cm = np.asarray(cm)
    n = cm.shape[0]
    off = cm.copy()
    np.fill_diagonal(off, 0)
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.92))
    ax.plot([0, n - 1], [0, n - 1], color=GRID, linewidth=1.2, zorder=1,
            label="Correct")
    shades = {1: (BLUE_RAMP[4], 14), 2: (BLUE_RAMP[6], 30), 3: (BLUE_RAMP[7], 48)}
    for count in sorted(set(off[off > 0].tolist())):
        colour, size = shades.get(count, (BLUE_RAMP[7], 48 + 12 * (count - 3)))
        t, p = np.nonzero(off == count)
        word = "image" if count == 1 else "images"
        ax.scatter(p, t, s=size, color=colour, edgecolor="white",
                   linewidth=0.6, zorder=3, label=f"{count} {word} wrong")
    ax.set_xlim(-5, n + 4)
    ax.set_ylim(n + 4, -5)
    ax.set_aspect("equal")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.4)
    ax.set_xlabel("Predicted species (index)")
    ax.set_ylabel("True species (index)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              frameon=False, handletextpad=0.4, columnspacing=1.2)
    save(fig, fig_dir, f"confusion_{model}")


def top_confusions(cm, class_names, k=6):
    cm = np.asarray(cm).copy()
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:k]
    pairs = []
    for idx in flat:
        t, p = np.unravel_index(idx, cm.shape)
        if cm[t, p] == 0:
            break
        pairs.append((int(t), int(p), int(cm[t, p])))
    return pairs


def fig_confused_pairs(cm, class_names, train_items, model, fig_dir, k=4, seed=0):
    pairs = top_confusions(cm, class_names, k)
    if not pairs:
        return
    rng = np.random.default_rng(seed)
    by_class = {}
    for path, label in train_items:
        by_class.setdefault(label, []).append(path)
    fig, axes = plt.subplots(2, len(pairs),
                             figsize=(PAGE_W * len(pairs) / 6, 2.6),
                             squeeze=False)
    for j, (t, p, n) in enumerate(pairs):
        for i, lab in enumerate([t, p]):
            ax = axes[i, j]
            ax.imshow(Image.open(rng.choice(by_class[lab])).convert("RGB"))
            ax.set_title(wrap(class_names[lab]), fontsize=6, pad=2,
                         linespacing=1.0)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    axes[0, 0].set_ylabel("True species", fontsize=7)
    axes[1, 0].set_ylabel("Predicted as", fontsize=7)
    for j, (t, p, n) in enumerate(pairs):
        axes[1, j].text(0.5, -0.08, f"{n} test image{'s' if n > 1 else ''}",
                        transform=axes[1, j].transAxes, ha="center", va="top",
                        fontsize=6, color=TEXT_2)
    fig.subplots_adjust(wspace=0.08, hspace=0.45)
    save(fig, fig_dir, f"confused_pairs_{model}")


def write_tables(results, path):
    def cell(m, key, scale=1.0, fmt=".2f"):
        mean, _, _, sd = seed_stats(results[m], key)
        if len(results[m]["runs"]) == 1:
            return f"{scale * mean:{fmt}}"
        return f"{scale * mean:{fmt}} $\\pm$ {scale * sd:{fmt}}"

    lines = [
        r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
        r"Model & Seeds & Params (M) & Epochs & Time (min) & Train acc. & Val. acc. \\",
        r"\midrule",
    ]
    for m, r in results.items():
        lines.append(
            f"{MODEL_LABELS[m]} & {len(r['runs'])} & "
            f"{r['eval']['params_trainable'] / 1e6:.1f} & "
            f"{cell(m, 'epochs_run', fmt='.0f')} & "
            f"{cell(m, 'train_minutes', fmt='.0f')} & "
            f"{cell(m, 'best_train_accuracy', 100)} & "
            f"{cell(m, 'best_val_accuracy', 100)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", "", ""]
    lines += [
        r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
        r"Model & Accuracy & Top-5 & Precision & Recall & F1 \\",
        r"\midrule",
    ]
    for m in results:
        lines.append(
            f"{MODEL_LABELS[m]} & {cell(m, 'accuracy', 100)} & "
            f"{cell(m, 'top5_accuracy', 100)} & "
            f"{cell(m, 'macro_precision', 1, '.3f')} & "
            f"{cell(m, 'macro_recall', 1, '.3f')} & "
            f"{cell(m, 'macro_f1', 1, '.3f')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def pretty(name):
    return " ".join(w.capitalize() for w in name.split())


def wrap(name, width=16):
    return "\n".join(textwrap.wrap(pretty(name), width))


def make_all(results_dir, fig_dir, models, seeds, class_names=None,
             train_items=None):
    set_style()
    results = load_results(results_dir, models, seeds)
    if not results:
        print("no results yet")
        return
    fig_curves(results, fig_dir)
    fig_metrics(results, fig_dir)
    fig_per_class_f1(results, fig_dir)
    best = max(results, key=lambda m: seed_stats(results[m], "macro_f1")[0])
    fig_hardest(results, best, fig_dir)
    cm = np.load(results[best]["confusion"])
    fig_confusion(cm, best, fig_dir)
    if class_names is not None and train_items is not None:
        fig_confused_pairs(cm, class_names, train_items, best, fig_dir)
    write_tables(results, os.path.join(results_dir, "tables.tex"))
    print("figures written to", fig_dir)
