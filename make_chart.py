"""Draw assets/part2_f1.png from results/part2_sentiment.csv."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / "results" / "part2_sentiment.csv").open()))
names = ["Zero-shot", "LoRA\n(all linear)", "Adapter\n(q/v only)"]
f1 = [float(r["f1"]) for r in rows]

BG, INK, MUT, BLUE, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#a9a8a3"
fig, ax = plt.subplots(figsize=(7, 4), dpi=200, facecolor=BG)
ax.set_facecolor(BG)
bars = ax.bar(names, f1, color=[GREY, BLUE, BLUE], width=0.55)
for b, v in zip(bars, f1):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", color=INK, fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_ylabel("F1 on 2,000 balanced test tweets", color=MUT)
ax.set_title("Fine-tuning lifts sentiment F1 from 0.69 to 0.86", loc="left", fontsize=11, fontweight="bold", color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=MUT)
fig.tight_layout()
out = root / "assets" / "part2_f1.png"
fig.savefig(out, facecolor=BG)
print("wrote", out)
