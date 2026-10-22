"""Главный график для README: доля accept с парным ДИ и итог каждой из 7 сделок.

Читает только готовые результаты, к модели не обращается:
    .venv/bin/python make_figure.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = Path("results/full.jsonl")
BOOT = Path("results/full-analysis/paired-bootstrap.json")
OUT = Path("results/figures/main.png")

COLORS = {"no_memory": "#8C8C8C", "memory": "#2F6DB5"}
AGENT_COLORS = {"A": "#D9822B", "B": "#2F6DB5", "C": "#5AA469"}


def main():
    boot = json.loads(BOOT.read_text(encoding="utf-8"))["metrics"]["accept_rate"]
    runs = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    deals = sorted(
        (r for r in runs if r["condition"] == "memory" and r["status"] == "agreed"),
        key=lambda r: r["index"],
    )
    n_deals = {c: sum(r["condition"] == c and r["status"] == "agreed" for r in runs) for c in COLORS}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})

    labels = {"no_memory": "Без памяти", "memory": "С памятью"}
    for i, cond in enumerate(["no_memory", "memory"]):
        m = boot[cond]
        lo, hi = m["ci95"]
        ax1.bar(i, m["value"] * 100, color=COLORS[cond], width=0.55)
        ax1.errorbar(i, m["value"] * 100, yerr=[[(m["value"] - lo) * 100], [(hi - m["value"]) * 100]],
                     color="black", capsize=6, lw=1.2)
        ax1.text(i, hi * 100 + 1.5, f"{m['value'] * 100:.1f}%".replace(".", ",") + f"\nсделок: {n_deals[cond]}/20",
                 ha="center", va="bottom", fontsize=10)
    d = boot["difference"]
    ax1.set_xticks([0, 1], [labels["no_memory"], labels["memory"]])
    ax1.set_ylabel("Доля голосов «за», %")
    ax1.set_ylim(0, 45)
    ax1.set_title((f"Разница +{d['value'] * 100:.1f} п.п., 95% ДИ "
                   f"[{d['ci95'][0] * 100:.1f}; {d['ci95'][1] * 100:.1f}]").replace(".", ",").replace("п,п,", "п.п."),
                  fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)

    for row, r in enumerate(deals):
        left = 0
        for agent in "ABC":
            share = r["allocation"][agent]
            ax2.barh(row, share, left=left, color=AGENT_COLORS[agent], edgecolor="white")
            if share >= 8:
                ax2.text(left + share / 2, row, f"{agent} {share}", ha="center", va="center",
                         color="white", fontsize=9)
            left += share
        zeros = [a for a in "ABC" if r["allocation"][a] == 0]
        note = f"раунд {r['rounds']}" + (f"; 0 у {', '.join(zeros)}" if zeros else "")
        ax2.text(102, row, note, va="center", fontsize=9)
    ax2.set_yticks(range(len(deals)), [f"прогон {r['index']}" for r in deals])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 135)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xlabel("Итоговое распределение 100 единиц")
    ax2.set_title("Все 7 сделок (только с памятью): кому что досталось", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(OUT)


if __name__ == "__main__":
    main()
