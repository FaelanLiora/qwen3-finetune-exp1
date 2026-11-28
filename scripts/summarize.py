"""汇总：把五次评测的结果变成一张主表和两张图。

输出：
    results/summary_table.md   可直接贴进报告的表格
    results/fig1_params_vs_math.png   可训练参数 vs 数学成绩
    results/fig2_abias_vs_science.png A 占比 vs 科学成绩
    results/summary_final.json        机器可读的完整数据

用法：
    python scripts/summarize.py
    python scripts/summarize.py --results /root/autodl-tmp/results
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")          # 无界面环境，直接出图
import matplotlib.pyplot as plt

# (标签, 显示名, 可训练参数百分比)
METHODS = [
    ("base",    "Base",         0.00),
    ("full",    "Full SFT",   100.00),
    ("lora",    "LoRA",         1.67),
    ("kllora",  "KL-LoRA",      1.67),
    ("ptuning", "P-Tuning v2",  0.19),
]


def read_summary(path: str) -> dict[str, dict]:
    """读 summary.csv，同名的取最后一次（后面的才是最终结果）。"""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tag = row["tag"].replace("demo_", "")
            if tag in {m[0] for m in METHODS}:
                out[tag] = row
    return out


def a_ratio(results_dir: str, tag: str) -> float | None:
    """统计该模型在科学题上答 A 的比例。"""
    path = os.path.join(results_dir, f"{tag}_arc.jsonl")
    if not os.path.exists(path):
        return None
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if not rows:
        return None
    c = Counter(r["pred"] for r in rows)
    return c.get("A", 0) / len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/root/autodl-tmp/results")
    args = ap.parse_args()

    summary = read_summary(os.path.join(args.results, "summary.csv"))

    rows = []
    for tag, name, params in METHODS:
        if tag not in summary:
            print(f"跳过 {name}：没有找到评测结果")
            continue
        s = summary[tag]
        rows.append({
            "tag": tag,
            "方法": name,
            "可训练参数": params,
            "数学": float(s["gsm8k_acc"]) * 100,
            "科学": float(s["arc_acc"]) * 100,
            "A占比": (a_ratio(args.results, tag) or 0) * 100,
        })

    if not rows:
        raise SystemExit("没有可汇总的数据")

    # ---------------- 主表 ----------------
    base_math = next((r["数学"] for r in rows if r["tag"] == "base"), None)
    base_sci = next((r["科学"] for r in rows if r["tag"] == "base"), None)

    lines = [
        "| 方法 | 可训练参数 | 数学 | Δ vs Base | 科学 | Δ vs Base | A 占比 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        d1 = f"{r['数学']-base_math:+.2f}" if base_math is not None else "—"
        d2 = f"{r['科学']-base_sci:+.2f}" if base_sci is not None else "—"
        p = "—" if r["可训练参数"] == 0 else f"{r['可训练参数']:.2f}%"
        lines.append(f"| {r['方法']} | {p} | {r['数学']:.2f}% | {d1} | {r['科学']:.2f}% | {d2} | {r['A占比']:.1f}% |")

    md = "\n".join(lines)
    print(md)
    with open(os.path.join(args.results, "summary_table.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")

    with open(os.path.join(args.results, "summary_final.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # ---------------- 图 1：参数 vs 数学 ----------------
    pts = [r for r in rows if r["可训练参数"] > 0] + [r for r in rows if r["可训练参数"] == 0]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)

    xs, ys, labels = [], [], []
    for r in pts:
        xs.append(max(r["可训练参数"], 0.12))       # 0% 画在对数轴上用 0.12 占位
        ys.append(r["数学"])
        labels.append(r["方法"])

    ax.scatter(xs, ys, s=70, c="#2b6cb0", zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (%, log scale)")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_title("More trainable parameters -> better math accuracy")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(args.results, "fig1_params_vs_math.png"))
    plt.close(fig)

    # ---------------- 图 2：A 占比 vs 科学 ----------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    xs = [r["A占比"] for r in rows]
    ys = [r["科学"] for r in rows]

    ax.scatter(xs, ys, s=70, c="#c53030", zorder=3)
    for r in rows:
        ax.annotate(r["方法"], (r["A占比"], r["科学"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)

    # 拟合一条直线，直观显示负相关
    n = len(xs)
    if n >= 2:
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0:
            k = num / den
            b = my - k * mx
            xr = [min(xs) - 2, max(xs) + 2]
            ax.plot(xr, [k * x + b for x in xr], "--", color="#666666", linewidth=1,
                    label=f"slope = {k:.2f}")

    ax.set_xlabel("Ratio of answering 'A' (%)")
    ax.set_ylabel("ARC-Challenge accuracy (%)")
    ax.set_title("A-bias almost fully determines science score")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.results, "fig2_abias_vs_science.png"))
    plt.close(fig)

    print(f"\n已生成：")
    for name in ["summary_table.md", "fig1_params_vs_math.png", "fig2_abias_vs_science.png"]:
        print(f"  {os.path.join(args.results, name)}")


if __name__ == "__main__":
    main()
