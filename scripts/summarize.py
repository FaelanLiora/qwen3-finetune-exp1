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
# 顺序按 PPT 第 2、9 页的方法排列：Base → Full SFT → LoRA → P-Tuning v2 → KL-LoRA(自研)
METHODS = [
    ("base",    "Base",         0.00),
    ("full",    "Full SFT",   100.00),
    ("lora",    "LoRA",         1.67),
    ("ptuning", "P-Tuning v2",  0.19),
    ("kllora",  "KL-LoRA",      1.67),
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

    # ---------------- 图 1：五方法双指标柱状图 ----------------
    # 不用散点：散点会把"基座 0 参数却最高"画成"参数越多越差"，容易误读。
    # 柱状图能直接看出每个方法在两项指标上的位置，以及相对基座的升降。
    names = [r["方法"] for r in rows]
    math_v = [r["数学"] for r in rows]
    sci_v = [r["科学"] for r in rows]
    xs = list(range(len(names)))

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=160)
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in xs], math_v, w, label="GSM8K (math)", color="#2b6cb0")
    b2 = ax.bar([i + w / 2 for i in xs], sci_v, w, label="ARC-Challenge (science)", color="#c53030")

    # 基座的两条参考线
    if base_math is not None:
        ax.axhline(base_math, ls="--", lw=1, color="#2b6cb0", alpha=0.45)
        ax.axhline(base_sci, ls="--", lw=1, color="#c53030", alpha=0.45)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=7)

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=12, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(30, 62)
    ax.set_title("Accuracy on two benchmarks (dashed lines = base model)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(args.results, "fig1_accuracy_bars.png"))
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
    ax.set_title("Science accuracy vs ratio of answering 'A'")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.results, "fig2_abias_vs_science.png"))
    plt.close(fig)

    print(f"\n已生成：")
    for name in ["summary_table.md", "fig1_accuracy_bars.png", "fig2_abias_vs_science.png"]:
        print(f"  {os.path.join(args.results, name)}")


if __name__ == "__main__":
    main()
