"""结果分析：查清"分数变了"到底是模型变了，还是判卷出了问题。

看四个指标：
    1. 正确率
    2. 输出里含 #### 的比例   —— 格式学会了吗
    3. 平均输出长度            —— 是不是太长被截断，或者太短在急着交卷
    4. 抽不出答案的条数        —— 判卷失败的比例

再和另一个方法对比，列出「原来对的、现在错了」的样例。

用法：
    python scripts/analyze.py --tag full
    python scripts/analyze.py --tag full --compare base
"""

from __future__ import annotations

import argparse
import json
import os
import statistics


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stats(rows: list[dict]) -> dict:
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    has_hash = sum("####" in r["output"] for r in rows) / n
    lens = [len(r["output"]) for r in rows]
    no_pred = sum(r["pred"] is None for r in rows)
    long_out = sum(l > 1200 for l in lens) / n      # 超长（可能是被截断的）
    return {
        "条数": n,
        "正确率": round(acc, 4),
        "含####的比例": round(has_hash, 3),
        "平均输出长度": round(statistics.mean(lens)),
        "中位长度": round(statistics.median(lens)),
        "超长占比": round(long_out, 3),
        "抽不出答案": no_pred,
    }


def show_examples(base_rows, cur_rows, k: int) -> None:
    base = {r["id"]: r for r in base_rows}
    cur = {r["id"]: r for r in cur_rows}
    print(f"\n===== 原来答对、现在答错的样例（最多 {k} 条）=====")
    shown = 0
    for i, b in base.items():
        c = cur.get(i)
        if c and b["correct"] and not c["correct"]:
            print(f"\n[标准答案] {b['gold']}")
            print(f"[原模型] {b['pred']!r}  正确")
            print(f"[新模型] {c['pred']!r}")
            print("[新模型的原话]")
            print(c["output"][:400])
            print("-" * 60)
            shown += 1
            if shown >= k:
                break
    if shown == 0:
        print("（没有这种情况）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/root/autodl-tmp/results")
    ap.add_argument("--tag", required=True, help="要分析的这次评测名，如 full")
    ap.add_argument("--compare", default=None, help="对比的方法名，如 base")
    ap.add_argument("--show", type=int, default=2, help="展示几条错误样例")
    args = ap.parse_args()

    cur_path = os.path.join(args.results, f"{args.tag}_gsm8k.jsonl")
    cur_rows = load(cur_path)

    print(f"===== {args.tag} 的数学评测 =====")
    s = stats(cur_rows)
    for k2, v in s.items():
        print(f"  {k2}: {v}")

    base_rows = None
    if args.compare:
        base_path = os.path.join(args.results, f"{args.compare}_gsm8k.jsonl")
        if os.path.exists(base_path):
            base_rows = load(base_path)
            print(f"\n===== {args.compare}（对比）=====")
            for k2, v in stats(base_rows).items():
                print(f"  {k2}: {v}")

            # 交叉统计：谁对谁错
            nb = sum(1 for b in base_rows if b["correct"])
            nc = sum(1 for c in cur_rows if c["correct"])
            both = sum(1 for b, c in zip(base_rows, cur_rows) if b["correct"] and c["correct"])
            print(f"\n  两边都对: {both}")
            print(f"  只有 {args.compare} 对: {nb - both}")
            print(f"  只有 {args.tag} 对: {nc - both}")
            print(f"  两边都错: {len(cur_rows) - (nb + nc - both)}")

    # ARC 部分
    arc_path = os.path.join(args.results, f"{args.tag}_arc.jsonl")
    if os.path.exists(arc_path):
        arc = load(arc_path)
        print(f"\n===== {args.tag} 的科学评测 =====")
        print(f"  条数: {len(arc)}")
        print(f"  正确率: {sum(r['correct'] for r in arc)/len(arc):.4f}")

    if base_rows:
        show_examples(base_rows, cur_rows, args.show)


if __name__ == "__main__":
    main()
