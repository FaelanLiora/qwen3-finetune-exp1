"""科学题：它是"按内容答题"还是"按位置猜"？

【要回答的问题】
全参微调后科学题涨了 15.8 分。这是"更会用已有知识"，还是"在作弊"？

【两个检验】
一、答题行为统计（读已有答卷，不花时间）
    - 预测字母的分布：是不是偏爱某个字母
    - 各字母位置的正确率：某个位置是否特别准
    - 逐题翻转：哪些题由错转对、由对转错

二、选项置换检验（核心）
    把四个选项整体挪一位，正确答案的位置跟着变，题目内容不变。
      答对 → 跟着内容走（真的在读题）
      答错且答案固定在原位置 → 跟着位置走（作弊）
      掉到 25%      → 瞎猜

用法：
    python scripts/probe_arc2.py --full /root/autodl-tmp/outputs/full
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ARC_SYSTEM


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(tok, question: str, labels: list[str], texts: list[str]) -> str:
    """按指定的选项顺序重新拼 prompt（和 build_data.py 里的格式完全一致）。"""
    body = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
    content = f"{question.strip()}\n\n{body}\n\n请只回答正确选项的字母。"
    msgs = [
        {"role": "system", "content": ARC_SYSTEM},
        {"role": "user", "content": content},
    ]
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


@torch.no_grad()
def score_letters(model, tok, prompts: list[str], label_sets: list[list[str]]) -> list[str]:
    """对每个 prompt，看四个选项字母谁的概率最大，返回预测字母。"""
    cache: dict[str, list[int]] = {}

    def ids_of(lab: str) -> list[int]:
        if lab not in cache:
            cache[lab] = sorted({
                tok.encode(lab, add_special_tokens=False)[0],
                tok.encode(" " + lab, add_special_tokens=False)[0],
            })
        return cache[lab]

    preds = []
    for i, (p, labels) in enumerate(zip(prompts, label_sets)):
        enc = tok(p, return_tensors="pt").to(model.device)
        logp = torch.log_softmax(model(**enc).logits[0, -1].float(), dim=-1)
        best = max(labels, key=lambda lab: max(logp[t].item() for t in ids_of(lab)))
        preds.append(best)
        if (i + 1) % 200 == 0:
            print(f"\r  {i+1}/{len(prompts)}", end="", flush=True)
    print()
    return preds


def behavior_report(base_rows: list[dict], full_rows: list[dict]) -> None:
    """从已有答卷里统计答题习惯。"""
    print("=" * 60)
    print("一、答题行为统计（读已有答卷，不重新计算）")
    print("=" * 60)

    for name, rows in [("原始模型", base_rows), ("全参微调", full_rows)]:
        cnt = Counter(r["pred"] for r in rows)
        acc = sum(r["correct"] for r in rows) / len(rows)
        print(f"\n{name}：正确率 {acc:.4f}")
        for lab in sorted(cnt):
            n = cnt[lab]
            hit = sum(1 for r in rows if r["pred"] == lab and r["correct"])
            print(f"    预测为 {lab}: {n:4d} 题（{n/len(rows):5.1%}），其中答对 {hit:4d}（准确率 {hit/n:5.1%}）")

    # 逐题翻转
    base = {r["id"]: r for r in base_rows}
    full = {r["id"]: r for r in full_rows}
    both_right = sum(1 for i in base if i in full and base[i]["correct"] and full[i]["correct"])
    gain = sum(1 for i in base if i in full and not base[i]["correct"] and full[i]["correct"])
    lose = sum(1 for i in base if i in full and base[i]["correct"] and not full[i]["correct"])
    both_wrong = len(base) - both_right - gain - lose
    print(f"\n逐题翻转：两边都对 {both_right}，由错转对 {gain}，由对转错 {lose}，都错 {both_wrong}")
    print(f"  → 净增 {gain - lose} 题")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/autodl-tmp/data/arc_test.jsonl")
    ap.add_argument("--base_model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--full", default="/root/autodl-tmp/outputs/full")
    ap.add_argument("--results", default="/root/autodl-tmp/results")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = load_jsonl(args.data)
    if args.limit > 0:
        rows = rows[: args.limit]

    base_file = os.path.join(args.results, "base_arc.jsonl")
    full_file = os.path.join(args.results, "full_arc.jsonl")
    if os.path.exists(base_file) and os.path.exists(full_file):
        behavior_report(load_jsonl(base_file), load_jsonl(full_file))

    # ---------- 选项置换 ----------
    print("\n" + "=" * 60)
    print("二、选项置换检验（题目不变，只把选项整体挪一位）")
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 原顺序的 prompt（用来验证重建是否和原来一致）
    orig_prompts, orig_labels, orig_golds = [], [], []
    # 置换后的 prompt
    perm_prompts, perm_labels, perm_golds = [], [], []

    for r in rows:
        labels, texts = r["labels"], r["texts"]
        gold_text = texts[labels.index(r["answer"])]

        orig_prompts.append(build_prompt(tok, r["question"], labels, texts))
        orig_labels.append(labels)
        orig_golds.append(r["answer"])

        # 整体右移一位：最后一个挪到最前面
        new_texts = [texts[-1]] + texts[:-1]
        new_gold = labels[new_texts.index(gold_text)]
        perm_prompts.append(build_prompt(tok, r["question"], labels, new_texts))
        perm_labels.append(labels)
        perm_golds.append(new_gold)

    for name, path in [("原始模型", args.base_model), ("全参微调", args.full)]:
        print(f"\n----- {name} -----")
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda").eval()

        print("  原顺序 ...")
        p1 = score_letters(model, tok, orig_prompts, orig_labels)
        acc1 = sum(a == b for a, b in zip(p1, orig_golds)) / len(rows)

        print("  置换后 ...")
        p2 = score_letters(model, tok, perm_prompts, perm_labels)
        acc2 = sum(a == b for a, b in zip(p2, perm_golds)) / len(rows)

        # 置换后，预测是否"仍然指向原来那个字母位置"
        stick = sum(1 for a, b in zip(p1, p2) if a == b) / len(rows)

        print(f"  原顺序正确率 = {acc1:.4f}")
        print(f"  置换后正确率 = {acc2:.4f}")
        print(f"  答案固定在原字母位置的比例 = {stick:.4f}（越高说明越依赖位置）")

        del model
        torch.cuda.empty_cache()

    print("""
===== 怎么读 =====
· 置换后正确率接近原顺序          → 跟着内容走，不是位置作弊
· 置换后正确率掉到 0.25 左右       → 纯瞎猜，之前的分数不可信
· 「固定在原位置」的比例很高        → 位置偏好，分数是假的
""")


if __name__ == "__main__":
    main()
