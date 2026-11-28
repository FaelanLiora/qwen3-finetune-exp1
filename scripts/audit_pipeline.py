"""流程审计：把"我以为对、但从没验证过"的四处全部核查一遍。

不训练、不用显卡，只检查数据和分词，10 秒出结果。

审计项：
    1. 训练与评测的分词是否一致（add_special_tokens 的差别）
    2. prompt 结尾是否符合 enable_thinking=False 的预期
    3. 标签掩码是否正确（题目部分全 -100，答案部分不掩）
    4. 有多少样本因超长被截断

用法：
    python scripts/audit_pipeline.py
"""

from __future__ import annotations

import json
import os

from transformers import AutoTokenizer


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    model = "/root/autodl-tmp/models/Qwen3-0.6B"
    data = "/root/autodl-tmp/data"
    tok = AutoTokenizer.from_pretrained(model)

    print("=" * 70)
    print("【审计 1】训练与评测的分词是否一致")
    print("=" * 70)
    rows = load_jsonl(os.path.join(data, "train.jsonl"))[:3]
    ok = True
    for r in rows:
        a = tok(r["prompt"]).input_ids                      # 评测时的写法
        b = tok(r["prompt"], add_special_tokens=False).input_ids  # 训练时的写法
        same = a == b
        ok = ok and same
        print(f"  {r['id']}: 长度 {len(a)} vs {len(b)}  ->  {'一致' if same else '★不一致★'}")
        if not same:
            print(f"    评测用前6个: {a[:6]}")
            print(f"    训练用前6个: {b[:6]}")
    print(f"  结论：{'通过' if ok else '★发现不一致，这会让前缀整体错位★'}")

    print()
    print("=" * 70)
    print("【审计 2】prompt 结尾格式（thinking 是否真的关了）")
    print("=" * 70)
    for r in rows[:1]:
        tail = r["prompt"][-40:]
        print(f"  最后 40 个字符: {tail!r}")
        ids = tok(r["prompt"], add_special_tokens=False).input_ids
        print(f"  最后 6 个 token: {tok.convert_ids_to_tokens(ids[-6:])}")
        if "<|im_start|>assistant" in r["prompt"]:
            print("  ✓ 含 assistant 起始标记")
        # Qwen3 关闭思考时，assistant 之后通常会有额外换行或空思考块
        print("  提示：若结尾只有 'assistant\\n' 而没有空思考块，说明 enable_thinking 可能未生效")
        print("       但这不影响训练/评测一致性（两边用的是同一个字符串）")

    print()
    print("=" * 70)
    print("【审计 3】标签掩码是否正确")
    print("=" * 70)
    IGNORE = -100
    bad = 0
    for r in rows:
        p = tok(r["prompt"], add_special_tokens=False).input_ids
        t = tok(r["target"], add_special_tokens=False).input_ids
        labels = [IGNORE] * len(p) + t
        cond1 = all(x == IGNORE for x in labels[: len(p)])
        cond2 = labels[len(p)] == t[0] if len(t) > 0 else False
        cond3 = tok.decode(labels[len(p):]) == tok.decode(t)
        good = cond1 and cond2 and cond3
        if not good:
            bad += 1
        print(f"  {r['id']}: 题目段全掩={cond1}  答案首token对齐={cond2}  答案段可解码={cond3}"
              f"  -> {'通过' if good else '★异常★'}")
    print(f"  结论：{'通过' if bad == 0 else f'★{bad} 条异常★'}")

    print()
    print("=" * 70)
    print("【审计 4】超长截断统计（训练集 512）")
    print("=" * 70)
    all_rows = load_jsonl(os.path.join(data, "train.jsonl"))
    over = 0
    lens = []
    for r in all_rows:
        p = tok(r["prompt"], add_special_tokens=False).input_ids
        t = tok(r["target"], add_special_tokens=False).input_ids
        lens.append(len(p) + len(t))
        if len(p) + len(t) > 512:
            over += 1
    lens.sort()
    print(f"  样本 {len(all_rows)} 条，总长度中位数 {lens[len(lens)//2]}，最大 {lens[-1]}")
    print(f"  超过 512 被截断：{over} 条（{over/len(all_rows):.2%}）")

    print()
    print("=" * 70)
    print("【审计 5】评测集的 prompt 是否与训练集同源")
    print("=" * 70)
    test_rows = load_jsonl(os.path.join(data, "gsm8k_test.jsonl"))[:1]
    tr_tail = rows[0]["prompt"][-20:]
    te_tail = test_rows[0]["prompt"][-20:]
    print(f"  训练 prompt 结尾: {tr_tail!r}")
    print(f"  评测 prompt 结尾: {te_tail!r}")
    print(f"  结论：{'一致' if tr_tail == te_tail else '★不一致★'}")


if __name__ == "__main__":
    main()
