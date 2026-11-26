"""公平基线：给原始模型看示例，它能不能自己"治好"字母偏好？

【要回答的问题】
全参微调把科学题从 43.09% 提到 58.87%（+15.8）。
但这 15.8 分里，有多少是"真的变强"，有多少只是"学会了按格式答"？

【办法：少样本提示（few-shot）】
给模型看 3 个「题目 + 选项 → 字母」的示例，再让它答真题。
如果原始模型看了示例也能到 58% 左右，
说明提升主要来自"格式/答题方式"，全参微调没有额外贡献。
如果它只到 47% 左右，而微调后仍是 58.87%，
说明全参微调确实多做了事。

【四种情况一次测完】
                      零示例（原始问法）   三示例（给例子）
    原始模型              ?                    ?
    全参微调              ?                    ?

同时记录「预测 A 的占比」——这是字母偏好的直接指标。

用法：
    python scripts/probe_fewshot.py --full /root/autodl-tmp/outputs/full
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ARC_SYSTEM

# 三个示例题（自己写的简单常识题，不来自 ARC 测试集，避免泄题）
EXAMPLES = [
    ("Which gas do plants mainly absorb for photosynthesis?",
     ["A", "B", "C", "D"],
     ["oxygen", "carbon dioxide", "nitrogen", "hydrogen"], "B"),
    ("At what temperature does water boil at standard pressure?",
     ["A", "B", "C", "D"],
     ["50 degrees", "80 degrees", "100 degrees", "120 degrees"], "C"),
    ("Which of these is a renewable energy source?",
     ["A", "B", "C", "D"],
     ["coal", "petroleum", "solar", "natural gas"], "C"),
]


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def question_block(question: str, labels: list[str], texts: list[str]) -> str:
    body = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
    return f"{question.strip()}\n\n{body}\n\n请只回答正确选项的字母。"


def build_prompt(tok, question, labels, texts, with_examples: bool) -> str:
    """拼对话。with_examples=True 时前面插入 3 轮「问—答」示例。"""
    msgs = [{"role": "system", "content": ARC_SYSTEM}]
    if with_examples:
        for q, labs, txts, ans in EXAMPLES:
            msgs.append({"role": "user", "content": question_block(q, labs, txts)})
            msgs.append({"role": "assistant", "content": ans})
    msgs.append({"role": "user", "content": question_block(question, labels, texts)})
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


@torch.no_grad()
def score(model, tok, prompts: list[str], label_sets: list[list[str]],
          batch_size: int = 16) -> list[str]:
    """批量算四个字母的概率，取最大的。左填充，取最后一个位置。"""
    cache: dict[str, list[int]] = {}

    def ids_of(lab: str) -> list[int]:
        if lab not in cache:
            cache[lab] = sorted({
                tok.encode(lab, add_special_tokens=False)[0],
                tok.encode(" " + lab, add_special_tokens=False)[0],
            })
        return cache[lab]

    preds = []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s:s + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=3072).to(model.device)
        logp = torch.log_softmax(model(**enc).logits[:, -1].float(), dim=-1)
        for j, labels in enumerate(label_sets[s:s + batch_size]):
            preds.append(max(labels, key=lambda lab: max(logp[j][t].item() for t in ids_of(lab))))
        print(f"\r  {min(s+batch_size, len(prompts))}/{len(prompts)}", end="", flush=True)
    print()
    return preds


def run_condition(model, tok, rows, with_examples: bool) -> tuple[float, float]:
    prompts = [build_prompt(tok, r["question"], r["labels"], r["texts"], with_examples)
               for r in rows]
    preds = score(model, tok, prompts, [r["labels"] for r in rows])
    acc = sum(p == r["answer"] for p, r in zip(preds, rows)) / len(rows)
    a_ratio = sum(1 for p in preds if p == "A") / len(preds)
    return acc, a_ratio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/autodl-tmp/data/arc_test.jsonl")
    ap.add_argument("--base_model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--full", default="/root/autodl-tmp/outputs/full")
    ap.add_argument("--out", default="/root/autodl-tmp/results")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    rows = load_jsonl(args.data)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"题目 {len(rows)} 道\n")

    tok = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    table = []
    for name, path in [("原始模型", args.base_model), ("全参微调", args.full)]:
        print(f"===== {name} =====")
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda").eval()

        print("  零示例 ...")
        acc0, a0 = run_condition(model, tok, rows, False)
        print(f"  → 正确率 {acc0:.4f}，预测A占比 {a0:.3f}")

        print("  三示例 ...")
        acc3, a3 = run_condition(model, tok, rows, True)
        print(f"  → 正确率 {acc3:.4f}，预测A占比 {a3:.3f}\n")

        table.append({"模型": name, "零示例正确率": round(acc0, 4), "零示例A占比": round(a0, 3),
                      "三示例正确率": round(acc3, 4), "三示例A占比": round(a3, 3)})

        del model
        torch.cuda.empty_cache()

    print("=" * 70)
    print("汇总")
    print("=" * 70)
    keys = list(table[0].keys())
    print(" | ".join(keys))
    for r in table:
        print(" | ".join(str(r[k]) for k in keys))

    with open(os.path.join(args.out, "fair_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)

    print("""
===== 怎么读 =====
· 原始模型「三示例」接近微调后的「零示例」  → 提升主要来自"学会按格式答"，全参微调无额外贡献
· 原始模型「三示例」仍明显低于微调后        → 全参微调确实多做了事
· 三示例后 A 占比明显下降                 → 字母偏好可以被"看例子"治好，不是模型固有顽疾
""")


if __name__ == "__main__":
    main()
