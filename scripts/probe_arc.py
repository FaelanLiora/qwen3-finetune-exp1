"""检验科学题分数是否可信：两种评分方式同时测。

【背景】
科学题目前的评分方式：看模型在 A/B/C/D 四个字母上的概率，取最大的。
这个方式有个前提：**模型本来就想输出一个字母**。
如果模型其实想说「The answer is C」，那它的字母概率就是噪声，
测出来的分数不代表它真实的科学水平。

【本脚本同时测两件事】
1. 字母概率质量：四个选项字母占了多少概率
       —— 很低（比如 5%）说明评分没在工作
       —— 很高（比如 90%）说明评分有效
2. 生成式复核：让模型自由写一句话，再从文字里抓字母
       —— 和第一种方法对比，看分数是否一致

两种方法都对同一个模型跑，再对比两个模型，就能分清：
    「模型真的变强了」 还是 「评分方式只在某个模型身上有效」

用法：
    python scripts/probe_arc.py --tag full --model /root/autodl-tmp/outputs/full
    python scripts/probe_arc.py            # 只测原始模型
"""

from __future__ import annotations

import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_letter(text: str, labels: list[str]) -> str | None:
    """从模型输出里抓第一个出现的选项字母/数字。"""
    best, best_pos = None, 10 ** 9
    for lab in labels:
        m = re.search(re.escape(lab), text)
        if m and m.start() < best_pos:
            best, best_pos = lab, m.start()
    return best


@torch.no_grad()
def logprob_pass(model, tok, rows: list[dict]) -> list[dict]:
    """评分方式一：看四个字母的对数概率。同时记录「字母概率质量」。"""
    cache: dict[str, list[int]] = {}

    def ids_of(lab: str) -> list[int]:
        if lab not in cache:
            cache[lab] = sorted({
                tok.encode(lab, add_special_tokens=False)[0],
                tok.encode(" " + lab, add_special_tokens=False)[0],
            })
        return cache[lab]

    out = []
    for i, r in enumerate(rows):
        enc = tok(r["prompt"], return_tensors="pt").to(model.device)
        logits = model(**enc).logits[0, -1]
        logp = torch.log_softmax(logits.float(), dim=-1)

        scores, mass = {}, 0.0
        for lab in r["labels"]:
            ids = ids_of(lab)
            scores[lab] = max(logp[t].item() for t in ids)
            mass += sum(logp[t].exp().item() for t in ids)

        pred = max(scores, key=scores.get)
        out.append({
            "id": r["id"], "gold": r["answer"], "pred": pred,
            "correct": pred == r["answer"],
            "letter_mass": mass,          # 四个字母一共占了多少概率
            "top_logp": max(scores.values()),
        })
        if (i + 1) % 200 == 0:
            print(f"\r  字母概率 {i+1}/{len(rows)}", end="", flush=True)
    print()
    return out


@torch.no_grad()
def generate_pass(model, tok, rows: list[dict], batch_size: int = 16,
                  max_new_tokens: int = 16) -> list[dict]:
    """评分方式二：让模型自由写，再从文字里抓字母。"""
    out = []
    for s in range(0, len(rows), batch_size):
        batch = rows[s:s + batch_size]
        enc = tok([r["prompt"] for r in batch], return_tensors="pt",
                  padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

        for r, t in zip(batch, texts):
            pred = parse_letter(t, r["labels"])
            out.append({
                "id": r["id"], "gold": r["answer"], "pred": pred,
                "correct": pred == r["answer"], "text": t,
            })
        print(f"\r  生成式 {min(s+batch_size, len(rows))}/{len(rows)}", end="", flush=True)
    print()
    return out


def summarize(name: str, lp: list[dict], gn: list[dict]) -> dict:
    n = len(lp)
    acc_lp = sum(x["correct"] for x in lp) / n
    acc_gn = sum(x["correct"] for x in gn) / n
    mass = sum(x["letter_mass"] for x in lp) / n
    agree = sum(a["pred"] == b["pred"] for a, b in zip(lp, gn)) / n
    return {
        "模型": name,
        "字母概率质量": round(mass, 3),
        "对数概率法正确率": round(acc_lp, 4),
        "生成式正确率": round(acc_gn, 4),
        "两法一致率": round(agree, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/autodl-tmp/data/arc_test.jsonl")
    ap.add_argument("--base_model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--model", default=None, help="要对比的模型目录，如 outputs/full")
    ap.add_argument("--out", default="/root/autodl-tmp/results")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = load_jsonl(args.data)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"题目 {len(rows)} 道\n")

    tok = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    targets = [("原始模型", args.base_model)]
    if args.model:
        targets.append(("微调后", args.model))

    results = []
    for name, path in targets:
        print(f"===== 测 {name}：{path} =====")
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda").eval()

        print("[1/2] 对数概率评分 ...")
        lp = logprob_pass(model, tok, rows)
        print("[2/2] 生成式评分 ...")
        gn = generate_pass(model, tok, rows)

        results.append(summarize(name, lp, gn))
        with open(os.path.join(args.out, f"probe_{name}.json"), "w", encoding="utf-8") as f:
            json.dump({"logprob": lp, "generate": gn}, f, ensure_ascii=False)

        del model
        torch.cuda.empty_cache()

    print("\n===== 汇总 =====")
    keys = list(results[0].keys())
    print(" | ".join(keys))
    for r in results:
        print(" | ".join(str(r[k]) for k in keys))

    print("""
怎么读这张表：
  · 「字母概率质量」低于 0.2  → 该模型的字母评分基本是噪声，分数不可比
  · 「两种方法一致率高」      → 分数可信
  · 「两种方法一致率低」      → 评分方式本身有问题，需要换
""")


if __name__ == "__main__":
    main()
