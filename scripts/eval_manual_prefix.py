"""用手写前缀路径评测 P-Tuning 模型（绕开 peft 的生成接口）。

【为什么单独一个文件】
peft 的前缀注入在训练（整段前向）时正常，但在逐字生成时与 transformers 5 的
缓存接口不匹配，导致模型陷入无限重复。本脚本用手写前缀 + 自己控制的解码，
得到 P-Tuning 的真实分数。

【评测协议与 eval.py 完全一致】
    GSM8K：贪心解码，max_new_tokens=512，抽 #### 后的数字精确匹配
    ARC  ：比较 A/B/C/D 四个字母的对数概率

用法：
    python scripts/eval_manual_prefix.py --adapter /root/autodl-tmp/outputs/ptuning6 --tag ptuning
    python scripts/eval_manual_prefix.py --adapter ... --tag ptuning --limit 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import is_correct, predict_final_answer
from decode_manual import load_trained_prefix, generate_one


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def arc_predict(model, prefix, tok, prompt: str, labels: list[str]) -> str:
    """带前缀前向一次，取最后一个位置的字母概率。"""
    nv = prefix.n_tokens
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    bsz, seq_len = ids.shape

    cache = prefix.build_cache(bsz, model.device)
    mask = torch.ones(bsz, nv + seq_len, dtype=torch.long, device=model.device)
    pos = torch.arange(nv, nv + seq_len, device=model.device)

    out = model(input_ids=ids, attention_mask=mask,
                past_key_values=cache, cache_position=pos)
    logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)

    cache_ids: dict[str, list[int]] = {}
    best, best_score = None, -1e9
    for lab in labels:
        if lab not in cache_ids:
            cache_ids[lab] = sorted({
                tok.encode(lab, add_special_tokens=False)[0],
                tok.encode(" " + lab, add_special_tokens=False)[0],
            })
        s = max(logp[t].item() for t in cache_ids[lab])
        if s > best_score:
            best, best_score = lab, s
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data_dir", default="/root/autodl-tmp/data")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/results")
    ap.add_argument("--tag", default="ptuning")
    ap.add_argument("--n_tokens", type=int, default=20)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_arc", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("加载基座与训练好的前缀 ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    model.config.use_cache = False

    prefix = load_trained_prefix(args.adapter, model.config, args.n_tokens, model.device)
    n_par = prefix.k.numel() + prefix.v.numel()
    print(f"前缀参数 {n_par/1e6:.3f}M")

    # ---------------- GSM8K ----------------
    rows = load_jsonl(os.path.join(args.data_dir, "gsm8k_test.jsonl"))
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"\n[1/2] GSM8K 评测（{len(rows)} 条）...", flush=True)
    out, t0 = [], time.time()
    for i, r in enumerate(rows):
        text = generate_one(model, prefix, tok, r["prompt"], args.max_new)
        pred = predict_final_answer(text)
        out.append({"id": r["id"], "gold": r["final"], "pred": pred,
                    "correct": is_correct(pred, r["final"]), "output": text})
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            acc = sum(x["correct"] for x in out) / len(out)
            print(f"\r  {i+1}/{len(rows)}  acc={acc:.4f}  {time.time()-t0:.0f}s",
                  end="", flush=True)
    print()
    gsm_acc = sum(x["correct"] for x in out) / len(out)

    with open(os.path.join(args.out_dir, f"{args.tag}_gsm8k.jsonl"), "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # ---------------- ARC ----------------
    if args.skip_arc:
        arc_acc, n_arc = -1.0, 0
    else:
        arc_rows = load_jsonl(os.path.join(args.data_dir, "arc_test.jsonl"))
        if args.limit > 0:
            arc_rows = arc_rows[: args.limit]
        print(f"[2/2] ARC-Challenge 评测（{len(arc_rows)} 条）...", flush=True)
        arc_out, t0 = [], time.time()
        for i, r in enumerate(arc_rows):
            pred = arc_predict(model, prefix, tok, r["prompt"], r["labels"])
            arc_out.append({"id": r["id"], "gold": r["answer"], "pred": pred,
                            "correct": pred == r["answer"]})
            if (i + 1) % 50 == 0 or i + 1 == len(arc_rows):
                acc = sum(x["correct"] for x in arc_out) / len(arc_out)
                print(f"\r  {i+1}/{len(arc_rows)}  acc={acc:.4f}  {time.time()-t0:.0f}s",
                      end="", flush=True)
        print()
        arc_acc, n_arc = sum(x["correct"] for x in arc_out) / len(arc_out), len(arc_out)

        with open(os.path.join(args.out_dir, f"{args.tag}_arc.jsonl"), "w", encoding="utf-8") as f:
            for x in arc_out:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    summary = {
        "tag": args.tag,
        "adapter": args.adapter,
        "gsm8k_acc": round(gsm_acc, 4),
        "gsm8k_n": len(out),
        "arc_acc": round(arc_acc, 4),
        "arc_n": n_arc,
        "limit": args.limit,
    }
    print("\n===== 结果 =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(os.path.join(args.out_dir, f"{args.tag}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(args.out_dir, "summary.csv")
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if is_new:
            w.writeheader()
        w.writerow(summary)
    print(f"\n已追加到汇总表: {csv_path}")


if __name__ == "__main__":
    main()
