"""比较各模型在科学题上的"字母概率分布"与基座的接近程度。

【为什么补这个检验】
之前用 Cohen's kappa 比较"最终选了哪个字母"（argmax），得出"KL-LoRA 更偏离基座"的结论。
但 argmax 是粗粒度的：分布可能整体向 A 偏移了，却没偏移到翻转 argmax 的程度。

本脚本直接比较**四个字母的概率向量**：
    对每道题，算出 p(A), p(B), p(C), p(D)（在四个字母上重新归一化）
    再计算它与基座分布的散度（JS 散度，对称且有界）

判据：
    JS(KL-LoRA, 基座) < JS(LoRA, 基座)  →  KL-LoRA 在分布层面确实更接近基座，
                                           "被拉回基座"的假设在分布层面成立

用法：
    python scripts/probe_letter_dist.py
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import transformers

_DTYPE_KW = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def letter_probs(model, tok, prompt: str, labels: list[str]) -> dict[str, float]:
    """返回四个字母上的归一化概率。"""
    enc = tok(prompt, return_tensors="pt").to(model.device)
    logits = model(**enc).logits[0, -1].float()

    ids = []
    for lab in labels:
        tid = tok.encode(lab, add_special_tokens=False)[0]
        ids.append(tid)
    sub = torch.softmax(logits[ids], dim=-1)
    return {lab: float(p) for lab, p in zip(labels, sub)}


def js_div(p: dict[str, float], q: dict[str, float], labels: list[str]) -> float:
    """JS 散度（对称，0 表示完全相同）。"""
    eps = 1e-12
    m = {k: 0.5 * (p[k] + q[k]) for k in labels}
    kl_pm = sum(p[k] * math.log((p[k] + eps) / (m[k] + eps)) for k in labels)
    kl_qm = sum(q[k] * math.log((q[k] + eps) / (m[k] + eps)) for k in labels)
    return 0.5 * kl_pm + 0.5 * kl_qm


def run_model(path: str, adapter: str | None, tok, rows, model_path: str) -> dict:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cuda", **{_DTYPE_KW: torch.bfloat16}).eval()
    model.config.use_cache = False
    if adapter:
        model = PeftModel.from_pretrained(model, adapter).eval()

    probs, acc = [], 0
    for i, r in enumerate(rows):
        p = letter_probs(model, tok, r["prompt"], r["labels"])
        probs.append(p)
        if max(p, key=p.get) == r["answer"]:
            acc += 1
        if (i + 1) % 300 == 0:
            print(f"\r  {i+1}/{len(rows)}", end="", flush=True)
    print()

    del model
    torch.cuda.empty_cache()
    return {"probs": probs, "acc": acc / len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--data", default="/root/autodl-tmp/data/arc_test.jsonl")
    ap.add_argument("--out", default="/root/autodl-tmp/outputs")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = load_jsonl(args.data)
    if args.limit > 0:
        rows = rows[: args.limit]
    labels = rows[0]["labels"]
    print(f"题目 {len(rows)} 道\n")

    tok = AutoTokenizer.from_pretrained(args.model)

    targets = [
        ("基座", None),
        ("LoRA", os.path.join(args.out, "lora")),
        ("KL-LoRA", os.path.join(args.out, "kllora")),
    ]

    results = {}
    for name, adapter in targets:
        print(f"===== {name} =====", flush=True)
        results[name] = run_model(None, adapter, tok, rows, args.model)
        print(f"  正确率 {results[name]['acc']:.4f}")

    base = results["基座"]["probs"]
    print("\n" + "=" * 66)
    print("与基座的分布距离（JS 散度，越小越接近）")
    print("=" * 66)
    print(f"{'模型':<10}{'JS 散度':>12}{'平均 p(A)':>14}{'正确率':>10}")

    for name in ["基座", "LoRA", "KL-LoRA"]:
        pr = results[name]["probs"]
        # 注意：每道题的选项标签可能不同（多数是 A/B/C/D，少数是 1/2/3/4），
        # 所以散度必须按每道题自己的标签计算，不能用统一的标签表。
        js = sum(js_div(pr[i], base[i], rows[i]["labels"]) for i in range(len(rows))) / len(rows)
        pa = sum(p.get("A", 0.0) for p in pr) / len(pr)
        print(f"{name:<10}{js:>12.4f}{pa:>14.4f}{results[name]['acc']:>10.4f}")

    print("""
===== 怎么读 =====
· JS(KL-LoRA, 基座) < JS(LoRA, 基座)  → KL-LoRA 在分布层面更接近基座，
                                        "被拉回基座"的假设在分布层面成立，
                                        此前基于 argmax 的 kappa 结论需要修正
· 反之                                → 假设在分布层面也被否定，只能靠 λ 消融进一步排查
""")


if __name__ == "__main__":
    main()
