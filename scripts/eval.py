"""统一评测脚本：GSM8K（生成式）+ ARC-Challenge（对数概率）。

【为什么四个方法都必须用这一个脚本】
评测协议一致，是实验可信的前提。本脚本固定了四件事：
    1. 同一份 prompt（build_data.py 里预先算好，存在 jsonl 里）
    2. 同样的贪心解码（do_sample=False，结果可复现）
    3. 同样的生成长度 max_new_tokens=512（对齐 PPT 的"评测长度 512"）
    4. 同样的答案抽取与判分（common.py，抽数字 + 精确匹配）

【两种评测方式的区别】
- GSM8K：让模型自由生成推理过程，再从文本里抓 #### 后面的数字 —— 生成式评测
- ARC  ：只让模型在 A/B/C/D 四个字母里选一个，比较四个字母的"对数概率" —— 判别式评测
        为什么不用生成？因为 0.6B 小模型容易在选择题上"解释一大堆"，
        直接比概率更稳定、更快、也更好复现。

用法：
    python scripts/eval.py --tag base
    python scripts/eval.py --tag base --limit 20          # 冒烟测试：只评 20 条
    python scripts/eval.py --tag lora --adapter /root/autodl-tmp/outputs/lora
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


# ---------------------------------------------------------------- 工具函数
def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model(model_path: str, adapter: str | None = None):
    """加载基座模型；如果给了 adapter 目录，就把 LoRA / Prefix 挂上去。"""
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"已加载适配器: {adapter}")
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n/1e6:.1f}M")
    return tok, model


# ---------------------------------------------------------------- GSM8K
@torch.no_grad()
def gsm8k_eval(model, tok, rows, batch_size: int, max_new_tokens: int) -> list[dict]:
    out_rows = []
    total = len(rows)
    t0 = time.time()

    for start in range(0, total, batch_size):
        batch = rows[start:start + batch_size]
        prompts = [r["prompt"] for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)

        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,                     # 贪心解码，保证可复现
            pad_token_id=tok.pad_token_id,
        )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)

        for r, text in zip(batch, texts):
            pred = predict_final_answer(text)
            out_rows.append({
                "id": r["id"],
                "gold": r["final"],
                "pred": pred,
                "correct": is_correct(pred, r["final"]),
                "output": text,
            })

        done = min(start + batch_size, total)
        acc = sum(x["correct"] for x in out_rows) / len(out_rows)
        print(f"\r  GSM8K {done}/{total}  acc={acc:.4f}  用时 {time.time()-t0:.0f}s",
              end="", flush=True)

    print()
    return out_rows


# ---------------------------------------------------------------- ARC
@torch.no_grad()
def arc_eval(model, tok, rows) -> list[dict]:
    """对每道题算 4 个选项字母的对数概率，取最大的那个作为预测。"""
    out_rows = []
    total = len(rows)
    t0 = time.time()

    # 选项标签有两种写法：多数题是字母 A/B/C/D，少数题是数字 1/2/3/4。
    # 所以不能写死字母表，要按每道题自己的标签去查 token 编号。
    label_ids: dict[str, list[int]] = {}

    def ids_of(lab: str) -> list[int]:
        """取某个标签对应的 token 编号，同时算「带空格」和「不带空格」两种写法。"""
        if lab not in label_ids:
            label_ids[lab] = sorted({
                tok.encode(lab, add_special_tokens=False)[0],
                tok.encode(" " + lab, add_special_tokens=False)[0],
            })
        return label_ids[lab]

    for i, r in enumerate(rows):
        enc = tok(r["prompt"], return_tensors="pt").to(model.device)
        logits = model(**enc).logits[0, -1]
        logp = torch.log_softmax(logits.float(), dim=-1)

        best_lab, best_score = None, -1e9
        for lab in r["labels"]:
            score = max(logp[t].item() for t in ids_of(lab))
            if score > best_score:
                best_lab, best_score = lab, score

        out_rows.append({
            "id": r["id"],
            "gold": r["answer"],
            "pred": best_lab,
            "correct": best_lab == r["answer"],
        })

        if (i + 1) % 50 == 0 or i + 1 == total:
            acc = sum(x["correct"] for x in out_rows) / len(out_rows)
            print(f"\r  ARC  {i+1}/{total}  acc={acc:.4f}  用时 {time.time()-t0:.0f}s",
                  end="", flush=True)

    print()
    return out_rows


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--adapter", default=None, help="LoRA / Prefix 适配器目录，Base 和 Full SFT 不填")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/data")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/results")
    ap.add_argument("--tag", default="base", help="这次评测的名字：base / full / lora / ptuning / kllora")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只评前 N 条（冒烟测试）")
    ap.add_argument("--skip_gsm8k", action="store_true",
                    help="跳过数学评测，直接读上次存下的成绩（用于只补考科学）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.limit > 0:            # 只考几道题的"试跑"，成绩标记为 demo_，不污染正式成绩表
        args.tag = "demo_" + args.tag
    tok, model = load_model(args.model, args.adapter)

    # ---- GSM8K ----
    gsm_path = os.path.join(args.out_dir, f"{args.tag}_gsm8k.jsonl")
    if args.skip_gsm8k:
        # 上次已经考过数学了，直接读盘上的成绩，不重复跑（省十分钟）
        print(f"\n[1/2] 跳过 GSM8K，读取已有成绩：{gsm_path}")
        gsm_out = load_jsonl(gsm_path)
        gsm_acc = sum(x["correct"] for x in gsm_out) / len(gsm_out)
        print(f"  已读取 {len(gsm_out)} 条，正确率 {gsm_acc:.4f}")
    else:
        gsm_rows = load_jsonl(os.path.join(args.data_dir, "gsm8k_test.jsonl"))
        if args.limit > 0:
            gsm_rows = gsm_rows[: args.limit]
        print(f"\n[1/2] GSM8K 评测（{len(gsm_rows)} 条）...")
        gsm_out = gsm8k_eval(model, tok, gsm_rows, args.batch_size, args.max_new_tokens)
        gsm_acc = sum(x["correct"] for x in gsm_out) / len(gsm_out)

        with open(gsm_path, "w", encoding="utf-8") as f:
            for x in gsm_out:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # ---- ARC ----
    arc_rows = load_jsonl(os.path.join(args.data_dir, "arc_test.jsonl"))
    if args.limit > 0:
        arc_rows = arc_rows[: args.limit]
    print(f"[2/2] ARC-Challenge 评测（{len(arc_rows)} 条）...")
    arc_out = arc_eval(model, tok, arc_rows)
    arc_acc = sum(x["correct"] for x in arc_out) / len(arc_out)

    with open(os.path.join(args.out_dir, f"{args.tag}_arc.jsonl"), "w", encoding="utf-8") as f:
        for x in arc_out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # ---- 汇总 ----
    summary = {
        "tag": args.tag,
        "adapter": args.adapter or "",
        "gsm8k_acc": round(gsm_acc, 4),
        "gsm8k_n": len(gsm_out),
        "arc_acc": round(arc_acc, 4),
        "arc_n": len(arc_out),
        "limit": args.limit,
    }
    print("\n===== 结果 =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(os.path.join(args.out_dir, f"{args.tag}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 追加到总的汇总表，方便最后直接贴进报告
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
