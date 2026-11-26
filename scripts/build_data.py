"""构建实验数据：下载 GSM8K / ARC-Challenge，模板化，切分，存 jsonl。

【输入】HuggingFace 数据集 openai/gsm8k、allenai/ai2_arc
【输出】<out_dir>/ 下四个文件
        train.jsonl   约 6726 条（GSM8K train 的 90%）
        val.jsonl     约  747 条（GSM8K train 的 10%，用于选 checkpoint）
        gsm8k_test.jsonl  1319 条
        arc_test.jsonl    1172 条
        meta.json     切分信息（seed、条数、模型路径），保证可复现

【为什么 prompt 要在这里算好】
训练时只需要「prompt 的 token 数」来做 loss mask，把字符串直接存下来，
就不用每次都重新套模板，也避免了四个方法各自套模板导致的不一致。

用法：
    python scripts/build_data.py
    python scripts/build_data.py --limit 50      # 先跑小样本验证流程
"""

from __future__ import annotations

import argparse
import json
import os
import random

from datasets import load_dataset
from transformers import AutoTokenizer

from common import (
    arc_choices,
    arc_messages,
    gsm8k_messages,
    label_final_answer,
)


def apply_template(tok, messages) -> str:
    """套 Qwen3 的 chat 模板。

    enable_thinking=False：关掉思考模式。理由：训练长度只有 512，
    思考段会把答案挤掉；四个方法统一关闭，保证公平。
    """
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def save_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {path}  ({len(rows)} 条)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B",
                    help="tokenizer 所在目录")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1, help="9:1 -> 验证集占 10%")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 条（调试用）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[1/4] 加载 tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"      eos_token = {tok.eos_token!r}")

    # ---------------- GSM8K ----------------
    print("[2/4] 下载 / 读取 GSM8K ...")
    ds_train = load_dataset("openai/gsm8k", "main", split="train")
    ds_test = load_dataset("openai/gsm8k", "main", split="test")
    print(f"      train={len(ds_train)}  test={len(ds_test)}")

    # 9:1 切分，seed 固定 —— 四个方法共用同一份，保证可比
    indices = list(range(len(ds_train)))
    random.Random(args.seed).shuffle(indices)
    n_val = int(len(indices) * args.val_ratio)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    def make_train_rows(idxs: list[int], prefix: str) -> list[dict]:
        rows = []
        for i in idxs:
            ex = ds_train[i]
            gold = label_final_answer(ex["answer"])
            if gold is None:          # 标注异常，跳过
                continue
            rows.append({
                "id": f"{prefix}-{i}",
                "question": ex["question"],
                "prompt": apply_template(tok, gsm8k_messages(ex["question"])),
                # target = 完整解题过程 + 结束符，让模型学会"推理完就停"
                "target": ex["answer"].strip() + tok.eos_token,
                "final": gold,
            })
        return rows

    if args.limit > 0:
        train_idx, val_idx = train_idx[: args.limit], val_idx[: args.limit]

    print("[3/4] 生成 GSM8K jsonl ...")
    train_rows = make_train_rows(train_idx, "train")
    val_rows = make_train_rows(val_idx, "val")
    test_rows = []
    for i in range(len(ds_test)):
        ex = ds_test[i]
        gold = label_final_answer(ex["answer"])
        if gold is None:
            continue
        test_rows.append({
            "id": f"test-{i}",
            "question": ex["question"],
            "prompt": apply_template(tok, gsm8k_messages(ex["question"])),
            "final": gold,
            "answer": ex["answer"],
        })

    save_jsonl(os.path.join(args.out_dir, "train.jsonl"), train_rows)
    save_jsonl(os.path.join(args.out_dir, "val.jsonl"), val_rows)
    save_jsonl(os.path.join(args.out_dir, "gsm8k_test.jsonl"), test_rows)

    # ---------------- ARC-Challenge ----------------
    print("[4/4] 下载 / 读取 ARC-Challenge ...")
    ds_arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    print(f"      test={len(ds_arc)}")
    arc_rows = []
    for row in ds_arc:
        pairs = arc_choices(row)
        arc_rows.append({
            "id": row["id"],
            "question": row["question"],
            "labels": [lab for lab, _ in pairs],
            "texts": [txt for _, txt in pairs],
            "answer": row["answerKey"],
            "prompt": apply_template(tok, arc_messages(row)),
        })
    save_jsonl(os.path.join(args.out_dir, "arc_test.jsonl"), arc_rows)

    # ---------------- meta ----------------
    meta = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "model": args.model,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_gsm8k_test": len(test_rows),
        "n_arc_test": len(arc_rows),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n===== 完成 =====")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    # 打印一条样本，方便肉眼检查格式
    print("\n===== 训练样本示例 =====")
    s = train_rows[0]
    print("prompt:")
    print(s["prompt"])
    print("--- target ---")
    print(s["target"])
    print("--- final ---", s["final"])


if __name__ == "__main__":
    main()
