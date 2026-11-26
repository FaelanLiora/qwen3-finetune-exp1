"""方法一：全参数监督微调（Full SFT）

【在干什么】
用 6726 条「题目 + 标准答案」，让模型学会"按格式交卷"。
**全部 5.96 亿参数都参与更新**，没有冻结任何东西。

【超参（对齐 PPT 第 9 页）】
    学习率 2e-5      —— 全参微调步子必须小，大了会把预训练的本事冲掉
    轮数   3
    长度   512
    批次   4 × 累积 4 = 等效 16

【显存账】
    权重(半精度) 1.2G + 梯度 1.2G + 优化器状态 4.8G ≈ 7.2G
    加上中间激活值，24G 的卡很宽裕。

用法：
    python scripts/train_full.py                       # 正式训练，约 15-20 分钟
    python scripts/train_full.py --limit 64 --epochs 1 --out /root/autodl-tmp/outputs/full_smoke
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from sft_dataset import SFTDataset, make_collate


def build_training_args(**kw):
    """兼容不同版本的 transformers：新版叫 eval_strategy，旧版叫 evaluation_strategy。"""
    try:
        return TrainingArguments(**kw, eval_strategy="epoch")
    except TypeError:
        return TrainingArguments(**kw, evaluation_strategy="epoch")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/data")
    ap.add_argument("--out", default="/root/autodl-tmp/outputs/full")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只用前 N 条（冒烟测试）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---------- 模型 ----------
    print("加载分词器与模型 ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.config.use_cache = False          # 开梯度检查点时必须关掉缓存
    model.gradient_checkpointing_enable()   # 用时间换显存

    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数 {n_all/1e6:.1f}M，可训练 {n_train/1e6:.1f}M（{'全部' if n_all == n_train else '部分'}）")

    # ---------- 数据 ----------
    train_set = SFTDataset(os.path.join(args.data_dir, "train.jsonl"), tok, args.max_len)
    val_set = SFTDataset(os.path.join(args.data_dir, "val.jsonl"), tok, args.max_len)
    if args.limit > 0:
        train_set.rows = train_set.rows[: args.limit]
        val_set.rows = val_set.rows[: min(32, len(val_set.rows))]
    print(f"训练样本 {len(train_set)} 条，验证样本 {len(val_set)} 条")

    # ---------- 训练配置 ----------
    targs = build_training_args(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=make_collate(tok.pad_token_id),
    )

    # ---------- 开训 ----------
    print("\n开始训练 ...")
    trainer.train()

    # ---------- 保存 ----------
    print(f"\n保存模型到 {args.out} ...")
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    if train_set.n_truncated:
        print(f"注意：有 {train_set.n_truncated} 条样本因超过 {args.max_len} 长度被截断")

    print("完成。接下来评测：")
    print(f"  python scripts/eval.py --tag full --model {args.out}")


if __name__ == "__main__":
    main()
