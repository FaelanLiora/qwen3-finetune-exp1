"""方法二：LoRA（低秩适配）

【在干什么】
原始权重 W 全部冻结，只在旁边挂两个小矩阵 A、B，学一个"增量"：
        h = W x + (α / r) · B A x

    A: 1024 → 16  （降维）
    B: 16 → 1024  （升维，初始化为 0，所以训练开始时输出和基座完全一样）

【和全参微调的差别】
    全参：改 596.0M 个参数（100%）
    LoRA：改约 5M 个参数（约 1%）——相当于 GB 和 MB 的差别

【超参（对齐 PPT 第 9 页）】
    学习率 2e-4（比全参大 10 倍：参数少，胆子可以大）
    r = 16, α = 32（α/r = 2 是实际的缩放倍数）
    轮数 3 / 长度 512 / 批次 4 × 累积 4

【保存的东西】
只保存 adapter（几十 MB），不保存整个模型。评测时用 --adapter 挂上去。

用法：
    python scripts/train_lora.py
    python scripts/train_lora.py --limit 64 --epochs 1 --out /root/autodl-tmp/outputs/lora_smoke
"""

from __future__ import annotations

import argparse
import math
import os

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import train_common
from sft_dataset import SFTDataset, make_collate

# Qwen3 的线性层名字：注意力 4 个 + 前馈 3 个
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/data")
    ap.add_argument("--out", default="/root/autodl-tmp/outputs/lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---------- 模型 + LoRA ----------
    print("加载分词器与基座 ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.config.use_cache = False

    cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()      # LoRA + 梯度检查点必须加这一行

    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数 {n_all/1e6:.1f}M，可训练 {n_train/1e6:.2f}M（{n_train/n_all:.2%}）")

    # ---------- 数据 ----------
    train_set = SFTDataset(os.path.join(args.data_dir, "train.jsonl"), tok, args.max_len)
    val_set = SFTDataset(os.path.join(args.data_dir, "val.jsonl"), tok, args.max_len)
    if args.limit > 0:
        train_set.rows = train_set.rows[: args.limit]
        val_set.rows = val_set.rows[: min(32, len(val_set.rows))]
    print(f"训练样本 {len(train_set)} 条，验证样本 {len(val_set)} 条")

    collate = make_collate(tok.pad_token_id)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False,
                            collate_fn=collate, num_workers=2)

    # ---------- 优化器 ----------
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.03))
    print(f"每轮 {steps_per_epoch} 次更新，共 {total_steps} 次，预热 {warmup} 次")

    # ---------- 训练 ----------
    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        print(f"\n===== 第 {ep}/{args.epochs} 轮 =====", flush=True)
        train_common.train(model, train_loader, opt, total_steps, warmup,
                           args.lr, args.grad_accum)

        val_loss = train_common.evaluate(model, val_loader)
        print(f"第 {ep} 轮结束，验证集损失 = {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            print(f"  验证损失创新低，保存适配器到 {args.out} ...", flush=True)
            model.save_pretrained(args.out)      # 只存 adapter，几十 MB

    print(f"\n训练完成，最佳验证损失 {best_val:.4f}")
    print(f"接下来评测（注意加 --adapter）：\n"
          f"  python scripts/eval.py --tag lora --adapter {args.out}")


if __name__ == "__main__":
    main()
