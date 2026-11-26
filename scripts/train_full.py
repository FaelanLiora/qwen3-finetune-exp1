"""方法一：全参数监督微调（Full SFT）—— 手写训练循环

【为什么不用 transformers 自带的 Trainer】
1. transformers 第 5 版把训练配置接口改了（warmup_ratio 等被移除），自带 Trainer 不稳定
2. 后面的 KL-LoRA 需要「同一批数据两次前向 + 自定义损失」，自带 Trainer 反而不好改
所以四个方法统一用这套手写循环，只有"模型怎么包、损失怎么算"不同。

【训练在干什么】
    1. 取一批数据（4 条）→ 模型前向 → 算出每条答案的交叉熵损失
    2. 累积 4 批（共 16 条）再更新一次参数（这叫梯度累积，省显存）
    3. 学习率从 0 慢慢升上去（预热），再按余弦曲线慢慢降下来
    4. 每轮结束在验证集上算一次损失，损失最小的那个版本存下来

【超参（对齐 PPT 第 9 页）】
    学习率 2e-5 / 轮数 3 / 长度 512 / 批次 4 × 累积 4

用法：
    python scripts/train_full.py
    python scripts/train_full.py --limit 64 --epochs 1 --out /root/autodl-tmp/outputs/full_smoke
"""

import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_dataset import SFTDataset, make_collate


def lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    """学习率曲线：先线性预热，再余弦下降到 0。"""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, max_batches: int = 200) -> float:
    """在验证集上算平均损失（越小说明拟合得越好）。"""
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        total += loss.item()
        n += 1
    model.train()
    return total / max(1, n)


def train(model, loader, opt, total_steps, warmup, base_lr, grad_accum, log_every=20) -> None:
    model.train()
    step = 0
    m = 0                 # 累积了几批
    loss_sum, n_loss = 0.0, 0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)

    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss

        loss_sum += loss.item()
        n_loss += 1
        (loss / grad_accum).backward()
        m += 1

        if m == grad_accum:
            m = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = lr_at(step, total_steps, warmup, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % log_every == 0:
                print(f"  step {step}/{total_steps}  loss={loss_sum/max(1,n_loss):.4f}"
                      f"  lr={lr:.2e}  {time.time()-t0:.0f}s", flush=True)
                loss_sum, n_loss = 0.0, 0

    # 收尾：最后不足 grad_accum 的残余梯度也更新一次
    if m > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/data")
    ap.add_argument("--out", default="/root/autodl-tmp/outputs/full")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只用前 N 条（冒烟测试）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---------- 模型 ----------
    print("加载分词器与模型 ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.config.use_cache = False

    # 梯度检查点：用时间换显存。不同版本方法名不同，做兼容。
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("已启用梯度检查点")
    except Exception as e:
        print(f"未启用梯度检查点（不影响正确性，只是更吃显存）: {e}")

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

    collate = make_collate(tok.pad_token_id)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=False, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False,
                            collate_fn=collate, num_workers=2)

    # ---------- 优化器 ----------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.03))
    print(f"每轮 {steps_per_epoch} 次参数更新，共 {total_steps} 次，预热 {warmup} 次")

    # ---------- 训练 ----------
    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        print(f"\n===== 第 {ep}/{args.epochs} 轮 =====", flush=True)
        train(model, train_loader, opt, total_steps, warmup, args.lr, args.grad_accum)

        val_loss = evaluate(model, val_loader)
        print(f"第 {ep} 轮结束，验证集损失 = {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            print(f"  验证损失创新低，保存模型到 {args.out} ...", flush=True)
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)

    if train_set.n_truncated:
        print(f"注意：有 {train_set.n_truncated} 条样本超过 {args.max_len} 长度被截断")

    print(f"\n训练完成，最佳验证损失 {best_val:.4f}")
    print(f"接下来评测：\n  python scripts/eval.py --tag full --model {args.out}")


if __name__ == "__main__":
    main()
