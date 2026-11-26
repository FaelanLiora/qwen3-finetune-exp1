"""训练共用件：学习率曲线、验证、训练循环。

四个方法（全参 / LoRA / P-Tuning v2 / KL-LoRA）共用同一套训练循环，
只有"模型怎么包、损失怎么算"不同。这样能保证训练过程完全一致，对比才公平。
"""

from __future__ import annotations

import math
import time

import torch


def lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    """学习率曲线：先线性预热，再余弦下降到 0。"""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, max_batches: int = 200) -> float:
    """在验证集上算平均损失。"""
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


def train(model, loader, opt, total_steps: int, warmup: int, base_lr: float,
          grad_accum: int, log_every: int = 20) -> None:
    """标准训练循环：梯度累积 + 梯度裁剪 + 余弦学习率。"""
    model.train()
    step = 0
    m = 0
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

    if m > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
