"""方法四：KL-LoRA（自研改进）

【和普通 LoRA 的唯一区别：损失函数】

    普通 LoRA   ：L = L_CE
    KL-LoRA    ：L = L_CE + λ · D_KL(q ‖ p)

其中：
    q = 关掉 LoRA 时的分布（基座，冻结，作为"锚点"）
    p = 开着 LoRA 时的分布（正在训练）
    D_KL(q‖p) = Σ_v q(v) · log[ q(v) / p(v) ]     —— 在答案 token 的位置上算，再取平均

【为什么这么写】
只用 CE 训练时，模型会一路朝"数学题最优"漂移，漂远了通用能力就受损。
加上 KL 惩罚后，每一步都在说："可以为了数学调参数，但输出的概率分布别离基座太远。"
λ 就是那个旋钮：λ=0 退化成普通 LoRA；λ 越大越保守。

【实现要点（答辩会问）】
1. q 必须 no_grad + 关掉适配器，不能让它产生梯度
2. KL 只在答案 token 上算（和 CE 的位置一致），否则会被大量 prompt 位置稀释
3. 全词表 15 万维，逐位置算 KL 很吃显存，所以分块累加

用法：
    python scripts/train_kl_lora.py                    # λ=0.1
    python scripts/train_kl_lora.py --lam 0.5 --out /root/autodl-tmp/outputs/kllora_05
    python scripts/train_kl_lora.py --limit 64 --epochs 1 --out /root/autodl-tmp/outputs/kl_smoke
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import train_common
from sft_dataset import IGNORE_INDEX, SFTDataset, make_collate

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def ce_and_kl(logits_p, logits_q, labels, lam: float, chunk: int = 64):
    """算 L_CE + λ·KL(q‖p)，只在答案位置上算。

    logits_p / logits_q : [B, T, V]
    labels              : [B, T]，题目位置为 -100
    """
    # 下一个 token 预测：位置 t 的输出对应标签 t+1
    p_l = logits_p[:, :-1].reshape(-1, logits_p.size(-1))
    q_l = logits_q[:, :-1].reshape(-1, logits_q.size(-1))
    y = labels[:, 1:].reshape(-1)

    keep = y.ne(IGNORE_INDEX)
    idx = keep.nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return None, 0.0

    # 交叉熵（用官方实现，省显存）
    ce = F.cross_entropy(p_l[keep].float(), y[keep], reduction="mean")

    # KL 散度：只取答案位置，分块累加，避免一次性开出 15 万 × 上千行的矩阵
    total_kl = 0.0
    for s in range(0, idx.numel(), chunk):
        sub = idx[s:s + chunk]
        lp = F.log_softmax(p_l[sub].float(), dim=-1)
        lq = F.log_softmax(q_l[sub].float(), dim=-1)
        total_kl += (lq.exp() * (lq - lp)).sum().item()
    kl = total_kl / idx.numel()

    return ce, kl


def train_kl(model, loader, opt, total_steps: int, warmup: int, base_lr: float,
             grad_accum: int, lam: float, log_every: int = 20) -> None:
    """训练循环：每步两次前向（关适配器 / 开适配器）。"""
    import time
    model.train()
    step, m = 0, 0
    ce_sum, kl_sum, n_log = 0.0, 0.0, 0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)

    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]

        # ---- 第一遍：关掉 LoRA，拿基座分布 q（不产生梯度）----
        with torch.no_grad(), model.disable_adapter():
            logits_q = model(input_ids=ids, attention_mask=mask).logits

        # ---- 第二遍：开着 LoRA，拿当前分布 p（要梯度）----
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits_p = model(input_ids=ids, attention_mask=mask).logits

        ce, kl = ce_and_kl(logits_p, logits_q, labels, lam)
        if ce is None:
            continue
        loss = ce + lam * kl

        (loss / grad_accum).backward()
        ce_sum += ce.item()
        kl_sum += kl
        n_log += 1
        m += 1

        if m == grad_accum:
            m = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = train_common.lr_at(step, total_steps, warmup, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % log_every == 0:
                print(f"  step {step}/{total_steps}  ce={ce_sum/max(1,n_log):.4f}"
                      f"  kl={kl_sum/max(1,n_log):.4f}  lr={lr:.2e}  {time.time()-t0:.0f}s",
                      flush=True)
                ce_sum, kl_sum, n_log = 0.0, 0.0, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/data")
    ap.add_argument("--out", default="/root/autodl-tmp/outputs/kllora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lam", type=float, default=0.1, help="KL 惩罚系数 λ")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("加载分词器与基座 ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.config.use_cache = False

    cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05,
                     target_modules=TARGET_MODULES, task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练 {n_train/1e6:.2f}M，λ = {args.lam}")

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

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.03))
    print(f"每轮 {steps_per_epoch} 次更新，共 {total_steps} 次，预热 {warmup} 次")

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        print(f"\n===== 第 {ep}/{args.epochs} 轮 =====", flush=True)
        train_kl(model, train_loader, opt, total_steps, warmup, args.lr,
                 args.grad_accum, args.lam)

        val_loss = train_common.evaluate(model, val_loader)
        print(f"第 {ep} 轮结束，验证集损失(仅CE) = {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            print(f"  验证损失创新低，保存适配器到 {args.out} ...", flush=True)
            model.save_pretrained(args.out)

    print(f"\n训练完成，最佳验证损失 {best_val:.4f}，λ={args.lam}")
    print(f"接下来评测：\n  python scripts/eval.py --tag kllora --adapter {args.out}")


if __name__ == "__main__":
    main()
