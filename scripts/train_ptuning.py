"""方法三：P-Tuning v2（深度前缀微调）

【在干什么】
整个模型全部冻结，只在**每一层**的注意力前面，拼上 20 个"虚拟 token"的 K、V：

        K̃ = [ P_K ; K ]        Ṽ = [ P_V ; V ]
        H = softmax( Q K̃ᵀ / √d_k + C ) Ṽ

那 20 个向量是可学习的参数，改的就是"模型在一层里看什么、取什么"。

【为什么比 LoRA 还"轻"】
    LoRA      ：7 个模块 × 28 层，约 1009 万参数（1.67%）
    P-Tuning  ：每层 20 个向量 × 28 层，约 115 万参数（0.19%）

【超参（对齐 PPT 第 9 页）】
    virtual_tokens = 20
    学习率 2e-4 / 轮数 3 / 长度 512 / 批次 4 × 累积 4

【保存的东西】
只保存 prefix 编码器（几 MB）。评测时用 --adapter 挂上去。

用法：
    python scripts/train_ptuning.py
    python scripts/train_ptuning.py --limit 64 --epochs 1 --out /root/autodl-tmp/outputs/ptuning_smoke
"""

from __future__ import annotations

import argparse
import math
import os

import torch
from peft import PrefixTuningConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import transformers

import train_common
from sft_dataset import SFTDataset, make_collate

# transformers 4.x 用 torch_dtype，5.x 用 dtype。这个变量让同一份代码两边都能跑。
_DTYPE_KW = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/data")
    ap.add_argument("--out", default="/root/autodl-tmp/outputs/ptuning")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--num_virtual_tokens", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---------- 模型 + 前缀 ----------
    print("加载分词器与基座 ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda", **{_DTYPE_KW: torch.bfloat16}
    )
    model.config.use_cache = False

    cfg = PrefixTuningConfig(
        num_virtual_tokens=args.num_virtual_tokens,
        task_type="CAUSAL_LM",
        prefix_projection=False,     # 不加额外的重参数化 MLP，参数更少、更好解释
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    # 【关键一步】缩小前缀的初始值
    # PEFT 默认用标准正态分布初始化前缀向量（数值约 ±1），而 K/V 的正常取值范围远小于此，
    # 训练开始时会把注意力彻底搅乱（实测初始损失高达 6~7）。
    # 这里把它缩小到 0.02，让训练从"几乎不影响模型"的状态起步，
    # 与 LoRA 把 B 初始化为 0 是同一个道理。
    n_reset = 0
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.Embedding) and mod.weight.shape[0] == args.num_virtual_tokens:
            torch.nn.init.normal_(mod.weight, mean=0.0, std=0.02)
            n_reset += 1
    print(f"已把 {n_reset} 个前缀向量的初始标准差缩放为 0.02")

    # 冻结模型只反传到前缀，开启输入梯度，避免梯度检查点下断链
    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数 {n_all/1e6:.1f}M，可训练 {n_train/1e6:.3f}M（{n_train/n_all:.3%}）")

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
            print(f"  验证损失创新低，保存前缀到 {args.out} ...", flush=True)
            model.save_pretrained(args.out)

    print(f"\n训练完成，最佳验证损失 {best_val:.4f}")
    print(f"接下来评测（注意加 --adapter）：\n"
          f"  python scripts/eval.py --tag ptuning --adapter {args.out}")


if __name__ == "__main__":
    main()
