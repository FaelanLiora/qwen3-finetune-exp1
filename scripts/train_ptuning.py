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
    ap.add_argument("--init", default="small", choices=["small", "tokens"],
                    help="small=小随机初始化；tokens=用真实 token 的 K/V 初始化（推荐）")
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

    # ---------- 用真实 token 的 K/V 初始化前缀（可选）----------
    # 原理：注意力打分的均值通常是负数，初始化为 0 等于给了前缀"偏高的分数"，
    # 会抢走大量注意力（实测初始损失 3.7）。改成真实 token 的 K/V，
    # 前缀一开始就像普通上下文，不抢注意力（预期初始损失接近基座 1.2）。
    init_kv = None
    if args.init == "tokens":
        print("采样真实 token 的 K/V 作为前缀初值 ...", flush=True)
        probe_tok = AutoTokenizer.from_pretrained(args.model)
        probe_ds = SFTDataset(os.path.join(args.data_dir, "val.jsonl"), probe_tok, args.max_len)
        probe_ds.rows = probe_ds.rows[:2]
        probe_batch = make_collate(probe_tok.pad_token_id)([probe_ds[0], probe_ds[1]])
        captured = {}

        handles = []
        for idx, layer in enumerate(model.model.layers):
            def mk(tag):
                def fn(module, inp, out):
                    captured[tag] = out.detach()
                return fn
            # 关键：K 要抓 k_norm 之后的输出（已归一化，量级与真实 K 一致）。
            # 抓 k_proj 的话数值过大，会让注意力打分爆炸（实测损失 12.3）。
            k_module = getattr(layer.self_attn, "k_norm", None) or layer.self_attn.k_proj
            handles.append(k_module.register_forward_hook(mk(f"k{idx}")))
            handles.append(layer.self_attn.v_proj.register_forward_hook(mk(f"v{idx}")))

        with torch.no_grad():
            model(input_ids=probe_batch["input_ids"].to(model.device),
                  attention_mask=probe_batch["attention_mask"].to(model.device))
        for h in handles:
            h.remove()

        cfg0 = model.config
        n_kv = getattr(cfg0, "num_key_value_heads", cfg0.num_attention_heads)
        hd = getattr(cfg0, "head_dim", cfg0.hidden_size // cfg0.num_attention_heads)
        nv = args.num_virtual_tokens

        init_kv = []
        for idx in range(cfg0.num_hidden_layers):
            t_len = captured[f"k{idx}"].shape[1]
            k = captured[f"k{idx}"].reshape(-1, t_len, n_kv, hd)
            v = captured[f"v{idx}"].reshape(-1, t_len, n_kv, hd)
            # 取前 nv 个真实位置，跨样本取平均
            init_kv.append((k[:, :nv].mean(0), v[:, :nv].mean(0)))   # 每层 [nv, kv, hd]
        print(f"  已采集 {len(init_kv)} 层，每层 [{n_kv}, {nv}, {hd}]")

    cfg = PrefixTuningConfig(
        num_virtual_tokens=args.num_virtual_tokens,
        task_type="CAUSAL_LM",
        prefix_projection=False,     # 不加额外的重参数化 MLP，参数更少、更好解释
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    # 【关键一步】初始化前缀
    # 默认 peft 用标准正态（数值约 ±1），会严重抢走注意力（实测初始损失 6~7）。
    # 这里提供两种初始化：
    #   small  —— 缩小到 0.02，但仍是"0 分"，在负均值分布的注意力里依然算高分（实测 3.7）
    #   tokens —— 用真实 token 的 K/V，前缀一开始就像普通上下文（推荐）
    n_set = 0
    if init_kv is not None:
        cfg1 = model.config
        n_kv = getattr(cfg1, "num_key_value_heads", cfg1.num_attention_heads)
        hd = getattr(cfg1, "head_dim", cfg1.hidden_size // cfg1.num_attention_heads)
        nv = args.num_virtual_tokens
        layers_k = torch.stack([k for k, v in init_kv], dim=0)      # [L, nv, kv, hd]
        layers_v = torch.stack([v for k, v in init_kv], dim=0)      # [L, nv, kv, hd]
        w = torch.stack([layers_k, layers_v], dim=2)                # [L, nv, 2, kv, hd]
        w = w.permute(1, 0, 2, 3, 4).reshape(nv, -1)                # [nv, L*2*kv*hd]
        for _, mod in model.named_modules():
            if isinstance(mod, torch.nn.Embedding) and mod.weight.shape[0] == nv:
                with torch.no_grad():
                    mod.weight.copy_(w.to(mod.weight.dtype))
                n_set += 1
        print(f"已用真实 token 的 K/V 初始化 {n_set} 个前缀向量")
    else:
        for _, mod in model.named_modules():
            if isinstance(mod, torch.nn.Embedding) and mod.weight.shape[0] == args.num_virtual_tokens:
                torch.nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                n_set += 1
        print(f"已把 {n_set} 个前缀向量的初始标准差缩放为 0.02")

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

    # 训练前先测一次损失，判断前缀初始化是否合理
    with torch.no_grad():
        b0 = next(iter(train_loader))
        b0 = {k: v.to(model.device) for k, v in b0.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            init_loss = model(**b0).loss.item()
    print(f"\n训练前的损失 = {init_loss:.4f}   （基座约 1.19；若 3.7 说明前缀在抢注意力）\n")

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
