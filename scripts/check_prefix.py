"""判定实验：前缀注入本身有没有破坏模型？

【要分清的两件事】
A. 前缀注入有 bug（版本兼容问题）→ 不训练的情况下，模型损失就已经异常
B. 只是训练没学好（学习率/初始化） → 不训练时损失正常，一训练才崩

【做法】
不训练，只做三次前向，比三个损失：
    1. 纯基座，不挂前缀            —— 参照值，应该 ~0.7
    2. 挂前缀，初始化缩到 0.02      —— 如果和 1 接近，说明注入正常
    3. 挂前缀，用默认初始化          —— 看看默认初始化到底破坏多大

用法：
    python scripts/check_prefix.py
"""

from __future__ import annotations

import argparse
import os

import torch
from peft import PrefixTuningConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_dataset import SFTDataset, make_collate


@torch.no_grad()
def mean_loss(model, loader, max_batches: int = 4) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total += model(**batch).loss.item()
        n += 1
    return total / max(1, n)


def shrink_prefix(model, num_virtual_tokens: int, std: float) -> int:
    n = 0
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.Embedding) and mod.weight.shape[0] == num_virtual_tokens:
            torch.nn.init.normal_(mod.weight, mean=0.0, std=std)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/data")
    ap.add_argument("--num_virtual_tokens", type=int, default=20)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = SFTDataset(os.path.join(args.data_dir, "val.jsonl"), tok, args.max_len)
    ds.rows = ds.rows[:32]
    loader = DataLoader(ds, batch_size=8, shuffle=False,
                        collate_fn=make_collate(tok.pad_token_id), num_workers=2)

    print("加载基座 ...", flush=True)
    # 【关键】attn_implementation="eager"
    # transformers 5 默认用 SDPA + is_causal 快捷方式算掩码，它假设查询和键等长。
    # 前缀会让键多出 20 个位置，导致掩码错位 20 格，模型看到的"上文"全错。
    # eager 实现会按真实长度生成掩码，前缀才能正确接上。
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager")
    base.config.use_cache = False
    l0 = mean_loss(base, loader)
    print(f"\n[1] 纯基座（无前缀）        损失 = {l0:.4f}")

    cfg = PrefixTuningConfig(num_virtual_tokens=args.num_virtual_tokens,
                             task_type="CAUSAL_LM", prefix_projection=False)

    # 情况 2：缩小初始化
    m2 = get_peft_model(base, cfg)
    n = shrink_prefix(m2, args.num_virtual_tokens, 0.02)
    print(f"    已缩放 {n} 个前缀向量（std=0.02）")
    l2 = mean_loss(m2, loader)
    print(f"[2] 挂前缀 + 小初始化       损失 = {l2:.4f}   （与 [1] 的差 = {l2-l0:+.4f}）")

    # 情况 3：默认初始化
    del m2
    torch.cuda.empty_cache()
    base2 = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager")
    base2.config.use_cache = False
    m3 = get_peft_model(base2, cfg)
    l3 = mean_loss(m3, loader)
    print(f"[3] 挂前缀 + 默认初始化     损失 = {l3:.4f}   （与 [1] 的差 = {l3-l0:+.4f}）")

    print("""
===== 结论怎么读 =====
· [2] 与 [1] 接近（差 < 0.2）  → 前缀注入正常，问题在"训练方式"（学习率太大等）
· [2] 明显高于 [1]（差 > 1）   → 前缀注入本身破坏了模型，是版本兼容问题，需要换实现方式
· [3] 远高于 [2]              → 默认初始化的破坏性确实很大，缩小初始化是必要的
""")


if __name__ == "__main__":
    main()
