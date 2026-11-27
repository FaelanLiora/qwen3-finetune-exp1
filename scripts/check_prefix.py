"""判定实验：前缀到底能不能正确注入？

不训练，只做前向，比四种情况的损失。目的是找出哪条实现路径可用。

    [1] 纯基座                    —— 参照值，应约 1.2
    [2] peft 每层 K/V 前缀         —— 原方案（已知有问题）
    [3] peft 输入层软提示          —— 备选（不是 v2，仅作对照）
    [4] 手写每层 K/V 前缀          —— 不依赖 peft，自己把 K/V 拼进缓存

判据：损失和 [1] 接近（差 < 0.2）的那条路可用。

用法：
    python scripts/check_prefix.py --impl all
    python scripts/check_prefix.py --impl manual
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from peft import (PrefixTuningConfig, PromptTuningConfig, PromptTuningInit,
                  get_peft_model)
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

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


class ManualPrefix(nn.Module):
    """手写前缀：每层 num_virtual_tokens 个可学习 K/V。

    做法：把前缀当成"已有的缓存"，直接塞进模型的 KV 缓存里，
    并把 attention_mask 和 cache_position 相应扩展，让真实 token 能看到这 20 个位置。
    这样每一层的注意力里都多出 20 个可学习的位置，正是 P-Tuning v2 的定义。
    """

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 n_tokens: int, dtype=torch.float32):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_layers = num_layers
        self.n_heads = num_kv_heads
        self.head_dim = head_dim
        # 初始化为 0：训练开始时前缀不产生任何影响，模型行为等于基座
        self.k = nn.Parameter(torch.zeros(num_layers, num_kv_heads, n_tokens, head_dim, dtype=dtype))
        self.v = nn.Parameter(torch.zeros(num_layers, num_kv_heads, n_tokens, head_dim, dtype=dtype))

    def build_cache(self, batch_size: int, device) -> DynamicCache:
        cache = DynamicCache()
        for layer in range(self.n_layers):
            k = self.k[layer].unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
            v = self.v[layer].unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
            cache.update(k, v, layer)
        return cache


def manual_loss(model, prefix: ManualPrefix, loader, max_batches: int = 4) -> float:
    """用手写前缀算损失。"""
    nv = prefix.n_tokens
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(model.device)
        mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)
        bsz, seq_len = ids.shape

        # 掩码左边补 20 个 1：让真实 token 可以关注到前缀
        ext_mask = torch.cat(
            [torch.ones(bsz, nv, dtype=mask.dtype, device=mask.device), mask], dim=1
        )
        cache = prefix.build_cache(bsz, model.device)
        # 真实 token 的绝对位置从 20 开始
        cache_position = torch.arange(nv, nv + seq_len, device=model.device)

        out = model(input_ids=ids, attention_mask=ext_mask, labels=labels,
                    past_key_values=cache, cache_position=cache_position)
        total += out.loss.item()
        n += 1
    return total / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/data")
    ap.add_argument("--num_virtual_tokens", type=int, default=20)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--impl", default="all",
                    choices=["prefix", "prompt", "manual", "all"],
                    help="测哪条实现路径")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = SFTDataset(os.path.join(args.data_dir, "val.jsonl"), tok, args.max_len)
    ds.rows = ds.rows[:32]
    loader = DataLoader(ds, batch_size=8, shuffle=False,
                        collate_fn=make_collate(tok.pad_token_id), num_workers=2)

    def fresh_model():
        m = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="eager")
        m.config.use_cache = False
        return m

    model = fresh_model()
    l0 = mean_loss(model, loader)
    print(f"\n[1] 纯基座                    损失 = {l0:.4f}   ← 参照值")

    if args.impl in ("prefix", "all"):
        cfg = PrefixTuningConfig(num_virtual_tokens=args.num_virtual_tokens,
                                 task_type="CAUSAL_LM", prefix_projection=False)
        m2 = get_peft_model(fresh_model(), cfg)
        shrink_prefix(m2, args.num_virtual_tokens, 0.02)
        l2 = mean_loss(m2, loader)
        print(f"[2] peft 每层 K/V 前缀        损失 = {l2:.4f}   （差 {l2-l0:+.4f}）")
        del m2
        torch.cuda.empty_cache()

    if args.impl in ("prompt", "all"):
        cfg_p = PromptTuningConfig(
            num_virtual_tokens=args.num_virtual_tokens,
            task_type="CAUSAL_LM",
            prompt_tuning_init=PromptTuningInit.RANDOM,
            tokenizer_name_or_path=args.model,
        )
        m3 = get_peft_model(fresh_model(), cfg_p)
        l3 = mean_loss(m3, loader)
        print(f"[3] peft 输入层软提示         损失 = {l3:.4f}   （差 {l3-l0:+.4f}）")
        del m3
        torch.cuda.empty_cache()

    if args.impl in ("manual", "all"):
        m4 = fresh_model()
        conf = m4.config
        prefix = ManualPrefix(
            num_layers=conf.num_hidden_layers,
            num_kv_heads=getattr(conf, "num_key_value_heads", conf.num_attention_heads),
            head_dim=getattr(conf, "head_dim", conf.hidden_size // conf.num_attention_heads),
            n_tokens=args.num_virtual_tokens,
            dtype=torch.bfloat16,
        ).to(m4.device)
        n_par = sum(p.numel() for p in prefix.parameters())
        print(f"    手写前缀可训练参数 {n_par/1e6:.3f}M")
        l4 = manual_loss(m4, prefix, loader)
        print(f"[4] 手写每层 K/V 前缀         损失 = {l4:.4f}   （差 {l4-l0:+.4f}）")

    print("""
===== 结论怎么读 =====
· 某个方案与 [1] 的差 < 0.2  → 该方案可用
· [2] / [3] 都差很多、只有 [4] 正常 → 用 [4]，参数规模一样，语义仍是"每层 20 个 K/V"
· [4] 也差很多               → 前缀路线在本框架下不通，需要换更低层的实现
""")


if __name__ == "__main__":
    main()
