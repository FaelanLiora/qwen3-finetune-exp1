"""用手写前缀路径做逐字解码，绕开 peft 的生成接口。

【为什么需要这个】
P-Tuning 训练是有效的（6 轮后验证损失 1.04，低于基座的 1.19），
但用 peft 的 generate 生成时会陷入无限重复。
原因：逐字解码时缓存不断变长，peft 的前缀注入与 transformers 5 的缓存接口不匹配。

【做法】
1. 从 peft 保存的适配器里读出训练好的前缀权重
2. 装进我们自己的 ManualPrefix
3. 自己写贪心解码：每步把"前缀 + 已生成 token"的缓存交给模型

用法：
    python scripts/decode_manual.py --adapter /root/autodl-tmp/outputs/ptuning6 --limit 8
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


class ManualPrefix(torch.nn.Module):
    """每层 num_virtual_tokens 个可学习 K/V，作为"已有缓存"注入。"""

    def __init__(self, num_layers, num_kv_heads, head_dim, n_tokens, dtype=torch.bfloat16):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_layers = num_layers
        self.k = torch.nn.Parameter(torch.zeros(num_layers, num_kv_heads, n_tokens, head_dim, dtype=dtype))
        self.v = torch.nn.Parameter(torch.zeros(num_layers, num_kv_heads, n_tokens, head_dim, dtype=dtype))

    def build_cache(self, batch_size: int, device) -> DynamicCache:
        cache = DynamicCache()
        for layer in range(self.n_layers):
            k = self.k[layer].unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
            v = self.v[layer].unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
            cache.update(k, v, layer)
        return cache


def load_trained_prefix(adapter_dir: str, model_config, n_tokens: int, device):
    """从 peft 的适配器文件里读出前缀权重，装进 ManualPrefix。

    不同版本的 peft 对同一份权重的命名不同（prefix_encoder / prompt_embeddings），
    所以这里按**形状**去找，不依赖键名：
        前缀权重形状 = [n_tokens, 层数 × 2 × KV头数 × head_dim]
    """
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    print(f"  适配器里的键：{list(sd.keys())}")

    num_layers = model_config.num_hidden_layers
    num_kv = getattr(model_config, "num_key_value_heads", model_config.num_attention_heads)
    head_dim = getattr(model_config, "head_dim",
                       model_config.hidden_size // model_config.num_attention_heads)
    token_dim = num_kv * head_dim
    expect_dim = num_layers * 2 * token_dim

    w = None
    for k, v in sd.items():
        if v.dim() == 2 and v.shape[0] == n_tokens and v.shape[1] == expect_dim:
            w = v.float()
            print(f"  按形状匹配到前缀权重：{k} {tuple(v.shape)}")
            break
    if w is None:
        for k, v in sd.items():
            print(f"    {k}: {tuple(v.shape)}")
        raise RuntimeError(
            f"没找到形状为 [{n_tokens}, {expect_dim}] 的前缀权重；"
            f"如果某个键的形状是 [{n_tokens}, {model_config.hidden_size}]，"
            f"那说明存的是软提示而不是每层 K/V 前缀"
        )

    # peft 的排列：[n_tokens, layers*2, token_dim] -> 拆成每层 K、V
    w = w.view(n_tokens, num_layers, 2, num_kv, head_dim)
    k = w[:, :, 0].permute(1, 2, 0, 3).contiguous()   # [layers, kv, n_tokens, head_dim]
    v = w[:, :, 1].permute(1, 2, 0, 3).contiguous()

    prefix = ManualPrefix(num_layers, num_kv, head_dim, n_tokens).to(device)
    with torch.no_grad():
        prefix.k.copy_(k.to(prefix.k.dtype))
        prefix.v.copy_(v.to(prefix.v.dtype))
    return prefix


@torch.no_grad()
def generate_one(model, prefix, tok, prompt: str, max_new: int = 512) -> str:
    """贪心解码，每步都带上前缀缓存。"""
    nv = prefix.n_tokens
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    bsz, seq_len = ids.shape

    cache = prefix.build_cache(bsz, model.device)
    mask = torch.ones(bsz, nv + seq_len, dtype=torch.long, device=model.device)
    pos = torch.arange(nv, nv + seq_len, device=model.device)

    out = model(input_ids=ids, attention_mask=mask,
                past_key_values=cache, cache_position=pos)
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)

    generated = [nxt]
    eos = tok.eos_token_id
    for _ in range(max_new - 1):
        if int(nxt[0, 0]) == eos:
            break
        mask = torch.cat([mask, torch.ones(bsz, 1, dtype=mask.dtype, device=mask.device)], dim=1)
        out = model(
            input_ids=nxt,
            attention_mask=mask,
            past_key_values=cache,
            cache_position=torch.tensor([mask.size(1) - 1], device=model.device),
        )
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(nxt)

    return tok.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", default="/root/autodl-tmp/data/gsm8k_test.jsonl")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--n_tokens", type=int, default=20)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("加载基座 ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    model.config.use_cache = False

    prefix = load_trained_prefix(args.adapter, model.config, args.n_tokens, model.device)

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()][: args.limit]

    for r in rows:
        text = generate_one(model, prefix, tok, r["prompt"], args.max_new)
        print(f"\n[{r['id']}] 标准={r['final']}")
        print(repr(text[:400]))
        print("-" * 60)


if __name__ == "__main__":
    main()
