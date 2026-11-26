"""把数据变成模型能"吃"的格式 —— 训练用的数据集与批处理。

【核心概念：label 掩码（loss mask）】
一条训练样本 = 「题目」+「标准答案」。
模型对每个位置都要预测下一个字，但我们**只关心答案部分**：
    题目部分：不罚不奖（label 设为 -100，PyTorch 会自动跳过）
    答案部分：错了就罚（参与算损失）

为什么？因为题目是"给定的输入"，模型不需要学怎么复述题目。
如果题目也参与算损失，模型会把大量力气花在"抄题目"上，效果反而差。

【为什么 prompt 和 target 分开存】
build_data.py 里已经算好了 prompt 字符串，这里分别切词再拼起来：
    input_ids = [题目 token] + [答案 token]
    labels    = [ -100 ... ] + [答案 token]
这样天然对齐，不需要去数字符位置，不会错位。
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class SFTDataset(Dataset):
    """读 jsonl，返回 input_ids / labels。"""

    def __init__(self, path: str, tokenizer, max_len: int = 512):
        with open(path, encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        self.tok = tokenizer
        self.max_len = max_len
        self.n_truncated = 0        # 被截断的样本数（诊断用）

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]

        prompt_ids = self.tok(r["prompt"], add_special_tokens=False).input_ids
        target_ids = self.tok(r["target"], add_special_tokens=False).input_ids

        # 超长就截答案的尾巴（题目不能截，截了就答非所问）
        if len(prompt_ids) + len(target_ids) > self.max_len:
            self.n_truncated += 1
            keep = max(self.max_len - len(prompt_ids), 1)
            target_ids = target_ids[:keep]

        input_ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_collate(pad_token_id: int):
    """生成一个批次拼接函数：把不等长的样本补齐到同一长度。

    训练时补在右边即可（补出来的位置 label 都是 -100，不参与算损失）。
    """

    def collate(batch: list[dict]) -> dict:
        max_len = max(x["input_ids"].size(0) for x in batch)
        input_ids, labels, attn = [], [], []
        for x in batch:
            n = x["input_ids"].size(0)
            pad = max_len - n
            input_ids.append(torch.cat([x["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)]))
            labels.append(torch.cat([x["labels"], torch.full((pad,), IGNORE_INDEX, dtype=torch.long)]))
            attn.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(labels),
        }

    return collate
