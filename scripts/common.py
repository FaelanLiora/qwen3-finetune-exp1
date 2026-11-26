"""公共工具：prompt 模板、答案抽取、判分。

【为什么要单独一个文件】
四个微调方法 + 评测脚本必须用**完全相同的 prompt 和判分逻辑**，
否则方法之间的差异可能来自 prompt 而不是算法。所有脚本都 import 这里。

【答辩要点】
- prompt 按 Qwen3 的 chat 模板构造，system 里明确要求"最后一行用 #### 数字"
- 判分是"抽数字 + 精确匹配"，不是 BLEU / 相似度
"""

from __future__ import annotations

import re

# ============================================================
# 一、GSM8K（数学应用题）
# ============================================================

SYSTEM_GSM8K = (
    "你是一个擅长小学数学应用题的助手。"
    "请一步一步地推理，最后一行必须以「#### 数字」的形式给出最终答案。"
)


def gsm8k_messages(question: str) -> list[dict]:
    """把一道题变成 chat 消息列表。"""
    return [
        {"role": "system", "content": SYSTEM_GSM8K},
        {"role": "user", "content": question.strip()},
    ]


# 匹配 "#### 42" / "#### 1,200" / "#### -3.5"
_LABEL_FINAL = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def label_final_answer(solution: str) -> str | None:
    """从 GSM8K 标注答案里抽最终数字（去掉千分位逗号）。"""
    hits = _LABEL_FINAL.findall(solution)
    if not hits:
        return None
    return hits[-1].replace(",", "")


def predict_final_answer(text: str) -> str | None:
    """从模型输出里抽答案。

    策略：优先取 "####" 之后的第一个数字；没有 #### 就取全文最后一个数字。
    """
    if "####" in text:
        tail = text.split("####")[-1]
        nums = _NUMBER.findall(tail)
        if nums:
            return nums[0].replace(",", "")
    nums = _NUMBER.findall(text)
    return nums[-1].replace(",", "") if nums else None


def _norm(x: str | None) -> str | None:
    """数字归一化：去逗号、去空格、5.0 -> 5。"""
    if x is None:
        return None
    x = str(x).replace(",", "").strip()
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return x


def is_correct(pred: str | None, gold: str | None) -> bool:
    """答案是否完全一致（归一化后比较）。"""
    return _norm(pred) is not None and _norm(pred) == _norm(gold)


# ============================================================
# 二、ARC-Challenge（科学常识选择题）
# ============================================================

ARC_SYSTEM = (
    "你是一个科学常识助手。请阅读题目和选项，"
    "只回答正确选项的字母（A、B、C 或 D），不要解释。"
)


def arc_choices(row: dict) -> list[tuple[str, str]]:
    """取选项，按字母排序，返回 [(label, text), ...]。"""
    pairs = list(zip(row["choices"]["label"], row["choices"]["text"]))
    pairs.sort(key=lambda p: p[0])
    return pairs


def arc_messages(row: dict) -> list[dict]:
    """把一道 ARC 题变成 chat 消息列表（题干 + 四个选项 + 只答字母）。"""
    body = "\n".join(f"{lab}. {txt}" for lab, txt in arc_choices(row))
    content = f"{row['question'].strip()}\n\n{body}\n\n请只回答正确选项的字母。"
    return [
        {"role": "system", "content": ARC_SYSTEM},
        {"role": "user", "content": content},
    ]
