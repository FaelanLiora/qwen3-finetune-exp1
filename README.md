# Qwen3-0.6B 微调方法对比实验（人工智能实验一）

在 Qwen3-0.6B 上对比四种微调方法，衡量「数学能力提升」与「通用能力保持」之间的权衡。

## 一、实验目标

以 `Qwen/Qwen3-0.6B` 为基座，在 GSM8K 上训练与评测，并用 ARC-Challenge 检验原有能力保持情况，
比较不同微调方法在**数学能力提升**与**能力保持**之间的差异。

## 二、方法一览

| 方法 | 训练对象 | 核心公式 | 关键参数 |
|---|---|---|---|
| Base | 不训练 | — | — |
| Full SFT | 全部参数 | `L_CE = -(1/\|M\|) Σ log p_θ(y_t \| x, y_<t)` | lr 2e-5, 3 epoch |
| LoRA | A、B 低秩旁路 | `h = Wx + (α/r)·BAx` | r=16, α=32, lr 2e-4 |
| P-Tuning v2 | 每层 K/V 前缀 | `H = softmax(QK̃ᵀ/√d_k + C)Ṽ` | virtual_tokens=20, lr 2e-4 |
| KL-LoRA（自研） | LoRA + KL 正则 | `L = L_CE + λ·D_KL(q‖p)` | r=16, α=32, λ=0.1 |

## 三、数据

| 数据 | 条数 | 用途 |
|---|---|---|
| GSM8K train | 7473（9:1 切分 → 约 6726 训练 / 747 验证） | 微调 |
| GSM8K test | 1319（全量） | 数学能力评测 |
| ARC-Challenge test | 1172（全量） | 通用能力保持评测 |

## 四、目录结构

```
qwen3-finetune-exp1/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/                 # 各方法的超参配置
├── data/                    # 数据缓存（不入库）
├── scripts/
│   ├── build_data.py        # 数据下载、格式化、切分、tokenize
│   ├── eval.py              # 统一评测（四个方法共用）
│   ├── train_full.py
│   ├── train_lora.py
│   ├── train_ptuning.py
│   └── train_kl_lora.py
├── results/                 # 评测输出（jsonl / csv，不入库）
└── notes/                   # 实验日志
```

## 五、环境

- 平台：AutoDL / 矩池云（租用 GPU）
- 系统：Ubuntu 22.04，Python 3.12，PyTorch 2.4.0，CUDA 12.1
- 显卡：参考 NVIDIA A30 24GB，实测最小 8GB
- 关键依赖：`transformers>=4.51`（Qwen3 架构支持）、`peft`、`datasets`、`accelerate`

## 六、进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| Step 1 | 工具、账号、仓库初始化 | ☐ |
| Step 2 | 租 GPU、连通服务器 | ☐ |
| Step 3 | 环境验证、模型与数据下载 | ☐ |
| Step 4 | 数据管线（9:1 切分、模板化、tokenize） | ☐ |
| Step 5 | 统一评测器 + Base 基线 | ☐ |
| Step 6 | Full SFT | ☐ |
| Step 7 | LoRA | ☐ |
| Step 8 | P-Tuning v2 | ☐ |
| Step 9 | KL-LoRA（含 λ 消融） | ☐ |
| Step 10 | 结果汇总与作图 | ☐ |
| Step 11 | 报告与答辩 | ☐ |

## 七、快速开始

（待 Step 5 完成后补充）
