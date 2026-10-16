# Short-Circuiting

私有通信架构实验：删除 IOI 局部电路中 A 头在关键位置的公共残差写回，
检验经私有通道的 A→B Query（Q-only）／Value 特征缩放（V-only）／双分支（Q+V）
调制能否恢复任务功能。规格见 `docs/architecture_spec.md`（实现 ↔ 规格第 1–20 节对应）。

## 架构一句话

`private_model(tokens, attention_mask, intervention)` 一次 forward 内完成：
原模型前向 → A 信号缓存（当前运行的真实 o_h，`s_B = concat(o_A1[p], …)`）→
A 公共写回删除（`attn_out[p] − Σ_{h∈A} o_h[p]`，b_O 不动）→ B 调制
（Q-only: `q' = q + g_Q(s_B)`；V-only: `z' = (1+γ)⊙z`，hook_z 处、W_O 前的精确代数等价）→
最终 logits。原模型冻结；适配器零初始化（初始恒等于 delete）。仅 donor 交换诊断使用
外部 forward，并明确标记。

## 目录

- `shared/`：公共架构（models/interventions/training/evaluation/utils），不 import questions、不硬编码 IOI
- `questions/q1_private_communication/`：IOI 数据、校准、训练、因果检验、报告
- `tests/shared/`：公共单元测试（第 12+20 节合并，31 项全部通过）
- `questions/q2_reserved/`、`questions/q3_reserved/`：预留，不臆造

## 环境

Python 3.13 / torch 2.12+cu126 / transformer_lens 3.2.1 / transformers 4.57.3。
本机无网络：GPT-2-small 从本地 HF 缓存 snapshot 加载（可用 `GPT2_SNAPSHOT` 覆盖路径）。
加载选项统一 `fold_ln=False, center_writing_weights=False, fold_value_biases=False`。

## 命令（可直接复制执行）

```bash
cd /home/zdzdhd/new/Short_Circuiting

# 1. 安装环境（依赖已预装；离线 editable 安装受 PEP 668 限制，repo 根目录 conftest.py 提供导入路径）
pip install -e . --no-build-isolation --no-deps || true

# 2. 运行公共及 Q1 测试（必须先通过，训练前验证零初始化=delete、off=delete、V 等价性）
python -m pytest tests/shared questions/q1_private_communication/tests -q

# 3. 生成和检查 Q1 数据（含 split overlap 检查）
python -m questions.q1_private_communication.src.cli data --config smoke.yaml
python -m questions.q1_private_communication.src.cli data --config full.yaml

# 4. 运行基线与校准（selected_circuit.json）
python -m questions.q1_private_communication.src.cli baseline --config full.yaml

# 5. 训练九组（original/delete/private_q/v/qv/public_adapter/constant_q/v/qv）
python -m questions.q1_private_communication.src.cli train --config full.yaml

# 6. 运行泛化评估与因果检验
python -m questions.q1_private_communication.src.cli evaluate --config full.yaml
python -m questions.q1_private_communication.src.cli causal --config full.yaml

# 7. 汇总报告
python -m questions.q1_private_communication.src.cli report --config full.yaml

# 8. 一条命令运行 Q1 smoke pipeline
python -m questions.q1_private_communication.src.cli smoke --config smoke.yaml
```

运行产物写入 `questions/q1_private_communication/results/<run_id>/`（resolved config、
environment.json、selected_circuit.json、checkpoints、predictions、statistics、figures、report.md），
永不覆盖已有 run。smoke 结果不代表正式结论；未执行项一律标记 `not_executed`，不编造数值。
