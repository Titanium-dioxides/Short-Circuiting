# Architecture Specification — 实现与规格对应关系

本文件记录规格（第 1–20 节，2026-10-16 版）与代码实现之间的对应关系，
以及所有显式偏离与数值约定。第 15–20 节优先级高于第 1–14 节的相应限制。

## 1. 环境与模型（规格第 5 节）

| 项目 | 值 |
|---|---|
| Python | 3.13（系统） |
| torch | 2.12.0+cu126 |
| transformer_lens | 3.2.1 |
| transformers | 4.57.3 |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB |
| 模型 | GPT-2 small（openai-community/gpt2, HF snapshot `607a30d…`，本地缓存加载，无网络） |
| 加载选项 | `fold_ln=False, center_writing_weights=False, fold_value_biases=False`（保留原始参数化；b_O 独立存在；全部组相同） |
| dtype / seed | FP32 训练与主测试；seed 由 config 指定并记录 |

**网络说明**：本机无法访问 huggingface.co。加载路径：先 `AutoModelForCausalLM.from_pretrained(<本地 snapshot>)`，再
`HookedTransformer.from_pretrained("gpt2", hf_model=hf, ...)`。见 `shared/models/backbone.py`。

## 2. Hook 语义（已核对 TL 3.2.1 源码并数值验证，非猜测）

GPT-2 走经典组件路径（`use_attn_result=False`）：

| hook | 形状 | 位置 | 用途 |
|---|---|---|---|
| `blocks.{l}.attn.hook_q` | [B, pos, 12, 64] | W_Q 投影之后、score 计算之前 | Q 调制：`q' = q + delta_q`（§15.3） |
| `blocks.{l}.attn.hook_z` | [B, pos, 12, 64] | `pattern @ v` 之后、W_O 之前 | V 缩放的精确等价位置：`z' = (1+γ)⊙z`（§15.4/15.5） |
| `blocks.{l}.hook_attn_out` | [B, pos, 768] | W_O + b_O 之后、残差相加之前 | A 写回删除（§2.3）与 public_adapter 插入点（§8） |
| `blocks.{l}.attn.hook_pattern` | [B, 12, qpos, kpos] | softmax 之后 | 姓名位置注意力测量 |
| `blocks.{l}.attn.hook_attn_scores` | [B, 12, qpos, kpos] | softmax 之前 | 诊断 |

**数值事实（已验证）**：`attn_out = Σ_h z_h @ W_O_h^T + b_O`，
其中 `W_O` 形状 [12, 64, 768]。故逐头贡献 `o_h[p] = z_h[p] @ W_O_h^T`（d_model 维，不含 b_O 份额），
删除为 `attn_out_private[p] = attn_out_current_run[p] − Σ_{h∈A@l} o_h[p]`。

`run_with_cache` 默认不缓存 z/q —— 本实现一律通过 `run_with_hooks` 的临时 `fwd_hooks` 自行缓存与修改，
每次 forward 使用独立、自动清理的 hook 与缓存对象（§2.6、§2.5）。

## 3. 实现 ↔ 规格对应

| 规格 | 实现 |
|---|---|
| §2.2 A/B 头集合、跨层约束 | `shared/models/routing.py::CircuitSpec`；校验 `A.layer < B.layer` 严格成立，第一版强制 `max(A.layer) < min(B.layer)`，违规报错 |
| §2.3 A 公共写回删除 | `hook_backend.py::DeleteHook`（hook_attn_out 处按位置 p 减去缓存的 o_h 和；不动 b_O、其他头、其他位置） |
| §2.4 s_B、低秩仿射 g_B、零初始化 | `adapters.py::LowRankAffine`（W_down 随机、W_up=0、b=0）；s_B = concat(o_A1[p], …) 仅含 A.layer < B.layer 的头，按 (layer, head) 排序 |
| §15.3 Q-only | `adapters.py::QBranch`（每接收头 B 独立 g_Q,B，输出 d_head）；`hook_backend.py::QHook` |
| §15.4/15.5 V-only | `adapters.py::VBranch`（每接收头 B 独立 g_V,B，输出 d_head）；`hook_backend.py::VScaleHook` 在 hook_z 处做 `(1+γ)⊙z`——即"加权聚合后、W_O 前"，为规格指定 V 调制形式的**精确代数等价**（§15.5 要求的参考实现验证见测试） |
| §15.6 Q+V | `adapters.py::DualBranch`（两分支独立低秩参数、不共享隐藏层、均零初始化、无跨分支模块） |
| §16 constant_q/v/qv | `adapters.py::ConstantQ/V/QV`（每 B 头可训练、样本无关 delta_q/gamma，零初始化；不添加无作用参数凑等参） |
| §8 public_adapter | `adapters.py::PublicAdapter`（删除 A 写回后，把 adapter(s_B) 写入最后一个 A 层 hook_attn_out 处残差位置 p；容量按对应私有模型实际参数预算配置） |
| §15.8 private_mode 配置 | `routing.py::PrivateMode`：`q_only | v_only | qv`；与 `architecture_mode`、`intervention_mode`、`budget_regime` 分开存放，绝不混用字段 |
| §17 参数预算 | `routing.py::resolve_budget`：方案 A（单分支固定 rank r）；方案 B（近似固定总预算 P，QV 尽量均分，记录实际 rank 与差额，不用无作用参数凑数） |
| §10/§18 因果检验 | `shared/interventions/signal_controls.py`（off/mean/shuffled/paired_swap + 分支消融 q_on/v_on + 选择性交换）；donor 信号来自独立外部 forward 并 detach，全部标记"外部 donor 干预" |
| §11/§19 指标与统计 | `shared/evaluation/statistics.py`、`q1…/metrics.py`（D、两种准确率、NLL、恢复率、paired bootstrap、delta_q/gamma 分布、contrib_j） |
| §6 数据 | `q1…/data.py`（模板/姓名顺序/重复主语/翻转配对/单 token 校验/split 与 overlap 检查） |
| §7 校准 | `q1…/calibration.py`（5 步诊断 + selected_circuit.json + 文献来源与 0/1 索引记录） |
| §12/§20 测试 | `tests/shared/`（公共 13+13 项合并）与 `questions/q1_private_communication/tests/` |

## 4. 显式数值约定与容差

1. **删除后精确加回的容差**：`attn_out − Σo_h` 与加回之间，以及"加回运行 vs 无删除运行"的最终 logits，
   因浮点舍入不保证比特级相等。约定（2026-10-16 实测 FP32）：局部（attn_out 层）加回误差 ~6e-8，断言 `atol=1e-5`；
   最终 logits 因 (x−y)+y 两次舍入经 4 层注意力放大，实测偏差 6.5e-3，断言 `atol=1e-2` 且干预位置之前的位置
   严格 `atol=1e-5`（干预不影响更早位置）。BF16/FP16 容差按其精度比例记录。均远小于可观测效应（O(1..10)），不使用过宽容差。
   注：加回必须使用**同一 run** 内删除 hook 刚计算的 o_sum（逐层恢复），跨 run 缓存的 o_h 会因上游状态不同而不等价。
2. **off 的精确定义**（§18）：Q 分支 `delta_q = 0`；V 分支 `gamma = 0`（恒等缩放）；QV 两者同时。置零的是**调制输出**，
   不是把适配器输入置零（适配器 bias 已为零初始化，但语义按规格区分）。三种模式完整 off 均必须等于 delete（测试断言）。
3. **mean_input**（§10.3）：均值来自**训练集正常改造运行**估计的 s_B（no_grad 收集、detach），不用测试集均值；适配器 bias 保留，记录在案。
4. **恢复率分母有效性规则**（§11，最终测试前确定）：分母 `meanD_original − meanD_delete` 必须 > 0 且 ≥ 0.05，
   否则标记"不具解释性"。本规则冻结于 2026-10-16，写入 resolved config。
5. **等效声明**：不用"不显著"表述"等效"；报告连续剩余差距，不预设等效界限。
6. **permutation（shuffled）**：固定循环移位 `i → (i+1) mod B`（B≥2 无不动点）；donor 的全部 A 信号（整个 s_B 向量）成组替换；保存 donor ids。

## 5. 显式偏离记录

1. **加载路径偏离**：因无网络，`from_pretrained` 不直接以仓库名联网下载，改为本地 snapshot + `hf_model=`。
   权重与官方 `openai-community/gpt2` 逐字节一致（同一 HF 缓存），不构成模型偏离。
2. **训练资源标记**：凡未能执行的实验（如某些预算方案、某些种子数）在 `report.md` 与结果 JSON 中标记 `not_executed`，不编造。
3. 无其他科学含义改变的偏离。若未来引入，必须追加在此并说明理由。

## 6. 报告必须回答的问题（§14、§20 验收）

实现忠实性 / 实验结果 / 解释边界三层，加上 §20 的 8 个补充验收问题（三架构是否全部实现、是否近似等参数比较、
QV 优势是否可能只来自更多参数、是否依赖样本特异 A 信号、V-only 是否仅静态缩放、两分支是否都被利用、
独立训练 vs 同 checkpoint 消融是否分开、是否如实说明 V 等价于目标头聚合后特征门控）。

## 7. 执行状态（2026-10-16）

| 步骤（§14） | 状态 |
|---|---|
| 1 环境检查 | 完成（无网络；GPT-2-small 走本地 HF 缓存） |
| 2 架构规格与代码边界 | 完成（shared 不 import questions；Q1 独有代码在 q1 包内） |
| 3 数据 | 完成：full 集 cal 880 / train 680 / val 200 / test_comb 880 / test_templ 352 / test_dist 880，overlap 检查通过。模板为自撰 IOI 变体（答案姓名是最终输入位置之后的**下一个 token**，前导空格单 token 已校验）；pronoun 形式仅用于异性别组合（同性别会歧义，manifest 记录） |
| 4 hook 语义与单测 | 完成：tests/shared 22 项 + Q1 10 项全部通过（训练前验证：零初始化=delete、off=delete、V 参考实现一致、分支消融一致性、冻结/梯度、单 forward、无早位污染、预算校验） |
| 5 校准 | 完成：基线 mean D=2.22（97%/68% top1）有意义的 IOI 能力；A=4 个 S-inhibition 候选单头删除损伤 0.32–0.53（稳定）；B 候选 q-patching 翻转率=0（证据记录：patching 后偏好反而远离 donor 方向，mean D_donor_dir ≈ −3），B 按文献候选选定 L9H6/L9H9 并如实记录弱证据；删除下 B query 变化 L2≈4.9–5.2；Query 修复诊断仅恢复 +0.04 D（B query 本身不携带大部分删除损伤） |
| 6 真实 private forward | 完成（GPU 冒烟验证） |
| 7 等价性/冻结/梯度 | 完成（测试套件） |
| 8 smoke 训练 | 完成（smoke-fixed_total_budget run；40 步非正式） |
| 9 九组完整训练 | 完成（两种预算方案 × 3 seeds × 9 组，共 42 次训练，checkpoint 齐全；方案 2 曾因 public_adapter rank 解析崩溃已修复重跑） |
| 10 泛化与因果 | 完成（evaluation.json + causal-*.json + measurements-*.json，两方案） |
| 11 多种子统计 | 完成（3 seeds 逐个报告 + paired bootstrap CI） |
| 12 报告 | 完成（report.md 319 行 + 图 4 类） |

**模板修正记录（显式偏离修正）**：初版模板把答案姓名放进输入末尾（任务无意义，top1=0），已修正为
经典 IOI 形式（答案=下一 token）。这是实现期修正，非科学含义偏离；旧数据已删除并重新校准。

## 8. 主要结果摘要（fixed_total_budget，等参数方案；test_combination / test_template / test_distractor）

| 组 | mean D（三 split 范围） | 恢复率 R（test_combination） |
|---|---|---|
| original | 2.18–2.40 | —（分母基线） |
| delete | 0.42–0.62 | 0 |
| private_q | 0.99–1.12 | ≈0.25 |
| private_v | 1.57–6.28 | ≈0.66（test_template ≈3.1） |
| private_qv | 13.5–15.1 | ≈6.6（>1，超过原模型） |
| public_adapter | 12.2–14.7 | —（对照） |
| constant_q/v/qv | 0.39–0.61 | ≈delete 水平 |

- 因果：off==delete 偏差 0.0；mean_input/shuffled 使功能大幅坍塌（信号特异性依赖成立）；QV 分支消融
  q_on_v_off≈1.2 / q_off_v_on≈−0.9 / 完整≈13.6（两分支都被利用、强协同）；匹配配对交换不翻转答案偏好
  （与校准中原始模型 q-patching 不翻转一致；OOD 风险已讨论）。
- 等参数 vs 双倍参数（fixed_branch_capacity，QV r=4+4）：QV R 从 ≈6.6 增至 ≈8.0——优势并非只来自参数数量。
