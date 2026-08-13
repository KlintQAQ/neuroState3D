# NeuroState-3D 代码现状与 P7 接入审计

审计日期：2026-08-13  
代码提交基线：`ee79a87`（审计开始时工作树干净）  
方案依据：NeuroState-3D Final Research & Sequential Execution Specification v3.0

## 1. 先回答：Diffusion 与 Drifting 要解决什么

它们不负责完成前端多模态 fusion，也不应替代 observed evidence path。它们只在确定性融合、full-evidence teacher、partial student 都成立以后，建模“给定不完整观测后仍然无法唯一确定的剩余不确定性”。

- 输入条件：真实 observed modalities 形成的 evidence/state、modality mask、quality，以及可选 target id。
- 学习目标：full-evidence teacher 的 latent reference（不是生物学 ground truth）。
- 输出：同一条件下的多个可能 latent state，用于 held-out prediction、posterior coverage、风险/校准和 posterior contraction。
- Diffusion：多步逐渐去噪，是稳健的 posterior reference；优先保证分布覆盖和诊断能力。
- Drifting：把分布演化放到训练过程，推理时直接从噪声一步映射到样本；优先解决 Diffusion 的高 NFE、高 latency，但必须重新验证 conditional 3D posterior 的 coverage/calibration。

因此，二者是并行可替换的 P7 posterior engines，不是串联模块，也不是“Diffusion 先生成、Drifting 再修复”。

## 2. 当前仓库已经具备的组成部分

| 阶段 | 已有代码 | 实际完成度判断 |
|---|---|---|
| BrainMVP 基础 | 官方 UniFormer/U-Net、`BrainMVPEncoder`、checkpoint 映射与覆盖率检查脚本 | 工程接口较完整；本机没有 `pretrained/BrainMVP_uniformer.pt`，本轮无法重跑真实 checkpoint 验证 |
| P1 adapter | identity/residual Conv modality adapter、shared encoder wrapper | 已有最小实现；没有五类 HCP view 的训练结果 |
| P2 baseline | mask-aware mean、fixed-slot concat + 1x1 Conv | 只覆盖两项；缺 scalar sum、spatial gate、普通 cross-attention 的同预算矩阵 |
| Missing mask | subject-level modality mask、random/fixed dropout；BraTS 四模态 15 个非空组合 | 工程逻辑存在；HCP 五模态 30 个 partial subset 与 held-out unseen subsets 尚未形成正式训练协议 |
| BraTS fusion | `EvidenceReliableFusion`：modality/state embedding、aux segmentation、class-conditioned spatial gates、conflict/reliability output | 这是 BraTS segmentation-oriented fusion，不等于 PDF 的 HCP quality-aware spatial target-aware modality-set fusion |
| HCP 数据路径 | T1/T2 manifest、NIfTI 方向/spacing/crop/normalization、affine 检查、synthetic tests | 真实 HCP 数据仍 pending authentication；FA/MD/FUNC、family-aware split、Q maps 尚未接通 |
| BraTS 数据路径 | download/prepare/inspect/smoke 脚本 | 当前工作区无本地 `data/`、无 smoke report，不能声称已完成真实数据训练 |
| Downstream | 原 BrainMVP BraTS training/testing 代码 | 属于官方下游基线，尚未成为 NeuroState state 的独立 downstream validation |
| P7 generator | 本轮加入两种统一 API 的 posterior engines、配置、测试、smoke benchmark | 工程可运行；由于 P3-P5 前置目标缺失，只能标为“接口已接入，科学实验未启动” |

## 3. 与最终方案之间仍缺什么

### P0 数据与 reliability

1. 真实 HCP T1/T2/FA/MD/FUNC subject manifest 和 family-aware train/val/test split。
2. DWI -> FA/MD、rs-fMRI -> ALFF/fALFF/ReHo 的可追溯 pipeline。
3. `Q_reg/Q_proc/Q_signal/Q_coverage` 空间图，不是目前 BraTS 的 present/missing/degraded 三值状态。
4. single-resampling、anti-alias、train-only normalization statistics、functional view D0 决策。

### P2-P3 确定性 fusion

1. 同 encoder/decoder/parameter budget 的 Mean、Conv+Concat、scalar、spatial gate、plain cross-attention baseline matrix。
2. 真正的 per-location 1-5 modality Set Attention。
3. modality/group/position/quality tokens、`q_state` 和 target-specific `q_target`。
4. permutation invariance、missing token isolation、Q shuffle/constant Q/corruption chain 测试。
5. severe missing 与 unseen-subset 下超过 Conv+Concat/spatial gate 的 Go 证据。

当前 `EvidenceReliableFusion` 依赖固定 BraTS slots，把 feature、aux segmentation、conflict 拼接后预测 ET/TC/WT gate；它没有 target modality query，也没有 preprocessing-derived Q。因此不能把它直接命名为方案中的 Core B。

### P4-P6 state learning

1. 五个 target decoder 与 full-evidence teacher；teacher 单独训练、验证后冻结。
2. balanced 30-subset partial student、state distillation、missing reconstruction、observed consistency。
3. held-out unseen subset 设计。
4. 可选 inferred proxy 的独立 provenance 和 `test_no_proxy_in_observed.py`。

这些是 P7 的训练 target 和 condition 来源。没有它们，generator 只能用 synthetic latent 做接口 smoke。

### P7-P9 生成、校准与最终验证

1. 训练 generator 的正式 script/dataloader/checkpoint/artifact protocol。
2. frozen latent autoencoder 或经过验证的 full-state latent codec。
3. 同条件、同 latent、同 split、同 backbone budget 的 Diffusion vs Drifting 训练。
4. fidelity 之外还要报告 diversity、coverage、uncertainty-error correlation、risk-coverage/AURC、OECE/HCCR、posterior contraction、NFE、latency、VRAM。
5. HCP held-out、BraTS transfer、site/scanner OOD 的逻辑分离。

## 4. 本轮 Diffusion 选择

选择：3D conditional latent diffusion，cosine schedule，v-prediction，DDIM sampler。

原因：

1. 方案的 target 本来就是 `Y_full = Enc_lat(B_full)`；在 latent 做生成与现有 3D state 接口一致，也显著降低 full-volume 显存。
2. 条件 latent diffusion 已是 missing-specific latent 和 3D medical translation 中更贴近本任务的成熟路线；pixel/voxel DDPM 不是当前硬件与目标下的合理首选。
3. v-prediction 对不同噪声级别的尺度更均衡；cosine schedule 对有限数据训练通常比简单 linear schedule 更稳。
4. DDIM 可用统一模型给出 10/25/50-step speed-quality curve，适合作为 1-NFE Drifting 的公平多步 reference。

这里的“最好”是“最适合当前 NeuroState-3D P7 reference”，不是宣称它在所有 3D 医学生成任务上绝对最优。正式实验仍需与 EDM-style sampler 或 flow-matching 做小规模验证，且不能使用 test set 选型。

## 5. 本轮新增实现

- `models/generators/common.py`：两分支共享的条件 3D residual backbone；显式接收 condition、modality mask、target id、可选 spatial quality。
- `models/generators/diffusion.py`：cosine latent diffusion、v objective、DDIM 多步采样、可选 latent-space consistency term。
- `models/generators/drifting.py`：官方 JAX drift field objective 的 PyTorch 移植；按连续 3D evidence condition 分组，以 subject full latent 为 positive、其他 subject latent 为可选 negatives；1 NFE 采样。
- `models/generators/posterior.py`：统一 factory。
- `NeuroState3D.sample_posterior`：显式 opt-in 接口；deterministic forward 不自动运行 generator。
- 两份 generator config、四个 unit tests、一个同形状 smoke benchmark。

共同采样契约：

```text
condition [B,Cc,D,H,W]
  + observed modality mask [B,5]
  + optional target/quality
  -> sample(condition, K, seed)
  -> posterior samples [B,K,C,D,H,W]
```

## 6. 验证结果与边界

- `compileall`：通过。
- `tests.test_posterior_generators`：4/4 通过，包括两分支 forward/backward、Drifting loss gradient、固定 seed、mask shape invariant、统一输出 shape。
- CPU synthetic smoke：Diffusion 4 NFE 与 Drifting 1 NFE 均产生 finite `[2,2,4,4,4,4]` 输出。
- smoke latency 只验证代码路径，不用于论文结论；网络未训练、CPU 规模极小。
- 本机无 HCP/BraTS 数据、无 BrainMVP checkpoint、无 CUDA PyTorch 环境，因此没有真实数据 fidelity/coverage/calibration 结果。

## 7. 推荐的严格下一顺序

1. 完成 P0 HCP 五模态 manifest/Q maps，并冻结 D0 functional view。
2. 补齐 P2 baseline matrix。
3. 把 target-aware Set Fusion 作为独立 HCP state 模型实现并通过 P3 Go/No-Go。
4. 训练/freeze full teacher，再训练 partial student。
5. 固定 teacher/student/latent 后，才正式训练本轮加入的 Diffusion 与 Drifting。
6. 按相同 seed/split/condition/latent 报告完整 posterior + efficiency 指标，再决定主方法。

当前最准确的项目状态是：**P0/P1/P2 工程脚手架 + 一个 BraTS segmentation fusion prototype + P7 双生成器工程接口；P3-P5 的研究主链尚未闭合。**
