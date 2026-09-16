# v3 T1c 训练检查（2026-09-16 UTC）

## 查看过程

当前实验：`h200_targetaware_t1c_stochastic_seed46_v3_transport`。
检查时已完成 3/20 个 epoch，正在第 4 轮（日志 epoch 从 0 开始），约 13,000 步。

实时日志：

```bash
tail -f /home/yuey21/NeuroState3D_data/logs/h200_pipeline/h200_targetaware_t1c_stochastic_seed46_v3/20260916_131750_cpu4_resume.stdout.log
```

每 20 步输出训练指标，每 100 步保存 checkpoint，每轮结束输出验证指标。
当前脚本没有 TensorBoard 写入；最终 JSON 报告在全部训练结束后才写入。
恢复训练之前的记录在同目录 `20260915_064722_train_transport.stdout.log`，首轮在 `20260914_072432_train_transport.stdout.log`。

## 最近验证结果

下表来自同一 v3 实验日志；MAE 和 selection score 越低越好，SSIM 和 Dice 越高越好。

| 完成轮数 | MAE | SSIM | ET MAE | 高亮肿瘤低估误差 | Prompt Dice 均值 | Selection score |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.049221 | 0.591997 | 0.214282 | 0.127345 | 0.665871 | 0.194248 |
| 2（当前最佳） | 0.045923 | 0.619976 | 0.210203 | 0.116470 | 0.692988 | 0.184541 |
| 3（最新完成） | 0.052869 | 0.493750 | 0.221883 | 0.166711 | 0.706959 | 0.208247 |

最新一轮相对上一轮：整体 MAE 增加 15.1%，SSIM 降低 0.1262，高亮肿瘤低估误差增加 43.1%。与此同时 WT MAE 从 0.173030 改善至 0.155064；并非所有区域都变差。

确认的现象是整体结构相似性下降、增强区域低估加重，而辅助分割仍改善。近期一个 batch 的 prompt_loss 为 18.6590，乘以 0.08 后为 1.4927，占总 loss 1.8071 的约 82.6%。这提示辅助目标与图像生成质量存在失衡风险；损失数值占比不等于梯度占比，不能仅凭这些日志断言因果，也不能断言是过拟合。应固定验证样本，对辅助损失权重进行消融后再决定调整。

当前仅完成 3 轮，不能与旧实验训练 20 轮后的最佳结果直接等同。训练 recon_l1 带区域权重，也不能直接当作验证整体 MAE。运行时 CPU 优化后日志报告 exact_resume；近期梯度非有限值计数为 0，未发现数值崩溃证据，但这不能证明恢复前后数值轨迹完全一致。

## 模型与旧对比图

在 `outputs/h200_targetaware_t1c_stochastic_seed46_v3_transport/` 中：

- `slice_virtual_modality_generator_best.pt`：当前验证最佳，第 2 轮。
- `slice_virtual_modality_generator_last.pt`：最近完成轮次，第 3 轮。
- `checkpoint_step_*.pt`：正在训练的中间快照，不能视作最佳结果。

`reports/visuals/h200_best_t1c_roi_sota/` 下的图片来自旧实验 stage2_noharm 的 best checkpoint，并非本次 v3 结果。代表病例的 mean_mae=0.018443 是选出的 16 张切片；该旧模型完整验证 best MAE=0.041871。把代表图的 0.018443 与 v3 最新完整验证的 0.052869 比较，会夸大差距。best/worst 图片也存在选样差异。

下一步应以同一批病例、同一切片、相同显示窗宽比较 v3 best 和 last，并对辅助损失权重做单因素实验。本次检查未修改正在运行的训练参数，也未重新启动训练。
