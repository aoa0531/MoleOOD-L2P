# 只训练预测头的 IIL 基线

使用原 Prompt 实验保存的 `stage_0.pt`，移除随机 Prompt 参数，保留完全相同的 D0 编码器和预测头。不会重训 D0，也不会加载已经经过增量训练的预测头。

完整图、子结构编码器及 BatchNorm 统计量冻结，编码器 Dropout 关闭。仅更新预测头，不创建 Prompt，不计算 key 损失。Adam、预测头学习率、类别权重、阶段划分、batch size、训练轮数与原检查点配置一致。

在模型构建和初始评估后恢复原 `stage_0.pt` 的随机数状态，使相同环境下的批次打乱和预测头 Dropout 随机序列与原 Prompt 实验对齐。不同软件版本或 GPU 内核不保证逐位一致。

## Colab 运行

先将新增代码提交到自己的 GitHub 仓库，再在原 Colab 中拉取：

```python
%cd /content/moleood
!git pull --ff-only origin work-from-9517ac2
```

确保 Drive 已挂载，数据文件仍在原位置。运行：

```bash
python -u train_head_only.py --source /content/drive/MyDrive/MoleOOD_IIL/run_01/stage_0.pt --output-dir /content/drive/MyDrive/MoleOOD_IIL/head_only_01
```

输出目录必须是新的或空目录，原 Prompt 结果不会被覆盖。新结果保存在 `head_only_01/metrics.json`。

断点续训使用基线自己的入口：

```bash
python -u train_head_only.py --resume /content/drive/MyDrive/MoleOOD_IIL/head_only_01/stage_2.pt
```

评估仍可使用 `evaluate.py --iil-checkpoint`。不要用 `train_iil.py --resume` 恢复此基线，该旧入口会为无 Prompt 模型添加 Prompt。

比较相同阶段、相同评估集的 AUC。基线只输出分类损失，而 Prompt 实验的总损失还包含 key 项，两者总损失数值不宜直接比较。一次对照不足以证明统计显著性，正式结论需要多个种子。
