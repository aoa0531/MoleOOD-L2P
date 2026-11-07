# 单个共享 Prompt 对照

所有分子共享一个长度为 1 的可学习向量，将其加到每个原子的嵌入上。没有 query 前向、keys、Top-1 检索或 key 损失。冻结完整图和子结构编码器，只训练共享 Prompt 与预测头。

从原 Prompt 池实验的 `stage_0.pt` 加载完全相同的 D0 主干和预测头，不重训 D0。共享向量固定取该检查点随机 Prompt 池的第 0 行初始化，未根据验证或测试表现选择。初始化范围因此与原实验一致。

数据划分、每阶段轮数、batch size、类别权重、预测头学习率和 Prompt 学习率都沿用源配置。模型构建和初始评估后恢复源检查点的随机数状态；在相同运行环境下使 batch 打乱和预测头 Dropout 的随机序列对齐。跨软件版本或 GPU 内核不保证逐位一致。

在原 Colab 中更新代码后运行（Drive 已挂载）：

```bash
python -u train_shared_prompt.py --source /content/drive/MyDrive/MoleOOD_IIL/run_01/stage_0.pt --output-dir /content/drive/MyDrive/MoleOOD_IIL/shared_prompt_01
```

输出目录必须为空或不存在。结果写入独立的 `metrics.json`，不覆盖已有实验。

断点续训：

```bash
python -u train_shared_prompt.py --resume /content/drive/MyDrive/MoleOOD_IIL/shared_prompt_01/stage_2.pt
```

评估：

```bash
python evaluate.py --iil-checkpoint /content/drive/MyDrive/MoleOOD_IIL/shared_prompt_01/stage_4.pt --splits iid_test ood_test --device cuda:0
```

不要使用原 `train_iil.py` 恢复此对照。日志中的 `prompt_counts` 只有一项，等于评估样本数量，这是共享 Prompt 的预期行为。

将本实验与只训练预测头、检索 Prompt 池两个实验按相同阶段比较。这里只使用分类损失，不能直接拿数值与含 key 损失的总损失比较。参数选择应使用验证集，多随机种子实验用于判断结果是否稳定。
