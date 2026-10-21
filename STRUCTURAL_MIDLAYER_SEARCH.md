# 中间层结构可塑性的验证集搜索

针对上一轮“中间层 8.0%→10.8%”的结果，做了真正的 train/validation/test 分离：

- 每类 3 张训练；
- 2 张验证，用于选参数；
- 5 张测试，完全不参与调参；
- 随机划分，避免顺序假象；
- 2 个隐藏层随机种子；
- 60 组 prune/split 配置。

模型：

- 固定 3D 编码器 → L5 特征；
- 1,000 个隐藏单元；
- 隐藏层 top-k=200；
- Ridge 线性读出；
- 分裂高频/中间/高方差单元；
- 子单元平分输入和输出权重；
- 删除低使用单元；
- 结构改变后重新训练读出。

## 结果

| 指标 | 数值 |
|---|---:|
| baseline 验证准确率 | 7–11% |
| baseline 测试准确率 | 7.2–10.4% |
| 最佳验证配置 | 12% |
| 该配置测试准确率 | **8.8%** |
| 最佳测试（不同 seed） | 10.4% |

- 验证集最佳 12%，只比 baseline 高 1–5 个百分点；
- 但对应测试只有 8.8%；
- 没有超过 baseline 的最好测试 10.4%；
- 60 组参数中，没有稳定超过 baseline 的配置。

## 结论

> **中间层 + 输出权重平分的方向是对的，但当前实现仍没有稳定提升。  
> 上一轮 10.8% 是单次、无验证集的结果；这次验证后没有复现。**

可能原因：

- 随机隐藏层本身太弱；
- L5 特征对 12×12 极难任务信息不足；
- 结构改变后只重训线性读出，不能弥补输入切分；
- 任务接近随机，参数差异被噪声淹没。

## 文件

```
threednet/structural_midlayer.py
threednet/structural_midlayer_search.py
results/structural_midlayer.json
results/structural_midlayer_search.json
STRUCTURAL_MIDLAYER_SEARCH.md
```
