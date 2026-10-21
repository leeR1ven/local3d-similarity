# 已经能分清的画面：分裂是否让 margin 更大？

你说得对：

> 有些画面本来就能分清；结构可塑性不一定改变 top-1，  
> 但可能让正确类与其他类的距离拉得更开。

这时应该看 **margin / gap**，而不是只看 accuracy。

## 1. 24×24 任务：accuracy 饱和，但 margin 有变化

| 变体 | 准确率 | 目标-最大他人 gap |
|---|---:|---:|
| baseline | 96.0% | 33.15 |
| 分裂中间 5% | 96.0% | **38.36** |
| 分裂中间 5% + 删除 1% | **96.8%** | 35.08 |
| 分裂高频 5%，保留全部输入 | 96.0% | 37.91 |

这里：

- accuracy 基本不变；
- split_mid_5 的 gap 从 **33.15 → 38.36**；
- 确实有“已经分得清，但分得更开”的效果；
- 但提升不大，而且非常难任务上没有复现。

## 2. 一像素差异的二分类任务

两个 12×12 pattern 只差一个像素，随机遮挡有时会盖住差异：

- baseline 准确率：**100%**；
- 所有 prune/split 变体：**100%**；
- margin 基本不变；
- 说明这个任务虽然看起来难，但当前 L5 已经能完全分开，没有提升空间。

## 3. 12×12 极难任务

| 变体 | 准确率 | gap |
|---|---:|---:|
| baseline | 11.2% | -25.35 |
| 只分裂中间 5% | 10.8% | -25.34 |
| 删除 + 分裂 | 11.2% | -25.35 |

- 任务已经接近随机，分裂没有提升；
- 这不是“已经分得清”的情况，而是信息不足。

## 4. 结论

- **“已经能分清，但分裂让它分得更开”** 在 24×24 任务上有弱证据：gap 33.15→38.36，accuracy 不变；
- 在极难任务上没有；
- 在本来 100% 的二分类任务上也没有；
- 所以结构可塑性可能更多是**增大 margin**，而不是提高 top-1；
- 评价应该同时看：
  - accuracy；
  - target-vs-other gap；
  - pairwise AUC / margin 分布；
  - 最小 margin，而不是平均 margin。

## 5. 文件

```
threednet/structural_tune.py
results/structural_tune.json
threednet/structural_binary.py
results/structural_binary.json
STRUCTURAL_MARGIN.md
```
