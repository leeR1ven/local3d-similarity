# 流式 tile 与单体全图：怎样才能完全一样？

你的判断是对的：

> **如果流式加载正确，tile 激活应该和单体全图完全一样。**

之前 `tiled_1960.py` 用的是**每个 tile 独立 top-k**，所以不一样：

| 层 | 单体 active | per-tile active |
|---|---:|---:|
| L1 | 635,040 | 635,040 |
| L2 | 127,008 | 158,760 |
| L3 | 95,256 | 127,008 |
| L4 | 63,504 | 95,256 |
| L5 | 31,752 | 63,504 |

per-tile top-k 每个 tile 独立选，激活数会偏多，而且边界缺少跨 tile 竞争。

## 1. 精确等价需要两个条件

1. **halo**：局部连接跨 tile 时，要能读到相邻 tile 的神经元；
2. **global top-k**：所有 tile 的分数放一起排序，全局取 top-k，而不是每个 tile 各取。

## 2. 已实现的精确版本

在 108×196 画布、2×2 个 54×98 tile 上：

- 每个 tile 计算自己的 score；
- 但 gather 上一层的全局激活（halo 自动包含）；
- 所有 tile 的 score 汇总后，做**一次全局 top-k**。

结果：

```json
{
  "exact_equal": true,
  "exact_active": [635040, 127008, 95256, 63504, 31752],
  "monolithic_active": [635040, 127008, 95256, 63504, 31752]
}
```

**逐层 bit-identical**，和单体全图完全一样。

文件：

```
threednet/tiled_exact.py
results/tiled_exact.json
tests/test_tiled_exact.py
```

## 3. 1960×1080 能不能完全一样？

能，但需要：

- 第一遍扫描所有 tile，收集/维护全局 top-k 阈值；
- 第二遍再扫描所有 tile，应用全局 mask；
- 每层保留完整激活（或分块 + halo）；
- 不能每个 tile 独立 top-k。

这次只在 108×196 / 2×2 tile 上做了精确等价证明；
1960×1080 全图的精确两遍流式还没有跑。

## 4. 人脸识别低是流式导致的吗？

**不是。**

`lfw_face_foveated.py` 是直接对单张 108×196 人脸 crop 跑模型，
没有经过 `tiled_1960.py` 的 per-tile top-k。

所以：

- 人脸 rank-1 低，不是流式加载导致的；
- 是单帧表示/原型读出的问题；
- 流式只影响 1960×1080 全图拼接，不影响单张人脸测试。

## 5. 结论

> **流式要等于单体，必须 halo + global top-k；  
> 之前 per-tile top-k 不等价；  
> 精确版本已在 2×2 tile 上验证 bit-identical；  
> 1960×1080 精确两遍流式还没实现；  
> 人脸低分不是流式造成的。**

