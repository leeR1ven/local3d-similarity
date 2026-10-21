# 三个 seed 的 margin 验证

随机模型 top-1 已经 100%，所以看不同画面的最小分离距离。

## 结果

| seed | baseline 最小不同距离 | L3 +10% | L5 +5% | L5 +10% |
|---:|---:|---:|---:|---:|
| 0 | 1893 | 1925 | 1982 | **2090** |
| 1 | 1888 | 1899 | 1993 | **2083** |
| 2 | 1863 | 1926 | 1942 | **2042** |

- L5 +10%：三个 seed 都提升约 **9.6–10.4%**；
- L5 +5%：提升约 **4.2–5.6%**；
- L3 +10%：只提升约 **0.6–3.4%**；
- AUC 仍为 1.0，top-1 没变，变化在 margin。

## 说明什么

> **增加 L5 神经元确实让不同画面拉得更开，而且三个 seed 一致。**

这不是“从不能分到能分”，而是：

- 原本就能分；
- 增加后最小距离更大；
- 区分更稳。

L5 比 L3 更有效；增加 10% 比 5% 更明显。

## 限制

- 随机 32×32 图像，不是真实复杂画面；
- 只看 margin，不看语义；
- 没有验证集选参，只有多 seed；
- L5 +10% 不是全局最优。

## 文件

```
threednet/discrimination_margin.py
threednet/margin_search.py
threednet/margin_validate.py
results/discrimination_margin.json
results/margin_search.json
results/margin_validate.json
MARGIN_SEARCH.md
MARGIN_VALIDATE.md
```
