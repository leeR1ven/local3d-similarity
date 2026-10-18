# 圆形中央清晰、边缘模糊的输入

## 1. 修正后的做法

之前用径向坐标变换做 foveation，会把画面几何拉伸，这是错的。  
现在改成：

- 几何完全不变，只改变清晰度；
- 圆形视野，外部置 0；
- 中心保持原分辨率；
- 边缘用低分辨率下采样再上采样得到模糊版本；
- 按半径平滑混合：中心 0% 模糊，边缘 100% 模糊。

不是把画面“卷”成 log-polar，而是：

> **同一个位置，同一个形状；中心清楚，边缘模糊。**

![foveated](results/fig_foveated.png)

## 2. 结果

10 个局部 pattern，每个 6 训练 + 6 测试背景；160×160 固定模型。

| 输入 | 检索准确率 | 目标-最大他人 gap |
|---|---:|---:|
| 均匀缩放 160×160 | 100% | 1,107 |
| 圆形 foveated（中心清晰、边缘模糊） | 100% | 1,080 |

- 两种输入都能 100% 认出；
- gap 基本相同（1,107 vs 1,080）；
- 在这个“目标在正中心”的任务上，foveated 没有带来分离度提升；
- 之前“gap 2,481”的结果来自几何扭曲，已撤回。

## 3. 说明什么

- foveation 不等于径向拉伸；
- 保持几何、只降低边缘清晰度，模型仍然能认出；
- 这个任务的目标本来就在中心，均匀输入已经足够，所以 foveated 没有优势；
- 要体现 foveated 的价值，需要 **目标偏离中心/背景干扰强/需要大视野** 的任务。

## 4. 限制

- 没有眼动/扫视；
- 没有真实边缘模糊/颜色衰减；
- 没有多尺度融合；
- 只有 10 个合成 pattern；
- 尚未与 1960×1080 结合。

## 5. 文件

```
threednet/foveated.py
results/foveated.json
results/fig_foveated.png
FOVEATED.md
```

运行：

```powershell
$py = "C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe"
& $py -m threednet.foveated
```
