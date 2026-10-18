# 1960×1080 彩色 tiled 实验

单体 1960×1080 网络约 22.5GB 连接，8GB GPU 放不下。  
这里用 **tile streaming**：把 1960×1080 切成 10×10 个 108×196 的 tile，
每个 tile 独立过同一个 196×108 网络，不构建全图单体网络。

## 1. 设置

- 画布 1960×1080 彩色；
- tile 108×196，10×10 共 100 个；
- 10 个 48×48 完整局部 patch；
- patch 随机放在某一个 tile 的正中心；
- 先扫描全部 100 个 tile，每个 tile 取中央区 L5 激活；
- 和 10 个 patch 原型比距离，取全局最高分；
- 同时输出预测的 patch 和 tile。

![tiled](results/fig_tiled_1960.png)

## 2. 结果

| 指标 | 结果 |
|---|---:|
| patch 识别准确率 | **100%** |
| 正确 tile 命中率 | **100%** |
| 相邻 tile 也正确 | 100% |
| 原始像素基线 | 100% |

10 个画布、每个 100 个 tile，全部找对 patch 和 tile。

## 3. 说明什么

- 全分辨率彩色图可以通过 tile streaming 处理，不需要 22.5GB 单体网络；
- 同一 tile 编码器在 100 个 tile 上复用；
- 中心 patch 能被定位到正确 tile；
- 这不是“全图全局网络”，tile 之间没有联合上下文。

## 4. 限制

- patch 放在 tile 正中心，没有跨 tile 边界；
- 每个画布只有一个目标 patch；
- 没有使用 foveated 输入；
- 没有真实人脸/物体检测；
- 不是单体 1960×1080 网络。

## 5. 文件

```
threednet/tiled_1960.py
results/tiled_1960.json
results/fig_tiled_1960.png
TILED_1960.md
```

运行：

```powershell
$py = "C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe"
& $py -m threednet.tiled_1960
```
