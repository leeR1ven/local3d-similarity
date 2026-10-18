# Local 3D Sparse Networks

固定权重、局部连接的三维稀疏网络。没有卷积、没有训练、没有标签。

研究目标：

1. 相似画面是否激活相似神经元；
2. 局部画面是否比整体画面更容易在前层区分；
3. 模型变大是否提高区分度；
4. 身份是否更像“连续观察的状态轨迹”，而不是固定神经元集合。

> 这是研究原型，不是人脸识别/AGI 产品。  
> 1960×1080 全分辨率尚未验证。

## 核心结构

每个神经元是三维体积 `H×W×Z` 中的一个点：

```
自己 + 3 个随机近邻（dx,dy∈{-1,0,1}, dz∈{-2..2}）
```

每层用全局 top-k 保证精确激活比例；权重固定，不训练。

## 主要结果

### 1. 相似画面 → 相似激活

LFW 400 张中心人脸：

- 像素 L1 距离 vs 第5层 Hamming 距离：**Spearman 0.989**；
- L1–L8 都保持：5 层 L5 0.989，8 层 L8 0.990。

### 2. 局部 patch 的逐层区分

20 个完整 48×48 patch × 40 个背景，196×108：

| 层 | 检索准确率 | gap |
|---|---:|---:|
| L1 | 100% | 4,589 |
| L2 | 100% | 3,729 |
| L3 | 100% | 2,766 |
| L4 | 100% | 1,863 |
| L5 | 100% | 947 |

前层 gap 更大：局部性主要在 early layers，整体性在 later layers。

### 3. 模型变大

20 个 32×32 patch × 30% 遮挡，隐藏层宽度 `z=15→240`：

| z | connections | L5 gap | L2 gap |
|---:|---:|---:|---:|
| 15 | 5.08M | 496 | 2,323 |
| 30 | 10.16M | 891 | 4,559 |
| 60 | 20.32M | 2,956 | 10,471 |
| 120 | 40.64M | 4,724 | 20,970 |
| 240 | 81.29M | **9,989** | **41,952** |

- 准确率 z=60 起饱和到 100%；
- gap 持续增大（L5 约 20 倍）；
- 前层 gap 始终大于后层；
- 原始像素基线 100%，null 3.3%–8.3%。

![scaling](results/fig_scaling_clean.png)

### 4. 静态身份神经元：负结果

LFW 157 人 / 4,286 张：

- Rank-1 7.65%（随机 0.64%）；
- 目标激活率 0.366，最大他人 0.747；
- A/B 固定神经元 Jaccard 0.0049；
- 单神经元 selectivity 与打乱身份的 null 几乎相同。

结论：**没有稳定的固定身份神经元/组合**。

### 5. 连续观察：身份是状态轨迹

10 个身份 × 3 条序列 × 8 帧；姿势/光线/遮挡/背景变化；196×108：

```
s_t = normalize(0.8 * s_{t-1} + 0.2 * a_t)
```

| 指标 | 数值 |
|---|---:|
| 连续帧激活余弦 | ≈0.00004 |
| 连续状态余弦 | 0.704 |
| 同身份最终状态余弦 | 0.685 |
| 不同身份最终状态余弦 | 0.534 |
| 换人 drop | 0.175 |
| 遮挡 3 帧后同/不同 | 0.714 / 0.545 |
| null 同/不同差 | 0.012 |

不同姿势/光线激活不同帧，但状态轨迹保持身份；不是固定神经元集合。

![temporal](results/fig_temporal_identity.png)

## 快速开始

```powershell
$py = "C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe"
pip install -r requirements.txt

# 局部 patch 逐层
& $py -m threednet.local_patterns_opaque

# 模型 scaling
& $py -m threednet.scaling_clean

# 连续观察
& $py -m threednet.temporal_identity
```

## LFW 复现

LFW 不包含在仓库中。下载并解压到：

```
data/lfw/lfw/<person_name>/<image.jpg>
```

然后：

```powershell
& $py -m threednet.prepare_lfw
& $py -m threednet.lfw_rigorous
& $py -m threednet.prepare_lfw_large
& $py -m threednet.lfw_large
```

LFW 仅用于研究，遵守原数据集许可；不得用于监控/部署。

## 测试

```powershell
& $py tests/test_threednet.py
& $py tests/test_scaling.py
& $py tests/test_temporal_identity.py
```

完整测试集 26 个通过。

## 限制

- 1960×1080 彩色 tiled 未实现；
- 单体 1960×1080 约 22.5GB 连接，8GB GPU 放不下；
- 连续状态目前是 EMA，不是完整 recurrent attractor/Hebbian；
- 只有合成 patch/背景，没有真实视频；
- 没有多 seed/置信区间；
- 没有平移/尺度不变性要求，也不应强加；
- 不证明“足够大就能识别人脸”。

## 文件

```
threednet/          网络与实验
tests/              单元测试
results/            JSON + 图
LOCAL_PATTERNS.md   局部 patch 逐层实验
MODEL_SCALING.md    模型变大实验
TEMPORAL_IDENTITY.md 连续观察身份
LFW_RIGOROUS.md     LFW 静态身份负结果
LFW_SIMILARITY.md   LFW 画面相似性
```

## License

代码 MIT。数据（LFW/Olivetti）遵循各自许可，不随仓库分发。


