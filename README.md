# Local 3D Sparse Networks

固定权重、局部连接的三维稀疏网络。没有卷积、没有训练、没有标签。

研究目标：

1. 相似画面是否激活相似神经元；
2. 局部画面是否比整体画面更容易在前层区分；
3. 模型变大是否提高区分度；
4. 身份是否更像“连续观察的状态轨迹”，而不是固定神经元集合。

> 这是研究原型，不是人脸识别/AGI 产品。  
> 1960×1080 只做了 tiled streaming 局部 patch，不是单体全图网络。

## 核心结构

每个神经元是三维体积 `H×W×Z` 中的一个点：

```
自己 + 3 个随机近邻（dx,dy∈{-1,0,1}, dz∈{-2..2}）
```

每层用全局 top-k 保证精确激活比例；权重固定，不训练。

## 实验索引

| 主题 | 代码 | 文档 | 结论 |
|---|---|---|---|
| 局部 patch 逐层 | `local_patterns_opaque.py` | `LOCAL_PATTERNS.md` | L1–L5 100%，前层 gap 更大 |
| 模型变大 | `scaling_clean.py` | `MODEL_SCALING.md` | gap 随 z 增大，准确率饱和 |
| 连续身份 | `temporal_identity.py` | `TEMPORAL_IDENTITY.md` | 单帧不同，状态轨迹保持身份 |
| 圆形 foveated | `foveated.py` | `FOVEATED.md` | 几何不变，仅中心清晰 |
| 分辨率变化 | `resolution_test.py` | `RESOLUTION.md` | 48–256 都能认出 |
| 1960×1080 tiled | `tiled_1960.py` | `TILED_1960.md` | patch/tile 100%（非精确等价） |
| 精确 tiled 等价 | `tiled_exact.py` | `TILED_EXACT.md` | halo + 全局 top-k，bit-identical |
| 1960 全图精确流式 | `tiled_1960_exact.py` | `TILED_1960_EXACT.md` | z=4 全图逐层 bit-identical |
| 种子/层/读出 | `seed_eval.py`, `layer_eval.py` | `EVALUATION.md` | 种子波动不大，top-5 明显更高 |
| Hebbian attractor | `recurrent_attractor.py` | `RECURRENT_ATTRACTOR.md` | 弱于 EMA，负结果 |
| 人脸中心 foveated | `lfw_face_foveated.py` | `LFW_FACE_FOVEATED.md` | 4.52%，仍很低 |
| 人脸码碰撞/读出 | `face_readout.py` | `FACE_READOUT.md` | 0 碰撞，PCA+logistic 9.03% |
| 不同人脸 pairwise | `face_pairwise.py` | `FACE_PAIRWISE.md` | AUC .614，判不同 precision 99.5% |
| 学习式 pair 读出 | `face_pair_readout.py` | `FACE_PAIR_READOUT.md` | AUC .600，BA .584 |
| LFW 静态身份 | `lfw_rigorous.py` | `LFW_RIGOROUS.md` | 固定身份神经元负结果 |
| 严格专属神经元 | `exclusive_neurons.py` | `EXCLUSIVE_NEURONS.md` | 157 人全部 0 个 |
| 结构可塑性 | `structural_plasticity.py` | `STRUCTURAL_PLASTICITY.md` | 删少用/分裂常用；100%→99% |
| 难任务结构可塑性 | `structural_hard2.py` | `STRUCTURAL_HARD.md` | 50 类 96%→64.8% |
| 极难任务 + 机制拆分 | `structural_ablation.py`, `structural_veryhard.py` | `STRUCTURAL_VERYHARD.md` | 12×12 难任务：baseline 11.2%，分裂 9.6%，删除+分裂 7.6% |
| 固定 top-k 分裂 | `structural_fixed_k.py` | `STRUCTURAL_FIXED_K.md` | 12×12：仍 10.8%；24×24 gap 33.15→38.36 |
| 已可分画面的 margin | `structural_tune.py`, `structural_binary.py` | `STRUCTURAL_MARGIN.md` | accuracy 饱和时 gap 可增大 |
| 中间层分裂 | `structural_midlayer.py` | `STRUCTURAL_MIDLAYER.md` | 单次 8.0%→10.8%，未复现 |
| 中间层验证搜索 | `structural_midlayer_search.py` | `STRUCTURAL_MIDLAYER_SEARCH.md` | 验证 12%，测试 8.8%，未超 baseline |
| 近阈值受控任务 | `structural_gate.py` | `STRUCTURAL_GATE.md` | baseline 84%，结构变体全一样 |
| 结构可塑性总判断 | 多个 | `STRUCTURAL_VERDICT.md` | 当前最优是不删除/不分裂 |
| 随机模型通用区分 | `random_capacity.py` | `RANDOM_CAPACITY.md` | 同底图/不同底图距离比 0.0055，0 碰撞 |
| 容量受限结构可塑性 | `structural_capacity.py`, `structural_capacity_confirm.py` | `STRUCTURAL_CAPACITY.md` | 5k 子集：52–60%→75–79.5% |
| 可逆上下文 | `reversible_context.py` | `REVERSIBLE_CONTEXT.md` | A→B→A 可恢复，但无增益 |
| 长尾删除/增加神经元 | `structural_longtail.py` | `STRUCTURAL_LONGTAIL.md` | 删除10% 84→85% |
| 增加神经元方式 | `structural_split_variants.py` | `NEURON_EXPANSION.md` | 中间频率+2+2：84→87%，常见 88→94% |
| 画面相似性 | `lfw_similarity.py` | `LFW_SIMILARITY.md` | ρ=0.989 |

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

### 5. 大模型 LFW：uniform vs foveated

155 人 / 1,805 张 108×196，L5 30k 神经元，不选固定组合：

| 输入 | Rank-1 全部 | 相似身份 | 不相似身份 |
|---|---:|---:|---:|
| uniform | 2.58% | 2.56% | 2.60% |
| foveated（人脸中心） | **4.52%** | 3.85% | **5.19%** |

foveated 约 1.75 倍于 uniform；方向符合直觉，但绝对水平仍远不能识别人脸。

- 不同身份激活码完全相同：**0 对**；
- 不同身份最小 Hamming：**556 / 30,000**；
- pairwise AUC：**0.614**；
- PCA + logistic rank-1：**9.03%**；
- 所以模型有区分信息，低分主要是固定原型 rank-1 太严。

按“不同人脸区分”看：AUC 0.614，判为不同的 precision 99.5%，但 recall 57.4%；判为相同的 precision 只有 0.87%。
学习式 pairwise 读出（PCA + logistic）也只有 AUC 0.600、balanced accuracy 0.584，说明不只是阈值设置问题。

![LFW foveated](results/fig_lfw_face_foveated.png)

### 6. 连续观察：身份是状态轨迹

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

### 7. 圆形 foveated 输入

整幅圆形视野，中心高分辨率、边缘低分辨率：

| 输入 | 检索准确率 | gap |
|---|---:|---:|
| 均匀 160×160 | 100% | 1,107 |
| 圆形 foveated 160×160 | 100% | 1,080 |

保持几何不变，只降低边缘清晰度；在这个中心目标任务上分离度与均匀输入基本相同。

![foveated](results/fig_foveated.png)

### 8. 固定模型，改变分辨率

模型固定 160×160；源分辨率 48–256；原型只在 128 训练：

| 源分辨率 | 准确率 | gap |
|---:|---:|---:|
| 48 | 100% | 757 |
| 128（训练） | 100% | **1,070** |
| 256 | 100% | 1,002 |

分辨率降到 48 仍能认出，gap 在训练分辨率附近最高。

![resolution](results/fig_resolution_test.png)

### 9. 1960×1080 彩色 tiled

10×10 个 108×196 tile，每个 tile 独立编码，不构建全图单体网络：

- patch 识别 100%；
- 正确 tile 100%；
- 原始像素基线 100%；
- 10 个画布 ×100 tile 全部找对。

![tiled](results/fig_tiled_1960.png)

### 10. Hebbian recurrent attractor：负结果

512 维投影 + Hebbian 转移矩阵：

| 指标 | EMA | Hebbian attractor |
|---|---:|---:|
| 同身份 - 不同身份 | **0.201** | 0.056 |
| 换人 drop | 0.005 | **0.027** |
| 遮挡后同 - 不同 | **0.311** | 0.062 |

Hebbian 版本能跑，但身份分离和遮挡恢复弱于 EMA；需要更好的 recurrent 结构，而不是只加 W。

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

# foveated / 分辨率
& $py -m threednet.foveated
& $py -m threednet.resolution_test

# 1960×1080 tiled / recurrent attractor
& $py -m threednet.tiled_1960
& $py -m threednet.recurrent_attractor
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
& $py -m threednet.lfw_face_foveated
```

LFW 仅用于研究，遵守原数据集许可；不得用于监控/部署。

## 测试

```powershell
& $py tests/test_threednet.py
& $py tests/test_scaling.py
& $py tests/test_temporal_identity.py
& $py tests/test_foveated.py
& $py tests/test_resolution.py
& $py tests/test_tiled.py
& $py tests/test_tiled_exact.py
& $py tests/test_lfw_face_foveated.py
```

仓库内包含 9 个关键测试；本地完整测试集 31 个通过（其余测试依赖旧分支，未全部打包）。

## 限制

- 1960×1080 只做了 tiled streaming，没有单体全图网络；
- 单体 1960×1080 约 22.5GB 连接，8GB GPU 放不下；
- Hebbian recurrent attractor 已实现，但身份分离弱于 EMA；
- 只有合成 patch/背景，没有真实视频；
- 没有多 seed/置信区间；
- 没有平移/尺度不变性要求，也不应强加；
- foveated 与 1960×1080 还没有结合；
- 不证明“足够大就能识别人脸”。

## 文件

```
threednet/          网络与实验
tests/              单元测试
results/            JSON + 图
LOCAL_PATTERNS.md   局部 patch 逐层实验
MODEL_SCALING.md    模型变大实验
TEMPORAL_IDENTITY.md 连续观察身份
LFW_FACE_FOVEATED.md LFW 人脸中心 foveated
FACE_READOUT.md     人脸码碰撞与读出方式
FACE_PAIRWISE.md    不同人脸 pairwise 区分
FACE_PAIR_READOUT.md 学习式同人/不同人读出
FOVEATED.md         圆形中央高分辨率输入
RESOLUTION.md       固定模型、改变分辨率
TILED_1960.md       1960×1080 tiled streaming（非精确等价）
TILED_EXACT.md      精确 tiled = 单体全图
TILED_1960_EXACT.md 1960×1080 全图精确流式
EVALUATION.md       种子/层/读出/评判方式
RECURRENT_ATTRACTOR.md Hebbian recurrent attractor
LFW_RIGOROUS.md     LFW 静态身份负结果
EXCLUSIVE_NEURONS.md 严格“只在 A 激活”测试
STRUCTURAL_PLASTICITY.md 少用删除/常用分裂
STRUCTURAL_HARD.md   50 类细粒度任务：96%→64.8%
STRUCTURAL_VERYHARD.md 删除/分裂拆开：极难任务
STRUCTURAL_FIXED_K.md 固定 top-k 的分裂修正
STRUCTURAL_MARGIN.md 已能分清时看 margin
STRUCTURAL_MIDLAYER.md 中间层平分输入/输出
STRUCTURAL_MIDLAYER_SEARCH.md 中间层验证集搜索
STRUCTURAL_GATE.md   近阈值任务：结构改变无效
STRUCTURAL_VERDICT.md 当前最优：baseline
RANDOM_CAPACITY.md  随机模型本来就能区分
STRUCTURAL_CAPACITY.md 容量受限时分裂可能有帮助
REVERSIBLE_CONTEXT.md A→B→A 可逆经验
STRUCTURAL_LONGTAIL.md 长尾：删除用于压缩
NEURON_EXPANSION.md 增加神经元的不同方式
LFW_SIMILARITY.md   LFW 画面相似性
```

## License

代码 MIT。数据（LFW/Olivetti）遵循各自许可，不随仓库分发。





































