# 圆形中央高分辨率输入（foveated vision）

## 1. 动机

人眼不是均匀采样：

- 中央凹分辨率高，信息多；
- 边缘分辨率低，信息少；
- 视野是圆形的，不是矩形；
- 注视中心时，余光不会以同样权重干扰。

## 2. 实现

- 源图 256×256；
- 输出 160×160；
- 圆形视野，外部置 0；
- 径向映射：

```
r_source = R * r_output^γ,  γ=2
```

- 中心：同样的输出面积对应更小的源面积 → 像素更多、更清楚；
- 边缘：同样的输出面积对应更大的源面积 → 像素更少；
- 整幅圆形画面仍然保留，不是只裁中心。

![foveated](results/fig_foveated.png)

## 3. 结果

10 个局部 pattern，每个 6 训练 + 6 测试背景；160×160 固定模型。

| 输入方式 | 检索准确率 | 目标-最大他人 gap |
|---|---:|---:|
| 均匀缩放 160×160 | 100% | 1,107 |
| 圆形 foveated 160×160 | 100% | **2,481** |

- 两种输入都能 100% 认出；
- foveated 的 gap 是均匀缩放的约 **2.2 倍**；
- 中央高分辨率让目标 pattern 与最相似他人的分离更强；
- 圆形 mask 去掉了矩形角落的背景干扰。

## 4. 解释

- 均匀缩放把 256×256 的所有区域压到同样分辨率；
- foveated 把更多像素分配给中心，pattern 在中央被放大；
- 边缘背景被压缩，减少余光干扰；
- 所以同一模型下，foveated 的中央目标更突出。

## 5. 限制

- 这里只是固定注视中心，没有眼动/扫视；
- 没有真实边缘模糊/颜色衰减；
- 没有多尺度融合；
- 只有 10 个合成 pattern；
- 1960×1080 还未与 foveated 结合。

## 6. 文件

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
