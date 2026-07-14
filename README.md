# 面向物理视频理解的高效微调实验

作者：周文煜（学号：3024244529）

本仓库复现论文中整个的 PEFT 实验。首次运行会自动下载数据集和预训练权重。所有训练均强制使用 CUDA，不会静默回退到 CPU。

## 实验环境

实验在单张 NVIDIA GeForce RTX 4070 Ti SUPER 16GB 上完成，训练使用 CUDA 12.8 wheel 和 AMP加速。完整实验至少需要占用 6GB 显存。

全部直接依赖及版本如下：

| 类别 | 依赖包 | 版本 | 用途 |
| --- | --- | --- | --- |
| 运行时 | Python | 3.12 | 代码运行 |
| 深度学习 | torch | 2.11.0+cu128 | CUDA 训练、AMP 和显存统计 |
| 深度学习 | torchvision | 0.26.0+cu128 | PyTorch 视觉算子 |
| 视觉模型 | timm | 1.0.27 | DeiT、DINOv2 和预训练权重 |
| 数值计算 | numpy | 2.4.6 | 数组与逐样本统计 |
| 数据处理 | pandas | 3.0.3 | manifest 和结果表 |
| 视频解码 | opencv-python-headless | 4.13.0.92 | 原始 MP4 解码与帧采样 |
| 指标计算 | scikit-learn | 1.9.0 | BAcc、F1 和 accuracy |
| 统计分析 | scipy | 1.17.1 | 配对区间与统计检验 |
| 数据工具 | remotezip | 0.12.3 | 数据归档兼容工具 |
| 实验日志 | tensorboard | 2.20.0 | 训练日志查看 |
| 测试 | pytest | 8.4.2 | 协议与参数预算测试 |
| 构建 | setuptools | 80.10.2 | 可编辑安装和 CLI 注册 |

在本仓库根目录逐行执行即可从空环境安装全部依赖：

```bash
conda create -n physlite-peft python=3.12 pip -y
conda activate physlite-peft

python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128

python -m pip install \
  numpy==2.4.6 pandas==3.0.3 opencv-python-headless==4.13.0.92 \
  scikit-learn==1.9.0 scipy==1.17.1 timm==1.0.27 \
  remotezip==0.12.3 tensorboard==2.20.0 pytest==8.4.2 setuptools==80.10.2

python -m pip install -e . --no-deps
```

建议创建名为 `physlite-peft` 的独立conda虚拟环境（运行代码本身不依赖该环境名），仓库提供等价的快捷conda环境安装方式：

```bash
conda env create -f environment.yml`
```

(可选)安装完成后检查环境、CUDA 和代码：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name()); assert torch.cuda.is_available()"
pytest -q
```

pytest测试不需要数据集或 GPU。

## 数据集下载

实验使用 Physion-Test-Core 的 1200 个视频。

在目录下运行以下命令即可自动下载约 271MiB 的公开压缩包、校验 SHA-256、解压到 `data/Physion/`，并检查主划分和五份重复划分不存在 family 泄漏：

```bash
physlite-prepare
```

数据集直链：[Physion.zip](https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/Physion.zip)。脚本校验值为:`1c80e51d9d299a54cc78bb20b9bb9b597d3b18067fd2f5a06e4e0a3a0c2c0c26`。

固定的无泄漏 CSV 划分已保存在 `data/manifests/`。DeiT 和 DINOv2 预训练权重由 `timm` 在首次训练时自动下载。

## 运行方式

下面给出三个粒度的复现，追求完整复现可直接跳到第三步。

### 1. 最小复现（测试）

下面的命令只训练冻结backbone的`head-only`组 和本文配置 D-SSF-LoRA，每个配置运行 seeds 0/1/2，共 6 次训练。它可以最快验证数据、训练、聚合和多种子结果是否完整连通。

```bash
physlite-run --suite main --only op_head allocation_q_last8_r4
physlite-report --suite main --allow-incomplete
```

结果位于：

```text
outputs/main/op_head/seed_*.json
outputs/main/allocation_q_last8_r4/seed_*.json
outputs/main/summary.md
outputs/main/summary.json
```

每个 JSON 均保存完整参数、随机种子、manifest 校验值、软件与 CUDA 版本、GPU、逐样本预测和场景指标。再次执行相同命令会自动跳过已完成任务；使用 `--force` 才会覆盖结果。

### 2. 复现论文主实验

主实验包含 28 个 PEFT 配置和 3 个随机种子，共 84 次训练，覆盖 head-only、BitFit、IA3、VPT、SSF、Adapter、AdaptFormer、LoRA、更新层、目标矩阵、rank、学习率以及 D-SSF-LoRA 消融：

```bash
physlite-run --suite main
physlite-report --suite main --verify
```

`--verify` 会将复现的 BAcc 均值与 `reference_results/main.json` 对比，默认允许 0.02 的数值误差。所有方法与超参数定义集中在 `configs/paper.json`。

### 3. 复现整篇论文

```bash
bash scripts/reproduce_all.sh
```

该脚本顺序执行 168 次单卡训练以及时间反事实评测，包括主矩阵、DeiT-B 迁移、分阶段训练、五份 grouped split、8/16 帧视频前缀、高分辨率与 DINOv2、任务头和反事实实验。单独复现某组实验时，可先查看并选择 suite：

```bash
physlite-run --list
physlite-run --suite deit_b
physlite-report --suite deit_b --verify
```

论文表格与 suite、输出文件的逐项对应关系见 [`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md)。

## 实验结果

下表是统一 DeiT-S、8 帧、family-grouped split 协议下的完整 28 配置矩阵。每行均为三个训练种子的均值与标准差；`Net PEFT` 扣除了所有方法共享的任务头参数，`Peak MiB` 为 PyTorch 记录的 CUDA 峰值分配显存。

| Stage | 配置 | 控制变量或作用 | Net PEFT | Test BAcc | Test F1 | Peak MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| A | Head-only | 冻结骨干参考 | 0 | .639 +/- .009 | .587 +/- .027 | 300 |
| A | BitFit | bias 更新 | 51,456 | .633 +/- .051 | .622 +/- .054 | 1,654 |
| A | IA3 last-4 q/v | 激活缩放 | 3,072 | .599 +/- .027 | .530 +/- .093 | 792 |
| A | LoRA last-4 q/v, r4 | 低秩更新，LR 3e-4 | 24,576 | .653 +/- .041 | .623 +/- .124 | 756 |
| A | Adapter last-4, b64 | 串行瓶颈 | 201,476 | .649 +/- .028 | .655 +/- .060 | 716 |
| A | AdaptFormer last-4, b64 | 并行瓶颈 | 198,400 | .665 +/- .021 | .679 +/- .015 | 640 |
| A | SSF all-layer | 特征仿射校准 | 19,200 | .664 +/- .005 | .622 +/- .027 | 2,091 |
| A | VPT, 8 tokens | prompt 更新 | 3,072 | .725 +/- .039 | .727 +/- .077 | 1,698 |
| A/B | SSF+LoRA last-4 q/v, r4 | 联合参考，LR 1e-3 | 43,776 | .723 +/- .036 | .737 +/- .027 | 2,167 |
| B | Matched LoRA last-4 q/v, r4 | 组合对照，LR 1e-3 | 24,576 | .716 +/- .017 | .720 +/- .061 | 756 |
| B | Joint LR 3e-4 | 优化设置对照 | 43,776 | .667 +/- .009 | .680 +/- .028 | 2,167 |
| B | Joint LR 7e-4 | 优化设置对照 | 43,776 | .691 +/- .041 | .689 +/- .037 | 2,167 |
| B | Joint LR 2e-3 | 优化设置对照 | 43,776 | .727 +/- .030 | .702 +/- .064 | 2,167 |
| C | SSF+LoRA last-2 q/v, r8 | 等预算层覆盖 | 43,776 | .701 +/- .069 | .655 +/- .150 | 2,129 |
| C | SSF+LoRA last-8 q/v, r2 | 等预算层覆盖 | 43,776 | .730 +/- .009 | .721 +/- .049 | 2,242 |
| C | Pure LoRA last-2 q/v, r8 | 纯 LoRA 等预算覆盖 | 24,576 | .687 +/- .034 | .692 +/- .023 | 488 |
| C | Pure LoRA last-8 q/v, r2 | 纯 LoRA 等预算覆盖 | 24,576 | .734 +/- .012 | .728 +/- .056 | 1,293 |
| C | Pure LoRA first-4 q/v, r4 | 等预算层窗口 | 24,576 | .685 +/- .036 | .687 +/- .038 | 1,681 |
| C | Pure LoRA middle-4 q/v, r4 | 等预算层窗口 | 24,576 | .732 +/- .021 | .729 +/- .020 | 1,218 |
| D | SSF+LoRA last-4 q-only, r4 | 目标矩阵对照 | 31,488 | .710 +/- .011 | .708 +/- .024 | 2,130 |
| D | SSF+LoRA last-4 v-only, r4 | 目标矩阵对照 | 31,488 | .666 +/- .003 | .635 +/- .006 | 2,130 |
| D | Pure LoRA last-8 q-only, r4 | SSF 贡献对照 | 24,576 | .740 +/- .012 | .742 +/- .037 | 1,220 |
| D | **D-SSF-LoRA: last-8 q-only, r4** | 等预算目标重分配 | 43,776 | **.756 +/- .020** | **.759 +/- .023** | 2,168 |
| D | SSF+LoRA last-8 v-only, r4 | 等预算目标重分配 | 43,776 | .734 +/- .009 | .729 +/- .040 | 2,168 |
| E | SSF+LoRA last-4 q/v, r1 | rank 扫描 | 25,344 | .682 +/- .014 | .694 +/- .042 | 2,166 |
| E | SSF+LoRA last-4 q/v, r2 | rank 扫描 | 31,488 | .684 +/- .007 | .654 +/- .029 | 2,166 |
| E | SSF+LoRA last-4 q/v, r8 | rank 扫描 | 68,352 | .730 +/- .009 | .726 +/- .033 | 2,168 |
| E | SSF+LoRA last-4 q/v, r16 | rank 扫描 | 117,504 | .749 +/- .031 | .752 +/- .045 | 2,170 |

论文附录B中有相同汇总。

最小复现完成后直接打开 `outputs/main/summary.md`。主实验完成后，机器可读的完整矩阵、配对差值、置信区间和 family bootstrap 位于 `outputs/main/summary.json`；论文原始运行的对应快照位于 `reference_results/main.json`。
