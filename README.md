# 面向物理视频理解的高效微调实验

本仓库复现课程论文中的 PEFT 实验。建议按下列四节从上到下执行；首次运行会自动下载数据集和预训练权重。所有训练均强制使用 CUDA，不会静默回退到 CPU。

## 实验环境

实验在单张 NVIDIA GeForce RTX 4070 Ti SUPER 16GB 上完成，使用 Python 3.12、PyTorch 2.11.0、CUDA 12.8 和 AMP。其他支持 CUDA 的 NVIDIA GPU 也可运行；完整实验建议至少具有 16GB 显存。

在本仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate cvpr_paper
pip install -e . --no-deps
```

若 `cvpr_paper` 环境已经存在，只需执行后两行。随后检查环境、CUDA 和代码安装：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name()); assert torch.cuda.is_available()"
pytest -q
```

测试不需要数据集或 GPU；正式训练入口会在 CUDA 不可用时直接终止。

## 数据集下载

实验使用 Physion-Test-Core 的 1200 个视频。运行以下命令即可自动下载约 271MiB 的公开压缩包、校验 SHA-256、解压到 `data/Physion/`，并检查主划分和五份重复划分不存在 family 泄漏：

```bash
physlite-prepare
```

数据集直链：[Physion.zip](https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/Physion.zip)。脚本校验值为：

```text
1c80e51d9d299a54cc78bb20b9bb9b597d3b18067fd2f5a06e4e0a3a0c2c0c26
```

若本机已有解压后的数据，将其放置或软链接为 `data/Physion/`，然后执行：

```bash
physlite-prepare --skip-download
```

固定的无泄漏 CSV 划分已保存在 `data/manifests/`。DeiT 和 DINOv2 预训练权重由 `timm` 在首次训练时自动下载。

## 运行方式

### 1. 最小复现：建议先运行

下面的命令只训练冻结骨干 `head-only` 和本文配置 D-SSF-LoRA，每个配置运行 seeds 0/1/2，共 6 次训练。它可以最快验证数据、训练、聚合和多种子结果是否完整连通。

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

下表给出主实验的代表性参考结果，指标为三个训练种子的均值与标准差。`Net PEFT` 扣除了所有方法共享的任务头参数。

| 配置 | Net PEFT | Test BAcc | Test F1 |
| --- | ---: | ---: | ---: |
| Head-only | 0 | .639 +/- .009 | .587 +/- .027 |
| VPT-8 | 3,072 | .725 +/- .039 | .727 +/- .077 |
| Pure LoRA, last-2 q/v, rank 8 | 24,576 | .687 +/- .034 | .692 +/- .023 |
| Pure LoRA, last-8 q/v, rank 2 | 24,576 | .734 +/- .012 | .728 +/- .056 |
| Pure LoRA, last-8 query, rank 4 | 24,576 | .740 +/- .012 | .742 +/- .037 |
| SSF + last-4 q/v LoRA（参考配置） | 43,776 | .723 +/- .036 | .737 +/- .027 |
| D-SSF-LoRA | 43,776 | **.756 +/- .020** | **.759 +/- .023** |

最小复现完成后直接打开 `outputs/main/summary.md`；主实验完成后，完整 28 配置结果、配对差值、置信区间和 family bootstrap 位于 `outputs/main/summary.json`。仓库内的 `reference_results/` 是论文原始运行的紧凑快照，可在不附带 checkpoint 和逐视频预测的情况下核对结果。
