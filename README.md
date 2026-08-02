<div align="center">
<h3> [AAAI 2026] SparseWorld：一种灵活、自适应且高效的4D占用世界模型，由稀疏动态查询驱动</h3>

<a href="https://arxiv.org/abs/2510.17482"><img src='https://img.shields.io/badge/arXiv-论文-red' alt='Paper PDF'></a>
[![Huggingface](https://img.shields.io/badge/Huggingface-模型-yellow?logo=Huggingface)](https://huggingface.co/MSunDYY2001/SparseWorld/tree/main)


<div align="center">

[党晨旭](https://msundyy.github.io)<sup>1,2\*</sup>，刘海燕<sup>3</sup>，鲍广军<sup>3</sup>，安培<sup>1</sup>，唐鑫月<sup>3</sup>，安攀<sup>4</sup>，马杰<sup>1†</sup>，\
孙秉川<sup>3†</sup>，王彦<sup>2†</sup>  


<sup>1</sup>华中科技大学  
<sup>2</sup>清华大学智能产业研究院（AIR） <sup>3</sup>联想集团\
<sup>4</sup>清华大学无锡应用技术研究院（AIRIC）

<div align="left">

## 摘要

语义占用已成为世界模型中一种强大的表征方式，因其能够捕捉丰富的空间语义信息。然而，现有的大多数占用世界模型依赖于静态且固定的嵌入或网格，这从根本上限制了感知的灵活性。此外，它们在网格上进行的"原地分类"方式与真实场景的动态性和连续性之间存在潜在的不对齐问题。

本文提出了 SparseWorld——一种新颖的4D占用世界模型，具有灵活性、自适应性和高效性，由稀疏动态查询驱动。我们提出了**范围自适应感知模块（Range-Adaptive Perception）**，其中可学习的查询由自车状态调制，并通过时空关联增强，从而实现扩展范围的感知。为了有效捕获场景动态，我们设计了**状态条件预测模块（State-Conditioned Forecasting）**，用回归引导的公式替代基于分类的预测，使动态查询精确地与4D环境的连续性对齐。此外，我们专门设计了**时序感知自调度训练策略（Temporal-Aware Self-Scheduling）**，以实现平稳高效的训练。

大量实验表明，SparseWorld 在感知、预测和规划任务上均达到了最先进的性能。全面的可视化和消融研究进一步验证了 SparseWorld 在灵活性、自适应性和效率方面的优势。

<div align="left">

## 概览

<img src="./pics/overview.png" width="1000">
</div>

<div align="left">

## 动态
- **`2026/1/13`**：我们发布了 SparseWorld 的升级版本 SparseOccVLA（[代码](https://github.com/MSunDYY/SparseOccVLA)，[论文](https://arxiv.org/abs/2601.06474)），成功将稀疏占用查询集成到大语言模型中，欢迎查看！
- **`2025/12/20`**：发布推理和训练代码以及预训练权重！
- **`2025/11/8`**：SparseWorld 被 AAAI 2026 接收！
- **`2025/10/10`**：论文在 [arXiv](https://arxiv.org/abs/2510.17482) 上发布。



## 快速开始
- [安装指南](docs/install.md)

- [数据集准备](docs/prepare_datasets.md)

- [训练与评估](docs/getting_started.md)


## 模型库


|          方法           |                            配置文件                            | 平均 mIoU | 平均 IoU |  日志  | 检查点 |
| :-----------------------: | :----------------------------------------------------------: | :------: | :-----: | :---: | :---------: |
| SparseWorld-R50               | [config](configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py) | 13.20 | 22.03 | [log](https://huggingface.co/MSunDYY2001/SparseWorld/blob/main/20260113_074102.log) | [model](https://huggingface.co/MSunDYY2001/SparseWorld/tree/main/ckpts) |

模型在 8 块 H20 GPU 上训练，实际仅占用约 17GB 显存，因此可在消费级显卡（如 RTX 4090）上复现。

## 实验结果与可视化

- 实验结果
<div align="left">
<img src="./pics/results.png" width="6000">

- 对比可视化

<img src="./pics/vis.png" width="6000">



## 致谢

本项目代码基于以下开源项目开发：
- [OPUS](https://github.com/jbwang1997/OPUS)
- [PreWorld](https://github.com/getterupper/PreWorld)

衷心感谢他们的杰出工作。

## 引用

如果您觉得我们的工作有帮助或有趣，请给我们一个 ⭐，感谢您的支持！

如果本工作对您的研究有帮助，请考虑引用：

```
@article{dang2025sparseworld,
  title={SparseWorld: A Flexible, Adaptive, and Efficient 4D Occupancy World Model Powered by Sparse and Dynamic Queries},
  author={Dang, Chenxu and Liu, Haiyan and Bao, Guangjun and An, Pei and Tang, Xinyue and Ma, Jie and Sun, Bingchuan and Wang, Yan},
  journal={arXiv preprint arXiv:2510.17482},
  year={2025}
}
```
```
@article{dang2026sparseoccvla,
  title={SparseOccVLA: Bridging Occupancy and Vision-Language Models via Sparse Queries for Unified 4D Scene Understanding and Planning}, 
  author={Dang, Chenxu and Wang, Jie and Guang, Li and Zihan, You and Hangjun, Ye and Jie, Ma and Long, Chen and Yan, Wang},
  journal={arXiv preprint arXiv:2601.06474},
  year={2026}
}
```
