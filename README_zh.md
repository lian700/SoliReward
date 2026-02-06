# SoliReward

**Video Reward Model Training Framework**

[![Paper](https://img.shields.io/badge/arXiv-2512.22170-b31b1b.svg)](https://arxiv.org/abs/2512.22170)

SoliReward 是一个用于训练和推理视频奖励模型的框架，支持多种模型架构，包括 InternVL3、InternVL3-5、Qwen2.5-VL 和 Qwen2-VL。该框架对应论文 **"SoliReward: Mitigating Susceptibility to Reward Hacking and Annotation Noise in Video Generation Reward Models"**。

## 快速开始

### 1. 环境安装

```bash
cd SoliReward
bash scripts/setup_env.sh
conda activate solireward
```

### 2. 训练

修改 `scripts/solireward_train.sh` 中的配置后运行：

```bash
bash scripts/solireward_train.sh
```

### 3. 推理

修改 `scripts/solireward_infer.sh` 中的配置后运行：

```bash
bash scripts/solireward_infer.sh
```

## 支持的模型

| 模型类型    | 参数名        | 说明                 |
| ----------- | ------------- | -------------------- |
| InternVL3   | `InternVL3`   | InternVL3 系列模型   |
| InternVL3.5 | `InternVL3-5` | InternVL3.5 系列模型 |
| Qwen2.5-VL  | `Qwen2.5-VL`  | Qwen2.5-VL 系列模型  |
| Qwen2-VL    | `Qwen2-VL`    | Qwen2-VL 系列模型    |

## 数据格式

### 训练数据

JSON 文件，包含 win/lose 配对数据：

```json
[
  {
    "win": [...],
    "lose": [...],
    "meta": {"win": {"quality": 1.0}, "lose": {"quality": 0.0}}
  }
]
```

### 推理数据

```json
[
  {"video_path": "/path/to/video.mp4"},
  {"video_path": "/path/to/video2.mp4", "prompt": "描述文本"}
]
```

## 损失函数

- **BT Loss**: Bradley-Terry 排序损失
- **BTT Loss**: Bradley-Terry-Tie 处理平局样本
- **BCE Loss**: 二元交叉熵用于绝对质量预测

## 主要参数

| 参数                               | 说明                                                                                                                                               | 默认值                               |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `--model_type`                     | 模型类型                                                                                                                                           | `InternVL3`                          |
| `--bt_loss_coeff`                  | BT 损失系数                                                                                                                                        | `1.0`                                |
| `--bce_loss_coeff`                 | BCE 损失系数                                                                                                                                       | `0.0`                                |
| `--reward_margin`                  | 奖励边界                                                                                                                                           | `3.0`                                |
| `--reduce_sequence`                | 奖励模型头部架构。定义将视频帧序列聚合为最终奖励分数的方法                                                                                         | `progressive_hierarchical_attention` |
| `--hierarchical_query_attn_layers` | 空格分隔的层索引,用于添加层次化查询注意力。仅在 `reduce_sequence='progressive_hierarchical_attention'` 时使用。示例: `'6 12 18 24'` 适用于 1B 模型 | `'6 12 18 24'`                       |
| `--enable_btt_loss`                | 启用 Bradley-Terry-Tie 损失(设置为 `1` 启用)。**当训练数据包含平局样本时必须启用**(即 win 和 lose 质量相近的样本)                                  | `0`                                  |

## Citation

如果您觉得本项目对您的研究有帮助，请引用我们的论文：

```bibtex
@article{lian2025solireward,
  title={SoliReward: Mitigating Susceptibility to Reward Hacking and Annotation Noise in Video Generation Reward Models},
  author={Lian, Jiesong and Zhong, Ruizhe and Zhou, Zixiang and Mi, Xiaoyue and Hao, Yixue and Zhou, Yuan and Lu, Qinglin and Hu, Long and Yan, Junchi},
  journal={arXiv preprint arXiv:2512.22170},
  year={2025}
}
```

## License

MIT License
