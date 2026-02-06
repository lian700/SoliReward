# SoliReward

**Video Reward Model Training Framework**

[![Paper](https://img.shields.io/badge/arXiv-2512.22170-b31b1b.svg)](https://arxiv.org/abs/2512.22170)

SoliReward is a framework for training and inferencing video reward models, supporting multiple model architectures including InternVL3, InternVL3-5, Qwen2.5-VL, and Qwen2-VL. This framework corresponds to the paper **"SoliReward: Mitigating Susceptibility to Reward Hacking and Annotation Noise in Video Generation Reward Models"**.

## Quick Start

### 1. Environment Setup

```bash
cd SoliReward
bash scripts/setup_env.sh
conda activate solireward
```

### 2. Training

Modify the configuration in `scripts/solireward_train.sh` and run:

```bash
bash scripts/solireward_train.sh
```

### 3. Inference

Modify the configuration in `scripts/solireward_infer.sh` and run:

```bash
bash scripts/solireward_infer.sh
```

## Supported Models

| Model Type  | Parameter Name | Description               |
| ----------- | -------------- | ------------------------- |
| InternVL3   | `InternVL3`    | InternVL3 series models   |
| InternVL3.5 | `InternVL3-5`  | InternVL3.5 series models |
| Qwen2.5-VL  | `Qwen2.5-VL`   | Qwen2.5-VL series models  |
| Qwen2-VL    | `Qwen2-VL`     | Qwen2-VL series models    |

## Data Format

### Training Data

JSON file containing win/lose pair data:

```json
[
  {
    "win": [...],
    "lose": [...],
    "meta": {"win": {"quality": 1.0}, "lose": {"quality": 0.0}}
  }
]
```

### Inference Data

```json
[
  {"video_path": "/path/to/video.mp4"},
  {"video_path": "/path/to/video2.mp4", "prompt": "description text"}
]
```

## Loss Functions

- **BT Loss**: Bradley-Terry ranking loss
- **BTT Loss**: Bradley-Terry-Tie loss for handling tie samples
- **BCE Loss**: Binary Cross Entropy for absolute quality prediction

## Main Arguments

| Argument                           | Description                                                                                                                                                                     | Default Value                        |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `--model_type`                     | Model type                                                                                                                                                                      | `InternVL3`                          |
| `--bt_loss_coeff`                  | BT loss coefficient                                                                                                                                                             | `1.0`                                |
| `--bce_loss_coeff`                 | BCE loss coefficient                                                                                                                                                            | `0.0`                                |
| `--reward_margin`                  | Reward margin                                                                                                                                                                   | `3.0`                                |
| `--reduce_sequence`                | Reward model head architecture. Defines the method for aggregating video frame sequences into a final reward score                                                              | `progressive_hierarchical_attention` |
| `--hierarchical_query_attn_layers` | Space-separated layer indices to add hierarchical query attention. Only used when `reduce_sequence='progressive_hierarchical_attention'`. Example: `'6 12 18 24'` for 1B models | `'6 12 18 24'`                       |
| `--enable_btt_loss`                | Enable Bradley-Terry-Tie loss (set to `1` to enable). **Required when training data contains tie samples** (samples where win and lose have similar quality)                    | `0`                                  |

## Citation

If you find this project helpful for your research, please cite our paper:

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
