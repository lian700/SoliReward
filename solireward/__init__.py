# SoliReward - Video Reward Model Training Framework
# This package contains the core modules for training and inference of reward models

from .config import (
    ModelArguments,
    InternVLArguments,
    QwenVLArguments,
    DataTrainingArguments,
    TrainerArguments,
    parse_args,
    create_config,
    save_config_to_json,
    load_config_from_json,
)
from .models import create_reward_model
from .data import create_dataset_from_json
from .trainer import BTWithLMHeadRewardTrainer
from .inference import (
    InferenceArguments,
    VideoInferenceDataset,
    RewardModelInference,
    load_data_from_json,
    prepare_messages_from_data,
)

__version__ = "0.1.0"
__all__ = [
    # Config
    "ModelArguments",
    "InternVLArguments", 
    "QwenVLArguments",
    "DataTrainingArguments",
    "TrainerArguments",
    "parse_args",
    "create_config",
    "save_config_to_json",
    "load_config_from_json",
    # Model
    "create_reward_model",
    # Data
    "create_dataset_from_json",
    # Trainer
    "BTWithLMHeadRewardTrainer",
    # Inference
    "InferenceArguments",
    "VideoInferenceDataset",
    "RewardModelInference",
    "load_data_from_json",
    "prepare_messages_from_data",
]
