"""
SoliReward Inference Module

This module provides inference capabilities for trained reward models,
supporting multiple architectures including InternVL3, InternVL3-5, 
Qwen2.5-VL and Qwen2-VL.

Key components:
- RewardModelInference: Main inference engine class
- InferenceArguments: Configuration dataclass for inference
- VideoInferenceDataset: Dataset class for video inference
"""

from .arguments import InferenceArguments
from .dataset import VideoInferenceDataset
from .engine import RewardModelInference
from .utils import load_data_from_json, prepare_messages_from_data

__all__ = [
    "InferenceArguments",
    "VideoInferenceDataset", 
    "RewardModelInference",
    "load_data_from_json",
    "prepare_messages_from_data",
]
