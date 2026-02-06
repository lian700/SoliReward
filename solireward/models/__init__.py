"""
SoliReward Models Module

This module contains implementations of various reward model architectures:
- InternVL3: InternVL3 reward model
- InternVL3-5: InternVL3.5 reward model
- QwenVL2.5: Qwen2-VL and Qwen2.5-VL reward models

Also includes:
- create_reward_model: Factory function for creating reward models
- load_reward_model_and_collator: Utility for loading trained models
"""

from .internvl3 import InternVL3RewardModel, InternVLDataCollator
from .internvl3_5 import InternVL35RewardModel
from .qwenvl2_5 import Qwen2VLRewardModel, Qwen25VLRewardModel, Qwen2_5VLDataCollator
from .reward_model import create_reward_model, load_reward_model_and_collator

__all__ = [
    # Factory functions
    "create_reward_model",
    "load_reward_model_and_collator",
    # InternVL3
    "InternVL3RewardModel",
    "InternVLDataCollator",
    # InternVL3.5
    "InternVL35RewardModel",
    # Qwen VL
    "Qwen2VLRewardModel",
    "Qwen25VLRewardModel",
    "Qwen2_5VLDataCollator",
]
