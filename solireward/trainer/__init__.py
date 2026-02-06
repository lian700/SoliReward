"""
SoliReward Trainer Module

This module provides the training components for reward model training:
- BTWithLMHeadRewardTrainer: Main trainer class
- Loss functions: BT loss, BTT loss, BCE loss
- Logging utilities: TensorBoard, console, evaluation storage
"""

from .reward_trainer import BTWithLMHeadRewardTrainer
from .loss import (
    compute_bt_loss,
    compute_btt_loss,
    compute_bce_loss,
    compute_combined_loss,
)
from .logging import LoggingMixin

__all__ = [
    "BTWithLMHeadRewardTrainer",
    "compute_bt_loss",
    "compute_btt_loss", 
    "compute_bce_loss",
    "compute_combined_loss",
    "LoggingMixin",
]
