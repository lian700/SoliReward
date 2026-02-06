"""
Loss computation utilities for reward model training.

This module contains functions and classes for computing various loss functions
used in reward model training, including Bradley-Terry loss, BTT loss, and BCE loss.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple


def compute_bt_loss(rewards_chosen: torch.Tensor, rewards_rejected: torch.Tensor,
                    reward_margin: float = 0.0, label_smoothing: float = 0.0) -> torch.Tensor:
    """
    Compute Bradley-Terry ranking loss with optional label smoothing.
    
    Args:
        rewards_chosen: Reward scores for chosen samples
        rewards_rejected: Reward scores for rejected samples
        reward_margin: Margin value for reward ranking loss
        label_smoothing: Epsilon for label smoothing (0.0 disables)
        
    Returns:
        Bradley-Terry loss tensor of shape [batch_size]
    """
    if label_smoothing > 0.0:
        # L = - [ (1-ε) log σ(d) + ε log σ(-d) ] where d = r_win - r_lose - margin
        bt_loss = -(
            (1.0 - label_smoothing) * nn.functional.logsigmoid(rewards_chosen - rewards_rejected - reward_margin)
            + label_smoothing * nn.functional.logsigmoid(rewards_rejected - rewards_chosen - reward_margin)
        )
    else:
        # Standard BT loss: -log σ(r_win - r_lose - margin)
        bt_loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - reward_margin)
    
    return bt_loss


def compute_btt_loss(rewards_chosen: torch.Tensor, rewards_rejected: torch.Tensor,
                     reward_margin: float = 0.0) -> torch.Tensor:
    """
    Compute Bradley-Terry-Tie (BTT) loss for handling tied samples.
    
    Args:
        rewards_chosen: Reward scores for chosen samples
        rewards_rejected: Reward scores for rejected samples
        reward_margin: Margin value for reward ranking loss
        
    Returns:
        BTT loss tensor of shape [batch_size]
    """
    btt_loss = (
        -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - reward_margin)
        - nn.functional.logsigmoid(rewards_rejected - rewards_chosen - reward_margin)
    )
    return btt_loss


def compute_bce_loss(rewards_chosen: torch.Tensor, rewards_rejected: torch.Tensor,
                     quality_win: torch.Tensor, quality_lose: torch.Tensor,
                     label_smoothing: float = 0.0) -> torch.Tensor:
    """
    Compute Binary Cross-Entropy loss for absolute quality prediction.
    
    Args:
        rewards_chosen: Reward scores for chosen samples
        rewards_rejected: Reward scores for rejected samples
        quality_win: Quality labels for chosen samples
        quality_lose: Quality labels for rejected samples
        label_smoothing: Epsilon for label smoothing (0.0 disables)
        
    Returns:
        BCE loss scalar tensor
    """
    quality_concat = torch.cat([quality_win, quality_lose], dim=0)
    reward_concat = torch.cat([rewards_chosen, rewards_rejected], dim=0)
    quality_concat_binary = (quality_concat > 0).float()
    
    # Apply label smoothing if enabled
    if label_smoothing > 0.0:
        quality_concat_binary = quality_concat_binary * (1.0 - label_smoothing) + 0.5 * label_smoothing
    
    bce_loss_fn = nn.BCEWithLogitsLoss()
    bce_loss = bce_loss_fn(reward_concat, quality_concat_binary)
    
    return bce_loss


def compute_combined_loss(bt_loss: torch.Tensor, btt_loss: torch.Tensor, 
                          bce_loss: torch.Tensor, 
                          quality_win: torch.Tensor, quality_lose: torch.Tensor,
                          bt_loss_coeff: float = 1.0, btt_loss_coeff: float = 1.0,
                          bce_loss_coeff: float = 0.0) -> torch.Tensor:
    """
    Combine different loss components with masking for tie/non-tie samples.
    
    Args:
        bt_loss: Bradley-Terry loss tensor
        btt_loss: Bradley-Terry-Tie loss tensor
        bce_loss: BCE loss tensor
        quality_win: Quality labels for chosen samples
        quality_lose: Quality labels for rejected samples
        bt_loss_coeff: Coefficient for BT loss
        btt_loss_coeff: Coefficient for BTT loss
        bce_loss_coeff: Coefficient for BCE loss
        
    Returns:
        Combined loss scalar tensor
    """
    tie_mask = (quality_win == quality_lose)
    non_tie_mask = (quality_win != quality_lose)
    
    loss = (
        bt_loss_coeff * bt_loss * non_tie_mask 
        + btt_loss_coeff * btt_loss * tie_mask 
        + bce_loss_coeff * bce_loss
    )
    
    return loss.mean()
