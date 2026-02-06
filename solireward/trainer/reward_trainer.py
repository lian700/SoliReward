"""
Reward model trainer module.

This module contains the main BTWithLMHeadRewardTrainer class for training
multimodal reward models with Bradley-Terry ranking loss.
"""

import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers.trainer import Trainer

from ..models.internvl3.reward_model import InternVL3RewardModel
from ..models.internvl3_5.reward_model import InternVL35RewardModel
from ..models.qwenvl2_5.reward_model import Qwen2VLRewardModel, Qwen25VLRewardModel
from ..utils.plot_utils import plot_eval_results, plot_training_metrics

from .logging import LoggingMixin
from .loss import compute_bt_loss, compute_btt_loss, compute_bce_loss, compute_combined_loss


# Filename for training config
_CONFIG_FILENAME = "training_config.json"


class BTWithLMHeadRewardTrainer(LoggingMixin, Trainer):
    """
    Custom Trainer for multimodal reward model training.
    
    This trainer extends HuggingFace's Trainer with:
    - Bradley-Terry ranking loss computation
    - Multi-model support (InternVL3, Qwen2-VL)
    - Gradient accumulation aware logging
    - Evaluation visualization
    """
    
    def __init__(self, *args, 
            use_global_metrics: bool = False,
            enable_btt_loss: int = 0,
            bt_loss_coeff: float = 1.0,
            btt_loss_coeff: float = 1.0,
            reward_margin: float = 0.0,
            bce_loss_coeff: float = 0.0,
            bt_label_smoothing: float = 0.0,
            bce_label_smoothing: float = 0.0,
        **kwargs):
        """
        Initialize the reward trainer.
        
        Args:
            use_global_metrics: Whether to gather metrics from all ranks
            enable_btt_loss: Whether to enable BTT loss component
            bt_loss_coeff: Coefficient for BT loss component
            btt_loss_coeff: Coefficient for BTT loss component
            reward_margin: Margin value for reward ranking loss
            bce_loss_coeff: Coefficient for BCE loss component
            bt_label_smoothing: Epsilon for BT loss label smoothing
            bce_label_smoothing: Epsilon for BCE loss label smoothing
        """
        # Initialize state variables
        self.is_train = True
        self.is_eval = not self.is_train
        self.eval_step = 0
        self.use_global_metrics = use_global_metrics
        self.eval_results = []
        self.print_log: bool = False
        self.training_metrics_history = []
        
        # Loss configuration
        self.enable_btt_loss = enable_btt_loss
        self.bt_loss_coeff = bt_loss_coeff
        self.btt_loss_coeff = btt_loss_coeff
        self.reward_margin = reward_margin
        self.bce_loss_coeff = bce_loss_coeff
        self.bt_label_smoothing = bt_label_smoothing
        self.bce_label_smoothing = bce_label_smoothing
        
        # Metrics accumulators
        self.train_metrics_accumulator = {
            'count': 0,
            'metric_sums': {}
        }
        self.eval_metrics_accumulator = {
            'count': 0,
            'metric_sums': {}
        }
        self.eval_steps_per_log = 1
        
        super().__init__(*args, **kwargs)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step with proper gradient accumulation handling.
        """
        if not hasattr(self, '_in_gradient_accumulation') or not self._in_gradient_accumulation:
            self._in_gradient_accumulation = True
            self._reset_train_metrics_accumulator()
        
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        is_sync_step = False
        if hasattr(self, 'accelerator') and hasattr(self.accelerator, 'gradient_state'):
            is_sync_step = self.accelerator.gradient_state.sync_gradients
        
        if is_sync_step:
            self._in_gradient_accumulation = False
        
        return loss
    
    def _prepare_model_inputs(self, model: nn.Module, inputs: Dict[str, Any]) -> tuple:
        """
        Prepare concatenated model inputs based on model type.
        
        Returns:
            Tuple of (batch_size, concatenated_kwargs)
        """
        if isinstance(model, (InternVL3RewardModel, InternVL35RewardModel)) or \
           (hasattr(model, 'module') and isinstance(model.module, (InternVL3RewardModel, InternVL35RewardModel))):
            batch_size = inputs["win"]["input_ids"].shape[0]
            
            concatenated_kwargs = {
                "input_ids": torch.cat([
                    inputs["win"]["input_ids"], 
                    inputs["lose"]["input_ids"]
                ], dim=0),
                "attention_mask": torch.cat([
                    inputs["win"]["attention_mask"], 
                    inputs["lose"]["attention_mask"]
                ], dim=0),
                "return_dict": True,
            }
            
            if "pixel_values" in inputs["win"] and "pixel_values" in inputs["lose"]:
                concatenated_kwargs["pixel_values"] = torch.cat([
                    inputs["win"]["pixel_values"], 
                    inputs["lose"]["pixel_values"]
                ], dim=0)
                concatenated_kwargs["image_flags"] = torch.cat([
                    inputs["win"]["image_flags"], 
                    inputs["lose"]["image_flags"]
                ], dim=0)
            else:
                if "pixel_values" not in inputs["win"] or "pixel_values" not in inputs["lose"]:
                    logging.warning("Missing pixel_values in either 'win' or 'lose' inputs.")
        
        elif isinstance(model, (Qwen2VLRewardModel, Qwen25VLRewardModel)) or \
             (hasattr(model, 'module') and isinstance(model.module, (Qwen2VLRewardModel, Qwen25VLRewardModel))):
            batch_size = inputs["batch_size"]
            concatenated_kwargs = inputs['inputs_concat']

        else:
            raise ValueError(f"Model must be an instance of InternVL3RewardModel or Qwen2VLRewardModel, "
                           f"got {type(model)}")
        
        return batch_size, concatenated_kwargs
    
    def _extract_rewards(self, outputs, batch_size: int) -> tuple:
        """
        Extract reward scores from model outputs.
        
        Returns:
            Tuple of (rewards_chosen, rewards_rejected)
        """
        if hasattr(outputs, 'reward_scores'):
            all_rewards = outputs.reward_scores
        elif isinstance(outputs, dict) and 'reward_scores' in outputs:
            all_rewards = outputs['reward_scores']
        else:
            logging.error("Model outputs do not contain 'reward_scores'.")
            all_rewards = outputs.get("logits", outputs)
            if isinstance(all_rewards, torch.Tensor) and all_rewards.dim() > 1:
                all_rewards = all_rewards.mean(dim=-1)
        
        rewards_chosen = all_rewards[:batch_size]
        rewards_rejected = all_rewards[batch_size:]
        
        return rewards_chosen, rewards_rejected
    
    def compute_loss(
        self,
        model: Union[InternVL3RewardModel, nn.Module],
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple]:
        """
        Compute the ranking loss for reward model training.
        """
        # Prepare model inputs
        batch_size, concatenated_kwargs = self._prepare_model_inputs(model, inputs)
        
        # Forward pass
        concatenated_outputs = model(**concatenated_kwargs)
        
        # Extract rewards
        rewards_chosen, rewards_rejected = self._extract_rewards(concatenated_outputs, batch_size)
        
        # Get quality labels
        quality_win = inputs["win"]["quality"]
        quality_lose = inputs["lose"]["quality"]
        
        # Compute losses
        bt_loss = compute_bt_loss(
            rewards_chosen, rewards_rejected,
            reward_margin=self.reward_margin,
            label_smoothing=self.bt_label_smoothing,
        )
        
        if self.enable_btt_loss:
            btt_loss = compute_btt_loss(rewards_chosen, rewards_rejected, self.reward_margin)
        else:
            btt_loss = torch.zeros_like(bt_loss)
        
        if self.bce_loss_coeff > 0.0:
            bce_loss = compute_bce_loss(
                rewards_chosen, rewards_rejected,
                quality_win, quality_lose,
                label_smoothing=self.bce_label_smoothing
            )
        else:
            bce_loss = torch.zeros_like(bt_loss)
        
        # Combine losses
        loss = compute_combined_loss(
            bt_loss, btt_loss, bce_loss,
            quality_win, quality_lose,
            bt_loss_coeff=self.bt_loss_coeff,
            btt_loss_coeff=self.btt_loss_coeff,
            bce_loss_coeff=self.bce_loss_coeff,
        )
        
        # Handle logging
        if self.state.global_step % self.args.logging_steps == 0 or self.is_eval:
            metrics_dict = {
                'bt_loss': bt_loss.mean(),
                'btt_loss': btt_loss.mean(),
                'bce_loss': bce_loss if isinstance(bce_loss, torch.Tensor) and bce_loss.numel() == 1 else torch.tensor(bce_loss),
                'rewards_chosen': rewards_chosen,
                'rewards_rejected': rewards_rejected,
                'quality_chosen': quality_win,
                'quality_rejected': quality_lose,
            }
            self._handle_logging(metrics_dict, inputs, loss)
                          
        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss

    def prediction_step(
        self,
        model: Union[InternVL3RewardModel, nn.Module],
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> tuple:
        """
        Prediction step for multimodal reward model evaluation.
        """
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            ignore_keys = []

        with torch.no_grad():
            loss_and_outputs = self.compute_loss(model, inputs, return_outputs=True)
            if isinstance(loss_and_outputs, tuple):
                loss, logits_dict = loss_and_outputs
            else:
                loss = loss_and_outputs
                logits_dict = {}

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        
        if isinstance(logits_dict, dict) and "rewards_chosen" in logits_dict and "rewards_rejected" in logits_dict:
            rewards_chosen = logits_dict["rewards_chosen"]
            rewards_rejected = logits_dict["rewards_rejected"]
            
            logits = torch.stack([rewards_chosen.squeeze(), rewards_rejected.squeeze()], dim=1)
            logits = torch.softmax(logits, dim=1)
        else:
            logging.error("Logits dictionary does not contain 'rewards_chosen' and 'rewards_rejected'.")
            if "win" in inputs and "input_ids" in inputs["win"]:
                batch_size = inputs["win"]["input_ids"].shape[0]
            elif "win" in inputs and "quality" in inputs["win"]:
                batch_size = len(inputs["win"]["quality"])
            else:
                batch_size = 1
                logging.warning("Could not determine batch size, using default of 1")
            logits = torch.zeros((batch_size, 2))

        labels = torch.zeros(logits.shape[0])

        return loss, logits, labels

    def evaluate(self, *args, **kwargs):
        """
        Evaluate the model and visualize samples.
        """
        self.is_eval = True
        self.is_train = not self.is_eval
        self.eval_step += int(self.args.eval_steps) if self.args.eval_steps is not None and self.args.eval_steps > 0 else 1
        
        self._calculate_eval_steps_per_log()
        self._reset_train_metrics_accumulator()
        self._reset_eval_metrics_accumulator()
        self.eval_results = []
        
        res = super().evaluate(*args, **kwargs)
        
        # Flush remaining eval metrics
        if self.eval_metrics_accumulator['count'] > 0 and hasattr(self, 'log'):
            count = self.eval_metrics_accumulator['count']
            log_dict = {}
            
            for metric_name, metric_sum in self.eval_metrics_accumulator['metric_sums'].items():
                log_dict[f"eval/{metric_name}"] = metric_sum / count
            
            self.log(log_dict)
            self._reset_eval_metrics_accumulator()
        
        # Plot results only on rank 0
        should_plot = (self.args.local_rank in [-1, 0] and len(self.eval_results) > 0)
        
        if should_plot:
            base_output_dir = getattr(self.args, 'output_dir', None)
            if base_output_dir:
                output_dir = os.path.join(base_output_dir, 'eval_plots')
            else:
                output_dir = 'eval_plots'
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                distribution_metrics = plot_eval_results(
                    eval_results=self.eval_results,
                    output_dir=output_dir,
                    eval_step=self.eval_step,
                    smooth_window=1
                )
                
                if isinstance(distribution_metrics, dict):
                    training_metrics = {
                        'global_step': self.state.global_step,
                        'sample_pair_positive_ratio': distribution_metrics.get('sample_pair_positive_ratio', 0.0),
                        'all_pair_positive_ratio': distribution_metrics.get('all_pair_positive_ratio', 0.0),
                        'best_f1_score': distribution_metrics.get('best_f1_score', 0.0)
                    }
                    self.training_metrics_history.append(training_metrics)
                    
                    if len(self.training_metrics_history) > 0:
                        plot_training_metrics(self.training_metrics_history, output_dir)
                
                if self.print_log:
                    print(f"Saved evaluation plots for step {self.eval_step} to {output_dir}")
            except Exception as e:
                print(f"Error plotting evaluation results: {e}")
        
        self.is_eval = False
        self.is_train = not self.is_eval
        self._reset_train_metrics_accumulator()
        return res

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        """
        Override checkpoint saving to include training config and tokenizer/processor.
        
        This ensures each checkpoint directory is self-contained with all necessary
        files for inference, including:
        - Model weights (handled by parent class)
        - training_config.json (copied from output_dir)
        - tokenizer/processor files (copied from original model path)
        """
        # Call parent's _save_checkpoint to handle model saving
        super()._save_checkpoint(model, trial, *args, **kwargs)
        
        # Only save additional files on rank 0
        if self.args.local_rank not in [-1, 0]:
            return
        
        # Determine the checkpoint directory that was just created
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        
        if not os.path.isdir(checkpoint_dir):
            logging.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return
        
        # Copy training_config.json to checkpoint directory
        self._copy_config_to_checkpoint(checkpoint_dir)
        
        # Copy tokenizer/processor to checkpoint directory
        self._copy_tokenizer_to_checkpoint(checkpoint_dir)
    
    def _copy_config_to_checkpoint(self, checkpoint_dir: str) -> None:
        """Copy training_config.json from output_dir to checkpoint directory."""
        source_config = os.path.join(self.args.output_dir, _CONFIG_FILENAME)
        target_config = os.path.join(checkpoint_dir, _CONFIG_FILENAME)
        
        if os.path.exists(source_config):
            try:
                shutil.copy2(source_config, target_config)
                logging.info(f"Copied {_CONFIG_FILENAME} to {checkpoint_dir}")
            except Exception as e:
                logging.warning(f"Failed to copy config to checkpoint: {e}")
        else:
            logging.warning(f"Config file not found at {source_config}")
    
    def _copy_tokenizer_to_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Copy tokenizer/processor files from the original model path to checkpoint directory.
        
        This looks for tokenizer files in the training config to find the original
        tokenizer path, then copies all relevant files.
        """
        import json
        
        # Read training config to get original tokenizer path
        config_path = os.path.join(self.args.output_dir, _CONFIG_FILENAME)
        if not os.path.exists(config_path):
            logging.warning("Cannot copy tokenizer: training config not found")
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            tokenizer_path = config.get("model_args", {}).get("tokenizer_name_or_path")
            if not tokenizer_path:
                tokenizer_path = config.get("model_args", {}).get("model_name_or_path")
            
            if not tokenizer_path or not os.path.isdir(tokenizer_path):
                logging.warning(f"Tokenizer path not found or invalid: {tokenizer_path}")
                return
            
            # Common tokenizer/processor files to copy
            tokenizer_files = [
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "added_tokens.json",
                "tokenizer.model",
                # Processor files (for multimodal models)
                "preprocessor_config.json",
                "processor_config.json",
                "chat_template.json",
                "image_processor_config.json",
                "video_processor_config.json",
            ]
            
            copied_count = 0
            for filename in tokenizer_files:
                source_file = os.path.join(tokenizer_path, filename)
                target_file = os.path.join(checkpoint_dir, filename)
                
                if os.path.exists(source_file) and not os.path.exists(target_file):
                    try:
                        shutil.copy2(source_file, target_file)
                        copied_count += 1
                    except Exception as e:
                        logging.warning(f"Failed to copy {filename}: {e}")
            
            if copied_count > 0:
                logging.info(f"Copied {copied_count} tokenizer/processor files to {checkpoint_dir}")
                
        except Exception as e:
            logging.warning(f"Failed to copy tokenizer to checkpoint: {e}")
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override save_model to also save tokenizer/processor files.
        
        This ensures the final saved model directory contains all necessary
        files for inference.
        """
        # Call parent's save_model
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        
        # Only save additional files on rank 0
        if self.args.local_rank not in [-1, 0]:
            return
        
        # Determine output directory
        save_dir = output_dir if output_dir is not None else self.args.output_dir
        
        # Copy tokenizer/processor to save directory
        self._copy_tokenizer_to_checkpoint(save_dir)
