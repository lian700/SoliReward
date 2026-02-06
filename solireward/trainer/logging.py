"""
Logging utilities for reward model training.

This module contains mixin classes that provide logging functionality
for TensorBoard, console output, and evaluation result storage.
"""

import json
import math
import os
from typing import Any, Dict, Optional

import torch


class LoggingMixin:
    """
    Mixin class providing logging functionality for reward model training.
    
    This mixin provides methods for:
    - Metrics computation and organization
    - TensorBoard logging with gradient accumulation handling
    - Console logging with detailed information
    - Evaluation result storage for plotting
    """
    
    def _reset_train_metrics_accumulator(self):
        """Reset the training metrics accumulator"""
        self.train_metrics_accumulator = {
            'count': 0,
            'metric_sums': {}  # Dynamic dictionary to store all metric sums
        }
    
    def _reset_eval_metrics_accumulator(self):
        """Reset the eval metrics accumulator"""
        self.eval_metrics_accumulator = {
            'count': 0,
            'metric_sums': {}  # Dynamic dictionary to store all metric sums
        }
    
    def _calculate_eval_steps_per_log(self):
        """
        Calculate how many eval steps to accumulate before logging to tensorboard
        Based on eval dataset length divided by world size, rounded up
        """
        try:
            # Get eval dataset length
            eval_dataset_length = 100  # Default fallback
            
            if hasattr(self, 'eval_dataset') and self.eval_dataset is not None:
                if hasattr(self.eval_dataset, '__len__'):
                    try:
                        eval_dataset_length = len(self.eval_dataset)
                    except (TypeError, AttributeError):
                        pass
                elif isinstance(self.eval_dataset, dict):
                    # Handle case where eval_dataset is a dict of datasets
                    for dataset in self.eval_dataset.values():
                        if hasattr(dataset, '__len__'):
                            try:
                                eval_dataset_length = len(dataset)
                                break
                            except (TypeError, AttributeError):
                                continue
                
            # Get world size for distributed training
            if self.args.local_rank != -1 and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
            else:
                world_size = 1
            
            # Calculate steps per log: ceil(dataset_length / world_size)
            steps_per_log = math.ceil(eval_dataset_length / world_size)
            
            # Ensure at least 1 step
            self.eval_steps_per_log = max(1, steps_per_log)
            
            if self.print_log and self.args.local_rank in [-1, 0]:
                print(f"Eval dataset length: {eval_dataset_length}, World size: {world_size}, "
                      f"Steps per log: {self.eval_steps_per_log}")
                      
        except Exception as e:
            if self.print_log:
                print(f"Error calculating eval steps per log: {e}. Using default value of 1.")
            self.eval_steps_per_log = 1
    
    def _compute_metrics(self, metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute and organize all metrics from model outputs
        
        Args:
            metrics_dict: Dictionary containing all raw metrics (tensors, values, etc.)
                Expected to contain keys like:
                - 'rewards_chosen', 'rewards_rejected' for reward computation
                - various loss tensors
                - any other custom metrics
            
        Returns:
            Dict of computed metrics with processed values ready for logging
        """
        processed_metrics = {}
        
        # Process reward-related metrics if available
        if 'rewards_chosen' in metrics_dict and 'rewards_rejected' in metrics_dict:
            rewards_chosen = metrics_dict['rewards_chosen']
            rewards_rejected = metrics_dict['rewards_rejected']
            
            # Compute reward difference and positive ratio
            reward_diff = rewards_chosen - rewards_rejected
            positive_diff_ratio = (reward_diff > 0).float().mean()
            
            # Compute reward statistics
            processed_metrics.update({
                'chosen_reward_mean': rewards_chosen.detach().mean(),
                'rejected_reward_mean': rewards_rejected.detach().mean(),
                'reward_diff_mean': reward_diff.detach().mean(),
                'positive_diff_ratio': positive_diff_ratio.detach(),
                'rewards_chosen': rewards_chosen,
                'rewards_rejected': rewards_rejected,
            })
        
        # Process all other metrics dynamically
        for key, value in metrics_dict.items():
            if key not in processed_metrics:
                processed_metrics[key] = value
                
        return processed_metrics
    
    def _log_to_tensorboard(self, metrics: Dict[str, Any]) -> None:
        """
        Log metrics to TensorBoard, handling gradient accumulation for training mode
        Supports logging any type of metrics dynamically
        
        Args:
            metrics: Dict of metrics returned from _compute_metrics
        """
        # Separate metrics into scalars and non-scalars for processing
        scalar_metrics = {}
        non_scalar_metrics = {}
        
        for key, value in metrics.items():
            # Skip non-loggable metrics (like raw tensors for detailed logging)
            if key in ['rewards_chosen', 'rewards_rejected', 'quality_chosen', 'quality_rejected']:
                non_scalar_metrics[key] = value
                continue
                
            # Convert tensor to scalar if needed
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    scalar_metrics[key] = value.item()
                elif value.dim() > 0:
                    # For multi-element tensors, use mean
                    scalar_metrics[key] = value.mean().item()
                else:
                    scalar_metrics[key] = value.item()
            else:
                scalar_metrics[key] = value
        
        # Apply all_reduce for TensorBoard logging if global metrics are enabled
        if self.use_global_metrics and self.args.local_rank != -1 and torch.distributed.is_initialized():
            for key, value in scalar_metrics.items():
                if isinstance(value, (int, float)):
                    tensor_value = torch.tensor(value, device=self.args.device, dtype=torch.float32)
                    torch.distributed.all_reduce(tensor_value, op=torch.distributed.ReduceOp.AVG)
                    scalar_metrics[key] = tensor_value.item()
        
        if self.is_eval:
            # For evaluation: handle accumulation based on eval_steps_per_log
            for key, value in scalar_metrics.items():
                if key not in self.eval_metrics_accumulator['metric_sums']:
                    self.eval_metrics_accumulator['metric_sums'][key] = 0.0
                self.eval_metrics_accumulator['metric_sums'][key] += value
            
            self.eval_metrics_accumulator['count'] += 1
            
            # Log accumulated metrics only when reaching eval_steps_per_log
            should_log = (self.eval_metrics_accumulator['count'] % self.eval_steps_per_log == 0)
            
            if should_log and hasattr(self, 'log') and self.eval_metrics_accumulator['count'] > 0:
                count = self.eval_metrics_accumulator['count']
                log_dict = {}
                
                for metric_name, metric_sum in self.eval_metrics_accumulator['metric_sums'].items():
                    log_dict[f"eval/{metric_name}"] = metric_sum / self.eval_steps_per_log
                
                self.log(log_dict)
                self._reset_eval_metrics_accumulator()
        else:
            # For training: handle gradient accumulation
            for key, value in scalar_metrics.items():
                if key not in self.train_metrics_accumulator['metric_sums']:
                    self.train_metrics_accumulator['metric_sums'][key] = 0.0
                self.train_metrics_accumulator['metric_sums'][key] += value
            
            self.train_metrics_accumulator['count'] += 1
            
            # Check if this is a gradient sync step
            is_sync_step = self._is_gradient_sync_step()
            
            # Log accumulated metrics only on sync steps
            if is_sync_step and hasattr(self, 'log') and self.train_metrics_accumulator['count'] > 0:
                count = self.train_metrics_accumulator['count']
                log_dict = {}
                
                for metric_name, metric_sum in self.train_metrics_accumulator['metric_sums'].items():
                    log_dict[f"train/{metric_name}"] = metric_sum / count
                
                self.log(log_dict)
                self._reset_train_metrics_accumulator()
    
    def _is_gradient_sync_step(self) -> bool:
        """
        Detect if this is a gradient synchronization step (end of gradient accumulation)
        
        Returns:
            bool: True if this is a sync step
        """
        if hasattr(self, 'accelerator') and hasattr(self.accelerator, 'gradient_state'):
            return self.accelerator.gradient_state.sync_gradients
        else:
            expected_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
            return (self.train_metrics_accumulator['count'] % expected_steps == 0)
    
    def _log_to_console(self, metrics: Dict[str, Any], inputs: Dict[str, Any], 
                        loss: torch.Tensor) -> Dict[str, Any]:
        """
        Log detailed information to console as JSON
        Supports dynamic metrics logging
        
        Args:
            metrics: Dict of metrics returned from _compute_metrics
            inputs: Model inputs containing quality info
            loss: Final loss tensor
            
        Returns:
            Dict of console log info
        """
        info = {
            'epoch': self.state.epoch,
            'global_step': self.state.global_step,
            'mode': 'eval' if self.is_eval else 'train',
            'loss': loss.item(),
        }
        
        for key, value in metrics.items():
            if key in ['rewards_chosen', 'rewards_rejected']:
                if key == 'rewards_chosen':
                    info['rewards_chosen'] = value.detach().cpu().tolist()
                elif key == 'rewards_rejected':
                    info['rewards_rejected'] = value.detach().cpu().tolist()
            else:
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        info[key] = value.item()
                    elif value.dim() > 0:
                        info[key] = value.mean().item()
                    else:
                        info[key] = value.item()
                else:
                    info[key] = value
        
        # Add quality information if available
        if inputs and "win" in inputs and "lose" in inputs:
            if "quality" in inputs["win"] and "quality" in inputs["lose"]:
                info['quality_chosen'] = inputs["win"]["quality"].detach().cpu().tolist()
                info['quality_rejected'] = inputs["lose"]["quality"].detach().cpu().tolist()
            else:
                batch_size = len(info.get('rewards_chosen', [1]))
                info['quality_chosen'] = [-1] * batch_size
                info['quality_rejected'] = [-1] * batch_size
        
        if self.print_log:
            print(f"LOG: {json.dumps(info, ensure_ascii=False)}")
        return info
    
    def _store_eval_results(self, console_info: Dict[str, Any]) -> None:
        """
        Store evaluation results for plotting (only during eval mode)
        Dynamically handles all metrics in console_info
        
        Args:
            console_info: Console log info dict returned from _log_to_console
        """
        if not self.is_eval:
            return
            
        if self.args.local_rank != -1 and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            
            info_tensors = {}
            detailed_keys = ['rewards_chosen', 'rewards_rejected', 'quality_chosen', 'quality_rejected']
            
            for key, value in console_info.items():
                if key not in ['epoch', 'global_step', 'mode'] + detailed_keys:
                    if isinstance(value, (int, float)):
                        info_tensors[key] = torch.tensor(value, device=self.args.device)
            
            gathered_tensors = {}
            for key, tensor in info_tensors.items():
                gathered_list = [torch.zeros_like(tensor) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_list, tensor)
                gathered_tensors[key] = [t.item() for t in gathered_list]
            
            if self.args.local_rank == 0:
                for rank_idx in range(world_size):
                    rank_info = {
                        'epoch': console_info['epoch'],
                        'global_step': console_info['global_step'],
                        'mode': console_info['mode'],
                    }
                    
                    for key, values_list in gathered_tensors.items():
                        rank_info[key] = values_list[rank_idx]
                    
                    for detail_key in detailed_keys:
                        if detail_key in console_info:
                            rank_info[detail_key] = console_info[detail_key] if rank_idx == 0 else []
                    
                    self.eval_results.append(rank_info)
        else:
            self.eval_results.append(console_info)
    
    def _handle_logging(self, metrics_dict: Dict[str, Any], inputs: Dict[str, Any], 
                        loss: torch.Tensor) -> None:
        """
        Complete logging pipeline: compute metrics, log to TensorBoard, console, and store eval results
        
        Args:
            metrics_dict: Dictionary containing all raw metrics (tensors, values, etc.)
            inputs: Model inputs
            loss: Final loss tensor
        """
        metrics = self._compute_metrics(metrics_dict)
        self._log_to_tensorboard(metrics)
        console_info = self._log_to_console(metrics, inputs, loss)
        self._store_eval_results(console_info)
