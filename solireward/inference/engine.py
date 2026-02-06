"""
Reward Model Inference Engine

This module provides the main inference engine class for running
reward model inference on video/image data.
"""

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models import load_reward_model_and_collator
from ..models.internvl3.data import InternVLDataCollator
from ..models.qwenvl2_5.data import Qwen2_5VLDataCollator

from .arguments import InferenceArguments, get_default_prompts
from .dataset import VideoInferenceDataset
from .utils import (
    load_data_from_json,
    prepare_messages_from_data,
    print_statistics,
)


class RewardModelInference:
    """
    Reward model inference engine for evaluating video/image quality.
    
    This class provides a high-level interface for running inference with
    trained reward models. It handles model loading, data processing, and
    batch inference with support for multiple model architectures.
    
    Attributes:
        args: Inference configuration arguments
        device: Device to run inference on
        dtype: Data type for model computations
        model: Loaded reward model
        data_collator: Data collator for the model
        model_type: Type of the loaded model
    
    Example:
        >>> from solireward.inference import RewardModelInference, InferenceArguments
        >>> args = InferenceArguments(model_name_or_path="/path/to/model")
        >>> engine = RewardModelInference(args)
        >>> scores = engine.predict_reward_batch(batch_messages)
    """
    
    def __init__(self, inference_args: InferenceArguments) -> None:
        """
        Initialize the reward model inference engine.
        
        Args:
            inference_args: Inference configuration arguments
        """
        self.args = inference_args
        self.device = inference_args.device
        self.dtype = inference_args.dtype

        self.model = None
        self.data_collator = None
        self.model_type: str = "unknown"

        self._load_model_and_collator()
        self._prepare_model()
        self._configure_prompts()
    
    def _load_model_and_collator(self) -> None:
        """Load the reward model and data collator from checkpoint."""
        print(f"Loading reward model and collator from: {self.args.model_name_or_path}")
        model, data_collator, model_type = load_reward_model_and_collator(
            self.args.model_name_or_path
        )

        self.model = model
        self.data_collator = data_collator
        self.model_type = model_type

        if self.data_collator is None:
            raise RuntimeError("Failed to initialize data collator from checkpoint")

        print(f"Loaded model type: {self.model_type}")

    def _prepare_model(self) -> None:
        """Prepare model for inference (device placement, dtype, eval mode)."""
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        # Move model to device
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            self.model = self.model.cuda()
        elif not self.device.startswith("cuda"):
            device_obj = torch.device(self.device)
            self.model = self.model.to(device_obj)
        
        # Set model dtype
        if self.dtype == "bf16":
            self.model = self.model.bfloat16()
        elif self.dtype == "fp16":
            self.model = self.model.half()
        elif self.dtype == "fp32":
            self.model = self.model.float()
        
        self.model.eval()
        print(f"Model ready on device: {self.device} with dtype: {self.dtype}")
    
    def _configure_prompts(self) -> None:
        """Configure system and user prompts based on task type."""
        task_type = self.args.reward_model_task_type
        
        # Validate task type and get defaults
        default_prompts = get_default_prompts(task_type)
        
        # Use default prompts if not specified
        if self.args.system_prompt is None:
            self.args.system_prompt = default_prompts["system_prompt"]
        if self.args.user_prompt is None:
            self.args.user_prompt = default_prompts["user_prompt"]
        
        print(f"Task type: {task_type}")
        print(f"System prompt configured")
        print(f"User prompt template: {self.args.user_prompt}")
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for DataLoader.
        
        Processes batch data using the appropriate data collator.
        
        Args:
            batch: List of items from VideoInferenceDataset
            
        Returns:
            Dict containing processed batch data ready for model input
        """
        if self.data_collator is None:
            raise ValueError("Data collator must be initialized")
        
        # Use the data collator to process the batch
        batch_data = self.data_collator(batch)
        
        # Extract model inputs based on collator type
        if isinstance(self.data_collator, InternVLDataCollator):
            model_input = {
                k: v for k, v in batch_data['win'].items() 
                if k not in ['quality', 'num_patches_list']
            }
        elif isinstance(self.data_collator, Qwen2_5VLDataCollator):
            batch_size = batch_data['batch_size']
            inputs_concat = batch_data['inputs_concat']
            model_input = {}
            
            for key, value in inputs_concat.items():
                if isinstance(value, torch.Tensor):
                    # For tensors, check if it's a 2D+ tensor with batch dimension
                    if value.dim() > 0 and value.shape[0] == 2 * batch_size:
                        model_input[key] = value[:batch_size]
                    elif key == 'pixel_values_videos' and 'video_grid_thw' in inputs_concat:
                        # Special handling for pixel_values_videos: slice based on video_grid_thw
                        video_grid_thw = inputs_concat['video_grid_thw']
                        if video_grid_thw.shape[0] == 2 * batch_size:
                            # Calculate number of visual tokens for first batch_size videos
                            num_tokens_per_video = video_grid_thw[:, 0] * video_grid_thw[:, 1] * video_grid_thw[:, 2]
                            total_tokens_first_half = num_tokens_per_video[:batch_size].sum().item()
                            model_input[key] = value[:int(total_tokens_first_half)]
                        else:
                            model_input[key] = value
                    else:
                        model_input[key] = value
                elif isinstance(value, (list, tuple)):
                    # For lists/tuples, check if length matches concatenated size
                    if len(value) == 2 * batch_size:
                        model_input[key] = value[:batch_size]
                    else:
                        model_input[key] = value
                else:
                    model_input[key] = value
        else:
            raise ValueError(
                f"Unsupported data collator type: {type(self.data_collator).__name__}"
            )
        
        return model_input
    
    def _prepare_batch_input(
        self, 
        batch_messages: Optional[List[List[Dict[str, Any]]]] = None,
        model_input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare model input from batch messages or use pre-processed input.
        
        Args:
            batch_messages: List of message lists in OpenAI format
            model_input: Pre-processed model input dict (from DataLoader)
            
        Returns:
            Model input dictionary ready for forward pass
        """
        if model_input is not None:
            return model_input
        
        if batch_messages is None:
            raise ValueError("Either batch_messages or model_input must be provided")
        
        if self.data_collator is None:
            raise ValueError("Data collator must be initialized")
        
        # Create fake pair batch for data collator
        fake_pair_batch = [{
            'win': messages,
            'lose': messages,
            'meta': {'win': {'quality': 1.0}, 'lose': {'quality': 0.0}}
        } for messages in batch_messages]
        
        batch_data = self.data_collator(fake_pair_batch)
        
        # Extract model inputs based on collator type
        if isinstance(self.data_collator, InternVLDataCollator):
            return {
                k: v for k, v in batch_data['win'].items() 
                if k not in ['quality', 'num_patches_list']
            }
        elif isinstance(self.data_collator, Qwen2_5VLDataCollator):
            result = {}
            batch_size = batch_data['batch_size']
            inputs_concat = batch_data['inputs_concat']
            
            for key, value in inputs_concat.items():
                if isinstance(value, torch.Tensor):
                    # For tensors, check if it's a 2D+ tensor with batch dimension
                    if value.dim() > 0 and value.shape[0] == 2 * batch_size:
                        result[key] = value[:batch_size]
                    elif key == 'pixel_values_videos' and 'video_grid_thw' in inputs_concat:
                        # Special handling for pixel_values_videos: slice based on video_grid_thw
                        video_grid_thw = inputs_concat['video_grid_thw']
                        if video_grid_thw.shape[0] == 2 * batch_size:
                            # Calculate number of visual tokens for first batch_size videos
                            num_tokens_per_video = video_grid_thw[:, 0] * video_grid_thw[:, 1] * video_grid_thw[:, 2]
                            total_tokens_first_half = num_tokens_per_video[:batch_size].sum().item()
                            result[key] = value[:int(total_tokens_first_half)]
                        else:
                            result[key] = value
                    else:
                        result[key] = value
                elif isinstance(value, (list, tuple)):
                    # For lists/tuples, check if length matches concatenated size
                    if len(value) == 2 * batch_size:
                        result[key] = value[:batch_size]
                    else:
                        result[key] = value
                else:
                    result[key] = value
            return result
        else:
            raise ValueError(
                f"Unsupported data collator type: {type(self.data_collator).__name__}"
            )
    
    def _move_to_device(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch tensors to the target device and dtype.
        
        Args:
            model_input: Dictionary of model inputs
            
        Returns:
            Input dictionary with tensors on target device
        """
        target_dtype = None
        if self.dtype == "bf16":
            target_dtype = torch.bfloat16
        elif self.dtype == "fp16":
            target_dtype = torch.float16
        elif self.dtype == "fp32":
            target_dtype = torch.float32
            
        for key, value in model_input.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point and target_dtype is not None:
                    value = value.to(target_dtype)
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    model_input[key] = value.cuda()
                else:
                    device_obj = torch.device(self.device)
                    model_input[key] = value.to(device_obj)
        
        return model_input
    
    def _extract_reward_scores(self, outputs: Any) -> List[float]:
        """
        Extract reward scores from model outputs.
        
        Args:
            outputs: Model output (dict or object with attributes)
            
        Returns:
            List of reward scores
            
        Raises:
            ValueError: If no reward field found in outputs
        """
        reward_scores = None
        
        if isinstance(outputs, dict):
            if 'reward_scores' in outputs:
                reward_scores = outputs['reward_scores'].squeeze().cpu().tolist()
            elif 'reward' in outputs:
                reward_scores = outputs['reward'].squeeze().cpu().tolist()
            elif 'logits' in outputs:
                reward_scores = outputs['logits'].squeeze().cpu().tolist()
        else:
            if hasattr(outputs, 'reward_scores'):
                reward_scores = outputs.reward_scores.squeeze().cpu().tolist()
            elif hasattr(outputs, 'reward'):
                reward_scores = outputs.reward.squeeze().cpu().tolist()
            elif hasattr(outputs, 'logits'):
                reward_scores = outputs.logits.squeeze().cpu().tolist()
        
        if reward_scores is None:
            available = (
                list(outputs.keys()) if isinstance(outputs, dict) 
                else [a for a in dir(outputs) if not a.startswith('_')]
            )
            raise ValueError(
                f"Model output does not contain 'reward_scores', 'reward' or "
                f"'logits' field. Available: {available}"
            )
        
        if isinstance(reward_scores, float):
            reward_scores = [reward_scores]
        
        return [float(score) for score in reward_scores]
    
    @torch.no_grad()
    def predict_reward_batch(
        self, 
        batch_messages: Optional[List[List[Dict[str, Any]]]] = None,
        model_input: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Predict reward scores for a batch of conversations.
        
        Args:
            batch_messages: List of message lists in OpenAI format (legacy method)
            model_input: Pre-processed model input dict (from DataLoader collate_fn)
            
        Returns:
            List of reward scores for each sample in the batch
        """
        if self.model is None:
            raise ValueError("Model must be initialized")
        
        # Prepare input
        model_input = self._prepare_batch_input(batch_messages, model_input)
        model_input.setdefault('return_dict', True)
        
        # Move to device
        model_input = self._move_to_device(model_input)
        
        # Get model output with autocast
        autocast_dtype = (
            torch.bfloat16 if self.dtype == 'bf16'
            else torch.float16 if self.dtype == 'fp16'
            else torch.float32
        )
        
        with torch.autocast(
            device_type='cuda' if self.device.startswith('cuda') else 'cpu',
            dtype=autocast_dtype
        ):
            outputs = self.model(**model_input)
        
        return self._extract_reward_scores(outputs)
    
    def run_inference(
        self,
        messages_list: List[List[Dict[str, Any]]],
        show_progress: bool = True
    ) -> List[float]:
        """
        Run inference on a list of messages.
        
        This method handles batching and optionally uses DataLoader for
        parallel data loading.
        
        Args:
            messages_list: List of message lists to process
            show_progress: Whether to show progress bar
            
        Returns:
            List of reward scores for all samples
        """
        all_scores = []
        batch_size = self.args.batch_size
        
        if self.args.use_dataloader:
            print(f"Using DataLoader with {self.args.num_workers} workers")
            dataset = VideoInferenceDataset(messages_list)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.args.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=self.device.startswith("cuda"),
                shuffle=False,
                drop_last=False,
            )
            
            iterator = tqdm(dataloader, desc="Processing batches") if show_progress else dataloader
            for model_input in iterator:
                scores = self.predict_reward_batch(model_input=model_input)
                all_scores.extend(scores)
        else:
            print("Using legacy processing without DataLoader")
            batches = [
                messages_list[i:i + batch_size] 
                for i in range(0, len(messages_list), batch_size)
            ]
            iterator = tqdm(batches, desc="Processing batches") if show_progress else batches
            for batch in iterator:
                scores = self.predict_reward_batch(batch_messages=batch)
                all_scores.extend(scores)
        
        return all_scores
    
    def infer_from_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        max_samples: int = -1,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run inference on data from a JSON file.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to save results (optional)
            max_samples: Maximum samples to process (-1 for all)
            show_progress: Whether to show progress bar
            
        Returns:
            List of result dictionaries with scores
        """
        # Load data
        print(f"Loading test data from: {input_file}")
        test_data = load_data_from_json(input_file)
        original_count = len(test_data)
        
        if max_samples >= 0:
            test_data = test_data[:max_samples]
            if len(test_data) != original_count:
                print(f"Limiting to first {len(test_data)} samples")
        
        print(f"Loaded {len(test_data)} test samples")
        
        # Prepare messages
        messages_list, valid_data = prepare_messages_from_data(
            test_data,
            self.args.reward_model_task_type,
            self.args.system_prompt,
            self.args.user_prompt
        )
        
        if not messages_list:
            print("No valid messages found in test data")
            return []
        
        # Run inference
        print("Processing batch data...")
        all_scores = self.run_inference(messages_list, show_progress)
        print(f"Processed {len(all_scores)} samples successfully")
        
        # Prepare results
        results = []
        for item, score in zip(valid_data, all_scores):
            result = dict(item)
            result['score'] = score
            results.append(result)
        
        # Save results
        if output_file:
            from pathlib import Path
            from .utils import save_results_to_json
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Saving results to: {output_file}")
            save_results_to_json(results, output_file)
            print(f"Results saved successfully!")
        
        # Print statistics
        if all_scores:
            print_statistics(all_scores)
        
        return results
