#!/usr/bin/env python3
"""
Reward Model Creation and Loading Utilities

This module provides factory functions for creating and loading reward models:
- create_reward_model: Create a new reward model from pretrained weights
- load_reward_model_and_collator: Load a trained reward model with its data collator
"""

from __future__ import annotations

import os
import warnings
import torch
import torch.nn as nn
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Type, TypeVar

T = TypeVar('T')

from easydict import EasyDict
from transformers import AutoProcessor, AutoTokenizer, AutoVideoProcessor
from transformers.training_args import TrainingArguments

from ..config import (
    DataTrainingArguments,
    InternVLArguments,
    ModelArguments,
    QwenVLArguments,
    TrainerArguments,
    load_config_from_json,
)
from .internvl3.reward_model import InternVL3RewardModel
from .internvl3.data import InternVLDataCollator
from .internvl3_5.reward_model import InternVL35RewardModel
from .qwenvl2_5.reward_model import Qwen2VLRewardModel, Qwen25VLRewardModel
from .qwenvl2_5.data import Qwen2_5VLDataCollator


_CONFIG_FILENAME = "training_config.json"


def _filter_kwargs_for_dataclass(dataclass_type: Type[T], kwargs: Dict[str, Any], class_name: str = "") -> Dict[str, Any]:
    """
    Filter kwargs to only include fields defined in the dataclass.
    Logs warnings for any unknown arguments that are filtered out.
    
    Args:
        dataclass_type: The dataclass type to filter kwargs for
        kwargs: The dictionary of keyword arguments to filter
        class_name: Optional class name for warning messages (defaults to dataclass name)
        
    Returns:
        Filtered dictionary containing only valid dataclass fields
    """
    valid_fields = {f.name for f in fields(dataclass_type)}
    filtered_kwargs = {}
    unknown_kwargs = []
    
    for key, value in kwargs.items():
        if key in valid_fields:
            filtered_kwargs[key] = value
        else:
            unknown_kwargs.append(key)
    
    if unknown_kwargs:
        cls_name = class_name or dataclass_type.__name__
        warnings.warn(
            f"[{cls_name}] Ignoring unknown configuration arguments: {unknown_kwargs}. "
            f"These arguments may be from a newer version of the model or are deprecated.",
            UserWarning
        )
    
    return filtered_kwargs


def create_reward_model(
    model_path: str, 
    tokenizer_path: Optional[str] = None, 
    model_args: Optional[ModelArguments] = None,
    training_args: Optional[TrainingArguments] = None,
    other_model_specific_args: Optional[Dict[str, Any]] = None
) -> Union[InternVL3RewardModel, InternVL35RewardModel, Qwen2VLRewardModel, Qwen25VLRewardModel]:
    """
    Factory function to create reward models with tokenizer setup.
    
    Args:
        model_path: Path to the pre-trained model
        tokenizer_path: Path to tokenizer (defaults to model_path)
        model_args: Model arguments including attention implementation and model_type
        training_args: Training arguments including dtype settings (bf16, fp16)
        other_model_specific_args: Dict containing model-specific arguments (internvl_args, qwenvl_args)
        
    Returns:
        Configured reward model (InternVL3, InternVL3-5, Qwen2.5-VL, or Qwen2-VL)
    """
    extra_model_args = {}
    
    # Handle attention implementation from model_args
    if model_args and model_args.attn_implementation:
        extra_model_args['attn_implementation'] = model_args.attn_implementation
    
    # Handle dtype from training_args (priority: bf16 > fp16 > default)
    if training_args:
        if training_args.bf16:
            extra_model_args['torch_dtype'] = torch.bfloat16
        elif training_args.fp16:
            extra_model_args['torch_dtype'] = torch.float16

    # Get model-specific arguments
    model_specific_config = None
    if other_model_specific_args:
        if model_args and model_args.model_type in ['InternVL3', 'InternVL3-5']:
            model_specific_config = other_model_specific_args.get('internvl_args')
        elif model_args and model_args.model_type in ['Qwen2.5-VL', 'Qwen2-VL']:
            model_specific_config = other_model_specific_args.get('qwenvl_args')

    # Select reward model class
    if model_args and model_args.model_type == 'InternVL3':
        RewardModel = InternVL3RewardModel
    elif model_args and model_args.model_type == 'InternVL3-5':
        RewardModel = InternVL35RewardModel
    elif model_args and model_args.model_type == 'Qwen2.5-VL':
        RewardModel = Qwen25VLRewardModel
    elif model_args and model_args.model_type == 'Qwen2-VL':
        RewardModel = Qwen2VLRewardModel
    else:
        raise ValueError(f"Unsupported model type: {model_args.model_type if model_args else 'None'}. "
                        f"Supported types: 'InternVL3', 'InternVL3-5', 'Qwen2.5-VL', 'Qwen2-VL'.")

    model = RewardModel.from_pretrained(
        model_path, 
        model_specific_config=model_specific_config,
        trust_remote_code=True, 
        **extra_model_args
    )
    
    # Load and set tokenizer
    tokenizer_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model.set_tokenizer(tokenizer)

    return model


def _resolve_config_path(model_path: str) -> str:
    """
    Locate the training_config.json file relative to the model directory.
    
    The config file is expected to be in the checkpoint directory itself.
    For backward compatibility, it also checks the parent directory if not found.
    """
    model_path = Path(model_path)
    config_path = model_path / _CONFIG_FILENAME
    if not config_path.exists():
        # Backward compatibility: check parent directory
        config_path = model_path.parent / _CONFIG_FILENAME
    assert config_path.exists(), (
        f"Could not find {_CONFIG_FILENAME} in {model_path} or its parent directory. "
        f"Make sure the checkpoint was saved with the updated trainer that copies config to each checkpoint."
    )
    return str(config_path)


def load_reward_model_and_collator(model_path: str) -> Tuple[Any, Any, str]:
    """
    Load a trained reward model and its data collator from a checkpoint directory.

    Args:
        model_path: Path to the trained reward model checkpoint directory.

    Returns:
        A tuple of (model, data_collator, model_type) ready for inference or further training.
    """
    config_path = _resolve_config_path(model_path)
    config = load_config_from_json(config_path)
    config["model_args"]["model_name_or_path"] = model_path
    config["model_args"]["tokenizer_name_or_path"] = model_path

    # Filter unknown kwargs for each Arguments class to handle version differences gracefully
    model_args = ModelArguments(**_filter_kwargs_for_dataclass(ModelArguments, config["model_args"]))
    _data_args = DataTrainingArguments(**_filter_kwargs_for_dataclass(DataTrainingArguments, config["data_args"]))
    _trainer_args = TrainerArguments(**_filter_kwargs_for_dataclass(TrainerArguments, config["trainer_args"]))
    internvl_args = InternVLArguments(**_filter_kwargs_for_dataclass(InternVLArguments, config["internvl_args"]))
    qwenvl_args = QwenVLArguments(**_filter_kwargs_for_dataclass(QwenVLArguments, config["qwenvl_args"]))
    training_args = EasyDict({"bf16": config["training_args"].get("bf16", True)})

    # Propagate model settings to modality-specific argument containers
    for key, value in asdict(model_args).items():
        setattr(internvl_args, key, value)
        setattr(qwenvl_args, key, value)

    # Instantiate the model using the stored configuration
    model = create_reward_model(
        model_path,
        tokenizer_path=model_path,
        model_args=model_args,
        training_args=training_args,
        other_model_specific_args={
            "internvl_args": internvl_args,
            "qwenvl_args": qwenvl_args,
        },
    )

    # Build the matching data collator
    if model_args.model_type in ["InternVL3", "InternVL3-5"]:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        num_image_token = getattr(model, "num_image_token", None)
        assert num_image_token and isinstance(num_image_token, int), \
            f"num_image_token should be int but got {type(num_image_token)}"
        data_collator = InternVLDataCollator(
            tokenizer=tokenizer,
            input_size=internvl_args.input_size,
            max_num=internvl_args.max_num,
            num_segments=internvl_args.num_segments,
            num_image_token=num_image_token,
            center_crop_video=internvl_args.center_crop_video,
        )
    elif model_args.model_type in ["Qwen2.5-VL", "Qwen2-VL"]:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
            image_factor=qwenvl_args.image_factor,
            min_pixels=qwenvl_args.min_pixels,
            max_pixels=qwenvl_args.max_pixels,
            max_ratio=qwenvl_args.max_ratio,
            video_min_pixels=qwenvl_args.video_min_pixels,
            video_max_pixels=qwenvl_args.video_max_pixels,
            video_total_pixels=qwenvl_args.video_total_pixels,
            frame_factor=qwenvl_args.frame_factor,
            fps=qwenvl_args.fps,
            fps_min_frames=qwenvl_args.fps_min_frames,
            fps_max_frames=qwenvl_args.fps_max_frames,
        )
        data_collator = Qwen2_5VLDataCollator(processor=processor)
    else:
        raise ValueError(
            f"Unsupported model type: {model_args.model_type}. Supported types: "
            "'InternVL3', 'InternVL3-5', 'Qwen2.5-VL', 'Qwen2-VL'."
        )

    return model, data_collator, model_args.model_type
