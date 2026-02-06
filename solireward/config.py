#!/usr/bin/env python3
"""
Configuration classes and argument parsing for InternVL Reward Model training
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or train from scratch.
    """
    
    model_name_or_path: str = field(
        default="OpenGVLab/InternVL3-1B",
        metadata={"help": "Path to the pre-trained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(
        default="InternVL3",
        metadata={"help": "Type of the model architecture (e.g., 'InternVL3', 'InternVL3-5', 'Qwen2.5-VL', 'Qwen2-VL')"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the tokenizer (defaults to model_name_or_path)"}
    )
    attn_implementation: Optional[str] = field(
        default='flash_attention_2',
        metadata={
            "help": "Attention implementation to use. Options: 'flash_attention_2', 'sdpa', 'eager'. "
                   "If None, will use model default. 'flash_attention_2' requires flash-attn>=2.0 to be installed."
        }
    )

    reduce_sequence: str = field(
        default='maxpool',
        metadata={"help": "Method to reduce sequence dimension: options include 'maxpool', 'meanpool', 'attention', hierarchical variants, or 'last_token_hidden_state'."}
    )
    reward_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for reward head"}
    )

    hierarchical_query_attn_layers: list[int] = field(
        default_factory = lambda: [6, 12, 18, 24],
        metadata={"help": "List of layer indices to apply hierarchical query attention"}
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ModelArguments to a dictionary, handling lists appropriately.
        """
        return asdict(self)

@dataclass
class InternVLArguments:
    """
    Arguments specific to InternVL model data collator
    """
    
    input_size: int = field(
        default=448,
        metadata={"help": "Input image size for vision processing"}
    )
    max_num: int = field(
        default=12,
        metadata={"help": "Maximum number of image tiles"}
    )
    num_segments: int = field(
        default=8,
        metadata={"help": "Number of video segments to process"}
    )

    center_crop_video: bool = field(
        default=False,
        metadata={"help": "Whether to center crop video frames before tiling"}
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert InternVLArguments to a dictionary, handling lists appropriately.
        """
        return asdict(self)


@dataclass
class QwenVLArguments:
    """
    Arguments specific to QwenVL model AutoProcessor
    """
    
    image_factor: int = field(
        default=28,
        metadata={"help": "IMAGE_FACTOR parameter for QwenVL processor"}
    )
    min_pixels: int = field(
        default=4 * 28 * 28,  # 4 * 28 * 28
        metadata={"help": "MIN_PIXELS parameter for QwenVL processor"}
    )
    max_pixels: int = field(
        default=16384 * 28 * 28,  # 16384 * 28 * 28
        metadata={"help": "MAX_PIXELS parameter for QwenVL processor"}
    )
    max_ratio: int = field(
        default=200,
        metadata={"help": "MAX_RATIO parameter for QwenVL processor"}
    )
    video_min_pixels: int = field(
        default=128 * 28 * 28,  # 128 * 28 * 28
        metadata={"help": "VIDEO_MIN_PIXELS parameter for QwenVL processor"}
    )
    video_max_pixels: int = field(
        default=768 * 28 * 28,  # 768 * 28 * 28
        metadata={"help": "VIDEO_MAX_PIXELS parameter for QwenVL processor"}
    )
    video_total_pixels: int = field(
        default=24576 * 28 * 28,  # 24576 * 28 * 28
        metadata={"help": "VIDEO_TOTAL_PIXELS parameter for QwenVL processor"}
    )
    frame_factor: int = field(
        default=2,
        metadata={"help": "FRAME_FACTOR parameter for QwenVL processor"}
    )
    fps: float = field(
        default=2.0,
        metadata={"help": "FPS parameter for QwenVL processor"}
    )
    fps_min_frames: int = field(
        default=4,
        metadata={"help": "FPS_MIN_FRAMES parameter for QwenVL processor"}
    )
    fps_max_frames: int = field(
        default=768,
        metadata={"help": "FPS_MAX_FRAMES parameter for QwenVL processor"}
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert QwenVLArguments to a dictionary, handling lists appropriately.
        """
        return asdict(self)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    train_data_path: List[str] = field(
        metadata={"help": "Path(s) to the training data JSON file(s). Can specify multiple files separated by space.", "nargs": "+"}
    )
    eval_data_path: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Path(s) to the evaluation data JSON file(s). Can specify multiple files separated by space.", "nargs": "+"}
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert DataTrainingArguments to a dictionary, handling lists appropriately.
        """
        return asdict(self)


@dataclass
class TrainerArguments:
    """
    Arguments pertaining to the custom trainer behavior
    """
    
    use_global_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to gather metrics from all ranks using all_reduce for more accurate distributed statistics. Default: False (only uses rank 0 metrics)"}
    )

    enable_btt_loss: int = field(
        default=0,
        metadata={"help": "Whether to enable BTT loss component. Default: 0 (disabled)"}
    )

    bt_loss_coeff: float = field(
        default=1.0,
        metadata={"help": "Coefficient for BT loss component. Default: 1.0"}
    )

    btt_loss_coeff: float = field(
        default=1.0,
        metadata={"help": "Coefficient for BTT loss component. Default: 1.0"}
    )

    reward_margin: float = field(
        default=0.0,
        metadata={"help": "Margin value for reward ranking loss"}
    )

    bce_loss_coeff: float = field(
        default=0.0,
        metadata={"help": "Coefficient for the BCE loss component"}
    )

    bt_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "Epsilon for Label Smoothing applied ONLY to BT loss (0.0 disables). Recommended small values like 0.05 or 0.1."}
    )

    bce_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "Epsilon for Label Smoothing applied to BCE loss (0.0 disables). Moves labels away from hard 0/1 towards 0.5. Recommended small values like 0.05 or 0.1."}
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert TrainerArguments to a dictionary, handling lists appropriately.
        """
        return asdict(self)


def parse_args():
    """
    Parse command line arguments using HfArgumentParser
    """
    parser = HfArgumentParser([ModelArguments, DataTrainingArguments, TrainerArguments, InternVLArguments, QwenVLArguments, TrainingArguments])  # type: ignore[arg-type]
    
    model_args, data_args, trainer_args, internvl_args, qwenvl_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set default tokenizer path
    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path
        
    # Set default logging directory
    if training_args.logging_dir is None:
        training_args.logging_dir = os.path.join(training_args.output_dir, "logs")
        
    return model_args, data_args, trainer_args, internvl_args, qwenvl_args, training_args


def create_config():
    """
    Create configuration objects from parsed arguments
    """
    return parse_args()


def save_config_to_json(model_args, data_args, trainer_args, internvl_args, qwenvl_args, training_args, output_dir, filename="training_config.json"):
    """
    Save all configuration arguments to a JSON file
    
    Args:
        model_args: ModelArguments instance
        data_args: DataTrainingArguments instance
        trainer_args: TrainerArguments instance
        internvl_args: InternVLArguments instance
        qwenvl_args: QwenVLArguments instance
        training_args: TrainingArguments instance
        output_dir: Directory to save the JSON file
        filename: Name of the JSON file (default: "training_config.json")
    
    Returns:
        str: Path to the saved JSON file
    """
    # Create a combined dictionary of all arguments
    combined_config = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "trainer_args": asdict(trainer_args),
        "internvl_args": asdict(internvl_args),
        "qwenvl_args": asdict(qwenvl_args),
        "training_args": training_args.to_dict(),  # TrainingArguments has a to_dict() method
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "config_version": "1.0",
            "created_by": "config.py"
        }
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to JSON file
    config_file_path = os.path.join(output_dir, filename)
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(combined_config, f, indent=2, ensure_ascii=False)
    
    print(f"Configuration saved to: {config_file_path}")
    return config_file_path


def load_config_from_json(json_path):
    """
    Load configuration from a JSON file
    
    Args:
        json_path: Path to the JSON configuration file
        
    Returns:
        dict: Dictionary containing all configuration sections
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from: {json_path}")
    return config