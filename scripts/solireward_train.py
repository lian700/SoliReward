#!/usr/bin/env python3
"""
Training script for SoliReward - Video Reward Model Training

This script is the main entry point for training reward models using the SoliReward framework.
It supports multiple model architectures including InternVL3, InternVL3-5, Qwen2.5-VL and Qwen2-VL.

Usage:
    python train.py --model_name_or_path <model_path> --train_data_path <data_path> --output_dir <output_dir>
    
For more options, run:
    python train.py --help
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for importing solireward package
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoProcessor
from dataclasses import asdict

# Import from solireward package
from solireward import (
    parse_args,
    save_config_to_json,
    create_reward_model,
    create_dataset_from_json,
    BTWithLMHeadRewardTrainer,
)
from solireward.models.internvl3.data import InternVLDataCollator
from solireward.models.qwenvl2_5.data import Qwen2_5VLDataCollator


def main():
    # Get rank information for distributed training
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            rank_prefix = f"[Rank {rank}/{world_size}] "
        else:
            rank_prefix = ""
    except:
        rank_prefix = ""
    
    # Parse arguments
    model_args, data_args, trainer_args, internvl_args, qwenvl_args, training_args = parse_args()
    for k, v in asdict(model_args).items():
        setattr(internvl_args, k, v)
        setattr(qwenvl_args, k, v)
    
    # Save configuration to JSON for reproducibility
    save_config_to_json(model_args, data_args, trainer_args, internvl_args, qwenvl_args, training_args, training_args.output_dir)
    
    # Load model and tokenizer
    print(f"{rank_prefix}Loading model and tokenizer...")
    model = create_reward_model(
        model_args.model_name_or_path, 
        model_args=model_args, 
        training_args=training_args, 
        other_model_specific_args={
            'internvl_args': internvl_args,
            'qwenvl_args': qwenvl_args,
        }
    )
    
    if model_args.model_type in ['Qwen2.5-VL', 'Qwen2-VL']:
        tokenizer = AutoProcessor.from_pretrained(
            model_args.tokenizer_name_or_path, 
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
    elif model_args.model_type in ['InternVL3', 'InternVL3-5']:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path, trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported model type: {model_args.model_type}. Supported types: 'InternVL3', 'InternVL3-5', 'Qwen2.5-VL', 'Qwen2-VL'.")

    
    # Create data collator
    if model_args.model_type in ['InternVL3', 'InternVL3-5']:
        data_collator = InternVLDataCollator(
            tokenizer=tokenizer,
            input_size=internvl_args.input_size,
            max_num=internvl_args.max_num,
            num_segments=internvl_args.num_segments,
            num_image_token=model.num_image_token,
            center_crop_video=internvl_args.center_crop_video,
        )
    elif model_args.model_type in ['Qwen2.5-VL', 'Qwen2-VL']:
        data_collator = Qwen2_5VLDataCollator(
            processor=tokenizer,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_args.model_type}. Supported types: 'InternVL3', 'InternVL3-5', 'Qwen2.5-VL', 'Qwen2-VL'.")


    # Load and prepare dataset
    print(f"{rank_prefix}Loading training dataset...")
    train_dataset = create_dataset_from_json(data_args.train_data_path)
    
    eval_dataset = None
    if data_args.eval_data_path and training_args.do_eval:
        print(f"{rank_prefix}Loading evaluation dataset...")
        eval_dataset = create_dataset_from_json(data_args.eval_data_path)
    
    # Update training arguments for multimodal data
    training_args.remove_unused_columns = False  # Important for multimodal data
    
    # Create trainer
    trainer = BTWithLMHeadRewardTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        use_global_metrics=trainer_args.use_global_metrics,
        enable_btt_loss=trainer_args.enable_btt_loss,
        bt_loss_coeff=trainer_args.bt_loss_coeff,
        btt_loss_coeff=trainer_args.btt_loss_coeff,
        reward_margin=trainer_args.reward_margin,
        bce_loss_coeff=trainer_args.bce_loss_coeff,
        bt_label_smoothing=trainer_args.bt_label_smoothing,
        bce_label_smoothing=trainer_args.bce_label_smoothing,
    )
    
    # Start training
    print(f"{rank_prefix}Starting training...")
    trainer.train()
    
    # Save the final model
    print(f"{rank_prefix}Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print(f"{rank_prefix}Training completed! Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
