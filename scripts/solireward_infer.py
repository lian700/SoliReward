#!/usr/bin/env python3
"""
Inference script for SoliReward - Video Reward Model Inference

This script is the main entry point for running inference with trained reward models
using the SoliReward framework. It supports multiple model architectures including 
InternVL3, InternVL3-5, Qwen2.5-VL and Qwen2-VL.

Usage:
    python solireward_infer.py --model_name_or_path <model_path> --input_file <input_json> --output_file <output_json>
    
For more options, run:
    python solireward_infer.py --help
"""

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# Add parent directory to path for importing solireward package
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import HfArgumentParser

# Import from solireward inference module
from solireward.inference import (
    InferenceArguments,
    RewardModelInference,
)


def main():
    """Main function for command line interface."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Parse arguments
    parser = HfArgumentParser(InferenceArguments)
    inference_args = parser.parse_args_into_dataclasses()[0]

    # Save inference arguments before starting inference
    if inference_args.output_file:
        output_file = Path(inference_args.output_file)
        args_output_file = output_file.parent / f"args_{output_file.stem}.json"
        print(f"Saving inference arguments to: {args_output_file}")
        args_dict = asdict(inference_args)
        with open(args_output_file, 'w', encoding='utf-8') as f:
            json.dump(args_dict, f, ensure_ascii=False, indent=2)
        print(f"Inference arguments saved successfully!")

    # Initialize inference engine
    print("Initializing reward model inference engine...")
    inference_engine = RewardModelInference(inference_args)
    print("Inference engine initialized successfully!")
    
    # Run inference if input file is provided
    if inference_args.input_file:
        results = inference_engine.infer_from_file(
            input_file=inference_args.input_file,
            output_file=inference_args.output_file,
            max_samples=inference_args.max_samples,
            show_progress=True
        )
    else:
        print("No input file specified. Use --input_file to specify input data")


if __name__ == "__main__":
    main()
