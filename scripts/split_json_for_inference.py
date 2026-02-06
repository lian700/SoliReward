#!/usr/bin/env python3
"""
Split input JSON file into multiple chunks for distributed inference.
"""
import json
try:
    import orjson
except ImportError:
    orjson = None
import os
import sys
import argparse
import random
from pathlib import Path
from tqdm import tqdm


def split_json(input_file: str, output_dir: str, num_splits: int, max_samples: int = -1, seed: int = 42):
    """
    Split JSON file into multiple chunks.
    
    Args:
        input_file: Path to input JSON file
        output_dir: Directory to save split files
        num_splits: Number of splits to create
        max_samples: Maximum number of samples to process (-1 for all)
        seed: Random seed for shuffling
    """
    print(f"Loading input JSON: {input_file}")
    file_size = os.path.getsize(input_file)
    with open(input_file, 'r', encoding='utf-8') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Loading {os.path.basename(input_file)}") as pbar:
            content = ''
            chunk_size = 8192
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                content += chunk
                pbar.update(len(chunk.encode('utf-8')))
            data = orjson.loads(content) if orjson else json.loads(content)
    
    original_count = len(data)
    print(f"Total samples in input: {original_count}")
    
    # Shuffle data
    print(f"Shuffling data with seed={seed}...")
    random.seed(seed)
    random.shuffle(data)
    
    # Limit samples if max_samples is set
    if max_samples > 0:
        data = data[:max_samples]
        print(f"Limited to first {len(data)} samples (max_samples={max_samples})")
    
    total_samples = len(data)
    samples_per_split = (total_samples + num_splits - 1) // num_splits  # Ceiling division
    
    print(f"Splitting {total_samples} samples into {num_splits} chunks")
    print(f"Samples per chunk: ~{samples_per_split}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Split data
    for split_id in range(num_splits):
        start_idx = split_id * samples_per_split
        end_idx = min(start_idx + samples_per_split, total_samples)
        
        if start_idx >= total_samples:
            # No more data for this split
            split_data = []
        else:
            split_data = data[start_idx:end_idx]
        
        output_file = os.path.join(output_dir, f"input_gpu_{split_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        print(f"Split {split_id}: {len(split_data)} samples -> {output_file}")
    
    print("\nJSON splitting completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Split JSON file for distributed inference")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for splits")
    parser.add_argument("--num_splits", type=int, required=True, help="Number of splits to create")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum samples to process (-1 for all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    
    args = parser.parse_args()
    
    try:
        split_json(args.input_file, args.output_dir, args.num_splits, args.max_samples, args.seed)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
