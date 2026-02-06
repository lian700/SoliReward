"""
Inference Utility Functions

This module provides utility functions for reward model inference,
including data loading and message formatting.
"""

import json
try:
    import orjson
except ImportError:
    orjson = None
import os
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm


def load_data_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load test data from JSON file.
    
    Args:
        json_path: Path to the JSON file containing test data
        
    Returns:
        List of dictionaries containing test data
    """
    file_size = os.path.getsize(json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Loading {os.path.basename(json_path)}") as pbar:
            content = ''
            chunk_size = 8192
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                content += chunk
                pbar.update(len(chunk.encode('utf-8')))
            data = orjson.loads(content) if orjson else json.loads(content)
    return data


def save_results_to_json(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save inference results to JSON file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the results
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def extract_video_path(item: Dict[str, Any]) -> Optional[str]:
    """
    Extract video path from a data item.
    
    Supports multiple field names for flexibility:
    - 'video_path': Standard format
    - 'input.video_local_path': Alternative format
    
    Args:
        item: Dictionary containing video information
        
    Returns:
        Video path string or None if not found
    """
    return item.get('video_path') or item.get('input.video_local_path')


def extract_prompt(item: Dict[str, Any]) -> Optional[str]:
    """
    Extract prompt/caption text from a data item.
    
    Supports multiple field names for flexibility:
    - 'prompt': Standard format
    - 'input.prompt': Alternative format
    - 'caption': Caption format
    
    Args:
        item: Dictionary containing prompt information
        
    Returns:
        Prompt string or None if not found
    """
    return item.get('prompt') or item.get('input.prompt') or item.get('caption')


def prepare_messages_from_data(
    test_data: List[Dict[str, Any]],
    task_type: str,
    system_prompt: str,
    user_prompt_template: str,
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Convert test data to messages format for inference.
    
    This function processes input data and creates message lists in the
    OpenAI-compatible format expected by the reward models.
    
    Args:
        test_data: List of test data items
        task_type: Task type ('text_alignment' or 'phy_deform')
        system_prompt: System prompt text
        user_prompt_template: User prompt template (may contain {prompt} placeholder)
        
    Returns:
        Tuple of (messages_list, valid_data) where:
        - messages_list: List of message lists ready for inference
        - valid_data: List of data items that were successfully processed
    """
    messages_list = []
    valid_data = []
    
    for item in test_data:
        if not isinstance(item, dict):
            print(f"Warning: Unsupported item format in test data (expected dict).")
            continue
        
        # Extract video path
        video_path = extract_video_path(item)
        if not video_path:
            print(f"Warning: No video path found in item. "
                  f"Expected 'video_path' or 'input.video_local_path' field.")
            continue
        
        # Format user prompt based on task type
        if task_type == "text_alignment":
            prompt_text = extract_prompt(item)
            if prompt_text is None:
                print(f"Warning: 'prompt', 'input.prompt', or 'caption' field "
                      f"missing for text_alignment task")
                continue
            user_prompt_text = user_prompt_template.format(prompt=prompt_text)
        else:
            user_prompt_text = user_prompt_template
        
        # Create message in OpenAI-compatible format
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt_text},
                    {"type": "video", "video": video_path}
                ]
            }
        ]
        
        messages_list.append(messages)
        valid_data.append(item)
    
    return messages_list, valid_data


def compute_statistics(scores: List[float]) -> Dict[str, float]:
    """
    Compute summary statistics for a list of scores.
    
    Args:
        scores: List of reward scores
        
    Returns:
        Dictionary containing statistics (count, mean, min, max, std)
    """
    if not scores:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0
        }
    
    count = len(scores)
    mean = sum(scores) / count
    min_val = min(scores)
    max_val = max(scores)
    
    # Calculate standard deviation
    variance = sum((x - mean) ** 2 for x in scores) / count
    std = variance ** 0.5
    
    return {
        "count": count,
        "mean": mean,
        "min": min_val,
        "max": max_val,
        "std": std
    }


def print_statistics(scores: List[float]) -> None:
    """
    Print summary statistics for inference results.
    
    Args:
        scores: List of reward scores
    """
    stats = compute_statistics(scores)
    print(f"\nSummary Statistics:")
    print(f"Total samples processed: {stats['count']}")
    print(f"Mean score: {stats['mean']:.4f}")
    print(f"Min score: {stats['min']:.4f}")
    print(f"Max score: {stats['max']:.4f}")
    print(f"Std deviation: {stats['std']:.4f}")
