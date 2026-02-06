import json
try:
    import orjson
except ImportError:
    orjson = None
import math
import time
import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
from typing import List, Dict, Union, Any
from collections import defaultdict
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding
from transformers.utils.generic import PaddingStrategy
import logging
from datasets import Dataset
from abc import ABC, abstractmethod
from collections import OrderedDict



def load_json(file_path) -> Union[List, Dict]:
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
    
    file_size = os.path.getsize(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"{rank_prefix}Loading {os.path.basename(file_path)}") as pbar:
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


def load_pair_json_data(file_path) -> List[Dict]:
    """
    list of dict, each dict has 'win' and 'lose' keys. 'win'/'lose' is a dict with 'video_path' and 'quality'.
    An example is:
    [
        {
            "win": {
                "video_path": "data/videos/1.mp4",
                "quality": 0.9
            },
            "lose": {
                "video_path": "data/videos/2.mp4",
                "quality": 0.1
            }
        },
        ...
    """
    data = load_json(file_path)
    assert isinstance(data, list), "The pair json data should be a list of dicts."
    assert len(data) > 0, "The pair json data is empty."
    assert isinstance(data[0], dict), "Each item in the pair json data should be a dict."
    return data


def convert_pair_json_data_to_openai_format(data: List[Dict]) -> List[Dict]:
    """
    Convert the pair json data to OpenAI format.
    Each item in the returned list is a dict with 'win' and 'lose' keys
    Each 'win'/'lose' is a list of messages in OpenAI format.
    Output example:
    [
        {
            "win": a message list in OpenAI format,
            "lose": a message list in OpenAI format,
            "meta": {
                "win": dict of meta info,
                "lose": dict of meta info,
            }
        },
        ...
    ]
    """
    ans = []
    for item in data:
        system_prompt = "You are a video quality assessment assistant. You must respond with only 'good' or 'bad' to indicate the video quality."
        user_prompt = "Please assess the quality of this video. Respond with only 'good' or 'bad'."
        pair = {}
        for win_or_lose in ['win', 'lose']:
            video_path = item[win_or_lose]['video_path']
            quality = item[win_or_lose]['quality']
            pair[win_or_lose] = [
                {   
                    "role": "system", 
                    "content": [
                        {
                            "type": "text", 
                            "text": system_prompt
                        },
                    ],
                },
                {   
                    "role": "user", 
                    "content": [
                        {
                            "type": "text", 
                            "text": user_prompt
                        },
                        {
                            "type": "video",
                            "video": video_path,
                        }
                    ]
                }
            ]

        pair['meta'] = {}
        for win_or_lose in ['win', 'lose']:
            pair['meta'][win_or_lose] = item[win_or_lose].copy()
        ans.append(pair)

    return ans


def convert_ms_swift_data_to_openai_format(data: List[Dict]) -> List[Dict]:
    """
    Convert the ms-swift format data to OpenAI format.
    
    ms-swift format structure:
    - messages: List of message dicts (system, user messages)
    - rejected_response: String containing the rejected response content
    - quality: List with [positive_score, negative_score]  
    - videos: List of video URLs corresponding to <video> placeholders
    - meta: Dictionary with positive and negative metadata
    
    For each item:
    - Positive sample: Use messages directly, replace <video> with actual video paths
    - Negative sample: Copy messages, replace last message content with rejected_response,
                      and replace <video> with actual video paths
    
    Returns the same format as convert_pair_json_data_to_openai_format:
    [
        {
            "win": a message list in OpenAI format,
            "lose": a message list in OpenAI format,
            "meta": {
                "win": dict of meta info,
                "lose": dict of meta info,
            }
        },
        ...
    ]
    """
    ans = []
    
    for item in data:
        # Validate required fields
        required_fields = ['messages', 'rejected_response', 'quality', 'videos']
        for field in required_fields:
            if field not in item:
                raise ValueError(f"Missing required field '{field}' in ms-swift data item")
        
        # Get base data
        messages = item['messages']
        rejected_response = item['rejected_response']
        quality_scores = item['quality']  # [positive_score, negative_score]
        video_urls = item['videos']  # [positive_video(s), negative_video(s)]
        
        # Validate quality scores
        if len(quality_scores) != 2:
            raise ValueError(f"Expected quality to have 2 scores [positive, negative], got {len(quality_scores)}")
        
        positive_score = quality_scores[0]
        negative_score = quality_scores[1]
        
        # Count <video> placeholders in messages to determine video distribution
        video_count_in_messages = sum(msg.get('content', '').count('<video>') for msg in messages)
        
        # Validate that we have the expected number of videos
        if len(video_urls) != video_count_in_messages * 2:
            raise ValueError(f"Expected {video_count_in_messages * 2} videos (2 per <video> placeholder), got {len(video_urls)}")
        
        # Split videos into positive and negative sets
        # Assuming first half are for positive sample, second half for negative sample
        half_point = len(video_urls) // 2
        positive_videos = video_urls[:half_point]
        negative_videos = video_urls[half_point:]
        
        # Helper function to process messages and add video content
        def process_messages_with_videos(messages: List[Dict], video_urls: List[str]) -> List[Dict]:
            """Process messages and add video content where <video> placeholders exist."""
            processed_messages = []
            video_idx = 0
            
            for message in messages:
                processed_message = {
                    "role": message["role"],
                    "content": []
                }
                
                content = message.get("content", "")
                
                # Handle the content - split by <video> and add video elements
                parts = content.split('<video>')
                
                for i, part in enumerate(parts):
                    # Add text part if not empty
                    if part.strip():
                        processed_message["content"].append({
                            "type": "text",
                            "text": part
                        })
                    
                    # Add video after each part except the last one
                    if i < len(parts) - 1 and video_idx < len(video_urls):
                        processed_message["content"].append({
                            "type": "video",
                            "video": video_urls[video_idx]
                        })
                        video_idx += 1
                
                # If content is empty after processing, add empty text
                if not processed_message["content"]:
                    processed_message["content"] = [{"type": "text", "text": ""}]
                
                processed_messages.append(processed_message)
            
            return processed_messages
        
        # Create positive sample (win) - use original messages
        win_messages = process_messages_with_videos(messages, positive_videos)
        
        # Create negative sample (lose) - copy messages and replace last message content
        lose_messages_raw = messages.copy()
        if lose_messages_raw:
            # Replace the content of the last message with rejected_response
            lose_messages_raw[-1] = {
                **lose_messages_raw[-1],  # Keep role and other fields
                "content": rejected_response
            }
        
        lose_messages = process_messages_with_videos(lose_messages_raw, negative_videos)
        
        # Create the result structure
        pair = {
            'win': win_messages,
            'lose': lose_messages,
            'meta': {
                'win': {
                    'quality': positive_score
                },
                'lose': {
                    'quality': negative_score
                }
            }
        }
        
        # Add original meta information if available
        if 'meta' in item:
            if 'positive' in item['meta']:
                pair['meta']['win'].update(item['meta']['positive'])
            if 'negative' in item['meta']:  
                pair['meta']['lose'].update(item['meta']['negative'])
        
        ans.append(pair)
    
    return ans


def create_dataset_from_json(json_paths: Union[str, List[str]]) -> Dataset:
    """
    Load and convert JSON data to Dataset format.
    
    Args:
        json_paths: A single JSON file path (str) or a list of JSON file paths (List[str])
    
    Returns:
        Dataset: Combined dataset from all JSON files
    """
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
    
    # Ensure json_paths is a list
    if isinstance(json_paths, str):
        json_paths = [json_paths]
    
    all_converted_data = []
    
    # Load and convert data from each file
    for json_path in json_paths:
        print(f"{rank_prefix}Loading data from: {json_path}")
        # Load the pair data
        raw_data = load_pair_json_data(json_path)
        
        # Convert to OpenAI format
        # converted_data = convert_pair_json_data_to_openai_format(raw_data)
        converted_data = convert_ms_swift_data_to_openai_format(raw_data)
        
        all_converted_data.extend(converted_data)
        print(f"{rank_prefix}Loaded {len(converted_data)} samples from {json_path}")
    
    print(f"{rank_prefix}Total samples loaded: {len(all_converted_data)}")
    
    # Create dataset from combined data
    dataset = Dataset.from_list(all_converted_data)
    return dataset

