from typing import List, Dict
import torch
from .vl_utils import load_image, load_video
from .image_placeholder import IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN
import logging
from dataclasses import dataclass
import json
import math
import time
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from typing import List, Dict, Union, Any
from collections import defaultdict
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding
from transformers.utils.generic import PaddingStrategy
import logging
from datasets import Dataset
from abc import ABC, abstractmethod
import functools

ENABLE_TIMING_LOGS = False

def timing_decorator(func):
    """Decorator: Record function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if ENABLE_TIMING_LOGS:
            logging.info(f"{func.__name__} execution time: {elapsed_time:.4f} seconds")
        return result
    return wrapper


@dataclass
class InternVLDataCollator:
    """Data collator specifically for InternVL models."""
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    return_tensors: str = "pt"
    input_size: int = 448
    max_num: int = 12
    num_segments: int = 32
    num_image_token: int = -1
    center_crop_video: bool = False

    @timing_decorator
    def _process_messages_and_extract_visual_data(self, messages: List[Dict]) -> tuple[List[Dict], torch.Tensor, List[int]]:
        """
        Process OpenAI format messages, add <image> tags, and extract visual content.
        Returns: (modified_messages, pixel_values, num_patches_list)
        """
        modified_messages = []
        all_pixel_values = []
        all_num_patches = []
        
        for message in messages:
            modified_message = {"role": message["role"], "content": []}
            
            if message["role"] == "system":
                # For system messages, just keep text content
                for content in message["content"]:
                    if content["type"] == "text":
                        modified_message["content"].append({"type": "text", "text": content["text"]})
                    else:
                        logging.warning(f"Unsupported content type '{content['type']}' in system message. Only 'text' is supported.")
                        
            elif message["role"] == "user":
                for content in message["content"]:
                    if content["type"] == "text":
                        modified_message["content"].append({"type": "text", "text": content["text"]})
                        
                    elif content["type"] == "video":
                        video_path = content["video"]
                        pixel_values, num_patches_list = load_video(
                            video_path,
                            input_size=self.input_size,
                            max_num=self.max_num,
                            num_segments=self.num_segments,
                            center_crop=self.center_crop_video,
                        )
                        all_pixel_values.append(pixel_values)
                        all_num_patches.extend(num_patches_list)
                        
                        # Add frame prefixes with <image> tags
                        video_text = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                        modified_message["content"].append({"type": "text", "text": video_text})
                        
                    elif content["type"] == "image":
                        image_path = content["image"]
                        pixel_values = load_image(
                            image_path, 
                            input_size=self.input_size, 
                            max_num=self.max_num
                        )
                        all_pixel_values.append(pixel_values)
                        all_num_patches.append(pixel_values.shape[0])
                        modified_message["content"].append({"type": "text", "text": "<image>"})
                    else:
                        logging.warning(f"Unsupported content type '{content['type']}' in user message. Supported types are 'text', 'image', and 'video'.")
                            
            modified_messages.append(modified_message)
        
        # Combine all pixel values
        if all_pixel_values:
            combined_pixel_values = torch.cat(all_pixel_values, dim=0) # total_patches x 3 x H x W
        else:
            combined_pixel_values = torch.empty(0)
            
        return modified_messages, combined_pixel_values, all_num_patches

    @timing_decorator
    def _replace_visual_placeholders(self, formatted_text: str, num_patches_list: List[int]) -> str:
        """Replace <image> placeholders with InternVL-specific image tokens."""
        # Calculate number of image tokens per patch (from InternVL model)
        num_image_token = self.num_image_token
        assert num_image_token > 0, "num_image_token must be set to a positive integer."

        for i, num_patches in enumerate(num_patches_list):
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
            formatted_text = formatted_text.replace('<image>', image_tokens, 1)
        
        # Assert: Check that IMG_CONTEXT_TOKEN count matches expected patch count
        expected_img_context_tokens = sum(num_image_token * num_patches for num_patches in num_patches_list)
        actual_img_context_tokens = formatted_text.count(IMG_CONTEXT_TOKEN)
        assert actual_img_context_tokens == expected_img_context_tokens, \
            f"IMG_CONTEXT_TOKEN count mismatch: expected {expected_img_context_tokens}, got {actual_img_context_tokens}. " \
            f"num_patches_list: {num_patches_list}, num_image_token: {num_image_token}"
        
        return formatted_text

    
    @timing_decorator
    def _flatten_message_content(self, messages: List[Dict]) -> List[Dict]:
        """Flatten message content to text-only format for tokenization."""
        flattened_messages = []
        for msg in messages:
            role = msg['role']
            contents = msg['content']
            text = ''
            for content in contents:
                if content['type'] == 'text':
                    text += content['text']
            flattened_messages.append({'role': role, 'content': text})
        return flattened_messages

    @timing_decorator
    def _apply_chat_template_and_tokenize(self, messages: List[Dict]) -> str:
        """Apply chat template and tokenize messages."""
        formatted_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
        )
        
        if not isinstance(formatted_text, str):
            raise ValueError(f"apply_chat_template should return a string, got {type(formatted_text)}")
        
        return formatted_text
    
    @timing_decorator
    def _get_pad_token_id(self) -> int:
        """Get pad token ID as integer."""
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
        # Ensure it's an integer
        if isinstance(pad_token_id, (str, list)):
            pad_token_id = 0
        return int(pad_token_id)
    
    @timing_decorator
    def _pad_sequences(self, input_ids_batch: List[torch.Tensor], attention_mask_batch: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad input sequences to same length."""
        from torch.nn.utils.rnn import pad_sequence
        
        pad_token_id = self._get_pad_token_id()
        padded_input_ids = pad_sequence(input_ids_batch, batch_first=True, padding_value=pad_token_id)
        padded_attention_mask = pad_sequence(attention_mask_batch, batch_first=True, padding_value=0)
        
        return padded_input_ids, padded_attention_mask
    
    @timing_decorator
    def _pad_visual_data(self, pixel_values_batch: List[torch.Tensor], num_patches_batch: List[List[int]], input_size: int) -> Dict[str, torch.Tensor]:
        """Process visual data with padding."""
        if not pixel_values_batch:
            # No visual data - create empty tensors
            return {
                'pixel_values': torch.empty(0, 3, input_size, input_size),
                'image_flags': torch.empty(0, dtype=torch.long)
            }
        
        # Handle variable sequence lengths by padding
        max_seq_len = max(pv.shape[0] for pv in pixel_values_batch)
        padded_pixel_values = []
        image_flags_list = []
        
        for pv in pixel_values_batch:
            original_len = pv.shape[0]
            if pv.shape[0] < max_seq_len:
                padding = torch.zeros(max_seq_len - pv.shape[0], *pv.shape[1:], dtype=pv.dtype)
                padded_pv = torch.cat([pv, padding], dim=0)
            else:
                padded_pv = pv
            padded_pixel_values.append(padded_pv)
            
            # Create image_flags: 1 for valid patches, 0 for padded patches
            flags = torch.cat([
                torch.ones(original_len, dtype=torch.long),  # valid patches
                torch.zeros(max_seq_len - original_len, dtype=torch.long)  # padded patches
            ])
            image_flags_list.append(flags)
        
        return {
            'pixel_values': torch.cat(padded_pixel_values, dim=0),  # Shape: [total_patches, 3, H, W]
            'image_flags': torch.cat(image_flags_list, dim=0)
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of pair data and return tokenized inputs for win and lose.
        
        Args:
            features: List of dicts, each containing 'win' and 'lose' message lists
            
        Returns:
            Dict with 'win' and 'lose' keys, each containing tokenized and processed data.
        """
        win_data = {'input_ids': [], 'attention_mask': [], 'pixel_values': [], 'num_patches': [], 'quality': []}
        lose_data = {'input_ids': [], 'attention_mask': [], 'pixel_values': [], 'num_patches': [], 'quality': []}
        
        for feature in features:
            # Process win and lose data
            for win_or_lose, data_dict in [('win', win_data), ('lose', lose_data)]:
                # Step 1: Process messages and extract visual data, convert video to frames
                modified_messages, pixel_values, num_patches_list = self._process_messages_and_extract_visual_data(
                    feature[win_or_lose]
                )

                # Step 2: Flatten messages to text-only format, convert content from list to string
                flattened_messages = self._flatten_message_content(modified_messages)
                
                # Step 3: Apply chat template
                formatted_text = self._apply_chat_template_and_tokenize(flattened_messages)
                
                # Step 4: Replace visual placeholders with model-specific tokens
                formatted_text_with_special_visual_tags = self._replace_visual_placeholders(formatted_text, num_patches_list)
                
                # Step 5: tokenize with visual tokens
                encoded = self.tokenizer(
                    formatted_text_with_special_visual_tags,
                    return_tensors=self.return_tensors,
                    truncation=False,
                    add_special_tokens=False
                )
                
                # Store results
                data_dict['input_ids'].append(encoded.input_ids[0])
                data_dict['attention_mask'].append(encoded.attention_mask[0])
                data_dict['quality'].append(feature['meta'][win_or_lose]['quality'])
                if pixel_values.numel() > 0:
                    data_dict['pixel_values'].append(pixel_values)
                    data_dict['num_patches'].append(num_patches_list)
        
        # Pad sequences - combine win and lose for unified padding
        all_input_ids = win_data['input_ids'] + lose_data['input_ids']
        all_attention_mask = win_data['attention_mask'] + lose_data['attention_mask']
        
        padded_input_ids, padded_attention_mask = self._pad_sequences(all_input_ids, all_attention_mask)
        
        # Split back to win and lose
        batch_size = len(win_data['input_ids'])
        win_input_ids = padded_input_ids[:batch_size]
        win_attention_mask = padded_attention_mask[:batch_size]
        lose_input_ids = padded_input_ids[batch_size:]
        lose_attention_mask = padded_attention_mask[batch_size:]
        
        # Build result
        result = {
            'win': {
                'input_ids': win_input_ids,
                'attention_mask': win_attention_mask,
                'quality': torch.tensor(win_data['quality'], dtype=torch.float32),
            },
            'lose': {
                'input_ids': lose_input_ids,
                'attention_mask': lose_attention_mask,
                'quality': torch.tensor(lose_data['quality'], dtype=torch.float32),
            }
        }
        
        # Add visual data if present
        for win_or_lose, data_dict in [('win', win_data), ('lose', lose_data)]:
            visual_result = self._pad_visual_data(
                data_dict['pixel_values'], 
                data_dict['num_patches'], 
                getattr(self, 'input_size', 448)
            )
            result[win_or_lose].update(visual_result)
            # Store num_patches_list as additional metadata (not a tensor)
            # setattr(result[win_or_lose], 'num_patches_list', data_dict['num_patches'])
            result[win_or_lose]['num_patches_list'] = data_dict['num_patches']
        
        return result