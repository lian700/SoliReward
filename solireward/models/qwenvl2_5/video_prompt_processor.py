from typing import List, Dict, Any, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding
from qwen_vl_utils import process_vision_info
from collections import defaultdict
import numpy as np
import time
import copy


class QwenVLVideoPromptProcessor(nn.Module):
    """
    Video-Prompt Processor for QwenVL Reward Model
    
    This class handles the processing of video pixel values and text prompts
    into the format expected by QwenVL processor while preserving gradients.
    """
    
    def __init__(self, 
                 processor,
                 max_num_frames: int = 10,
                 min_pixels: int = 256 * 28 * 28,
                 max_pixels: int = 1280 * 28 * 28,
                 factor: int = 28,
                 system_prompt: Optional[str] = "你是一个专业的视频质量评估专家。请根据提供的视频内容，判断是否同时满足以下所有问题的合格标准：\n\n{questions_str}\n\n回答要求：\n- 只有当所有问题的答案都是\"合格\"时，才输出：good\n- 如果任何一个问题的答案是\"部分合格\"或\"不合格\"，则输出：bad\n- 不要输出任何其他内容\n- 答案要准确、客观\n", 
                 questions: Optional[List[str]] = [
                     "物理规律是否合格？",
                     "是否存在人物或动物畸形？"
                 ],
                 fake_video_url: str = "http://example.com/fake_video.mp4"):
        """
        Initialize the processor.
        
        Args:
            processor: QwenVL processor with apply_chat_template method
            max_num_frames: Maximum number of frames to process
            min_pixels: Minimum number of pixels for resize
            max_pixels: Maximum number of pixels for resize
            factor: Factor for smart resize (similar to IMAGE_FACTOR in vision_process.py)
            system_prompt: System prompt template
            questions: List of evaluation questions
            fake_video_url: Fake video URL to use in message construction
        """
        super().__init__()
        self.processor = processor
        self.max_num_frames = max_num_frames
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.factor = factor
        self.system_prompt = system_prompt if system_prompt is not None else "You are a helpful assistant that accurately evaluates videos based on the user's questions."
        self.questions = questions if questions is not None else []
        self.fake_video_url = fake_video_url
        self.has_semantic_question = any('语义' in q.lower() for q in self.questions)
        
        # Add numbering to questions
        for i in range(len(self.questions)):
            self.questions[i] = f"{i+1}. {self.questions[i]}"
    
    def _round_by_factor(self, number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    def _ceil_by_factor(self, number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        import math
        return math.ceil(number / factor) * factor

    def _floor_by_factor(self, number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        import math
        return math.floor(number / factor) * factor

    def _smart_resize(self, height: int, width: int) -> tuple[int, int]:
        """
        Rescales the image so that dimensions are divisible by factor and pixels are within range.
        Similar to vision_process.py smart_resize function.
        """
        import math
        
        MAX_RATIO = 200
        if max(height, width) / min(height, width) > MAX_RATIO:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
            )
        
        h_bar = max(self.factor, self._round_by_factor(height, self.factor))
        w_bar = max(self.factor, self._round_by_factor(width, self.factor))
        
        if h_bar * w_bar > self.max_pixels:
            beta = math.sqrt((height * width) / self.max_pixels)
            h_bar = self._floor_by_factor(height / beta, self.factor)
            w_bar = self._floor_by_factor(width / beta, self.factor)
        elif h_bar * w_bar < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h_bar = self._ceil_by_factor(height * beta, self.factor)
            w_bar = self._ceil_by_factor(width * beta, self.factor)
            
        return h_bar, w_bar

    def _sample_video_frames(self, video_tensor: torch.Tensor, num_segments: Optional[int] = None) -> torch.Tensor:
        """
        Sample frames from video tensor using uniform sampling.
        
        Args:
            video_tensor: Video tensor of shape (batch_size, num_frames, 3, H, W) or (num_frames, 3, H, W)
            num_segments: Number of frames to sample (defaults to self.max_num_frames)
            
        Returns:
            torch.Tensor: Sampled video tensor
        """
        if num_segments is None:
            num_segments = self.max_num_frames
            
        if video_tensor.dim() == 4:  # (num_frames, 3, H, W)
            total_frames = video_tensor.shape[0]
            if total_frames <= num_segments:
                return video_tensor
            
            # Uniform sampling
            idx = torch.linspace(0, total_frames - 1, num_segments).round().long()
            sampled_frames = video_tensor[idx]  # (num_segments, 3, H, W)
            return sampled_frames
        elif video_tensor.dim() == 5:  # (batch_size, num_frames, 3, H, W)
            batch_size, total_frames = video_tensor.shape[:2]
            if total_frames <= num_segments:
                return video_tensor
                
            # Sample same indices for all samples in batch
            idx = torch.linspace(0, total_frames - 1, num_segments).round().long()
            sampled_frames = video_tensor[:, idx]  # (batch_size, num_segments, 3, H, W)
            return sampled_frames
        else:
            raise ValueError(f"Unsupported video tensor shape: {video_tensor.shape}")

    def _resize_video_frames(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize video frames while preserving gradients.
        
        Args:
            video_tensor: Video tensor of shape (batch_size, num_frames, 3, H, W) or (num_frames, 3, H, W)
            
        Returns:
            torch.Tensor: Resized video tensor
        """
        if video_tensor.dim() == 4:  # (num_frames, 3, H, W)
            num_frames, channels, orig_height, orig_width = video_tensor.shape
            new_height, new_width = self._smart_resize(orig_height, orig_width)
            
            # Resize using bilinear interpolation (differentiable)
            resized_video = F.interpolate(
                video_tensor, 
                size=(new_height, new_width), 
                mode='bilinear', 
                align_corners=False
            )  # (num_frames, 3, new_height, new_width)
            return resized_video
            
        elif video_tensor.dim() == 5:  # (batch_size, num_frames, 3, H, W)
            batch_size, num_frames, channels, orig_height, orig_width = video_tensor.shape
            new_height, new_width = self._smart_resize(orig_height, orig_width)
            
            # Reshape for batch processing
            video_reshaped = video_tensor.view(batch_size * num_frames, channels, orig_height, orig_width)
            
            # Resize using bilinear interpolation (differentiable)
            resized_video = F.interpolate(
                video_reshaped, 
                size=(new_height, new_width), 
                mode='bilinear', 
                align_corners=False
            )  # (batch_size * num_frames, 3, new_height, new_width)
            
            # Reshape back
            resized_video = resized_video.view(batch_size, num_frames, channels, new_height, new_width)
            return resized_video
        else:
            raise ValueError(f"Unsupported video tensor shape: {video_tensor.shape}")

    def _build_chat_messages_batch(self, prompts: List[str]) -> List[List[Dict]]:
        """
        Build OpenAI-style chat messages with video for a batch of inputs.
        
        Args:
            prompts: List of text prompts which generate the videos
            
        Returns:
            List of message lists, one for each sample in the batch
        """
        batch_messages = []
        
        for prompt in prompts:
            if self.has_semantic_question:
                user_instruction = "生成视频的文本提示词是: {video_gen_prompt}。根据该提示词生成的视频为：".format(video_gen_prompt=prompt)
            else:
                user_instruction = "请评估以下视频："
            
            messages = [
                {
                    "role": "system", 
                    "content": [
                        {"type": "text", "text": self.system_prompt.format(questions_str='\n'.join(self.questions))}
                    ]
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_instruction},
                        {"type": "video", "video": self.fake_video_url}
                    ]
                }
            ]
            batch_messages.append(messages)
        
        return batch_messages

    def _reorder_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reorder messages to QwenVL format (similar to data collator).
        """
        res = []
        for message in messages:
            new_message = defaultdict(list)
            role = message["role"]
            content = message["content"]

            new_message["role"] = role
            for c in content:
                type_c = c["type"]
                new_message["content"].append({
                    "type": type_c,
                    type_c: c[type_c]
                })

            res.append(new_message)
        return res

    def prepare_inputs_batch(self, 
                            prompts: List[str],
                            video_pixel_values: torch.Tensor,
                            ) -> Dict[str, Any]:
        """
        Prepare batch of videos and prompts for QwenVL processor input while preserving gradients.
        
        Args:
            prompts: List of text prompts describing the video tasks
            video_pixel_values: Video tensor of shape (batch_size, num_frames, 3, H, W) with gradients.
                The range of video_pixel_values should be [0, 1].
            quality_scores: Optional quality scores for each sample, shape (batch_size,)
            
        Returns:
            Dictionary containing inputs in QwenVL data collator format:
            {
                'inputs_concat': BatchEncoding,     # the processed inputs from QwenVL processor
                'quality_concat': torch.Tensor,    # the quality scores, shape = (batch_size,)
                'batch_size': int,                 # the batch size
            }
        """
        # Ensure video tensor requires gradients
        if not video_pixel_values.requires_grad:
            video_pixel_values = video_pixel_values.requires_grad_(True)
        
        batch_size = video_pixel_values.shape[0]
        
        # Check that number of prompts matches batch size
        if len(prompts) != batch_size:
            raise ValueError(f"Number of prompts ({len(prompts)}) must match batch size ({batch_size})")
        
        # Step 1: Sample frames from video if we have more frames than the maximum allowed
        total_frames = video_pixel_values.shape[1]
        if total_frames > self.max_num_frames:
            video_pixel_values = self._sample_video_frames(video_pixel_values, self.max_num_frames)
        
        # Step 2: Resize video frames while preserving gradients
        video_pixel_values = self._resize_video_frames(video_pixel_values)
        
        # Step 3: Build chat messages for all samples
        batch_messages = self._build_chat_messages_batch(prompts)
        
        # Step 4: Reorder messages to QwenVL format
        messages_concat = [self._reorder_messages(messages) for messages in batch_messages]
        
        # Step 5: Apply chat template
        texts_concat = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_concat
        ]
        
        # Step 6: Process vision info with retry mechanism (similar to data collator)
        pass

        # Step 7: Process with QwenVL processor
        # Convert video tensor to the expected format for QwenVL
        # QwenVL expects videos as a list of tensors for each sample
        # Based on vision_process.py, QwenVL uses (T, C, H, W) format with values in [0, 255] range
        batch_videos = []
        for i in range(batch_size):
            # Convert from (num_frames, 3, H, W) to (T, C, H, W) format
            # Keep the same format but scale to [0, 255] range
            video_sample = video_pixel_values[i]  # (num_frames, 3, H, W)
            video_sample = (video_sample * 255).clamp(0, 255)  # Convert to [0, 255] range
            batch_videos.append(video_sample)
        
        # Use the processor to handle both text and videos
        inputs_concat = self.processor(
            text=texts_concat,
            videos=batch_videos,
            padding=True,
            return_tensors="pt",
        )

        # Step 9: Return in QwenVL data collator format
        res = {
            "inputs_concat": inputs_concat,
            "batch_size": batch_size,
        }
        
        return res

    def forward(self, 
                 prompts: List[str], 
                 video_pixel_values: torch.Tensor,
                ) -> Dict[str, Any]:
        """
        Convenience method that calls prepare_inputs_batch.
        """
        return self.prepare_inputs_batch(prompts, video_pixel_values)