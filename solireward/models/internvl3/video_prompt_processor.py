import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Union, Optional
import math
import numpy as np

from .image_placeholder import IMG_CONTEXT_TOKEN, IMG_START_TOKEN, IMG_END_TOKEN
from .vl_utils import IMAGENET_MEAN, IMAGENET_STD

class InternVLVideoPromptProcessor(nn.Module):
    """
    Video-Prompt Processor for InternVL3 Reward Model
    
    This class handles the processing of video pixel values and text prompts
    into the format expected by InternVL3RewardModel while preserving gradients.
    """
    
    def __init__(self, 
                 tokenizer,
                 input_size: int = 448,
                 max_num_frames: int = 8,
                 max_num: int = 12, 
                 num_image_token: int = 256,
                 system_prompt: Optional[str] = "你是一个专业的视频质量评估专家。请根据提供的视频内容，判断是否同时满足以下所有问题的合格标准：\n\n{questions_str}\n\n回答要求：\n- 只有当所有问题的答案都是\"合格\"时，才输出：good\n- 如果任何一个问题的答案是\"部分合格\"或\"不合格\"，则输出：bad\n- 不要输出任何其他内容\n- 答案要准确、客观\n", 
                 questions: Optional[List[str]] = [
                     "物理规律是否合格？",
                     "是否存在人物或动物畸形？"
                 ]):
        """
        Initialize the processor.
        
        Args:
            tokenizer: Tokenizer with apply_chat_template method
            input_size: Base size for image patches
            max_num_frames: Maximum number of frames to process (frames will be sampled if video has more)
            max_num: Maximum number of patches per frame
            num_image_token: Number of image tokens per patch
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.input_size = input_size
        self.max_num_frames = max_num_frames
        self.max_num = max_num
        self.num_image_token = num_image_token
        self.system_prompt = system_prompt if system_prompt is not None else "You are a helpful assistant that accurately evaluates videos based on the user's questions."
        self.questions = questions if questions is not None else []
        self.has_semantic_question = any('语义' in q.lower() for q in self.questions)
        
        for i in range(len(self.questions)):
            self.questions[i] = f"{i+1}. {self.questions[i]}" # add numbering
        
        # ImageNet normalization constants - register as buffers
        self.register_buffer('imagenet_mean', torch.tensor(IMAGENET_MEAN))
        self.register_buffer('imagenet_std', torch.tensor(IMAGENET_STD))
    
    def _build_chat_messages_batch(self, prompts: List[str], num_frames_list: List[int]) -> List[List[Dict]]:
        """
        Build OpenAI-style chat messages with video frames for a batch of inputs.
        
        Args:
            prompts: List of text prompts which generate the videos
            num_frames_list: List of number of video frames for each sample
            
        Returns:
            List of message lists, one for each sample in the batch
        """
        batch_messages = []
        
        for prompt, num_frames in zip(prompts, num_frames_list):
            # Create frame prefixes for each video frame
            if self.has_semantic_question:
                user_instruction = "生成视频的文本提示词是: {video_gen_prompt}。根据该提示词生成的视频为：".format(video_gen_prompt=prompt)
            else:
                user_instruction = "请评估以下视频："

            frame_text = ''.join([f'Frame{i+1}: <image>\n' for i in range(num_frames)])
            
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
                        {"type": "text", "text": user_instruction + frame_text}
                    ]
                }
            ]
            batch_messages.append(messages)
        
        return batch_messages
    
    def _apply_chat_template_batch(self, batch_messages: List[List[Dict]]) -> List[str]:
        """
        Apply chat template to a batch of messages.
        
        Args:
            batch_messages: List of message lists for each sample
            
        Returns:
            List of formatted text strings
        """
        formatted_texts = []
        
        for messages in batch_messages:
            # Convert messages to simple format for tokenizer
            simple_messages = []
            for msg in messages:
                role = msg['role']
                content_parts = msg['content']
                text = ''
                for part in content_parts:
                    if part['type'] == 'text':
                        text += part['text']
                simple_messages.append({'role': role, 'content': text})
            
            formatted_text = self.tokenizer.apply_chat_template(
                simple_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_texts.append(formatted_text)
        
        return formatted_texts
    
    def _replace_image_placeholders_batch(self, texts: List[str], num_patches_lists: List[List[int]]) -> List[str]:
        """
        Replace <image> placeholders with InternVL-specific image tokens for a batch.
        
        Args:
            texts: List of texts containing <image> placeholders
            num_patches_lists: List of patch count lists for each sample
            
        Returns:
            List of texts with replaced image tokens
        """
        processed_texts = []
        
        for text, num_patches_list in zip(texts, num_patches_lists):
            processed_text = text
            for num_patches in num_patches_list:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                processed_text = processed_text.replace('<image>', image_tokens, 1)
            processed_texts.append(processed_text)
        
        return processed_texts
    
    def _get_frame_indices(self, total_frames: int, num_segments: Optional[int] = None, bound: Optional[List[float]] = None, first_idx: int = 0) -> torch.Tensor:
        """
        Get frame indices for sampling frames from video tensor, based on vl_utils.get_index logic.
        
        Args:
            total_frames: Total number of frames in the video
            num_segments: Number of segments to sample from (defaults to self.max_num_frames)
            bound: Optional time bounds [start_time, end_time] in seconds (currently unused for tensors)
            first_idx: First frame index to consider
            
        Returns:
            torch.Tensor: Frame indices to sample, shape (num_segments,)
        """
        if num_segments is None:
            num_segments = self.max_num_frames
            
        # For tensor processing, we assume bound is in frame indices rather than time
        if bound:
            start_idx, end_idx = int(bound[0]), int(bound[1])
        else:
            start_idx, end_idx = first_idx, total_frames - 1
            
        start_idx = max(first_idx, start_idx)
        end_idx = min(end_idx, total_frames - 1)
        
        if start_idx >= end_idx:
            # Edge case: if start >= end, just repeat the start frame
            frame_indices = torch.full((num_segments,), start_idx, dtype=torch.long)
        else:
            seg_size = float(end_idx - start_idx) / num_segments
            frame_indices = torch.tensor([
                int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
                for idx in range(num_segments)
            ], dtype=torch.long)
            
            # Clamp indices to valid range
            frame_indices = torch.clamp(frame_indices, min=first_idx, max=total_frames - 1)
        
        return frame_indices
    
    def _sample_video_frames(self, video_tensor: torch.Tensor, num_segments: Optional[int] = None, bound: Optional[List[float]] = None) -> torch.Tensor:
        """
        Sample frames from video tensor using the same logic as vl_utils.get_index.
        
        Args:
            video_tensor: Video tensor of shape (batch_size, num_frames, 3, H, W) or (num_frames, 3, H, W)
            num_segments: Number of frames to sample (defaults to self.max_num_frames)
            bound: Optional frame bounds [start_frame, end_frame] for sampling range
            
        Returns:
            torch.Tensor: Sampled video tensor with shape (batch_size, num_segments, 3, H, W) or (num_segments, 3, H, W)
        """
        if num_segments is None:
            num_segments = self.max_num_frames
            
        if video_tensor.dim() == 4:  # (num_frames, 3, H, W)
            total_frames = video_tensor.shape[0]
            frame_indices = self._get_frame_indices(total_frames, num_segments, bound)
            sampled_frames = video_tensor[frame_indices]  # (num_segments, 3, H, W)
            return sampled_frames
        elif video_tensor.dim() == 5:  # (batch_size, num_frames, 3, H, W)
            batch_size, total_frames = video_tensor.shape[:2]
            frame_indices = self._get_frame_indices(total_frames, num_segments, bound)
            # Sample frames for all batch items
            sampled_frames = video_tensor[:, frame_indices]  # (batch_size, num_segments, 3, H, W)
            return sampled_frames
        else:
            raise ValueError(f"Expected video tensor with 4 or 5 dimensions, got {video_tensor.dim()}")
    
    def _preprocess_video_frames_batch(self, pixel_values: torch.Tensor) -> tuple:
        """
        Process video tensor batch using dynamic preprocessing while preserving gradients.
        
        Args:
            pixel_values: Video tensor of shape (batch_size, num_frames, 3, H, W)
            
        Returns:
            Tuple containing:
            - List of lists of processed frame tensors for each sample in batch
            - List of lists of patch counts for each sample in batch
        """
        batch_size, num_frames = pixel_values.shape[:2]
        batch_processed_frames = []
        batch_num_patches_lists = []
        
        for batch_idx in range(batch_size):
            sample_frames = []
            sample_patch_counts = []
            
            for frame_idx in range(num_frames):
                frame = pixel_values[batch_idx, frame_idx]  # Shape: (3, H, W)
                
                # Apply dynamic preprocessing to this frame
                processed_patches = self._dynamic_preprocess_tensor(
                    frame, 
                    min_num=1, 
                    use_thumbnail=True
                )  # Shape: (num_patches, 3, input_size, input_size)
                
                sample_frames.append(processed_patches)
                sample_patch_counts.append(processed_patches.shape[0])
            
            batch_processed_frames.append(sample_frames)
            batch_num_patches_lists.append(sample_patch_counts)
        
        return batch_processed_frames, batch_num_patches_lists
    
    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor with ImageNet statistics (differentiable).
        
        Args:
            tensor: Input tensor of shape (..., 3, H, W) where 3 is the channel dimension
            
        Returns:
            Normalized tensor
        """
        # Create tensors from buffer values with proper dtype and device
        mean = self.imagenet_mean.to(dtype=tensor.dtype, device=tensor.device)  # Shape: (3,)
        std = self.imagenet_std.to(dtype=tensor.dtype, device=tensor.device)    # Shape: (3,)

        # For tensor of shape (N, 3, H, W), we need mean and std to be (1, 3, 1, 1)
        # Reshape for broadcasting to match tensor dimensions
        if tensor.dim() == 4:  # (N, 3, H, W)
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)
        elif tensor.dim() == 3:  # (3, H, W)
            mean = mean.view(3, 1, 1)
            std = std.view(3, 1, 1)
        else:
            # General case: add dimensions to match tensor shape
            target_shape = [1] * tensor.dim()
            # Find channel dimension (should be 3)
            channel_dim = -1
            for i, size in enumerate(tensor.shape):
                if size == 3:
                    channel_dim = i
                    break
            if channel_dim == -1:
                raise ValueError(f"Could not find channel dimension of size 3 in tensor shape {tensor.shape}")
            
            target_shape[channel_dim] = 3
            mean = mean.view(target_shape)
            std = std.view(target_shape)
        
        # Apply normalization
        normalized = (tensor - mean) / std
        
        return normalized
    
    def _find_closest_aspect_ratio(self, aspect_ratio: float, target_ratios: list, 
                                   width: int, height: int, image_size: int) -> tuple:
        """
        Find the closest aspect ratio from target ratios (differentiable version).
        
        Args:
            aspect_ratio: Original image aspect ratio (w/h)
            target_ratios: List of (w, h) ratio tuples
            width: Original image width
            height: Original image height
            image_size: Target patch size
            
        Returns:
            Best ratio tuple (w, h)
        """
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def _dynamic_preprocess_tensor(self, frame_tensor: torch.Tensor, 
                                   min_num: int = 1, use_thumbnail: bool = True) -> torch.Tensor:
        """
        Dynamic preprocessing for a single frame tensor while preserving gradients.
        
        Args:
            frame_tensor: Input tensor of shape (1, 3, H, W) or (3, H, W)
            min_num: Minimum number of patches
            use_thumbnail: Whether to add thumbnail patch
            
        Returns:
            Preprocessed tensor with dynamic patches: (num_patches, 3, input_size, input_size)
        """
        if frame_tensor.dim() == 3:
            frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dim: (1, 3, H, W)
        
        _, _, orig_height, orig_width = frame_tensor.shape
        aspect_ratio = orig_width / orig_height
        
        # Calculate target ratios
        target_ratios = set(
            (i, j) for n in range(min_num, self.max_num + 1) 
            for i in range(1, n + 1) for j in range(1, n + 1) 
            if i * j <= self.max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        # Find closest aspect ratio
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, self.input_size
        )
        
        # Calculate target dimensions
        target_width = self.input_size * target_aspect_ratio[0]
        target_height = self.input_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        # Resize the frame using bilinear interpolation (differentiable)
        resized_frame = F.interpolate(
            frame_tensor, 
            size=(target_height, target_width), 
            mode='bilinear', 
            align_corners=False
        )  # (1, 3, target_height, target_width)
        
        # Extract patches using unfold operation (differentiable)
        patches = []
        patches_per_row = target_width // self.input_size
        patches_per_col = target_height // self.input_size
        
        for i in range(blocks):
            row = i // patches_per_row
            col = i % patches_per_row
            
            # Extract patch using tensor slicing
            patch = resized_frame[:, :, 
                                  row * self.input_size:(row + 1) * self.input_size,
                                  col * self.input_size:(col + 1) * self.input_size]
            patches.append(patch)
        
        # Add thumbnail if requested and we have more than 1 patch
        if use_thumbnail and len(patches) != 1:
            thumbnail = F.interpolate(
                frame_tensor,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False
            )
            patches.append(thumbnail)
        
        # Stack all patches
        processed_patches = torch.cat(patches, dim=0)  # (num_patches, 3, input_size, input_size)
        
        return processed_patches
    
    def prepare_inputs_batch(self, 
                            prompts: List[str],
                            video_pixel_values: torch.Tensor) -> Dict[str, Any]:
        """
        Prepare batch of videos and prompts for InternVL3RewardModel input while preserving gradients.
        
        Args:
            prompts: List of text prompts describing the video tasks
            video_pixel_values: Video tensor of shape (batch_size, num_frames, 3, H, W) with gradients.
                The range of video_pixel_values should be [0, 1].
            
        Returns:
            Dictionary containing batched model inputs with preserved gradients:
            {
                'input_ids': torch.Tensor,           # Shape: (batch_size, max_seq_len)
                                                     # Tokenized text input with padding
                'attention_mask': torch.Tensor,      # Shape: (batch_size, max_seq_len) 
                                                     # Attention mask for padded sequences (1 for valid tokens, 0 for padding)
                'pixel_values': torch.Tensor,        # Shape: (total_patches, 3, input_size, input_size)
                                                     # Concatenated normalized image patches from all samples and frames
                                                     # where total_patches = sum of patches from all frames in all samples
                'image_flags': torch.Tensor          # Shape: (total_patches,)
                                                     # Flags indicating valid patches (1) vs padded patches (0)
                                                     # Corresponds to pixel_values tensor alignment
            }
            
        """
        # Ensure video tensor requires gradients
        if not video_pixel_values.requires_grad:
            video_pixel_values = video_pixel_values.requires_grad_(True)
        
        batch_size, total_frames = video_pixel_values.shape[:2]
        
        # Check that number of prompts matches batch size
        if len(prompts) != batch_size:
            raise ValueError(f"Number of prompts ({len(prompts)}) must match batch size ({batch_size})")
        
        # Step 0: Sample frames from video if we have more frames than the maximum allowed
        if total_frames > self.max_num_frames:
            print(f"Sampling {self.max_num_frames} frames from {total_frames} total frames")
            video_pixel_values = self._sample_video_frames(video_pixel_values, num_segments=self.max_num_frames)
        
        batch_size, num_frames = video_pixel_values.shape[:2]  # Update num_frames after sampling
        
        # Step 1: Build chat messages for all samples
        num_frames_list = [num_frames] * batch_size  # All samples have same number of frames
        batch_messages = self._build_chat_messages_batch(prompts, num_frames_list)
        
        # Step 2: Apply chat template to all samples
        formatted_texts = self._apply_chat_template_batch(batch_messages)
        print(f"Formatted texts: {formatted_texts}")
        
        # Step 3: Process video frames for all samples (preserving gradients)
        batch_processed_frames, batch_num_patches_lists = self._preprocess_video_frames_batch(video_pixel_values)
        
        # Step 4: Normalize frames (differentiable) and combine
        all_pixel_values_list = []
        
        for sample_frames in batch_processed_frames:
            sample_pixel_values_list = []
            
            # Normalize each frame's patches for this sample
            for frame_patches in sample_frames:
                normalized_patches = self._normalize_tensor(frame_patches)
                sample_pixel_values_list.append(normalized_patches)
            
            # Combine all patches for this sample
            sample_pixel_values = torch.cat(sample_pixel_values_list, dim=0)  # (total_patches_per_sample, 3, input_size, input_size)
            all_pixel_values_list.append(sample_pixel_values)
        
        # Step 5: Replace image placeholders for all samples
        texts_with_image_tokens = self._replace_image_placeholders_batch(
            formatted_texts, 
            batch_num_patches_lists
        )
        
        # Step 6: Tokenize all texts together for proper padding
        encoded = self.tokenizer(
            texts_with_image_tokens,
            return_tensors='pt',
            truncation=False,
            padding=True,  # Enable padding for batch processing
            add_special_tokens=False
        )
        
        # Step 7: Handle pixel values using concatenate (aligned with internvl3/data.py)
        # First pad each sample to the same number of patches
        max_patches = max(pv.shape[0] for pv in all_pixel_values_list)
        padded_pixel_values = []
        batch_image_flags = []
        
        for pixel_values in all_pixel_values_list:
            current_patches = pixel_values.shape[0]
            
            if current_patches < max_patches:
                # Pad with zeros
                padding_shape = (max_patches - current_patches, *pixel_values.shape[1:])
                padding = torch.zeros(padding_shape, dtype=pixel_values.dtype, device=pixel_values.device)
                padded_pixel_values.append(torch.cat([pixel_values, padding], dim=0))
                
                # Create image flags: 1 for valid patches, 0 for padded patches
                image_flags = torch.cat([
                    torch.ones(current_patches, dtype=torch.long, device=pixel_values.device),
                    torch.zeros(max_patches - current_patches, dtype=torch.long, device=pixel_values.device)
                ])
            else:
                padded_pixel_values.append(pixel_values)
                image_flags = torch.ones(current_patches, dtype=torch.long, device=pixel_values.device)
            
            batch_image_flags.append(image_flags)
        
        # Step 8: Concatenate all pixel values and image flags (aligned with internvl3/data.py)
        batch_pixel_values = torch.cat(padded_pixel_values, dim=0)  # (total_patches, 3, input_size, input_size)
        batch_image_flags_tensor = torch.cat(batch_image_flags, dim=0)  # (total_patches,)
        
        # Step 9: Prepare final inputs
        model_inputs = {
            'input_ids': encoded.input_ids,  # (batch_size, max_seq_len)
            'attention_mask': encoded.attention_mask,  # (batch_size, max_seq_len)
            'pixel_values': batch_pixel_values,  # (total_patches, 3, input_size, input_size) - concatenated
            'image_flags': batch_image_flags_tensor  # (total_patches,) - concatenated
        }
        
        return model_inputs
