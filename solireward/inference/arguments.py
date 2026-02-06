"""
Inference Arguments Configuration

This module defines the configuration dataclass for reward model inference.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceArguments:
    """
    Arguments for inference configuration.
    
    Input file format should be JSON with the following structure:
    
    Standard format:
    [
        {"video_path": "/path/to/video1.mp4"},
        {"video_path": "/path/to/video2.mp4"}
    ]
    
    Alternative format (will use input.video_local_path):
    [
        {
            "input.model": "veo3",
            "input.prompt": "A bicycle speeds along...",
            "input.video_local_path": "/path/to/video.mp4",
            "input.video_url": "http://...",
            "output.deformity": "Pass",
            "output.physics": "Pass",
            "output.text_alignment": "Pass"
        }
    ]
    
    For text_alignment task, input should include 'prompt' or 'input.prompt' field.
    
    Attributes:
        model_name_or_path: Path to the trained reward model checkpoint
        input_file: Path to input JSON file containing test data
        output_file: Path to save inference results
        batch_size: Number of samples to process in each batch
        device: Device to run inference on ('cuda' or 'cpu')
        dtype: Data type for model ('bf16', 'fp16', or 'fp32')
        reward_model_task_type: Task type ('text_alignment' or 'phy_deform')
        system_prompt: Custom system prompt (optional, uses default based on task)
        user_prompt: Custom user prompt template (optional, uses default based on task)
        max_samples: Maximum number of samples to process (-1 for all)
        num_workers: Number of DataLoader workers for parallel video processing
        use_dataloader: Whether to use DataLoader for acceleration
    """
    
    model_name_or_path: str
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    batch_size: int = 4
    device: str = "cuda"
    dtype: str = "bf16"
    reward_model_task_type: str = "phy_deform"  # Options: "text_alignment" or "phy_deform"
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    max_samples: int = -1
    num_workers: int = 8
    use_dataloader: bool = True


# Default prompts for different task types
DEFAULT_PROMPTS = {
    "text_alignment": {
        "system_prompt": (
            "你是一个专业的视频质量评估专家。请根据提供的视频内容，判断是否同时满足以下所有问题的合格标准：\n\n"
            "1. 视频是否符合prompt的语义？\n\n"
            "回答要求：\n"
            "- 只有当所有问题的答案都是\"合格\"时，才输出：good\n"
            "- 如果任何一个问题的答案是\"部分合格\"或\"不合格\"，则输出：bad\n"
            "- 不要输出任何其他内容\n"
            "- 答案要准确、客观\n"
        ),
        "user_prompt": "生成视频的文本提示词是: {prompt}。根据该提示词生成的视频为："
    },
    "phy_deform": {
        "system_prompt": (
            "你是一个专业的视频质量评估专家。请根据提供的视频内容，判断是否同时满足以下所有问题的合格标准：\n\n"
            "1. 物理规律是否合格？\n"
            "2. 是否存在人物或动物畸形？\n\n"
            "回答要求：\n"
            "- 只有当所有问题的答案都是\"合格\"时，才输出：good\n"
            "- 如果任何一个问题的答案是\"部分合格\"或\"不合格\"，则输出：bad\n"
            "- 不要输出任何其他内容\n"
            "- 答案要准确、客观\n"
        ),
        "user_prompt": "请评估以下视频："
    }
}


def get_default_prompts(task_type: str) -> dict:
    """
    Get default prompts for a given task type.
    
    Args:
        task_type: The task type ('text_alignment' or 'phy_deform')
        
    Returns:
        Dict containing 'system_prompt' and 'user_prompt' keys
        
    Raises:
        ValueError: If task_type is not supported
    """
    if task_type not in DEFAULT_PROMPTS:
        raise ValueError(
            f"Invalid task type: {task_type}. "
            f"Must be one of: {list(DEFAULT_PROMPTS.keys())}"
        )
    return DEFAULT_PROMPTS[task_type]
