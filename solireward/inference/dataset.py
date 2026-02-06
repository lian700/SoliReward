"""
Inference Dataset Module

This module provides dataset classes for reward model inference.
"""

from typing import Any, Dict, List

from torch.utils.data import Dataset


class VideoInferenceDataset(Dataset):
    """
    Dataset class for video inference with message format.
    
    This dataset wraps a list of messages in the format expected by the
    reward model data collators. Each item is converted to a fake pair
    format for compatibility with the existing data collation infrastructure.
    
    Attributes:
        messages_list: List of message lists in OpenAI-compatible format
    """
    
    def __init__(self, messages_list: List[List[Dict[str, Any]]]):
        """
        Initialize dataset with list of messages.
        
        Args:
            messages_list: List of message lists in OpenAI format, where each
                          message list represents a conversation with role
                          and content fields.
        """
        self.messages_list = messages_list
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.messages_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item as a fake pair for the data collator.
        
        The data collators expect paired data (win/lose format) for training,
        so we create a fake pair where both win and lose are the same message.
        This allows us to reuse the existing data collation infrastructure
        for inference.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dict with 'win', 'lose', and 'meta' keys for compatibility 
            with data collator
        """
        messages = self.messages_list[idx]
        return {
            'win': messages,
            'lose': messages,
            'meta': {
                'win': {'quality': 1.0},
                'lose': {'quality': 0.0}
            }
        }
