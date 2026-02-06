
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from einops import rearrange, repeat

class AttentionQuery(nn.Module):
    """
    Learnable query vector class for attention-based sequence reduction.
    """
    
    def __init__(self, hidden_size: int, num_queries: int = 1, num_heads: int = 8):
        """
        Initialize the attention query.
        
        Args:
            hidden_size: Dimension of the hidden features
            num_queries: Number of query vectors (default: 1)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        
        # Initialize learnable query parameters
        self.query = nn.Embedding(num_queries, hidden_size)
        self.attention = nn.ModuleList([nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True) for _ in range(num_queries)])
        # Defer initialization to avoid ZeRO-3 issues
        self._initialized = False

    
    def reset_parameters(self):
        """Reset parameters with DeepSpeed ZeRO-3 compatibility."""
        # Check if deepspeed is available and ZeRO-3 is enabled
        try:
            import deepspeed
            if hasattr(self.query.weight, 'ds_id'):
                # ZeRO-3 is active, use GatheredParameters context
                with deepspeed.zero.GatheredParameters([self.query.weight], modifier_rank=0):
                    if deepspeed.comm.get_rank() == 0:
                        self.query.reset_parameters()
                # Synchronize the initialized parameters across all ranks
                for attn in self.attention:
                    for name, param in attn.named_parameters():
                        if hasattr(param, 'ds_id'):
                            with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                                if deepspeed.comm.get_rank() == 0:
                                    # Let the default initialization happen on rank 0
                                    pass
            else:
                # Not using ZeRO-3, standard initialization
                self.query.reset_parameters()
                for attn in self.attention:
                    attn._reset_parameters()
        except (ImportError, AttributeError):
            # DeepSpeed not available or not using ZeRO-3
            self.query.reset_parameters()
            for attn in self.attention:
                attn._reset_parameters()
        
        self._initialized = True
    
    def forward(self, seq_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to compute attention-based sequence reduction.
        Args:
            seq_list: List of input sequences, each of shape (batch_size, seq_len, hidden_size)
        Returns:
            Tensor of shape (batch_size, num_queries, hidden_size) after attention-based reduction
        """
        num_query = len(seq_list)
        assert num_query == self.num_queries, f"Expected {self.num_queries} sequences, but got {num_query}"
        outputs = []
        for i, seq in enumerate(seq_list):
            batch_size = seq.size(0)
            query = repeat(self.query.weight[i], 'hidden_size -> batch_size 1 hidden_size', batch_size=batch_size)  # (batch_size, 1, hidden_size)
            attn_output, _ = self.attention[i](query, seq, seq)  # (batch_size, 1, hidden_size)
            outputs.append(attn_output)
        outputs = torch.cat(outputs, dim=1)  # (batch_size, num_queries, hidden_size)
        return outputs  # (batch_size, num_queries, hidden_size)


def create_reward_head(hidden_size: int, reward_dropout: float = 0.0) -> nn.Module:
    """Create 2-layer MLP for reward prediction."""
    reward_head = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),  # SiLU activation as requested
        nn.Dropout(reward_dropout),
        nn.Linear(hidden_size, 1, bias=False)  # Output single reward score without bias
    )
    
    return reward_head