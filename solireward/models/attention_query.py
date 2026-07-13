
import torch
import torch.nn as nn
from typing import Optional, Union

from einops import repeat


def normalize_attention_mask(
    attention_mask: Optional[torch.Tensor],
    sequence: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Validate and normalize a token attention mask to a boolean tensor."""
    if attention_mask is None:
        return None
    if attention_mask.dim() != 2:
        raise ValueError(
            f"attention_mask must have shape (batch_size, seq_len), got {tuple(attention_mask.shape)}"
        )
    if attention_mask.shape != sequence.shape[:2]:
        raise ValueError(
            "attention_mask shape must match the sequence batch and length dimensions: "
            f"got {tuple(attention_mask.shape)} and {tuple(sequence.shape[:2])}"
        )

    mask = attention_mask.to(device=sequence.device, dtype=torch.bool)
    if not mask.any(dim=1).all():
        raise ValueError("attention_mask contains a sample with no non-PAD tokens")
    return mask


def select_last_non_padding(
    sequence: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Select each sample's last real token, independent of padding side."""
    mask = normalize_attention_mask(attention_mask, sequence)
    if mask is None:
        return sequence[:, -1, :]

    positions = torch.arange(sequence.size(1), device=sequence.device).unsqueeze(0)
    last_indices = positions.masked_fill(~mask, -1).max(dim=1).values
    batch_indices = torch.arange(sequence.size(0), device=sequence.device)
    return sequence[batch_indices, last_indices]


def masked_sequence_pool(
    sequence: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    mode: str,
) -> torch.Tensor:
    """Mean- or max-pool a sequence while excluding PAD positions."""
    mask = normalize_attention_mask(attention_mask, sequence)
    if mask is None:
        return sequence.mean(dim=1) if mode == "mean" else sequence.max(dim=1).values

    if mode == "mean":
        weights = mask.unsqueeze(-1).to(sequence.dtype)
        return (sequence * weights).sum(dim=1) / weights.sum(dim=1)
    if mode == "max":
        return sequence.masked_fill(~mask.unsqueeze(-1), torch.finfo(sequence.dtype).min).max(dim=1).values
    raise ValueError(f"Unsupported pooling mode: {mode}")

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
    
    def forward(
        self,
        seq_list: list[torch.Tensor],
        attention_mask: Optional[Union[torch.Tensor, list[Optional[torch.Tensor]]]] = None,
    ) -> torch.Tensor:
        """
        Forward pass to compute attention-based sequence reduction.
        Args:
            seq_list: List of input sequences, each of shape (batch_size, seq_len, hidden_size)
            attention_mask: Token mask shared by all sequences, or one mask per sequence.
        Returns:
            Tensor of shape (batch_size, num_queries, hidden_size) after attention-based reduction
        """
        num_query = len(seq_list)
        assert num_query == self.num_queries, f"Expected {self.num_queries} sequences, but got {num_query}"
        if isinstance(attention_mask, list) and len(attention_mask) != num_query:
            raise ValueError(
                f"Expected {num_query} attention masks, but got {len(attention_mask)}"
            )
        outputs = []
        for i, seq in enumerate(seq_list):
            batch_size = seq.size(0)
            query = repeat(self.query.weight[i], 'hidden_size -> batch_size 1 hidden_size', batch_size=batch_size)  # (batch_size, 1, hidden_size)
            seq_mask = attention_mask[i] if isinstance(attention_mask, list) else attention_mask
            seq_mask = normalize_attention_mask(seq_mask, seq)
            key_padding_mask = None if seq_mask is None else ~seq_mask
            attn_output, _ = self.attention[i](
                query,
                seq,
                seq,
                key_padding_mask=key_padding_mask,
            )  # (batch_size, 1, hidden_size)
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
