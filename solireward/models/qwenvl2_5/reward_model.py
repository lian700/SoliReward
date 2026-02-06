from typing import Optional

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration

from ..attention_query import AttentionQuery, create_reward_head
from ...config import QwenVLArguments


class BaseQwenVLRewardModel:
    """
    Base class containing shared reward model logic for QwenVL models.
    Provides configurable sequence reduction strategies mirroring InternVL models.
    """

    def __init__(self):
        self._good_token_id: Optional[int] = None
        self._bad_token_id: Optional[int] = None
        self.tokenizer = None

        self.reduce_sequence: str = 'good_token'
        self.reward_head: Optional[nn.Module] = None
        self.reward_dropout: float = 0.0
        self.hierarchical_query_attn_layers: list[int] = []

        # Optional modules for different reduction strategies
        self.attention_query: Optional[AttentionQuery] = None
        self.attention_query_multi_layer: Optional[AttentionQuery] = None
        self.attention_query_multi_layer_reduce: Optional[AttentionQuery] = None
        self.attention_query_final_layer: Optional[AttentionQuery] = None
        self.attention_query_first_layer: Optional[AttentionQuery] = None
        self.attention_query_other_layers: Optional[nn.ModuleList] = None

    def set_tokenizer(self, tokenizer):
        """Set tokenizer and cache good/bad token IDs for efficient lookup"""
        self.tokenizer = tokenizer

        vocab = tokenizer.get_vocab()
        self._good_token_id = vocab.get("good")
        self._bad_token_id = vocab.get("bad")
        assert self._good_token_id is not None, "'good' token not found in tokenizer vocabulary."
        assert self._bad_token_id is not None, "'bad' token not found in tokenizer vocabulary."

    def initialize_sequence_reducer(self, hidden_size: int, model_specific_config: Optional[QwenVLArguments]):
        """Configure sequence reduction modules based on model-specific arguments."""
        if model_specific_config is None:
            self.reduce_sequence = 'good_token'
            self.reward_head = None
            self.reward_dropout = 0.0
            self.hierarchical_query_attn_layers = []
            self.attention_query = None
            self.attention_query_multi_layer = None
            self.attention_query_multi_layer_reduce = None
            self.attention_query_final_layer = None
            self.attention_query_first_layer = None
            self.attention_query_other_layers = None
            return

        # Reset modules before applying new configuration
        self.attention_query = None
        self.attention_query_multi_layer = None
        self.attention_query_multi_layer_reduce = None
        self.attention_query_final_layer = None
        self.attention_query_first_layer = None
        self.attention_query_other_layers = None
        self.reward_head = None

        self.reduce_sequence = getattr(model_specific_config, 'reduce_sequence', 'good_token')
        self.reward_dropout = getattr(model_specific_config, 'reward_dropout', 0.0)
        self.hierarchical_query_attn_layers = getattr(model_specific_config, 'hierarchical_query_attn_layers', [])

        if self.reduce_sequence == 'attention':
            self.attention_query = AttentionQuery(hidden_size)
            self.reward_head = create_reward_head(hidden_size, reward_dropout=self.reward_dropout)

        elif self.reduce_sequence == 'hierarchical_attention':
            num_queries = len(self.hierarchical_query_attn_layers)
            if num_queries == 0:
                raise ValueError("hierarchical_attention requires non-empty hierarchical_query_attn_layers")
            self.attention_query_multi_layer = AttentionQuery(hidden_size, num_queries=num_queries)
            self.attention_query_multi_layer_reduce = AttentionQuery(hidden_size)
            self.attention_query_final_layer = AttentionQuery(hidden_size)
            self.reward_head = create_reward_head(hidden_size, reward_dropout=self.reward_dropout)

        elif self.reduce_sequence == 'hierarchical_attention-shared':
            num_queries = len(self.hierarchical_query_attn_layers)
            if num_queries == 0:
                raise ValueError("hierarchical_attention-shared requires non-empty hierarchical_query_attn_layers")
            self.attention_query_multi_layer = AttentionQuery(hidden_size, num_queries=num_queries)
            self.attention_query_multi_layer_reduce = AttentionQuery(hidden_size)
            self.attention_query_final_layer = AttentionQuery(hidden_size)
            self.reward_head = create_reward_head(hidden_size, reward_dropout=self.reward_dropout)

        elif self.reduce_sequence == 'hierarchical_attention-v2':
            num_queries = len(self.hierarchical_query_attn_layers)
            if num_queries == 0:
                raise ValueError("hierarchical_attention-v2 requires non-empty hierarchical_query_attn_layers")
            self.attention_query_multi_layer = AttentionQuery(hidden_size, num_queries=num_queries)
            self.attention_query_multi_layer_reduce = AttentionQuery(hidden_size)
            self.reward_head = create_reward_head(hidden_size, reward_dropout=self.reward_dropout)

        elif self.reduce_sequence == 'progressive_hierarchical_attention':
            if len(self.hierarchical_query_attn_layers) < 1:
                raise ValueError("progressive_hierarchical_attention requires at least one hierarchical_query_attn_layer")
            self.attention_query_first_layer = AttentionQuery(hidden_size)
            other_layers = len(self.hierarchical_query_attn_layers) - 1
            self.attention_query_other_layers = nn.ModuleList([
                nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True) for _ in range(other_layers)
            ])
            self.attention_query_final_layer = AttentionQuery(hidden_size)
            self.reward_head = create_reward_head(hidden_size, reward_dropout=self.reward_dropout)

        elif self.reduce_sequence == 'last_token_hidden_state':
            self.reward_head = create_reward_head(hidden_size, reward_dropout=self.reward_dropout)

        elif self.reduce_sequence in {'maxpool', 'meanpool'}:
            self.reward_head = create_reward_head(hidden_size, reward_dropout=self.reward_dropout)

        elif self.reduce_sequence == 'good_token':
            self.reward_head = None

        else:
            raise ValueError(f"Unsupported reduce_sequence: {self.reduce_sequence}")

    def process_outputs_for_reward(self, outputs, return_dict: bool = True):
        """Process model outputs to extract reward scores and supporting tensors."""
        hidden_states = getattr(outputs, 'hidden_states', None)

        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

        last_token_logits = logits[:, -1, :]
        good_logits = last_token_logits[:, self._good_token_id]
        bad_logits = last_token_logits[:, self._bad_token_id]

        if self.reduce_sequence == 'good_token':
            reward_scores = good_logits

        elif self.reduce_sequence == 'attention':
            if hidden_states is None:
                raise ValueError("Hidden states are required when reduce_sequence='attention'. Set output_hidden_states=True.")
            if self.attention_query is None or self.reward_head is None:
                raise RuntimeError("Attention modules are not initialized for reduce_sequence='attention'.")
            last_hidden_states = hidden_states[-1]
            attn_output = self.attention_query([last_hidden_states]).squeeze(1)
            reward_scores = self.reward_head(attn_output).squeeze(-1)

        elif self.reduce_sequence == 'hierarchical_attention':
            if hidden_states is None:
                raise ValueError("Hidden states are required when reduce_sequence='hierarchical_attention'. Set output_hidden_states=True.")
            if self.attention_query_multi_layer is None or self.attention_query_multi_layer_reduce is None or self.attention_query_final_layer is None or self.reward_head is None:
                raise RuntimeError("Hierarchical attention modules are not initialized correctly.")
            selected_hidden_states = [hidden_states[i] for i in self.hierarchical_query_attn_layers]
            attn_output = self.attention_query_multi_layer(selected_hidden_states)
            attn_output = self.attention_query_multi_layer_reduce([attn_output]).squeeze(1)
            attn_output_final = self.attention_query_final_layer([hidden_states[-1]]).squeeze(1)
            reward_scores = self.reward_head(attn_output + attn_output_final).squeeze(-1)

        elif self.reduce_sequence == 'hierarchical_attention-shared':
            if hidden_states is None:
                raise ValueError("Hidden states are required when reduce_sequence='hierarchical_attention-shared'. Set output_hidden_states=True.")
            if self.attention_query_multi_layer is None or self.attention_query_multi_layer_reduce is None or self.attention_query_final_layer is None or self.reward_head is None:
                raise RuntimeError("Hierarchical shared attention modules are not initialized correctly.")
            shared_query = self.attention_query_final_layer.query.weight[0].unsqueeze(0).unsqueeze(0)
            selected_hidden_states = [hidden_states[i] for i in self.hierarchical_query_attn_layers]
            attn_outputs = []
            for idx, seq in enumerate(selected_hidden_states):
                batch_size = seq.size(0)
                query = shared_query.repeat(batch_size, 1, 1)
                attn_output, _ = self.attention_query_multi_layer.attention[idx](query, seq, seq)
                attn_outputs.append(attn_output)
            attn_outputs = torch.cat(attn_outputs, dim=1)
            attn_output = self.attention_query_multi_layer_reduce([attn_outputs]).squeeze(1)
            attn_output_final = self.attention_query_final_layer([hidden_states[-1]]).squeeze(1)
            reward_scores = self.reward_head(attn_output + attn_output_final).squeeze(-1)

        elif self.reduce_sequence == 'hierarchical_attention-v2':
            if hidden_states is None:
                raise ValueError("Hidden states are required when reduce_sequence='hierarchical_attention-v2'. Set output_hidden_states=True.")
            if self.attention_query_multi_layer is None or self.attention_query_multi_layer_reduce is None or self.reward_head is None:
                raise RuntimeError("Hierarchical attention v2 modules are not initialized correctly.")
            selected_hidden_states = [hidden_states[i] for i in self.hierarchical_query_attn_layers]
            attn_output = self.attention_query_multi_layer(selected_hidden_states)
            attn_output = self.attention_query_multi_layer_reduce([attn_output]).squeeze(1)
            reward_scores = self.reward_head(attn_output).squeeze(-1)

        elif self.reduce_sequence == 'progressive_hierarchical_attention':
            if hidden_states is None:
                raise ValueError("Hidden states are required when reduce_sequence='progressive_hierarchical_attention'. Set output_hidden_states=True.")
            if self.attention_query_first_layer is None or self.attention_query_other_layers is None or self.attention_query_final_layer is None or self.reward_head is None:
                raise RuntimeError("Progressive hierarchical attention modules are not initialized correctly.")
            query = self.attention_query_first_layer([hidden_states[self.hierarchical_query_attn_layers[0]]])
            for idx, layer_idx in enumerate(self.hierarchical_query_attn_layers[1:]):
                key_value = hidden_states[layer_idx]
                query, _ = self.attention_query_other_layers[idx](query, key_value, key_value)
            query = query.squeeze(1)
            attn_output_final = self.attention_query_final_layer([hidden_states[-1]]).squeeze(1)
            reward_scores = self.reward_head(query + attn_output_final).squeeze(-1)

        elif self.reduce_sequence == 'maxpool':
            if hidden_states is None:
                raise ValueError("Hidden states are required when reduce_sequence='maxpool'. Set output_hidden_states=True.")
            if self.reward_head is None:
                raise RuntimeError("Reward head is not initialized for reduce_sequence='maxpool'.")
            pooled = hidden_states[-1].max(dim=1).values
            reward_scores = self.reward_head(pooled).squeeze(-1)

        elif self.reduce_sequence == 'meanpool':
            if hidden_states is None:
                raise ValueError("Hidden states are required when reduce_sequence='meanpool'. Set output_hidden_states=True.")
            if self.reward_head is None:
                raise RuntimeError("Reward head is not initialized for reduce_sequence='meanpool'.")
            pooled = hidden_states[-1].mean(dim=1)
            reward_scores = self.reward_head(pooled).squeeze(-1)

        elif self.reduce_sequence == 'last_token_hidden_state':
            if hidden_states is None:
                raise ValueError("Hidden states are required when reduce_sequence='last_token_hidden_state'. Set output_hidden_states=True.")
            if self.reward_head is None:
                raise RuntimeError("Reward head is not initialized for reduce_sequence='last_token_hidden_state'.")
            last_hidden_states = hidden_states[-1]
            last_token_state = last_hidden_states[:, -1, :]
            reward_scores = self.reward_head(last_token_state).squeeze(-1)

        else:
            raise ValueError(f"Unsupported reduce_sequence: {self.reduce_sequence}")

        result = {
            'logits': logits,
            'reward_scores': reward_scores,
            'good_logits': good_logits,
            'bad_logits': bad_logits,
        }

        loss = getattr(outputs, 'loss', None)
        if loss is not None:
            result['loss'] = loss

        past_key_values = getattr(outputs, 'past_key_values', None)
        if past_key_values is not None:
            result['past_key_values'] = past_key_values

        hidden_states_attr = getattr(outputs, 'hidden_states', None)
        if hidden_states_attr is not None:
            result['hidden_states'] = hidden_states_attr

        attentions = getattr(outputs, 'attentions', None)
        if attentions is not None:
            result['attentions'] = attentions

        if not return_dict:
            output_tuple = (result['logits'], result['reward_scores'])
            if 'loss' in result:
                output_tuple = (result['loss'],) + output_tuple
            return output_tuple

        return result


class Qwen2VLRewardModel(Qwen2VLForConditionalGeneration, BaseQwenVLRewardModel):
    """
    Qwen2VL Reward Model that extracts the logit of "good" token from the last output position
    to use as reward signal for preference learning.
    """
    
    def __init__(self, config, model_specific_config: Optional[QwenVLArguments] = None):
        super().__init__(config)
        BaseQwenVLRewardModel.__init__(self)
        # Store model-specific configuration
        self.model_specific_config = model_specific_config

        hidden_size = getattr(config, 'hidden_size', None)
        if hidden_size is None:
            raise ValueError("Qwen2VL config must provide hidden_size for reward modeling.")
        self.initialize_sequence_reducer(hidden_size, self.model_specific_config)

    def _init_weights(self, module):  # type: ignore[override]
        print("init reward model weights: {}".format(module.__class__.__name__))
        super()._init_weights(module)
        if isinstance(module, AttentionQuery):
            # Only initialize if not already done (ZeRO-3 safe)
            if not getattr(module, '_initialized', False):
                module.reset_parameters()
    
    def forward(self, *args, **kwargs):  # type: ignore[override]
        """
        Forward pass that returns both the standard model outputs and reward scores.
        
        Returns:
            Dict containing:
            - logits: Standard model logits
            - reward_scores: Logit of "good" token at last position for each sample
            - good_logits: Logit of "good" token  
            - bad_logits: Logit of "bad" token
            - Other standard outputs (loss, past_key_values, etc.)
        """
        return_dict = kwargs.get('return_dict', True)
        kwargs['output_hidden_states'] = True

        outputs = super().forward(*args, **kwargs)
        
        # Step 2: Process outputs to add reward scores
        return self.process_outputs_for_reward(outputs, return_dict)


class Qwen25VLRewardModel(Qwen2_5_VLForConditionalGeneration, BaseQwenVLRewardModel):
    """
    Qwen2.5VL Reward Model that extracts the logit of "good" token from the last output position
    to use as reward signal for preference learning.
    """
    
    def __init__(self, config, model_specific_config: Optional[QwenVLArguments] = None):
        super().__init__(config)
        BaseQwenVLRewardModel.__init__(self)
        # Store model-specific configuration
        self.model_specific_config = model_specific_config

        hidden_size = getattr(config, 'hidden_size', None)
        if hidden_size is None:
            raise ValueError("Qwen2.5VL config must provide hidden_size for reward modeling.")
        self.initialize_sequence_reducer(hidden_size, self.model_specific_config)

    def _init_weights(self, module):  # type: ignore[override]
        print("init reward model weights: {}".format(module.__class__.__name__))
        super()._init_weights(module)
        if isinstance(module, AttentionQuery):
            # Only initialize if not already done (ZeRO-3 safe)
            if not getattr(module, '_initialized', False):
                module.reset_parameters()
    
    def forward(self, *args, **kwargs):  # type: ignore[override]
        """
        Forward pass that returns both the standard model outputs and reward scores.
        
        Returns:
            Dict containing:
            - logits: Standard model logits
            - reward_scores: Logit of "good" token at last position for each sample
            - good_logits: Logit of "good" token  
            - bad_logits: Logit of "bad" token
            - Other standard outputs (loss, past_key_values, etc.)
        """
        return_dict = kwargs.get('return_dict', True)
        kwargs['output_hidden_states'] = True

        outputs = super().forward(*args, **kwargs)
        
        # Step 2: Process outputs to add reward scores
        return self.process_outputs_for_reward(outputs, return_dict)
