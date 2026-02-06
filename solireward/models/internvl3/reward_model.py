from typing import Optional
from .image_placeholder import IMG_CONTEXT_TOKEN
from .modeling_internvl_chat import InternVLChatModel
from ...config import InternVLArguments
from ..attention_query import AttentionQuery, create_reward_head
import torch
import torch.nn as nn

class InternVLBaseRewardModel:
    """
    Base reward model that provides core reward scoring functionality.
    Can be mixed with different chat models.
    """
    
    def __init__(self):
        self._good_token_id = None
        self._bad_token_id = None
        self.tokenizer = None
        self.img_context_token_id = None
        self.reduce_sequence: str = 'good_token'
        self.reward_head: Optional[nn.Module] = None
        self.hierarchical_query_attn_layers: list[int] = []
        self.attention_query: Optional[AttentionQuery] = None
        self.attention_query_multi_layer: Optional[AttentionQuery] = None
        self.attention_query_multi_layer_reduce: Optional[AttentionQuery] = None
        self.attention_query_final_layer: Optional[AttentionQuery] = None
        self.attention_query_first_layer: Optional[AttentionQuery] = None
        self.attention_query_other_layers: Optional[nn.ModuleList] = None
        self.reward_dropout: float = 0.0
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer and cache good/bad token IDs for efficient lookup"""
        self.tokenizer = tokenizer
        
        # Method 1: Direct vocabulary lookup
        vocab = tokenizer.get_vocab()
        self._good_token_id = vocab.get("good")
        self._bad_token_id = vocab.get("bad")
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    def configure_sequence_reducer(self, hidden_size: int, model_specific_config: Optional[InternVLArguments]):
        """Configure sequence reduction modules according to model-specific settings."""
        if model_specific_config is None:
            # Default to using good-token logits if no configuration is provided
            self.reduce_sequence = 'good_token'
            self.reward_head = None
            self.hierarchical_query_attn_layers = []
            self.reward_dropout = 0.0
            return

        # Reset optional modules before configuring
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
    
    def process_reward_outputs(self, outputs):
        """
        Process model outputs to extract reward scores.
        
        Args:
            outputs: Model outputs containing logits
            
        Returns:
            Dict containing reward-related outputs
        """
        hidden_states = getattr(outputs, 'hidden_states', None)

        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            # If outputs is a tuple, logits should be the first element
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Extract last token logits (batch_size, vocab_size)
        last_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
        
        # Extract logits for "good" and "bad" tokens (without softmax)
        good_logits = last_token_logits[:, self._good_token_id]  # Shape: (batch_size,)
        bad_logits = last_token_logits[:, self._bad_token_id]  # Shape: (batch_size,)

        # how to reduce sequence of logits to a single reward score
        if self.reduce_sequence == 'attention':
            if hidden_states is None:
                raise ValueError("Hidden states are required when reduce_sequence='attention'. Ensure output_hidden_states=True.")
            if self.attention_query is None or self.reward_head is None:
                raise RuntimeError("Attention modules are not initialized for reduce_sequence='attention'.")
            last_hidden_states = hidden_states[-1]  # (batch_size, seq_len, hidden_size)
            attn_output = self.attention_query([last_hidden_states])  # (batch_size, num_queries=1, hidden_size)
            attn_output = attn_output.squeeze(1)  # (batch_size, hidden_size)
            reward_scores = self.reward_head(attn_output).squeeze(-1)  # (batch_size,)

        elif self.reduce_sequence == 'hierarchical_attention':
            if hidden_states is None:
                raise ValueError("Hidden states are required when reduce_sequence='hierarchical_attention'. Ensure output_hidden_states=True.")
            if self.attention_query_multi_layer is None or self.attention_query_multi_layer_reduce is None or self.attention_query_final_layer is None or self.reward_head is None:
                raise RuntimeError("Hierarchical attention modules are not initialized correctly.")
            selected_hidden_states = [hidden_states[i] for i in self.hierarchical_query_attn_layers]
            attn_output = self.attention_query_multi_layer(selected_hidden_states)  # (batch_size, num_queries, hidden_size)
            attn_output = self.attention_query_multi_layer_reduce([attn_output])  # (batch_size, 1, hidden_size)
            attn_output = attn_output.squeeze(1)  # (batch_size, hidden_size)

            last_hidden_states = hidden_states[-1]  # (batch_size, seq_len, hidden_size)
            attn_output_final_layer = self.attention_query_final_layer([last_hidden_states])
            attn_output_final_layer = attn_output_final_layer.squeeze(1)  # (batch_size, hidden_size)
            attn_output = attn_output + attn_output_final_layer  # (batch_size, hidden_size)
            reward_scores = self.reward_head(attn_output).squeeze(-1)  # (batch_size,)
        
        elif self.reduce_sequence == 'hierarchical_attention-shared':
            if hidden_states is None:
                raise ValueError("Hidden states are required when reduce_sequence='hierarchical_attention-shared'. Ensure output_hidden_states=True.")
            if self.attention_query_multi_layer is None or self.attention_query_multi_layer_reduce is None or self.attention_query_final_layer is None or self.reward_head is None:
                raise RuntimeError("Hierarchical shared attention modules are not initialized correctly.")
            shared_query = self.attention_query_final_layer.query.weight[0]  # (hidden_size,)
            shared_query = shared_query.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)

            selected_hidden_states = [hidden_states[i] for i in self.hierarchical_query_attn_layers]
            
            attn_outputs = []
            for idx, seq in enumerate(selected_hidden_states):
                batch_size = seq.size(0)
                query = shared_query.repeat(batch_size, 1, 1)  # (batch_size, 1, hidden_size)
                attn_output, _ = self.attention_query_multi_layer.attention[idx](query, seq, seq)  # (batch_size, 1, hidden_size)
                attn_outputs.append(attn_output)
            attn_outputs = torch.cat(attn_outputs, dim=1)  # (batch_size, num_queries, hidden_size)
            attn_output = self.attention_query_multi_layer_reduce([attn_outputs])  # (batch_size, 1, hidden_size)
            attn_output = attn_output.squeeze(1)  # (batch_size, hidden_size)

            last_hidden_states = hidden_states[-1]  # (batch_size, seq_len, hidden_size)
            attn_output_final_layer = self.attention_query_final_layer([last_hidden_states])
            attn_output_final_layer = attn_output_final_layer.squeeze(1)  # (batch_size, hidden_size)
            attn_output = attn_output + attn_output_final_layer  # (batch_size, hidden_size)
            reward_scores = self.reward_head(attn_output).squeeze(-1)  # (batch_size,)
        
        elif self.reduce_sequence == 'hierarchical_attention-v2':
            if hidden_states is None:
                raise ValueError("Hidden states are required when reduce_sequence='hierarchical_attention-v2'. Ensure output_hidden_states=True.")
            if self.attention_query_multi_layer is None or self.attention_query_multi_layer_reduce is None or self.reward_head is None:
                raise RuntimeError("Hierarchical attention v2 modules are not initialized correctly.")
            selected_hidden_states = [hidden_states[i] for i in self.hierarchical_query_attn_layers]
            attn_output = self.attention_query_multi_layer(selected_hidden_states)  # (batch_size, num_queries, hidden_size)
            attn_output = self.attention_query_multi_layer_reduce([attn_output])  # (batch_size, 1, hidden_size)
            attn_output = attn_output.squeeze(1)  # (batch_size, hidden_size)
            reward_scores = self.reward_head(attn_output).squeeze(-1)  # (batch_size,)
        
        elif self.reduce_sequence == 'progressive_hierarchical_attention':
            if hidden_states is None:
                raise ValueError("Hidden states are required when reduce_sequence='progressive_hierarchical_attention'. Ensure output_hidden_states=True.")
            if self.attention_query_first_layer is None or self.attention_query_other_layers is None or self.attention_query_final_layer is None or self.reward_head is None:
                raise RuntimeError("Progressive hierarchical attention modules are not initialized correctly.")
            query = self.attention_query_first_layer([hidden_states[self.hierarchical_query_attn_layers[0]]])  # (batch_size, 1, hidden_size)
            for layer_idx in self.hierarchical_query_attn_layers[1:]:
                key_value = hidden_states[layer_idx]  # (batch_size, seq_len, hidden_size)
                query, _ = self.attention_query_other_layers[self.hierarchical_query_attn_layers.index(layer_idx)-1](query, key_value, key_value)  # (batch_size, 1, hidden_size)
            query = query.squeeze(1)  # (batch_size, hidden_size)
            attn_output_final_layer = self.attention_query_final_layer([hidden_states[-1]])
            attn_output_final_layer = attn_output_final_layer.squeeze(1)  # (batch_size, hidden_size)
            query = query + attn_output_final_layer  # (batch_size, hidden_size)
            reward_scores = self.reward_head(query).squeeze(-1)  # (batch_size,)

        elif self.reduce_sequence == 'maxpool':
            if hidden_states is None:
                raise ValueError("Hidden states are required when reduce_sequence='maxpool'. Ensure output_hidden_states=True.")
            if self.reward_head is None:
                raise RuntimeError("Reward head is not initialized for reduce_sequence='maxpool'.")
            last_hidden_states = hidden_states[-1]
            pooled = last_hidden_states.max(dim=1).values
            reward_scores = self.reward_head(pooled).squeeze(-1)

        elif self.reduce_sequence == 'last_token_hidden_state':
            if hidden_states is None:
                raise ValueError("Hidden states are required when reduce_sequence='last_token_hidden_state'. Ensure output_hidden_states=True.")
            if self.reward_head is None:
                raise RuntimeError("Reward head is not initialized for reduce_sequence='last_token_hidden_state'.")
            last_hidden_states = hidden_states[-1]
            last_token_state = last_hidden_states[:, -1, :]
            reward_scores = self.reward_head(last_token_state).squeeze(-1)
        elif self.reduce_sequence == 'meanpool':
            if hidden_states is None:
                raise ValueError("Hidden states are required when reduce_sequence='meanpool'. Ensure output_hidden_states=True.")
            if self.reward_head is None:
                raise RuntimeError("Reward head is not initialized for reduce_sequence='meanpool'.")
            last_hidden_states = hidden_states[-1]
            pooled = last_hidden_states.mean(dim=1)
            reward_scores = self.reward_head(pooled).squeeze(-1)
            

        elif self.reduce_sequence == 'good_token':
            # Use "good" logits as reward score
            reward_scores = good_logits
        
        else:
            raise ValueError(f"Unsupported reduce_sequence: {self.reduce_sequence}")
        
        result = {
            'logits': logits,
            'reward_scores': reward_scores,
            'good_logits': good_logits,
            'bad_logits': bad_logits,
        }
        
        # Include other outputs from original outputs if available
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
            
        return result

class InternVL3RewardModel(InternVLChatModel, InternVLBaseRewardModel):
    """
    InternVL Reward Model that extracts the logit of "good" token from the last output position
    to use as reward signal for preference learning.
    """
    
    def __init__(self, config, model_specific_config: Optional[InternVLArguments] = None):
        InternVLChatModel.__init__(self, config)
        InternVLBaseRewardModel.__init__(self)
        # Store model-specific configuration
        self.model_specific_config = model_specific_config

        hidden_size = self.hidden_size if hasattr(self, 'hidden_size') else config.hidden_size
        assert isinstance(hidden_size, int), "hidden_size should be an integer"
        self.configure_sequence_reducer(hidden_size, self.model_specific_config)
    
    def _init_weights(self, module):
        print("init reward model weights: {}".format(module.__class__.__name__))
        super()._init_weights(module)
        if isinstance(module, AttentionQuery):
            # Only initialize if not already done (ZeRO-3 safe)
            if not getattr(module, '_initialized', False):
                module.reset_parameters()
    
    def forward(self, *args, **kwargs):
        """
        Forward pass that returns both the standard model outputs and reward scores.
        
        Returns:
            Dict containing:
            - logits: Standard model logits
            - reward_scores: Logit of "good" token at last position for each sample
            - good_logits: Logit of "good" token  
            - bad_logits: Logit of "bad" token
        """
        return_dict = kwargs.get('return_dict', True)
        kwargs['output_hidden_states'] = True  # Ensure hidden states are returned for reward processing
        
        # Call the chat model's forward pass
        outputs = InternVLChatModel.forward(self, *args, **kwargs)
        
        # Process outputs through base reward model
        result = self.process_reward_outputs(outputs)
        
        if not return_dict:
            # Return as tuple if return_dict is False
            output_tuple = (result['logits'], result['reward_scores'])
            if 'loss' in result:
                output_tuple = (result['loss'],) + output_tuple
            return output_tuple
            
        return result