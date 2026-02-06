from ..attention_query import AttentionQuery
from ..internvl3.reward_model import InternVLBaseRewardModel
from .modeling_internvl_chat import InternVLChatModel
from typing import Optional
from ...config import InternVLArguments

class InternVL35RewardModel(InternVLChatModel, InternVLBaseRewardModel):
    """
    InternVL3.5 Reward Model that extracts the logit of "good" token from the last output position
    to use as reward signal for preference learning.
    
    Inherits from InternVL3.5 chat model and the base reward model functionality.
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
        kwargs['output_hidden_states'] = True

        # Call the InternVL3.5 chat model's forward pass
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