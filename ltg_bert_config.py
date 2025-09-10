"""
Configuration class for LtgBertForMaskedLM model.
"""

import os
import json
from transformers import PretrainedConfig
from typing import Optional


class LtgBertConfig(PretrainedConfig):
    """
    Configuration class for LtgBertForMaskedLM model.
    
    This is the configuration class to store the configuration of a [`LtgBertForMaskedLM`]. It is used to instantiate
    an LTG BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BERT
    [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 16384):
            Vocabulary size of the LTG BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LtgBertForMaskedLM`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        position_bucket_size (`int`, *optional*, defaults to 32):
            The size of the position bucket for relative position encoding. This is a specific parameter for the LTG
            BERT architecture.
        layer_norm_eps (`float`, *optional*, defaults to 1e-7):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        share_layer_weights (`bool`, *optional*, defaults to `False`):
            Whether to share weights across transformer layers (ALBERT-style). When enabled, all transformer
            layers share the same parameters, significantly reducing model size while potentially improving
            generalization for limited data scenarios.
        use_dynamic_masking (`bool`, *optional*, defaults to `False`):
            Whether to use RoBERTa-style dynamic masking. When enabled, tokens are masked differently on
            each epoch rather than using static pre-computed masks, effectively multiplying the training
            signal without requiring additional data.

    Examples:

    ```python
    >>> from ltg_bert_config import LtgBertConfig
    >>> from ltg_bert import LtgBertForMaskedLM

    >>> # Initializing an LTG BERT configuration
    >>> configuration = LtgBertConfig()

    >>> # Initializing a model from the configuration
    >>> model = LtgBertForMaskedLM(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "ltg_bert"
    
    def __init__(
        self,
        vocab_size: int = 16384,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 2048,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        position_bucket_size: int = 32,
        layer_norm_eps: float = 1e-7,
        use_cache: bool = True,
        classifier_dropout: Optional[float] = None,
        num_labels: int = 2,
        problem_type: Optional[str] = None,
        share_layer_weights: bool = False,
        use_dynamic_masking: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.position_bucket_size = position_bucket_size
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.share_layer_weights = share_layer_weights
        self.use_dynamic_masking = use_dynamic_masking

        # Validate configuration
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

    @property
    def head_size(self) -> int:
        """The size of each attention head."""
        return self.hidden_size // self.num_attention_heads

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = super().to_dict()
        return output

    def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
        """
        Save a configuration object and tokenizer files to the directory `save_directory`.
        
        This method extends the base save_pretrained to also create necessary tokenizer files
        for compatibility with transformers AutoTokenizer and copy the source files.
        """
        # Call the parent save_pretrained to save config.json
        super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)
        
        # Create tokenizer files if they don't exist
        self._create_tokenizer_files(save_directory)
        
        # Copy source files for remote code loading
        self._copy_source_files(save_directory)
    
    def _copy_source_files(self, save_directory):
        """
        Copy the necessary Python source files to the save directory for remote code loading.
        """
        import shutil
        
        # Files to copy
        source_files = ['ltg_bert_config.py', 'ltg_bert.py']
        
        for filename in source_files:
            if os.path.exists(filename):
                dest_path = os.path.join(save_directory, filename)
                shutil.copy2(filename, dest_path)
                print(f"Copied {filename} to {save_directory}")
            else:
                print(f"Warning: {filename} not found in current directory")
    
    def _create_tokenizer_files(self, save_directory):
        """
        Create the necessary tokenizer files for transformers compatibility.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Create tokenizer_config.json for PreTrainedTokenizerFast (works with custom tokenizer.json)
        tokenizer_config_path = os.path.join(save_directory, "tokenizer_config.json")
        if not os.path.exists(tokenizer_config_path):
            tokenizer_config = {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "clean_up_tokenization_spaces": True,
                "do_lower_case": False,
                "model_max_length": self.max_position_embeddings,
                "strip_accents": None,
                "tokenize_chinese_chars": True,
                "cls_token": "[CLS]",
                "mask_token": "[MASK]",
                "pad_token": "[PAD]",
                "sep_token": "[SEP]",
                "unk_token": "[UNK]",
                "added_tokens_decoder": {
                    "0": {"content": "[PAD]", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
                    "1": {"content": "[UNK]", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
                    "2": {"content": "[CLS]", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
                    "3": {"content": "[SEP]", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
                    "4": {"content": "[MASK]", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True}
                }
            }
            
            with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
                json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
        
        # Create special_tokens_map.json
        special_tokens_path = os.path.join(save_directory, "special_tokens_map.json")
        if not os.path.exists(special_tokens_path):
            special_tokens = {
                "cls_token": {"content": "[CLS]", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False},
                "mask_token": {"content": "[MASK]", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False},
                "pad_token": {"content": "[PAD]", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False},
                "sep_token": {"content": "[SEP]", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False},
                "unk_token": {"content": "[UNK]", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False}
            }
            
            with open(special_tokens_path, 'w', encoding='utf-8') as f:
                json.dump(special_tokens, f, indent=2, ensure_ascii=False)
        
        # Create vocab.txt from existing tokenizer.json if it exists
        tokenizer_json_path = os.path.join(save_directory, "tokenizer.json")
        vocab_txt_path = os.path.join(save_directory, "vocab.txt")
        
        if os.path.exists(tokenizer_json_path) and not os.path.exists(vocab_txt_path):
            self._extract_vocab_from_tokenizer_json(tokenizer_json_path, vocab_txt_path)
    
    def _extract_vocab_from_tokenizer_json(self, tokenizer_json_path, vocab_txt_path):
        """
        Extract vocabulary from tokenizer.json and create vocab.txt
        """
        try:
            with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
            
            # Extract vocabulary from tokenizer.json
            vocab = tokenizer_data.get('model', {}).get('vocab', {})
            
            if vocab:
                # Sort by token ID
                sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
                
                with open(vocab_txt_path, 'w', encoding='utf-8') as f:
                    for token, _ in sorted_vocab:
                        f.write(f"{token}\n")
                
                print(f"Created vocab.txt with {len(sorted_vocab)} tokens")
            else:
                print("Warning: Could not extract vocabulary from tokenizer.json")
                
        except Exception as e:
            print(f"Warning: Failed to create vocab.txt: {e}")
    
    @classmethod
    def register_for_auto_class(cls, auto_class="AutoConfig"):
        """
        Register this config class with a given auto class. This will only work if the auto class already exists.
        """
        super().register_for_auto_class(auto_class)


# Register the configuration with transformers auto classes
def register_ltg_bert():
    """Register LtgBertConfig and LtgBert models with transformers auto classes."""
    try:
        from transformers import AutoConfig, AutoModelForMaskedLM, AutoModelForSequenceClassification
        
        # Register the config - use more robust checking
        try:
            # Check if already registered by trying to get it
            AutoConfig.get_config_class("ltg_bert")
        except (KeyError, AttributeError):
            # Not registered yet, so register it
            AutoConfig.register("ltg_bert", LtgBertConfig)
        
        # Register the models - we need to import it here to avoid circular imports
        try:
            from . import ltg_bert
            if hasattr(ltg_bert, 'LtgBertForMaskedLM'):
                AutoModelForMaskedLM.register(LtgBertConfig, ltg_bert.LtgBertForMaskedLM)
            if hasattr(ltg_bert, 'LtgBertForSequenceClassification'):
                AutoModelForSequenceClassification.register(LtgBertConfig, ltg_bert.LtgBertForSequenceClassification)
        except ImportError:
            # Fallback for when imported as a module
            try:
                import ltg_bert
                if hasattr(ltg_bert, 'LtgBertForMaskedLM'):
                    AutoModelForMaskedLM.register(LtgBertConfig, ltg_bert.LtgBertForMaskedLM)
                if hasattr(ltg_bert, 'LtgBertForSequenceClassification'):
                    AutoModelForSequenceClassification.register(LtgBertConfig, ltg_bert.LtgBertForSequenceClassification)
            except ImportError:
                pass
                
        print("LtgBert configuration and models registered successfully")
    except Exception as e:
        print(f"Warning: Failed to register LtgBert classes: {e}")


# Auto-register when the module is imported
register_ltg_bert()
