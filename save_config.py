#!/usr/bin/env python3
"""
Save a complete LtgBertConfig configuration file
"""

from config import BertConfig
from ltg_bert_config import LtgBertConfig
import transformers

def save_ltg_bert_config(custom_config_path, output_path):
    """Save a complete LtgBertConfig configuration"""
    # Load custom config
    custom_config = BertConfig.from_json_file(custom_config_path)
    
    # Create LtgBertConfig
    transformers_config = LtgBertConfig(
        vocab_size=custom_config.vocab_size,
        hidden_size=custom_config.hidden_size,
        num_hidden_layers=custom_config.num_hidden_layers,
        num_attention_heads=custom_config.num_attention_heads,
        intermediate_size=custom_config.intermediate_size,
        max_position_embeddings=custom_config.max_position_embeddings,
        attention_probs_dropout_prob=custom_config.attention_probs_dropout_prob,
        hidden_dropout_prob=custom_config.hidden_dropout_prob,
        layer_norm_eps=custom_config.layer_norm_eps,
    )
    
    # Add custom attributes
    transformers_config.position_bucket_size = custom_config.position_bucket_size
    
    # Add transformers-specific fields
    transformers_config.architectures = ["LtgBertForMaskedLM"]
    transformers_config.auto_map = {
        "AutoConfig": "ltg_bert_config.LtgBertConfig",
        "AutoModelForMaskedLM": "ltg_bert.LtgBertForMaskedLM"
    }
    transformers_config.torch_dtype = "float32"
    transformers_config.transformers_version = transformers.__version__
    
    # Save the configuration
    transformers_config.save_pretrained(output_path)
    print(f"Configuration saved to {output_path}/config.json")

if __name__ == "__main__":
    save_ltg_bert_config(
        "./configs/base.json", 
        "./model_babylm_bert_ltg/checkpoint"
    )
