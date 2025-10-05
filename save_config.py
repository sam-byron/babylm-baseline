#!/usr/bin/env python3
"""
Convert a local training JSON config into a Hugging Face-compatible LtgBertConfig.

This script reads a minimal training-time BertConfig JSON (fields like hidden_size,
num_layers, etc.) and writes an AutoConfig-compatible config.json suitable for
transformers Auto* loading with trust_remote_code.

Examples
  python save_config.py --in ./configs/base.json --out ./model_babylm_bert_ltg/checkpoint
"""

import argparse
from config import BertConfig
from configuration_ltgbert import LtgBertConfig
import transformers


def save_ltg_bert_config(custom_config_path: str, output_path: str) -> str:
    """Load a training JSON, build LtgBertConfig, and save to output_path; returns config.json path."""
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
        pad_token_id=getattr(custom_config, "pad_token_id", 4),
    )

    # Add custom attributes
    transformers_config.position_bucket_size = getattr(custom_config, "position_bucket_size", 32)

    # Add transformers-specific fields
    transformers_config.architectures = [
        "LtgBertForMaskedLM",
        "LtgBertForSequenceClassification",
    ]
    transformers_config.auto_map = {
        "AutoConfig": "configuration_ltgbert.LtgBertConfig",
        "AutoModelForMaskedLM": "modeling_ltgbert.LtgBertForMaskedLM",
        "AutoModelForSequenceClassification": "modeling_ltgbert.LtgBertForSequenceClassification",
    }
    transformers_config.torch_dtype = "float32"
    transformers_config.transformers_version = transformers.__version__

    # Save the configuration
    save_dir = transformers_config.save_pretrained(output_path)
    print(f"Configuration saved to {output_path}/config.json")
    return str(save_dir)


def main():
    ap = argparse.ArgumentParser(description="Write a transformers-compatible config.json for LTG-BERT")
    ap.add_argument("--in", dest="in_path", required=False, default="./configs/base.json", help="Input training JSON path (defaults to ./configs/base.json)")
    ap.add_argument("--out", dest="out_path", required=False, default="./model_babylm_bert_ltg/checkpoint", help="Output directory (defaults to ./model_babylm_bert_ltg/checkpoint)")
    args = ap.parse_args()

    save_ltg_bert_config(args.in_path, args.out_path)


if __name__ == "__main__":
    main()
