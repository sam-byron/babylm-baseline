# coding=utf-8
# Copyright 2023 Language Technology Group from University of Oslo and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LTG-BERT configuration

Defines the configuration for the custom LTG-BERT model compatible with Hugging Face
Transformers' PretrainedConfig. This config mirrors standard BERT fields and introduces
an additional relative position bucket size used by the attention implementation.

Usage
- Python API:
    from configuration_ltgbert import LtgBertConfig
    cfg = LtgBertConfig(vocab_size=16384, hidden_size=768, num_hidden_layers=12)
- From pretrained directory:
    cfg = LtgBertConfig.from_pretrained("/path/to/checkpoint")

Notes
- model_type is set to "bert" for broad compatibility with tools that expect a BERT-like model.
- position_bucket_size controls the number of relative position buckets on each side plus the center bucket.
"""


from transformers.configuration_utils import PretrainedConfig


class LtgBertConfig(PretrainedConfig):
    r"""
    Configuration container for [`LtgBertModel`].

    This is used to instantiate an LTG-BERT model according to the specified arguments, defining
    the model architecture. Configuration objects inherit from [`PretrainedConfig`].

    Parameters
    - vocab_size (int, default 16384):
        Size of the tokenizer vocabulary; determines valid token id range.
    - hidden_size (int, default 768):
        Embedding dimension and per-token hidden feature size.
    - num_hidden_layers (int, default 12):
        Number of Transformer encoder layers.
    - num_attention_heads (int, default 12):
        Attention heads per layer; must divide hidden_size.
    - intermediate_size (int, default 2048):
        FFN hidden size before projection back to hidden_size.
    - hidden_dropout_prob (float, default 0.1):
        Dropout applied to embeddings and layer outputs.
    - attention_probs_dropout_prob (float, default 0.1):
        Dropout applied to attention probability tensors.
    - max_position_embeddings (int, default 512):
        Maximum supported sequence length for position bucketing.
    - position_bucket_size (int, default 32):
        Relative position bucketing granularity used by the custom attention.
    - layer_norm_eps (float, default 1e-7):
        Epsilon for LayerNorm stability.
    - pad_token_id (int, default 4):
        Token id treated as padding.
    - output_all_encoded_layers (bool, default True):
        Whether encoder returns all layer activations (internal use).
    - classifier_dropout (float | None):
        Optional dropout override for classification heads.
    """
    model_type = "bert"
    def __init__(
        self,
        vocab_size=16384,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        hidden_size=768,
        intermediate_size=2048,
        max_position_embeddings=512,
        position_bucket_size=32,
        num_attention_heads=12,
        num_hidden_layers=12,
        layer_norm_eps=1.0e-7,
        pad_token_id=4,
        output_all_encoded_layers=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.output_all_encoded_layers = output_all_encoded_layers
        self.position_bucket_size = position_bucket_size
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout = classifier_dropout
