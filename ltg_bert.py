import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _softmax_backward_data as _softmax_backward_data
from torch.utils import checkpoint
from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from ltg_bert_config import LtgBertConfig


class LtgBertForMaskedLM(PreTrainedModel):
    """
    Custom BERT model for Masked Language Modeling that subclasses PreTrainedModel
    """
    config_class = LtgBertConfig
    base_model_prefix = "ltg_bert"
    supports_gradient_checkpointing = True
    _no_split_modules = ["EncoderLayer"]

    def __init__(self, config: LtgBertConfig, classifier="basic", activation_checkpointing=False):
        super().__init__(config)
        self.config = config
        self.embedding = Embedding(config)
        self.transformer = Encoder(config, activation_checkpointing)
        self.classifier = MaskClassifier(config, self.embedding.word_embedding.weight)
        if classifier != "basic":
            self.next_sentence_classifier = NextSentenceClassifier(config)
        else:
            self.next_sentence_classifier = None
        
        # Initialize weights
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Initialize linear layers with truncated normal distribution
            std = math.sqrt(2.0 / (5.0 * module.in_features))
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Initialize embedding layers with truncated normal distribution
            std = math.sqrt(2.0 / (5.0 * module.embedding_dim))
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm layers are already initialized correctly by PyTorch
            pass
        elif isinstance(module, nn.Parameter):
            # Initialize parameter tensors (like relative embeddings)
            std = math.sqrt(2.0 / (5.0 * module.size(-1)))
            nn.init.trunc_normal_(module, mean=0.0, std=std, a=-2*std, b=2*std)

    def get_contextualized(self, input_ids, attention_mask):
        # Transpose input_ids from [batch_size, seq_len] to [seq_len, batch_size]
        input_ids = input_ids.transpose(0, 1)
        static_embeddings, relative_embedding = self.embedding(input_ids)
        # attention_mask should be [batch_size, 1, 1, seq_len] for broadcasting
        # Convert from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        contextualized_embeddings = self.transformer(static_embeddings, attention_mask, relative_embedding)
        return contextualized_embeddings

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass that returns MaskedLMOutput for compatibility with transformers
        """
        contextualized_embeddings = self.get_contextualized(input_ids, attention_mask)[-1]
        # Transpose back from [seq_len, batch_size, hidden_size] to [batch_size, seq_len, hidden_size]
        contextualized_embeddings = contextualized_embeddings.transpose(0, 1)
        
        # Get predictions from classifier
        prediction_scores = self.classifier(contextualized_embeddings, labels)
        
        loss = None
        if labels is not None:
            # Compute MLM loss
            active_loss = labels.view(-1) != -100
            num_active = active_loss.sum().item()
            if active_loss.any():
                active_labels = labels.view(-1)[active_loss]
                # prediction_scores already contains only predictions for masked tokens
                loss = F.cross_entropy(prediction_scores, active_labels)
                
                # Debug: Print loss computation details for first batch
                if hasattr(self, '_debug_step_count'):
                    self._debug_step_count += 1
                else:
                    self._debug_step_count = 1
                    
                if self._debug_step_count == 1:
                    print(f"Loss computation debug:")
                    print(f"  - Active tokens: {num_active}")
                    print(f"  - Prediction scores shape: {prediction_scores.shape}")
                    print(f"  - Active labels shape: {active_labels.shape}")
                    print(f"  - Active labels min/max: {active_labels.min().item()}/{active_labels.max().item()}")
                    print(f"  - Cross entropy loss: {loss.item()}")
            else:
                loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=None,
            attentions=None,
        )

    def get_input_embeddings(self):
        return self.embedding.word_embedding

    def set_input_embeddings(self, value):
        self.embedding.word_embedding = value

    def get_output_embeddings(self):
        return self.classifier.nonlinearity[-1]

    def set_output_embeddings(self, new_embeddings):
        self.classifier.nonlinearity[-1] = new_embeddings

    def register_for_auto_class(self):
        from transformers import AutoConfig, AutoModelForMaskedLM
        AutoConfig.register("ltg_bert", LtgBertConfig)
        AutoModelForMaskedLM.register(LtgBertConfig, LtgBertForMaskedLM)


class LtgBertForSequenceClassification(PreTrainedModel):
    """
    Custom BERT model for Sequence Classification that subclasses PreTrainedModel
    """
    config_class = LtgBertConfig
    base_model_prefix = "ltg_bert"
    supports_gradient_checkpointing = True
    _no_split_modules = ["EncoderLayer"]

    def __init__(self, config: LtgBertConfig, activation_checkpointing=False):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        
        # Shared BERT components
        self.embedding = Embedding(config)
        self.transformer = Encoder(config, activation_checkpointing)
        
        # Classification head
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Initialize linear layers with truncated normal distribution
            std = math.sqrt(2.0 / (5.0 * module.in_features))
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Initialize embedding layers with truncated normal distribution
            std = math.sqrt(2.0 / (5.0 * module.embedding_dim))
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm layers are already initialized correctly by PyTorch
            pass
        elif isinstance(module, nn.Parameter):
            # Initialize parameter tensors (like relative embeddings)
            std = math.sqrt(2.0 / (5.0 * module.size(-1)))
            nn.init.trunc_normal_(module, mean=0.0, std=std, a=-2*std, b=2*std)

    def get_contextualized(self, input_ids, attention_mask):
        # Transpose input_ids from [batch_size, seq_len] to [seq_len, batch_size]
        input_ids = input_ids.transpose(0, 1)
        static_embeddings, relative_embedding = self.embedding(input_ids)
        # attention_mask should be [batch_size, 1, 1, seq_len] for broadcasting
        # Convert from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        contextualized_embeddings = self.transformer(static_embeddings, attention_mask, relative_embedding)
        return contextualized_embeddings

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass that returns SequenceClassifierOutput for compatibility with transformers
        """
        contextualized_embeddings = self.get_contextualized(input_ids, attention_mask)[-1]
        # Transpose back from [seq_len, batch_size, hidden_size] to [batch_size, seq_len, hidden_size]
        contextualized_embeddings = contextualized_embeddings.transpose(0, 1)
        
        # Use [CLS] token representation (first token) for classification
        # Shape: [batch_size, hidden_size]
        cls_representation = contextualized_embeddings[:, 0, :]
        
        # Apply dropout and classifier
        cls_representation = self.dropout(cls_representation)
        logits = self.classifier(cls_representation)
        
        loss = None
        if labels is not None:
            # Determine problem type without modifying config
            problem_type = self.config.problem_type
            if problem_type is None:
                if self.num_labels == 1:
                    problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    problem_type = "single_label_classification"
                else:
                    problem_type = "multi_label_classification"

            if problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def get_input_embeddings(self):
        return self.embedding.word_embedding

    def set_input_embeddings(self, value):
        self.embedding.word_embedding = value


class Encoder(nn.Module):
    def __init__(self, config, activation_checkpointing=False):
        super().__init__()
        
        self.config = config
        self.activation_checkpointing = activation_checkpointing
        
        if config.share_layer_weights:
            # ALBERT-style parameter sharing: create only one layer and reuse it
            self.shared_layer = EncoderLayer(config)
            self.layers = nn.ModuleList([self.shared_layer for _ in range(config.num_hidden_layers)])
            print(f"🔄 ALBERT-style parameter sharing enabled: {config.num_hidden_layers} layers sharing weights")
        else:
            # Standard approach: each layer has its own parameters
            self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

        # Apply layer-wise scaling only for non-shared layers
        if not config.share_layer_weights:
            for i, layer in enumerate(self.layers):
                layer.mlp.mlp[1].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
                layer.mlp.mlp[-2].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
    
    def forward(self, hidden_states, attention_mask, relative_embedding):
        hidden_states = [hidden_states]
        
        if self.config.share_layer_weights:
            # ALBERT-style: use the same layer multiple times
            for _ in range(self.config.num_hidden_layers):
                if self.activation_checkpointing:
                    hidden_states.append(
                        checkpoint.checkpoint(self.shared_layer, hidden_states[-1], attention_mask, relative_embedding)
                    )
                else:
                    hidden_states.append(
                        self.shared_layer(hidden_states[-1], attention_mask, relative_embedding)
                    )
        else:
            # Standard: use different layers
            for layer in self.layers:
                if self.activation_checkpointing:
                    hidden_states.append(
                        checkpoint.checkpoint(layer, hidden_states[-1], attention_mask, relative_embedding)
                    )
                else:
                    hidden_states.append(
                        layer(hidden_states[-1], attention_mask, relative_embedding)
                    )

        return hidden_states


class MaskClassifier(nn.Module):
    def __init__(self, config, subword_embedding):
        super().__init__()
        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(subword_embedding.size(1), subword_embedding.size(0))
        )
        self.initialize(config.hidden_size, subword_embedding)

    def initialize(self, hidden_size, embedding):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.nonlinearity[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.nonlinearity[-1].weight = embedding
        self.nonlinearity[1].bias.data.zero_()
        self.nonlinearity[-1].bias.data.zero_()

    def forward(self, x, masked_lm_labels=None):
        if masked_lm_labels is not None:
            x = torch.index_select(x.flatten(0, 1), 0, torch.nonzero(masked_lm_labels.flatten() != -100).squeeze())
        x = self.nonlinearity(x)
        return x


class NextSentenceClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, 1)
        )
        self.initialize(config.hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.nonlinearity[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.nonlinearity[-1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.nonlinearity[1].bias.data.zero_()
        self.nonlinearity[-1].bias.data.zero_()

    def forward(self, x):
        x = self.nonlinearity(x[0, :, :]).squeeze(-1)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = FeedForward(config)

    def forward(self, x, padding_mask, relative_embedding):
        x = x + self.attention(x, padding_mask, relative_embedding)
        x = x + self.mlp(x)
        return x


class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        x = x * F.gelu(gate, approximate='tanh')
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, 2*config.intermediate_size, bias=False),
            GeGLU(),
            nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.initialize(config.hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.mlp[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.mlp[-2].weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        return self.mlp(x)


class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, dim):
        # Store dim for backward
        ctx.dim = dim
        # Convert mask to boolean if it's not already
        # Assuming mask is 1 for valid positions, 0 for padding
        # We need to invert it: True for positions to mask (padding)
        if mask.dtype != torch.bool:
            mask = (mask == 0)  # Convert 0s (padding) to True (mask these positions)
        
        # Ensure mask can broadcast to x's shape
        # x shape: [batch_size, num_heads, seq_len, seq_len]
        # mask shape should be: [batch_size, 1, 1, seq_len] or broadcastable
        mask = mask.expand_as(x)
        
        # Use in-place operations more carefully for distributed training
        x_masked = x.masked_fill(mask, float('-inf'))
        result = torch.softmax(x_masked, ctx.dim)
        result = result.masked_fill(mask, 0.0)
        
        # Save for backward - be more careful about what we save
        ctx.save_for_backward(result, mask)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, mask = ctx.saved_tensors
        # Use the standard softmax backward implementation
        grad_input = _softmax_backward_data(grad_output, result, ctx.dim, result.dtype)
        # Zero out gradients for masked positions
        grad_input = grad_input.masked_fill(mask, 0.0)
        return grad_input, None, None


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        self.in_proj_qk = nn.Linear(config.hidden_size, 2*config.hidden_size, bias=True)
        self.in_proj_v = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.pre_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)
        self.post_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        position_indices = torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(1) \
            - torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(0)
        position_indices = self.make_log_bucket_position(position_indices, config.position_bucket_size, config.max_position_embeddings)
        position_indices = config.position_bucket_size - 1 + position_indices
        self.register_buffer("position_indices", position_indices, persistent=True)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = 1.0 / math.sqrt(3 * self.head_size)
        self.initialize()

    def make_log_bucket_position(self, relative_pos, bucket_size, max_position):
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos))
        log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position-1) / mid) * (mid - 1)).int() + mid
        bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()
        return bucket_pos

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.in_proj_qk.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.in_proj_v.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.in_proj_qk.bias.data.zero_()
        self.in_proj_v.bias.data.zero_()
        self.out_proj.bias.data.zero_()

    def compute_attention_scores(self, hidden_states, relative_embedding):
        key_len, batch_size, _ = hidden_states.size()
        query_len = key_len

        hidden_states = self.pre_layer_norm(hidden_states)

        query, key = self.in_proj_qk(hidden_states).chunk(2, dim=2)  # shape: [T, B, D]
        value = self.in_proj_v(hidden_states)  # shape: [T, B, D]

        pos = self.in_proj_qk(self.dropout(relative_embedding))  # shape: [2T-1, 2D]
        pos = F.embedding(self.position_indices[:query_len, :key_len], pos)  # shape: [T, T, 2D]
        pos = pos.view(query_len, key_len, self.num_heads, 2*self.head_size)
        query_pos, key_pos = pos.chunk(2, dim=3)

        query = query.reshape(query_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        key = key.reshape(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        value = value.view(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)

        attention_scores = torch.bmm(query, key.transpose(1, 2) * self.scale)

        query = query.view(batch_size, self.num_heads, query_len, self.head_size)
        key = key.view(batch_size, self.num_heads, query_len, self.head_size)
        attention_scores = attention_scores.view(batch_size, self.num_heads, query_len, key_len)
        attention_scores.add_(torch.einsum("bhqd,qkhd->bhqk", query, key_pos * self.scale))
        attention_scores.add_(torch.einsum("bhkd,qkhd->bhqk", key * self.scale, query_pos))

        return attention_scores, value

    def compute_output(self, attention_probs, value):
        attention_probs = self.dropout(attention_probs)
        context = torch.bmm(attention_probs.flatten(0, 1), value)  # shape: [B*H, Q, D]
        context = context.transpose(0, 1).reshape(context.size(1), -1, self.hidden_size)  # shape: [Q, B, H*D]
        context = self.out_proj(context)
        context = self.post_layer_norm(context)
        context = self.dropout(context)
        return context

    def forward(self, hidden_states, attention_mask, relative_embedding):
        attention_scores, value = self.compute_attention_scores(hidden_states, relative_embedding)
        attention_probs = MaskedSoftmax.apply(attention_scores, attention_mask, -1)
        return self.compute_output(attention_probs, value)


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.relative_embedding = nn.Parameter(torch.empty(2 * config.position_bucket_size - 1, config.hidden_size))
        self.relative_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.initialize()

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.relative_embedding, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.word_embedding.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, input_ids):
        word_embedding = self.dropout(self.word_layer_norm(self.word_embedding(input_ids)))
        relative_embeddings = self.relative_layer_norm(self.relative_embedding)
        return word_embedding, relative_embeddings


# Register the model configuration
from transformers import AutoConfig, AutoModelForMaskedLM, AutoModelForSequenceClassification

# Register the configuration
AutoConfig.register("ltg_bert", LtgBertConfig)

# Register the models
AutoModelForMaskedLM.register(LtgBertConfig, LtgBertForMaskedLM)
AutoModelForSequenceClassification.register(LtgBertConfig, LtgBertForSequenceClassification)
