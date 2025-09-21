import math
import tempfile
import pytest
import torch

from ltg_bert import LtgBertForMaskedLM
from ltg_bert_config import LtgBertConfig


SEED = 1337
torch.manual_seed(SEED)


def make_config(
    vocab_size=16384,
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=8,
    intermediate_size=512,
    max_position_embeddings=512,
    position_bucket_size=32,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    layer_norm_eps=1e-5,
    share_layer_weights=False,
):
    assert hidden_size % num_attention_heads == 0
    return LtgBertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        position_bucket_size=position_bucket_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        layer_norm_eps=layer_norm_eps,
        share_layer_weights=share_layer_weights,
    )


def random_inputs(batch_size, seq_len, vocab_size, device="cpu", mask_ratio=0.15):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, device=device, dtype=torch.long)
    num_to_mask = max(1, int(seq_len * mask_ratio))
    for b in range(batch_size):
        idx = torch.randperm(seq_len, device=device)[:num_to_mask]
        labels[b, idx] = input_ids[b, idx]
    return input_ids, attention_mask, labels


@pytest.mark.parametrize("batch_size,seq_len", [(1, 8), (2, 32), (4, 128), (2, 512)])
@pytest.mark.parametrize("share", [False, True])
def test_forward_shapes_and_loss_cpu(batch_size, seq_len, share):
    cfg = make_config(hidden_size=256, num_attention_heads=8, share_layer_weights=share, max_position_embeddings=512)
    model = LtgBertForMaskedLM(cfg).cpu().train()

    input_ids, attention_mask, labels = random_inputs(batch_size, seq_len, cfg.vocab_size, device="cpu", mask_ratio=0.2)

    out_no_labels = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
    assert out_no_labels.logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert out_no_labels.loss is None

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    n_masked = (labels != -100).sum().item()
    assert out.logits.shape == (n_masked, cfg.vocab_size)
    assert out.loss is not None and out.loss.ndim == 0

    out.loss.backward()
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "Found non-finite gradient"
            grad_norm += p.grad.data.norm().item()
    assert math.isfinite(grad_norm)


def test_attention_mask_variants_cpu():
    cfg = make_config()
    model = LtgBertForMaskedLM(cfg).cpu().eval()
    B, L = 2, 32
    ids = torch.randint(0, cfg.vocab_size, (B, L))
    labels = torch.full((B, L), -100, dtype=torch.long)
    labels[0, :4] = ids[0, :4]

    mask_2d = torch.ones(B, L, dtype=torch.long)
    out1 = model(ids, mask_2d, labels)
    assert torch.isfinite(out1.loss)

    mask_bool = torch.ones(B, L, dtype=torch.bool)
    out2 = model(ids, mask_bool, labels)
    assert torch.isfinite(out2.loss)

    mask_4d = torch.ones(B, 1, 1, L, dtype=torch.long)
    out3 = model(ids, mask_4d, labels)
    assert torch.isfinite(out3.loss)

    all_zero = torch.zeros(B, L, dtype=torch.long)
    out4 = model(ids, all_zero, labels=None)
    assert torch.isfinite(out4.logits).all()


def test_no_masked_tokens_returns_zero_loss():
    cfg = make_config()
    model = LtgBertForMaskedLM(cfg).cpu().train()
    B, L = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (B, L))
    mask = torch.ones(B, L, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    out = model(ids, mask, labels)
    assert out.loss is not None
    assert out.loss.item() == 0.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_precision_inference(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for mixed-precision test")
    cfg = make_config()
    model = LtgBertForMaskedLM(cfg).cuda().eval()
    B, L = 2, 64
    ids = torch.randint(0, cfg.vocab_size, (B, L), device="cuda")
    mask = torch.ones(B, L, device="cuda", dtype=torch.long)

    with torch.cuda.amp.autocast(dtype=dtype):
        out = model(ids, mask, labels=None)
    assert out.logits.dtype in (torch.float16, torch.bfloat16)


@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_activation_checkpointing(activation_checkpointing):
    cfg = make_config()
    model = LtgBertForMaskedLM(cfg, activation_checkpointing=activation_checkpointing).cpu().train()
    B, L = 2, 32
    ids, mask, labels = random_inputs(B, L, cfg.vocab_size, device="cpu")
    out = model(ids, mask, labels)
    out.loss.backward()


def test_weight_tying():
    cfg = make_config()
    model = LtgBertForMaskedLM(cfg).cpu()
    in_emb = model.get_input_embeddings().weight
    out_proj = model.get_output_embeddings()
    assert out_proj.weight.data_ptr() == in_emb.data_ptr(), "Output head should be tied to input embedding"

    with torch.no_grad():
        in_emb[0, 0] += 1.0
    assert torch.equal(in_emb, out_proj.weight)


def test_serialization_roundtrip():
    cfg = make_config()
    model = LtgBertForMaskedLM(cfg).cpu().eval()
    B, L = 1, 8
    ids = torch.randint(0, cfg.vocab_size, (B, L))
    mask = torch.ones(B, L, dtype=torch.long)

    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d, safe_serialization=False)
        reloaded = LtgBertForMaskedLM.from_pretrained(d)
        out1 = model(ids, mask, labels=None).logits
        out2 = reloaded(ids, mask, labels=None).logits
        assert out1.shape == out2.shape
        torch.testing.assert_close(out1, out2, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("batch_size,seq_len", [(8, 128), (8, 256)])
def test_large_batch_does_not_oob(batch_size, seq_len):
    cfg = make_config(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=8, max_position_embeddings=512)
    model = LtgBertForMaskedLM(cfg).cpu().eval()
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    out = model(ids, mask, labels=None)
    assert out.logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert torch.isfinite(out.logits).all()
