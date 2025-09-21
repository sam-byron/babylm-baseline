#!/usr/bin/env python3
"""
Comprehensive stress testing suite for LtgBertForMaskedLM
Run with: PYTHONPATH=.. python run_stress_tests.py
"""

# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import os
import math
import tempfile
import time
import random
import argparse
import torch
import torch.nn as nn

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
    # labels: -100 for non-masked, token id for masked positions
    labels = torch.full((batch_size, seq_len), -100, device=device, dtype=torch.long)
    num_to_mask = max(1, int(seq_len * mask_ratio))
    for b in range(batch_size):
        idx = torch.randperm(seq_len, device=device)[:num_to_mask]
        labels[b, idx] = input_ids[b, idx]
    return input_ids, attention_mask, labels


def test_basic_shapes_and_loss(device="cpu"):
    """Test basic forward pass with various batch sizes and sequence lengths"""
    print("🔍 Testing basic shapes and loss...")
    
    test_cases = [(1, 8), (2, 32), (4, 128), (2, 512)]
    share_options = [False, True]
    
    passed = 0
    total = len(test_cases) * len(share_options)
    
    for batch_size, seq_len in test_cases:
        for share in share_options:
            try:
                cfg = make_config(hidden_size=256, num_attention_heads=8, 
                                share_layer_weights=share, max_position_embeddings=512)
                model = LtgBertForMaskedLM(cfg).to(device).train()

                input_ids, attention_mask, labels = random_inputs(batch_size, seq_len, cfg.vocab_size, device=device, mask_ratio=0.2)

                # Test without labels -> [B, L, V]
                out_no_labels = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
                assert out_no_labels.logits.shape == (batch_size, seq_len, cfg.vocab_size)
                assert out_no_labels.loss is None

                # Test with labels -> [N_masked, V], loss is scalar
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                n_masked = (labels != -100).sum().item()
                assert out.logits.shape == (n_masked, cfg.vocab_size)
                assert out.loss is not None and out.loss.ndim == 0

                # Test backward pass
                out.loss.backward()
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all(), "Found non-finite gradient"
                        grad_norm += p.grad.data.norm().item()
                assert math.isfinite(grad_norm)
                
                passed += 1
                print(f"  ✅ B={batch_size}, L={seq_len}, share={share}")
                
            except Exception as e:
                print(f"  ❌ B={batch_size}, L={seq_len}, share={share}: {e}")
    
    print(f"✅ Basic tests: {passed}/{total} passed\n")
    return passed == total


def test_attention_masks(device="cpu"):
    """Test different attention mask formats"""
    print("🔍 Testing attention mask variants...")
    
    cfg = make_config()
    model = LtgBertForMaskedLM(cfg).to(device).eval()
    B, L = 2, 32
    ids = torch.randint(0, cfg.vocab_size, (B, L), device=device)
    labels = torch.full((B, L), -100, dtype=torch.long, device=device)
    labels[0, :4] = ids[0, :4]
    
    tests = [
        ("2D long mask", torch.ones(B, L, dtype=torch.long, device=device)),
        ("2D bool mask", torch.ones(B, L, dtype=torch.bool, device=device)),
        ("4D broadcastable mask", torch.ones(B, 1, 1, L, dtype=torch.long, device=device)),
        ("All zeros mask", torch.zeros(B, L, dtype=torch.long, device=device)),
    ]
    
    passed = 0
    for name, mask in tests:
        try:
            if "All zeros" in name:
                out = model(ids, mask, labels=None)  # Don't use labels for all-zero mask
                assert torch.isfinite(out.logits).all()
            else:
                out = model(ids, mask, labels)
                assert torch.isfinite(out.loss)
            passed += 1
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    
    print(f"✅ Mask tests: {passed}/{len(tests)} passed\n")
    return passed == len(tests)


def test_weight_tying(device="cpu"):
    """Test that weight tying works correctly"""
    print("🔍 Testing weight tying...")
    
    try:
        cfg = make_config()
        model = LtgBertForMaskedLM(cfg).to(device)
        in_emb = model.get_input_embeddings().weight
        out_proj = model.get_output_embeddings()
        
        # Check weight tying
        assert out_proj.weight.data_ptr() == in_emb.data_ptr(), "Output head should be tied to input embedding"
        
        # Test modification propagates
        with torch.no_grad():
            original_value = in_emb[0, 0].clone()
            in_emb[0, 0] += 1.0
            assert torch.equal(in_emb, out_proj.weight), "Changes to input embedding should reflect in output head"
            in_emb[0, 0] = original_value  # restore
        
        print("  ✅ Weight tying works correctly")
        print("✅ Weight tying test passed\n")
        return True
    except Exception as e:
        print(f"  ❌ Weight tying test failed: {e}")
        print("❌ Weight tying test failed\n")
        return False


def test_mixed_precision(device="cpu"):
    """Test mixed precision if CUDA is available"""
    if device == "cpu":
        print("⏭️  Skipping mixed precision tests (CPU only)\n")
        return True
        
    print("🔍 Testing mixed precision...")
    
    dtypes = [torch.float16, torch.bfloat16]
    passed = 0
    
    for dtype in dtypes:
        try:
            cfg = make_config()
            model = LtgBertForMaskedLM(cfg).to(device).eval()
            B, L = 2, 64
            ids = torch.randint(0, cfg.vocab_size, (B, L), device=device)
            mask = torch.ones(B, L, device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(dtype=dtype):
                out = model(ids, mask, labels=None)
            assert out.logits.dtype in (torch.float16, torch.bfloat16)
            passed += 1
            print(f"  ✅ {dtype}")
        except Exception as e:
            print(f"  ❌ {dtype}: {e}")
    
    print(f"✅ Mixed precision tests: {passed}/{len(dtypes)} passed\n")
    return passed == len(dtypes)


def test_serialization(device="cpu"):
    """Test model serialization and loading"""
    print("🔍 Testing serialization...")
    
    try:
        cfg = make_config()
        model = LtgBertForMaskedLM(cfg).to(device).eval()
        B, L = 1, 8
        ids = torch.randint(0, cfg.vocab_size, (B, L), device=device)
        mask = torch.ones(B, L, dtype=torch.long, device=device)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Use safe_serialization=False to handle shared tensors
            model.save_pretrained(temp_dir, safe_serialization=False)
            reloaded = LtgBertForMaskedLM.from_pretrained(temp_dir).to(device)
            out1 = model(ids, mask, labels=None).logits
            out2 = reloaded(ids, mask, labels=None).logits
            assert out1.shape == out2.shape
            torch.testing.assert_close(out1, out2, atol=1e-4, rtol=1e-4)
        
        print("  ✅ Serialization works correctly")
        print("✅ Serialization test passed\n")
        return True
    except Exception as e:
        print(f"  ❌ Serialization test failed: {e}")
        print("❌ Serialization test failed\n")
        return False


def benchmark_performance(device="cpu", steps=20):
    """Benchmark training performance"""
    print(f"🔍 Benchmarking performance on {device}...")
    
    cfg = make_config(hidden_size=512, num_attention_heads=8, num_hidden_layers=6)
    model = LtgBertForMaskedLM(cfg).to(device)
    
    test_cases = [(4, 128), (8, 128), (4, 256)]
    
    for batch_size, seq_len in test_cases:
        if seq_len > cfg.max_position_embeddings:
            continue
            
        try:
            ids, mask, labels = random_inputs(batch_size, seq_len, cfg.vocab_size, device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            model.train()
            
            # Warmup
            for _ in range(3):
                optimizer.zero_grad()
                out = model(ids, mask, labels)
                out.loss.backward()
                optimizer.step()
            
            # Benchmark
            if device != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            tokens = 0
            for _ in range(steps):
                optimizer.zero_grad()
                out = model(ids, mask, labels)
                out.loss.backward()
                optimizer.step()
                tokens += batch_size * seq_len
            
            if device != "cpu":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            
            tps = tokens / dt
            mem = torch.cuda.max_memory_allocated() / 1e9 if device != "cpu" else 0.0
            
            print(f"  📊 B={batch_size:2d} L={seq_len:3d}: {tps:8.1f} tok/s  {mem:5.2f}GB")
            
            if device != "cpu":
                torch.cuda.reset_peak_memory_stats()
                
        except Exception as e:
            print(f"  ❌ B={batch_size}, L={seq_len}: {e}")
    
    print("✅ Performance benchmark completed\n")
    return True


def fuzz_test_configs(n_tests=20, device="cpu"):
    """Fuzz test with random configurations"""
    print(f"🔍 Fuzz testing with {n_tests} random configs...")
    
    def rand_divisor(n):
        divs = [d for d in range(1, n+1) if n % d == 0]
        return random.choice(divs)
    
    passed = 0
    for i in range(n_tests):
        try:
            vocab = random.choice([1024, 4096, 8192])
            hidden = random.choice([128, 256, 384])
            heads = rand_divisor(hidden)
            layers = random.choice([1, 2, 3])
            interm = random.choice([hidden*2, hidden*3])
            max_pos = random.choice([128, 256, 512])
            
            cfg = LtgBertConfig(
                vocab_size=vocab,
                hidden_size=hidden,
                num_attention_heads=heads,
                num_hidden_layers=layers,
                intermediate_size=interm,
                max_position_embeddings=max_pos,
                position_bucket_size=32,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                layer_norm_eps=1e-5,
                share_layer_weights=random.choice([False, True]),
            )
            
            model = LtgBertForMaskedLM(cfg).to(device).eval()
            B = random.choice([1, 2])
            L = random.choice([16, 32, min(64, cfg.max_position_embeddings)])
            
            ids = torch.randint(0, cfg.vocab_size, (B, L), device=device)
            mask = torch.ones(B, L, device=device, dtype=torch.long)
            
            out = model(ids, mask, labels=None)
            assert out.logits.shape == (B, L, cfg.vocab_size)
            assert torch.isfinite(out.logits).all()
            
            passed += 1
            if (i + 1) % 5 == 0:
                print(f"  ✅ {i+1}/{n_tests} configs passed")
                
        except Exception as e:
            print(f"  ❌ Config {i+1}: {e}")
    
    print(f"✅ Fuzz test: {passed}/{n_tests} configs passed\n")
    return passed == n_tests


def main():
    parser = argparse.ArgumentParser(description='Stress test LtgBertForMaskedLM')
    parser.add_argument('--device', choices=['cpu', 'cuda'], 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip-bench', action='store_true', help='Skip performance benchmark')
    parser.add_argument('--bench-steps', type=int, default=10, help='Steps for benchmark')
    parser.add_argument('--fuzz-tests', type=int, default=15, help='Number of fuzz tests')
    args = parser.parse_args()
    
    device = args.device
    print(f"🚀 Starting stress tests on {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()
    
    results = []
    results.append(("Basic shapes and loss", test_basic_shapes_and_loss(device)))
    results.append(("Attention masks", test_attention_masks(device)))
    results.append(("Weight tying", test_weight_tying(device)))
    results.append(("Mixed precision", test_mixed_precision(device)))
    results.append(("Serialization", test_serialization(device)))
    results.append(("Fuzz configs", fuzz_test_configs(args.fuzz_tests, device)))
    
    if not args.skip_bench:
        results.append(("Performance benchmark", benchmark_performance(device, args.bench_steps)))
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("=" * 50)
    print("📊 STRESS TEST SUMMARY")
    print("=" * 50)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\n🎯 Overall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All stress tests PASSED!")
        return 0
    else:
        print("💥 Some tests FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())