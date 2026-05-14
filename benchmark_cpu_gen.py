#!/usr/bin/env python3
"""Quick CPU generation benchmark using only tokenizer"""
import torch
from transformers import AutoTokenizer
import time

print("Loading LLaMA tokenizer...")
model_id = "./checkpoints/ShapeLLM_7B_gapartnet_v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Simulate multi-step text generation by tokenizing a longer text
print("Estimating generation speed based on tokenizer processing...")

# Create a representative input (432 tokens for system + prompt)
prompt = "The quick brown fox jumps over the lazy dog. " * 50

start = time.time()
for _ in range(10):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
tokenize_time = time.time() - start

avg_tokenize_per_call = tokenize_time / 10
print(f"Average tokenization time per prompt: {avg_tokenize_per_call:.4f}s")

# Rough estimate: LLM generation loops roughly 512 times for 512 tokens
# Each loop involves tokenization + forward pass
# CPU forward pass is roughly 10-100x slower than tokenization
rough_total_per_sample = 512 * (avg_tokenize_per_call + 0.1)  # 0.1s per token = rough estimate
hours_for_full_test = (3935 * rough_total_per_sample) / 3600

print(f"\nRough estimate for full test:")
print(f"  Per sample (512 tokens): {rough_total_per_sample:.1f}s")
print(f"  Full test (3935 samples): {3935 * rough_total_per_sample / 3600:.1f} hours")
print(f"\nNote: This is a ROUGH estimate. Actual time depends heavily on CPU hardware and model implementation.")
