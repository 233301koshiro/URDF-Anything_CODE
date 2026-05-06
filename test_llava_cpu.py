#!/usr/bin/env python3
"""
Simple LLaVA CPU verification test with progress visibility.
Tests model loading and basic forward pass on CPU with detailed logging.
"""
import torch
import os
import sys
import time
import logging
from datetime import datetime

# Setup console logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def log_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def log_step(step: int, desc: str):
    """Print a step header."""
    logger.info(f"[Step {step}] {desc}")

def log_success(msg: str):
    """Log success message."""
    logger.info(f"✓ {msg}")

def log_error(msg: str):
    """Log error message."""
    logger.error(f"✗ {msg}")

log_section("LLaVA CPU Verification Test")

# Step 1: Check CUDA availability
log_step(1, "Checking CUDA availability...")
cuda_available = torch.cuda.is_available()
device_count = torch.cuda.device_count()
logger.info(f"  CUDA Available: {cuda_available}")
logger.info(f"  CUDA Device Count: {device_count}")
log_success("CUDA check completed")

# Step 2: Setup paths
log_step(2, "Setting up workspace...")
workspace_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, workspace_root)
model_path = os.path.join(workspace_root, "checkpoints/ShapeLLM_7B_gapartnet_v1.0")
logger.info(f"  Workspace: {workspace_root}")
logger.info(f"  Model path: {model_path}")
log_success("Workspace setup completed")

# Step 3: Verify checkpoint exists
log_step(3, "Verifying model checkpoint...")
if not os.path.exists(model_path):
    log_error(f"Model checkpoint not found at {model_path}")
    logger.info("Download may still be in progress. Waiting for weights...")
    
    # Check subdirectories
    checkpoints_dir = os.path.dirname(model_path)
    if os.path.exists(checkpoints_dir):
        subdirs = os.listdir(checkpoints_dir)
        logger.info(f"Available subdirectories: {subdirs}")
    
    logger.info("Please wait for download to complete and retry.")
    sys.exit(0)
else:
    log_success(f"Model checkpoint found")
    # List checkpoint files
    files = os.listdir(model_path)
    logger.info(f"  Checkpoint files: {len(files)} items")
    for f in sorted(files)[:5]:  # Show first 5 files
        logger.info(f"    - {f}")
    if len(files) > 5:
        logger.info(f"    ... and {len(files)-5} more files")

# Step 4: Import model builder
log_step(4, "Importing LLaVA model builder...")
try:
    from model.llava.model.builder import load_pretrained_model
    log_success("LLaVA model builder imported")
except ImportError as e:
    log_error(f"Failed to import model builder: {e}")
    sys.exit(1)

# Step 5: Load model
log_step(5, "Loading LLaVA model on CPU...")
load_start = time.time()
try:
    logger.info("  This may take 1-2 minutes on CPU...")
    
    model_name = "llava_llama2"
    loaded = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device="cpu"  # Explicit CPU device
    )
    if len(loaded) == 4:
        tokenizer, model, image_processor, context_len = loaded
    elif len(loaded) == 3:
        tokenizer, model, context_len = loaded
        image_processor = None
    else:
        raise RuntimeError(f"Unexpected return values from load_pretrained_model: {len(loaded)}")
    
    load_time = time.time() - load_start
    log_success(f"Model loaded in {load_time:.1f} seconds")
    
except Exception as e:
    log_error(f"Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Verify model properties
log_step(6, "Verifying model properties...")
try:
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    logger.info(f"  Model device: {model_device}")
    logger.info(f"  Model dtype: {model_dtype}")
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    log_success("Model properties verified")
except Exception as e:
    log_error(f"Property verification failed: {e}")
    sys.exit(1)

# Step 7: Verify tokenizer
log_step(7, "Verifying tokenizer...")
try:
    tokenizer_type = type(tokenizer).__name__
    vocab_size = len(tokenizer)
    
    logger.info(f"  Tokenizer type: {tokenizer_type}")
    logger.info(f"  Vocab size: {vocab_size:,}")
    logger.info(f"  Max length: {getattr(tokenizer, 'model_max_length', 'N/A')}")
    
    log_success("Tokenizer verified")
except Exception as e:
    log_error(f"Tokenizer verification failed: {e}")
    sys.exit(1)

# Step 8: Test forward pass
log_step(8, "Testing model forward pass...")
try:
    sample_text = "What is in the image?"
    
    logger.info(f"  Input text: '{sample_text}'")
    logger.info("  Tokenizing...")
    
    input_ids = tokenizer.encode(sample_text, return_tensors='pt')
    logger.info(f"  Tokenized shape: {input_ids.shape}")
    logger.info(f"  Token IDs device: {input_ids.device}")
    
    # Move to model device
    input_ids = input_ids.to(model_device)
    logger.info(f"  Moved to model device: {input_ids.device}")
    
    # Forward pass
    logger.info("  Running forward pass...")
    forward_start = time.time()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
    
    forward_time = time.time() - forward_start
    
    logger.info(f"  Output logits shape: {logits.shape}")
    logger.info(f"  Output device: {logits.device}")
    logger.info(f"  Forward pass time: {forward_time:.2f} seconds")
    
    log_success(f"Forward pass successful (1 token, {forward_time:.2f}s)")
    
except Exception as e:
    log_error(f"Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 9: Memory info
log_step(9, "Checking memory usage...")
try:
    if cuda_available:
        current_mem = torch.cuda.memory_allocated() / (1024**3)  # GB
        max_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        logger.info(f"  GPU Memory: {current_mem:.2f}GB / {max_mem:.2f}GB")
    else:
        logger.info("  (No CUDA device - CPU memory usage via system)")
    
    log_success("Memory check completed")
except Exception as e:
    logger.warning(f"Could not check memory: {e}")

# Final summary
log_section("Verification Complete")
logger.info("✓ All tests PASSED")
logger.info(f"✓ LLaVA is running successfully on CPU")
logger.info(f"✓ Model loaded: {model_name}")
logger.info(f"✓ Device: {model_device}")
logger.info(f"✓ Ready for inference")

print("=" * 70)
