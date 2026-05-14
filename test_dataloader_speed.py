#!/usr/bin/env python3
"""Test dataloader speed"""
import sys
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from train_lightning import ModelArguments, DataArguments, TrainingArguments, LISADataModule
from transformers import HfArgumentParser, AutoTokenizer

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    logger.info(f"Data root: {data_args.data_root}")
    logger.info(f"Sample points: {data_args.sample_points_num}")
    
    # Create datamodule
    datamodule = LISADataModule(
        model_args, data_args, training_args,
        tokenizer=AutoTokenizer.from_pretrained(model_args.model_name_or_path),
    )
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    
    logger.info(f"Test dataloader created. Len test_dataset = {len(datamodule.test_dataset)}")
    logger.info(f"Batch size = {training_args.per_device_eval_batch_size}")
    
    # Test loading first few batches
    for batch_idx, batch in enumerate(test_loader):
        batch_start = time.time()
        batch_time = time.time() - batch_start
        logger.info(f"Batch {batch_idx}: keys={list(batch.keys())}, values shapes: {[(k, v.shape if hasattr(v, 'shape') else len(v)) for k, v in batch.items()]}")
        
        if batch_idx >= 2:
            break
    
    logger.info("Dataloader test completed successfully")

if __name__ == "__main__":
    main()
