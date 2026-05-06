"""
Progress visualization utilities for CPU-based training and evaluation.
Provides enhanced logging and progress bars for long-running operations.
"""

import sys
import time
import logging
from datetime import datetime
from typing import Optional
from tqdm import tqdm

class ProgressLogger:
    """Enhanced logger for CPU training/evaluation with progress timestamps."""
    
    def __init__(self, name: str, log_interval: int = 10):
        """
        Initialize progress logger.
        
        Args:
            name: Logger name
            log_interval: Interval (in seconds) between progress logs
        """
        self.logger = logging.getLogger(name)
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.start_time = time.time()
        
        # Configure if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s] %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)
    
    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
    
    def progress(self, msg: str):
        """
        Log progress message at specified interval.
        Only logs if log_interval seconds have passed since last progress log.
        """
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            elapsed = current_time - self.start_time
            self.logger.info(f"[{elapsed:.1f}s] {msg}")
            self.last_log_time = current_time
    
    def elapsed_time(self) -> float:
        """Return elapsed time in seconds."""
        return time.time() - self.start_time


class ProgressBar:
    """Enhanced progress bar with ETA and throughput info."""
    
    def __init__(self, 
                 total: int,
                 desc: str = "Processing",
                 unit: str = "it",
                 log_interval: int = 5):
        """
        Initialize progress bar.
        
        Args:
            total: Total number of iterations
            desc: Description/title of progress bar
            unit: Unit name (iterations, batches, etc.)
            log_interval: Interval between detailed updates
        """
        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            ncols=100,
            ascii=True,
            dynamic_ncols=False
        )
        self.current = 0
        self.total = total
        self.start_time = time.time()
        self.log_interval = log_interval
        self.last_update = 0
    
    def update(self, n: int = 1):
        """Update progress bar."""
        self.pbar.update(n)
        self.current += n
    
    def set_postfix(self, **kwargs):
        """Set postfix info (e.g., loss, accuracy)."""
        self.pbar.set_postfix(kwargs)
    
    def close(self):
        """Close progress bar."""
        self.pbar.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.close()


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def setup_logging(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging for a module.
    
    Args:
        name: Logger name
        log_file: Optional log file path
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s] %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")
    
    return logger
