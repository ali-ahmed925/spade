"""Debug logger utility for SPADE model debugging.

Logs debug information to a file instead of cluttering the console.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


# Global variable to track the log file path (shared across all debug loggers)
_shared_log_file = None


def get_debug_logger(name: str = "spade_debug") -> logging.Logger:
    """Get or create a debug logger that writes to a file.
    
    All debug loggers share the same log file to avoid clutter.
    
    Args:
        name: Logger name (default: "spade_debug")
        
    Returns:
        Configured logger instance
    """
    global _shared_log_file
    
    logger = logging.getLogger(name)
    
    # If logger already has handlers, return it (avoid duplicate handlers)
    if logger.handlers:
        return logger
    
    # Set level to DEBUG
    logger.setLevel(logging.DEBUG)
    
    # Create debug_logs directory
    debug_dir = Path("debug_logs")
    debug_dir.mkdir(exist_ok=True)
    
    # Use shared log file (create once, reuse for all loggers)
    if _shared_log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _shared_log_file = debug_dir / f"spade_debug_{timestamp}.log"
    
    # Create file handler
    file_handler = logging.FileHandler(_shared_log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger (avoid console output)
    logger.propagate = False
    
    # Log initialization message (only once)
    if not hasattr(get_debug_logger, '_initialized'):
        logger.debug(f"Debug logger initialized. Logging to: {_shared_log_file}")
        get_debug_logger._initialized = True
    
    return logger

