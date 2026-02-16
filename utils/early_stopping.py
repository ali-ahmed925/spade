"""Early stopping utility for training."""

import numpy as np


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping.
        mode: 'max' for metrics to maximize (e.g., AUROC), 'min' for metrics to minimize (e.g., loss).
        min_delta: Minimum change to qualify as an improvement.
        verbose: Whether to print early stopping messages.
    """
    
    def __init__(
        self,
        patience: int = 7,
        mode: str = "max",
        min_delta: float = 0.0,
        verbose: bool = True,
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation metric score.
            
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: Score improved to {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs (best: {self.best_score:.4f})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping training after {self.patience} epochs without improvement")
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == "max":
            return current > best + self.min_delta
        else:  # mode == "min"
            return current < best - self.min_delta
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False



