# utils/repro.py

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set seed for reproducibility across Python, NumPy, and PyTorch."""
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # This enforces deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)

def get_dataloader_seed_components(seed: int = 42):
    """
    Returns:
        - worker_init_fn: function to seed workers
        - generator: torch.Generator with the same seed
    """
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return seed_worker, generator