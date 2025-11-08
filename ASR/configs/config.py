import os
import torch
import numpy as np
import random

# Ensure reproducibility
torch.manual_seed(1111)
torch.cuda.manual_seed(1111)
np.random.seed(1111)
random.seed(1111)

class Config:
    """Configuration class for the Conformer ASR model and training."""
    def __init__(self):
        # Data paths
        self.train_data_path = "/ASR/input/train-clean-100"
        self.test_data_path = "/ASR/input/test-clean"
        
        # Model Parameters (Default values, will be updated by SimpleTokenizer)
        self.vocab_size = 29
        self.n_mels = 80
        self.sample_rate = 16000
        
        # Feature/Preprocessing Parameters
        self.wav_len = 30  # Max audio length in seconds for training
        self.downsample = 4 # Downsampling factor in DownSampler
        
        # Conformer Parameters
        self.dropout = 0.1
        self.hidDim = 512
        self.n_layer = 12
        self.nhead = 8
        self.headDim = 64
        self.conv_kernel_size = 31

        # Training Parameters
        self.nepochs = 10
        self.batch_size = 16
        self.base_lr = 1e-4
        self.clip = 1.0
        self.anneal_strategy = 'linear'
        self.pct_start = 0.3
        self.div_factor = 10
        self.final_div_factor = 1e8

        # Saving/Loading
        self.save_dir = "/ASR/working/checkpoints"
        self.model_save_path = "/ASR/input/final/final_model.pth"
        
        os.makedirs(self.save_dir, exist_ok=True)
        # Create a generic path for the final model save to be used in test.py
        self.generic_model_path = os.path.join(self.save_dir, "final_model.pth")


cfg = Config()