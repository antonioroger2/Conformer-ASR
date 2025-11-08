import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from configs.config import cfg # Relative import

class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Self-Attention from Transformer."""
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()[:2]

        # 1. Linear projections and split heads
        q = self.w_q(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Masking: shape [B, 1, 1, S] -> [B, H, S, S]
            expanded_mask = mask.unsqueeze(1).unsqueeze(2) 
            expanded_mask = expanded_mask.expand(batch_size, self.n_head, seq_len, seq_len)
            scores = scores.masked_fill(~expanded_mask, -1e4) # Fill non-mask positions with a large negative number

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 3. Final linear layer
        output = self.w_o(context)
        return output

class FeedForward(nn.Module):
    """The Feed-Forward Module in Conformer (uses Swish/SiLU activation)."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # d_ff is typically 4 * d_model
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Conformer uses Swish/SiLU, but the original notebook used ReLU, so we'll stick to ReLU
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class ConvolutionModule(nn.Module):
    """The Convolution Module in Conformer."""
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Pointwise Conv 1 (to 2*d_model for GLU)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        
        # Depthwise Conv (groups=d_model for per-channel convolution)
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, padding=padding)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU() # SiLU is the standard Conformer activation
        
        # Pointwise Conv 2 (back to d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, T, D)
        x = self.layer_norm(x)
        x = x.transpose(1, 2) # (B, D, T) for Conv1d

        # 1. Pointwise Conv 1 + GLU
        x = self.pointwise_conv1(x)
        x = self.glu(x) # Output is (B, D, T)

        # 2. Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        # 3. Pointwise Conv 2
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2) # (B, T, D)
        return x

class ConformerBlock(nn.Module):
    """Single Conformer block, composed of four modules in a specific order."""
    def __init__(self, d_model, n_head, d_ff, kernel_size=31, dropout=0.1):
        super().__init__()
        
        # Feed Forward Module (with 0.5 scaling)
        self.ff1 = FeedForward(d_model, d_ff, dropout)
        
        # Multi-Head Self Attention Module
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        
        # Convolution Module
        self.conv_module = ConvolutionModule(d_model, kernel_size, dropout)
        
        # Second Feed Forward Module (with 0.5 scaling)
        self.ff2 = FeedForward(d_model, d_ff, dropout)

        # Layer Normalization layers for the attention, conv, and final output
        self.norm1 = nn.LayerNorm(d_model) # before FF1
        self.norm2 = nn.LayerNorm(d_model) # before Attention
        # Note: Convolution module has its own internal LayerNorm
        self.norm4 = nn.LayerNorm(d_model) # before FF2
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Feed-Forward (First Half)
        residual = x
        x = self.norm1(x)
        x = self.ff1(x)
        x = self.dropout(x * 0.5) + residual # Swish has replaced ReLU here, but the notebook used ReLU in FeedForward

        # 2. Multi-Head Self-Attention
        residual = x
        x = self.norm2(x)
        x = self.self_attn(x, mask)
        x = self.dropout(x) + residual

        # 3. Convolution Module (with internal norm)
        residual = x
        x = self.conv_module(x)
        x = x + residual

        # 4. Feed-Forward (Second Half)
        residual = x
        x = self.norm4(x) # The original code uses norm4, which is technically after the convolution, before the second FF
        x = self.ff2(x)
        x = self.dropout(x * 0.5) + residual
        
        # Final LayerNorm is applied after the Conformer stack in ConformerCTC
        return x

class DownSampler(nn.Module):
    """
    Downsamples the input Mel-spectrogram features (B, T, F) using 2D convolutions.
    Reduces the sequence length (T) and transforms the feature dimension (F) to hidden_dim.
    """
    def __init__(self, n_mels, hidden_dim, downsample_factor=4):
        super().__init__()
        # 2x2 downsampling via two strides of 2
        self.conv = nn.Sequential(
            # Input (B, 1, T, F). 2D Conv over T and F.
            nn.Conv2d(1, 256, kernel_size=3, stride=2, padding=1), # -> (B, 256, T/2, F/2)
            nn.SiLU(),
            # Depthwise Separable Conv part
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256), # -> (B, 256, T/4, F/4)
            nn.Conv2d(256, 256, kernel_size=1), # Pointwise Conv
            nn.SiLU(),
        )

        # Final projection to hidden_dim
        # F/4 is the remaining feature dimension after 2x2 downsampling.
        # This module assumes n_mels (F) is divisible by 4.
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(256 * (n_mels // downsample_factor), hidden_dim)
        )
        # Store for reference
        self.downsample_factor = downsample_factor

    def forward(self, x):
        # x shape: (B, T, F)
        x = x.unsqueeze(1) # Add channel dim: (B, 1, T, F)
        x = self.conv(x)   # (B, C=256, T/4, F/4)
        
        # Transpose and reshape to merge C and F dimensions
        x = x.permute(0, 2, 1, 3) # (B, T/4, C, F/4)
        B, T_new, C, F_new = x.size()
        x = x.reshape(B, T_new, C * F_new) # (B, T/4, C*F/4)
        
        x = self.out(x) # (B, T/4, hidden_dim)
        return x

class ConformerCTC(nn.Module):
    """The complete Conformer ASR model using CTC for training."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 1. Downsampler (reduces T and maps F to hidDim)
        self.downsampler = DownSampler(cfg.n_mels, cfg.hidDim, cfg.downsample)

        # 2. Conformer Blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=cfg.hidDim,
                n_head=cfg.nhead,
                d_ff=cfg.hidDim * 4,
                kernel_size=cfg.conv_kernel_size,
                dropout=cfg.dropout
            ) for _ in range(cfg.n_layer)
        ])

        # 3. Final LayerNorm and Classifier
        self.layer_norm = nn.LayerNorm(cfg.hidDim)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidDim, cfg.vocab_size)
        )
        
    def _create_length_mask(self, lengths, max_len):
        """Creates an attention mask from sequence lengths."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).expand(
            batch_size, max_len
        ) < lengths.unsqueeze(1)
        return mask.bool()

    def forward(self, x, lengths=None):
        # x shape: (B, T, F)

        # 1. Downsampling
        x = self.downsampler(x) # (B, T', D) where T' = T / downsample

        # 2. Create Attention Mask
        mask = None
        if lengths is not None:
            max_len = x.size(1)
            # The input lengths are already downsampled in collate_fn
            mask = self._create_length_mask(lengths, max_len)

        # 3. Pass through Conformer Blocks
        for block in self.conformer_blocks:
            x = block(x, mask)

        # 4. Final Layers
        x = self.layer_norm(x)
        logits = self.classifier(x) # (B, T', V)

        # 5. Log Softmax for CTC loss
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs