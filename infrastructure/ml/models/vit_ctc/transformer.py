import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, n_block, nhead, dim, drop_out, use_causal_mask=True):
        super().__init__()
        self.use_causal_mask = use_causal_mask
        self.transformer = self._init_transformer(n_block, nhead, dim, drop_out)

    def _init_transformer(self, n_block, nhead, dim, drop_out):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim,
            activation='relu',
            dropout=drop_out,
            batch_first=True
        )
        return nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_block,
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # For CTC, we want causal attention (each position only sees past)
        if self.use_causal_mask:
            # Use PyTorch's built-in causal masking
            seq_len = X.size(1)
            # Create attention mask: True = block, False = allow
            # Upper triangle (excluding diagonal) should be blocked
            attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=X.device), diagonal=1).bool()
            # DEBUG: Print mask info first time
            if not hasattr(self, '_mask_printed'):
                print(f"[DEBUG] Causal mask created: shape={attn_mask.shape}, device={attn_mask.device}")
                print(f"[DEBUG] Mask[0,:5]={attn_mask[0,:5].tolist()}")
                self._mask_printed = True
            return self.transformer(X, mask=attn_mask)
        else:
            return self.transformer(X)