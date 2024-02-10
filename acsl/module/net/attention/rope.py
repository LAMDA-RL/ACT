import torch

class RoPEEncoding(torch.nn.Module):
    def __init__(
        self, 
        dim, 
        max_len, 
        base=10000
    ) -> None:
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.matmul(t.unsqueeze(1), inv_freq.unsqueeze(0))
        self.register_buffer("sin", freqs.sin())
        self.register_buffer("cos", freqs.cos())
        
    def forward(self, x, timesteps):
        B, L, *_ = x.shape
        sin, cos = self.sin[timesteps, :], self.cos[timesteps, :]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        ret = torch.stack([
            x1*cos - x2*sin, x1*sin+x2*cos
        ], axis=-1).reshape(B, L, *_)
        return ret