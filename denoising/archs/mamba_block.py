import torch
import torch.nn as nn
from timm.models.layers import DropPath
from typing import Optional, Tuple
from functools import partial
import math

class MambaBlock(nn.Module):
    """
    A Mamba block implementing the selective state space model.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Projection layers
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, device=device, dtype=dtype)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            device=device,
            dtype=dtype,
        )
        
        # State space parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, device=device, dtype=dtype)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, device=device, dtype=dtype)
        
        # Initialize dt
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner, device=device, dtype=dtype) * 
            (math.log(dt_max) - math.log(dt_min)) + 
            math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # State space matrices
        self.A_log = nn.Parameter(torch.log(torch.rand(self.d_inner, self.d_state, device=device, dtype=dtype)))
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device, dtype=dtype))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, device=device, dtype=dtype)
        
        self.layer_idx = layer_idx
        self.use_fast_path = use_fast_path

    def forward(self, hidden_states: torch.Tensor, inference_params=None) -> torch.Tensor:
        """
        Forward pass of the Mamba block.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, d_model)
            inference_params: Optional inference parameters
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project input
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        
        # Convolution
        x = x.transpose(1, 2)
        x = self.conv1d(x)[..., :seq_len]
        x = x.transpose(1, 2)
        
        # State space
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.transpose(1, 2)
        dt = dt.transpose(1, 2)
        
        # Selective scan
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        y = self.selective_scan(x, dt, A, B, C, D)
        
        # Output projection
        y = y * torch.silu(z)
        output = self.out_proj(y)
        
        return output

    def selective_scan(self, u, delta, A, B, C, D):
        """
        Selective scan operation.
        """
        batch_size, seq_len, d_inner = u.shape
        n = A.shape[1]
        
        # Discretize A and B
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        
        # Initialize state
        x = torch.zeros(batch_size, d_inner, n, device=u.device, dtype=u.dtype)
        
        # Scan
        ys = []
        for i in range(seq_len):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            y = torch.einsum('bnd,bn->bd', x, C[:, :, i])
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        y = y + u * D.unsqueeze(1)
        
        return y

def create_mamba_block(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    dt_rank: int = "auto",
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    dt_init: str = "random",
    dt_scale: float = 1.0,
    dt_init_floor: float = 1e-4,
    conv_bias: bool = True,
    bias: bool = False,
    use_fast_path: bool = True,
    layer_idx: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> MambaBlock:
    """
    Factory function to create a Mamba block with specified parameters.
    """
    return MambaBlock(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dt_rank=dt_rank,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_init=dt_init,
        dt_scale=dt_scale,
        dt_init_floor=dt_init_floor,
        conv_bias=conv_bias,
        bias=bias,
        use_fast_path=use_fast_path,
        layer_idx=layer_idx,
        device=device,
        dtype=dtype,
    ) 