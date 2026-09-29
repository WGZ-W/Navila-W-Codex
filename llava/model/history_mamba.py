"""Journal-style temporal visual encoder for OpenFly history frames."""

from typing import Optional

import torch
from mamba_ssm import Mamba
from torch import nn


class VisionMambaBlock(nn.Module):
    def __init__(self, hidden_size: int, state_size: int = 16, conv_size: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.mamba = Mamba(
            d_model=hidden_size,
            d_state=state_size,
            d_conv=conv_size,
            expand=expand,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mamba(hidden_states)


class HistoryMamba(nn.Module):
    """Encode a fixed sequence of resized history frames into patch tokens.

    This follows Journal's dual-stream design: current-frame visual tokens are
    produced by NaVILA's existing vision tower, while past frames pass through
    this independent patch-embedding and Mamba stream.
    """

    def __init__(
        self,
        num_frames: int = 4,
        image_size: int = 384,
        patch_size: int = 32,
        hidden_size: int = 768,
        depth: int = 4,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("History image size must be divisible by the patch size")
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        module_dtype = dtype or torch.float32

        self.patch_embed = nn.Conv2d(
            3,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            dtype=module_dtype,
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_frames * self.num_patches_per_frame, hidden_size, dtype=module_dtype)
        )
        self.blocks = nn.ModuleList([VisionMambaBlock(hidden_size) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_size, dtype=module_dtype)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"Expected history images [B,T,3,H,W], received {tuple(images.shape)}")
        batch_size, num_frames, channels, height, width = images.shape
        if num_frames != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} history frames, received {num_frames}")
        if channels != 3 or (height, width) != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected history frames [B,{self.num_frames},3,{self.image_size},{self.image_size}], "
                f"received {tuple(images.shape)}"
            )

        images = images.reshape(batch_size * num_frames, channels, height, width)
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        patches = patches.reshape(batch_size, num_frames * self.num_patches_per_frame, self.hidden_size)
        hidden_states = patches + self.position_embedding.to(dtype=patches.dtype)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.norm(hidden_states)


def build_history_mamba(config, checkpoint_path: Optional[str] = None) -> HistoryMamba:
    dtype = getattr(config, "model_dtype", "torch.float16")
    dtype = eval(dtype) if isinstance(dtype, str) else dtype
    model = HistoryMamba(
        num_frames=getattr(config, "history_num_frames", 4),
        image_size=getattr(config, "history_image_size", 384),
        patch_size=getattr(config, "history_patch_size", 32),
        hidden_size=getattr(config, "history_hidden_size", 768),
        depth=getattr(config, "history_depth", 4),
        dtype=dtype,
    )
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    return model.to(dtype=dtype)
