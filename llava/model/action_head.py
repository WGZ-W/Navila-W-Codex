"""Discrete OpenFly action classification utilities.

The OpenFly simulator exposes ten discrete navigation actions.  The model
predicts one action ID and this module converts that ID to the eight-element
vector expected by the simulator.
"""

from typing import Sequence, Union

import numpy as np
import torch
from torch import nn


OPENFLY_ACTION_VECTORS = torch.tensor(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],  # 0: stop
        [0, 3, 0, 0, 0, 0, 0, 0],  # 1: move forward 3 m
        [0, 0, 15, 0, 0, 0, 0, 0],  # 2: turn left
        [0, 0, 0, 15, 0, 0, 0, 0],  # 3: turn right
        [0, 0, 0, 0, 2, 0, 0, 0],  # 4: move up
        [0, 0, 0, 0, 0, 2, 0, 0],  # 5: move down
        [0, 0, 0, 0, 0, 0, 5, 0],  # 6: move left
        [0, 0, 0, 0, 0, 0, 0, 5],  # 7: move right
        [0, 6, 0, 0, 0, 0, 0, 0],  # 8: move forward 6 m
        [0, 9, 0, 0, 0, 0, 0, 0],  # 9: move forward 9 m
    ],
    dtype=torch.float32,
)


def format_openfly_action_prompt(instruction: str) -> str:
    """Build the shared classifier prompt used for training and inference."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("OpenFly instructions must be non-empty strings")
    return f"Select the next OpenFly navigation action (0-9) for this instruction: {instruction.strip()}"


def action_ids_to_vectors(action_ids: torch.LongTensor) -> torch.Tensor:
    """Map a tensor of OpenFly action IDs to eight-dimensional vectors."""
    action_ids = torch.as_tensor(action_ids, dtype=torch.long)
    if action_ids.numel() and ((action_ids < 0).any() or (action_ids >= len(OPENFLY_ACTION_VECTORS)).any()):
        raise ValueError("OpenFly action IDs must be in the range [0, 9]")
    vectors = OPENFLY_ACTION_VECTORS.to(action_ids.device)
    return vectors[action_ids]


def action_vector_to_id(action: Union[Sequence[float], np.ndarray, torch.Tensor]) -> int:
    """Return the ID of an exact OpenFly action vector.

    Training data must contain a valid discrete action.  Silently mapping an
    unknown vector to STOP would corrupt the classifier labels, so invalid
    vectors are rejected here.
    """
    action_tensor = torch.as_tensor(action, dtype=torch.float32).cpu()
    if action_tensor.shape != (8,):
        raise ValueError(f"An OpenFly action vector must have shape (8,), got {tuple(action_tensor.shape)}")
    matches = torch.all(torch.isclose(OPENFLY_ACTION_VECTORS, action_tensor), dim=1)
    matched_ids = torch.where(matches)[0]
    if matched_ids.numel() != 1:
        raise ValueError(f"Unknown OpenFly action vector: {action_tensor.tolist()}")
    return int(matched_ids.item())


class OpenFlyActionHead(nn.Module):
    """Pool one multimodal sequence and classify it into an OpenFly action."""

    def __init__(self, hidden_size: int, num_actions: int = 10, dropout: float = 0.0) -> None:
        super().__init__()
        if num_actions != len(OPENFLY_ACTION_VECTORS):
            raise ValueError(f"OpenFly requires exactly {len(OPENFLY_ACTION_VECTORS)} actions, got {num_actions}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Action-head dropout must be in [0, 1)")
        self.num_actions = num_actions
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_actions)

    @staticmethod
    def pool_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Select the last non-padding token for right- or left-padded batches."""
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected hidden states with shape [B, L, D], got {tuple(hidden_states.shape)}")
        if attention_mask is None:
            return hidden_states[:, -1]
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError(
                f"Attention mask shape {tuple(attention_mask.shape)} does not match hidden states "
                f"{tuple(hidden_states.shape[:2])}"
            )
        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        last_positions = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
        if (last_positions < 0).any():
            raise ValueError("Cannot classify a sequence containing only padding")
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_indices, last_positions]

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        pooled = self.pool_last_token(hidden_states, attention_mask)
        return self.classifier(self.dropout(self.norm(pooled)))

    @staticmethod
    def action_ids_to_vectors(action_ids: torch.LongTensor) -> torch.Tensor:
        return action_ids_to_vectors(action_ids)
