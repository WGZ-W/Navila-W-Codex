import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llava.model.action_head import (
    OPENFLY_ACTION_VECTORS,
    OpenFlyActionHead,
    action_ids_to_vectors,
    action_vector_to_id,
    format_openfly_action_prompt,
)
from llava.data.openfly import OpenFlyActionDataset, OpenFlyRLDSActionDataset


def test_openfly_action_mapping_round_trip():
    action_ids = torch.arange(10)
    vectors = action_ids_to_vectors(action_ids)
    assert torch.equal(vectors, OPENFLY_ACTION_VECTORS)
    assert [action_vector_to_id(vector) for vector in vectors] == list(range(10))


def test_unknown_action_vector_is_rejected():
    with pytest.raises(ValueError, match="Unknown OpenFly action vector"):
        action_vector_to_id(np.ones(8, dtype=np.float32))


def test_action_prompt_is_shared_and_validated():
    assert format_openfly_action_prompt("go to the tower").endswith("go to the tower")
    with pytest.raises(ValueError, match="non-empty"):
        format_openfly_action_prompt(" ")


def test_legacy_vertical_action_ids_are_canonicalized():
    assert OpenFlyActionDataset._coerce_action_id(-1) == 4
    assert OpenFlyActionDataset._coerce_action_id(-2) == 5


def test_rlds_dataset_accepts_root_or_dataset_directory(tmp_path):
    version_dir = tmp_path / "vln_history" / "1.0.0"
    version_dir.mkdir(parents=True)
    (tmp_path / "vln_history" / "corruption").mkdir()
    (version_dir / "dataset_info.json").write_text("{}", encoding="utf-8")
    (version_dir / "dataset_statistics_test.json").write_text(json.dumps({"num_transitions": 123}), encoding="utf-8")
    data_args = SimpleNamespace(num_video_frames=8)

    from_root = OpenFlyRLDSActionDataset(tmp_path, None, None, data_args)
    from_dataset_dir = OpenFlyRLDSActionDataset(tmp_path / "vln_history", None, None, data_args)

    assert len(from_root) == 123
    assert from_root.data_root == tmp_path
    assert from_dataset_dir.data_root == tmp_path


def test_rlds_history_frames_are_trimmed_and_left_padded(tmp_path):
    version_dir = tmp_path / "vln_history" / "1.0.0"
    version_dir.mkdir(parents=True)
    (version_dir / "dataset_info.json").write_text("{}", encoding="utf-8")
    (version_dir / "dataset_statistics_test.json").write_text(json.dumps({"num_transitions": 1}), encoding="utf-8")
    dataset = OpenFlyRLDSActionDataset(tmp_path, None, None, SimpleNamespace(num_video_frames=4))
    history = np.stack([np.full((2, 2, 3), value, dtype=np.uint8) for value in (1, 2)])

    frames = dataset._history_images({"observation": {"history_images": history}})

    assert [np.asarray(frame)[0, 0, 0] for frame in frames] == [1, 1, 1, 2]


def test_pool_last_token_handles_left_and_right_padding():
    hidden_states = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    attention_mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=torch.bool)
    pooled = OpenFlyActionHead.pool_last_token(hidden_states, attention_mask)
    assert torch.equal(pooled[0], hidden_states[0, 1])
    assert torch.equal(pooled[1], hidden_states[1, 2])


def test_action_head_outputs_ten_logits_and_backpropagates():
    head = OpenFlyActionHead(hidden_size=8, dropout=0.0)
    hidden_states = torch.randn(3, 5, 8, requires_grad=True)
    attention_mask = torch.ones(3, 5, dtype=torch.bool)
    logits = head(hidden_states, attention_mask)
    assert logits.shape == (3, 10)
    logits.sum().backward()
    assert hidden_states.grad is not None
