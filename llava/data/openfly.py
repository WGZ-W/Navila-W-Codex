"""OpenFly datasets for discrete action classification."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from llava.constants import IGNORE_INDEX
from llava.mm_utils import process_images
from llava.model.action_head import action_vector_to_id, format_openfly_action_prompt
from llava.utils.media import extract_media
from llava.utils.tokenizer import tokenize_conversation


class OpenFlyRLDSActionDataset(IterableDataset):
    """Stream classification samples from the local ``vln_history`` TFDS dataset."""

    DATASET_NAME = "vln_history"

    def __init__(self, data_path, image_folder, tokenizer, data_args, training_args=None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.num_video_frames = int(getattr(data_args, "history_num_frames", 4))
        if self.num_video_frames <= 0:
            raise ValueError("num_video_frames must be positive")

        self.data_root, self.version_dir = self._resolve_data_root(data_path)
        self.num_samples = self._read_num_transitions(self.version_dir)
        self.seed = int(getattr(training_args, "seed", 42))
        self.shuffle = bool(getattr(training_args, "do_train", True))
        self.shuffle_buffer_size = int(os.environ.get("OPENFLY_RLDS_SHUFFLE_BUFFER", "10000"))
        if self.shuffle_buffer_size <= 0:
            raise ValueError("OPENFLY_RLDS_SHUFFLE_BUFFER must be positive")
        self._iteration = 0

    @classmethod
    def _resolve_data_root(cls, data_path) -> tuple[Path, Path]:
        path = Path(data_path).expanduser().resolve()
        data_root = path.parent if path.name == cls.DATASET_NAME else path
        dataset_dir = data_root / cls.DATASET_NAME
        if not dataset_dir.is_dir():
            raise FileNotFoundError(
                f"Cannot find the {cls.DATASET_NAME} TFDS directory under {data_root}. "
                "Set OPENFLY_RLDS_ROOT to the directory containing vln_history."
            )
        versions = []
        for candidate in dataset_dir.iterdir():
            version_parts = candidate.name.split(".")
            if candidate.is_dir() and version_parts and all(part.isdigit() for part in version_parts):
                versions.append(candidate)
        versions.sort(key=lambda candidate: tuple(int(part) for part in candidate.name.split(".")))
        if not versions:
            raise FileNotFoundError(f"No version directory found under {dataset_dir}")
        version_dir = versions[-1]
        if not (version_dir / "dataset_info.json").is_file():
            raise FileNotFoundError(f"Missing TFDS metadata: {version_dir / 'dataset_info.json'}")
        return data_root, version_dir

    @staticmethod
    def _read_num_transitions(version_dir: Path) -> int:
        statistics_paths = sorted(version_dir.glob("dataset_statistics_*.json"))
        if not statistics_paths:
            raise FileNotFoundError(f"Missing dataset_statistics_*.json under {version_dir}")
        with statistics_paths[-1].open("r", encoding="utf-8") as handle:
            num_transitions = int(json.load(handle)["num_transitions"])
        if num_transitions <= 0:
            raise ValueError(f"Invalid num_transitions={num_transitions} in {statistics_paths[-1]}")
        return num_transitions

    @staticmethod
    def _decode_instruction(value) -> str:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Every vln_history step must contain a non-empty language_instruction")
        return value.strip()

    def _history_images(self, step) -> List[Image.Image]:
        try:
            history = step["observation"]["history_images"]
        except KeyError as error:
            raise ValueError("A vln_history step is missing observation.history_images") from error
        if getattr(history, "ndim", None) != 4 or history.shape[-1] != 3:
            raise ValueError(f"Expected history_images shaped [T,H,W,3], got {getattr(history, 'shape', None)}")
        if len(history) == 0:
            raise ValueError("A vln_history step contains no history images")
        history = list(history[-self.num_video_frames :])
        history = [history[0]] * (self.num_video_frames - len(history)) + history
        return [Image.fromarray(frame).convert("RGB") for frame in history]

    def _transform_step(self, step) -> Dict[str, torch.Tensor]:
        instruction = self._decode_instruction(step["language_instruction"])
        action_id = action_vector_to_id(step["action"])
        history_images = self._history_images(step)
        current = Image.fromarray(step["observation"]["image_1"]).convert("RGB")
        conversation = [
            {
                "from": "human",
                "value": [current, format_openfly_action_prompt(instruction)],
            }
        ]
        media = extract_media(conversation, self.data_args)
        input_ids = tokenize_conversation(conversation, self.tokenizer, add_generation_prompt=True)
        history_pixels = self.data_args.image_processor.preprocess(history_images, return_tensors="pt")["pixel_values"]
        return {
            "input_ids": input_ids,
            "labels": torch.full_like(input_ids, IGNORE_INDEX),
            "image": process_images(media["image"], self.data_args.image_processor, self.data_args),
            "history_images": history_pixels,
            "action_labels": torch.tensor(action_id, dtype=torch.long),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        try:
            import tensorflow as tf
            import tensorflow_datasets as tfds
        except ImportError as error:
            raise ImportError(
                "OpenFly RLDS training requires tensorflow and tensorflow-datasets. "
                "Install the project's 'rlds' optional dependencies."
            ) from error

        # TensorFlow is only a CPU input pipeline here. Do not let it reserve
        # accelerator memory needed by PyTorch and DeepSpeed.
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass

        iteration_seed = self.seed + self._iteration
        self._iteration += 1
        read_config = tfds.ReadConfig(shuffle_seed=iteration_seed)
        builder = tfds.builder(self.DATASET_NAME, data_dir=str(self.data_root))
        episodes = builder.as_dataset(split="train", shuffle_files=self.shuffle, read_config=read_config)

        worker = get_worker_info()
        if worker is not None:
            episodes = episodes.shard(num_shards=worker.num_workers, index=worker.id)
        if self.shuffle:
            episodes = episodes.shuffle(1024, seed=iteration_seed, reshuffle_each_iteration=True)

        steps = episodes.interleave(
            lambda episode: episode["steps"],
            cycle_length=16,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not self.shuffle,
        )
        if self.shuffle:
            steps = steps.shuffle(
                self.shuffle_buffer_size,
                seed=iteration_seed,
                reshuffle_each_iteration=True,
            )
        steps = steps.prefetch(2)
        for step in tfds.as_numpy(steps):
            yield self._transform_step(step)

    def __len__(self) -> int:
        return self.num_samples


class OpenFlyActionDataset(Dataset):
    """Expand OpenFly episodes into image-history/action training samples.

    Supported annotation forms are:

    * OpenFly episodes with ``image_path``, ``gpt_instruction`` and an
      ``action``/``actions`` sequence of IDs.
    * Flattened samples with ``image``/``images``, ``instruction`` and either
      ``action_id`` or an eight-dimensional ``action_vector``.
    """

    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

    def __init__(self, data_path, image_folder, tokenizer, data_args, training_args=None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = Path(image_folder or ".")
        self.num_video_frames = int(getattr(data_args, "history_num_frames", 4)) + 1
        if self.num_video_frames <= 0:
            raise ValueError("num_video_frames must be positive")

        with open(data_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("episodes", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("OpenFly annotations must be a list or a dictionary containing 'episodes'")

        self.samples: List[Dict[str, Any]] = []
        for record in records:
            self.samples.extend(self._expand_record(record))
        if not self.samples:
            raise ValueError(f"No OpenFly action samples were found in {data_path}")

    @staticmethod
    def _instruction(record: Dict[str, Any]) -> str:
        instruction = record.get("instruction", record.get("gpt_instruction"))
        if isinstance(instruction, dict):
            instruction = instruction.get("instruction_text")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Every OpenFly record must contain a non-empty instruction")
        return instruction.strip()

    @staticmethod
    def _coerce_action_id(action: Any) -> int:
        if isinstance(action, bool):
            raise ValueError("Boolean values are not valid OpenFly actions")
        if isinstance(action, (int, float)) and int(action) == action:
            action_id = int(action)
        elif isinstance(action, (list, tuple)) and len(action) == 8:
            action_id = action_vector_to_id(action)
        else:
            raise ValueError(f"Unsupported OpenFly action value: {action!r}")
        # Some raw OpenFly trajectory files encode vertical movement as -1/-2,
        # while the evaluator uses canonical IDs 4/5 for the same actions.
        action_id = {-1: 4, -2: 5}.get(action_id, action_id)
        if not 0 <= action_id < 10:
            raise ValueError(f"OpenFly action ID must be in [0, 9], got {action_id}")
        return action_id

    def _expand_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        instruction = self._instruction(record)
        if "action_id" in record:
            return [
                {
                    "instruction": instruction,
                    "action_id": self._coerce_action_id(record["action_id"]),
                    "images": record.get("images", record.get("image")),
                    "episode_dir": record.get("image_path"),
                    "frame_index": record.get("frame_index", 0),
                    "frames": record.get("frames"),
                }
            ]

        if "action_vector" in record:
            return [
                {
                    "instruction": instruction,
                    "action_id": self._coerce_action_id(record["action_vector"]),
                    "images": record.get("images", record.get("image")),
                    "episode_dir": record.get("image_path"),
                    "frame_index": record.get("frame_index", 0),
                    "frames": record.get("frames"),
                }
            ]

        actions = record.get("actions", record.get("action"))
        if actions is None:
            raise ValueError("Every OpenFly record must contain action_id, action_vector, action, or actions")
        if not isinstance(actions, (list, tuple)):
            actions = [actions]

        return [
            {
                "instruction": instruction,
                "action_id": self._coerce_action_id(action),
                "images": None,
                "episode_dir": record.get("image_path"),
                "frame_index": step,
                "frames": record.get("frames"),
            }
            for step, action in enumerate(actions)
        ]

    def _resolve_path(self, path: str, episode_dir: str = None) -> Path:
        if isinstance(path, dict):
            path = path.get("path")
        if not path:
            raise ValueError("An OpenFly frame entry is missing its path")
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        if episode_dir:
            episode_candidate = self.image_folder / episode_dir / candidate
            if episode_candidate.is_file():
                return episode_candidate
        return self.image_folder / candidate

    def _numbered_frame(self, episode_dir: str, frame_index: int) -> Path:
        if not episode_dir:
            raise ValueError("Episode annotations require image_path or an explicit frames list")
        directory = Path(episode_dir)
        if not directory.is_absolute():
            directory = self.image_folder / directory
        # OpenFly trajectory builders commonly append two STOP labels while
        # reusing the last recorded image.  Search backwards to reproduce that
        # behavior when the final action has no same-index frame file.
        for index in range(frame_index, -1, -1):
            for extension in self.IMAGE_EXTENSIONS:
                candidate = directory / f"{index}{extension}"
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(f"Cannot find frame {frame_index} under {directory}")

    def _frame_path(self, sample: Dict[str, Any], frame_index: int) -> Path:
        frames = sample.get("frames")
        if frames:
            frame_index = min(frame_index, len(frames) - 1)
            return self._resolve_path(frames[frame_index], sample.get("episode_dir"))
        return self._numbered_frame(sample.get("episode_dir"), frame_index)

    def _history_paths(self, sample: Dict[str, Any]) -> Sequence[Path]:
        explicit_images = sample.get("images")
        if explicit_images:
            if isinstance(explicit_images, (str, os.PathLike)):
                explicit_images = [explicit_images]
            paths = [self._resolve_path(path, sample.get("episode_dir")) for path in explicit_images]
            return paths[-self.num_video_frames :]

        current = int(sample["frame_index"])
        first = max(0, current - self.num_video_frames + 1)
        indices = list(range(first, current + 1))
        indices = [indices[0]] * (self.num_video_frames - len(indices)) + indices
        return [self._frame_path(sample, index) for index in indices]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        paths = self._history_paths(sample)
        pil_images = []
        for path in paths:
            with Image.open(path) as image:
                pil_images.append(image.convert("RGB").copy())

        current_image = pil_images[-1]
        history_images = pil_images[:-1]
        if not history_images:
            history_images = [current_image]
        history_images = [history_images[0]] * (self.num_video_frames - min(len(history_images), self.num_video_frames)) + history_images[-self.num_video_frames :]
        conversation = [
            {
                "from": "human",
                "value": [current_image, format_openfly_action_prompt(sample["instruction"])],
            }
        ]
        media = extract_media(conversation, self.data_args)
        input_ids = tokenize_conversation(conversation, self.tokenizer, add_generation_prompt=True)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        images = process_images(media["image"], self.data_args.image_processor, self.data_args)
        history_pixels = self.data_args.image_processor.preprocess(history_images, return_tensors="pt")["pixel_values"]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "image": images,
            "history_images": history_pixels,
            "action_labels": torch.tensor(sample["action_id"], dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def lengths(self) -> List[int]:
        return [len(sample["instruction"].split()) + 128 * self.num_video_frames for sample in self.samples]

    @property
    def modality_lengths(self) -> List[int]:
        return self.lengths
