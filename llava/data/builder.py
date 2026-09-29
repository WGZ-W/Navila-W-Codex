import os
from typing import Any, Optional

from hydra.utils import instantiate
from torch.utils.data import ConcatDataset, Dataset, IterableDataset
from transformers import PreTrainedTokenizer

from llava.data.datasets_mixture import DATASETS_LEGACY
from llava.train.args import DataArguments, TrainingArguments
from llava.utils import io
from llava.utils.logging import logger

__all__ = ["DATASETS", "register_datasets", "build_dataset"]


def register_datasets(name: Optional[str] = None):
    if name is None:
        name = os.environ.get("VILA_DATASETS", "default")
        logger.info(f"Registering datasets from `{name}`.")
    return io.load(os.path.join(os.path.dirname(__file__), "registry", f"{name}.yaml"))


DATASETS = register_datasets()


class RepeatedDataset(Dataset):
    def __init__(self, dataset: Dataset, times: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.times = times

    def __len__(self) -> int:
        return len(self.dataset) * self.times

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % len(self.dataset)]


def build_dataset(
    mixture: str,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    datasets = []
    for name in mixture.strip().lower().split("+"):
        if "*" in name:
            name, times = name.split("*")
            times = int(times)
        else:
            times = 1

        if name in DATASETS_LEGACY:
            logger.warning(f"Dataset {name} is registered under legacy mode.")
            dataset = build_dataset_legacy(
                name,
                data_args=data_args,
                training_args=training_args,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError(f"Dataset {name} is not registered.")

        if times > 1 and isinstance(dataset, IterableDataset):
            raise ValueError("Iterable RLDS datasets cannot use the '*N' repetition syntax")
        if times > 1:
            dataset = RepeatedDataset(dataset, times)
        datasets.append(dataset)
    iterable_datasets = [dataset for dataset in datasets if isinstance(dataset, IterableDataset)]
    if iterable_datasets:
        if len(datasets) != 1:
            raise ValueError("The OpenFly RLDS stream cannot currently be mixed with map-style datasets")
        return iterable_datasets[0]
    return ConcatDataset(datasets)


def build_dataset_legacy(
    name: str,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    from llava.data.dataset import DummyDataset, LazyEnvDropDataset, LazySupervisedDataset, LazyVLNCEDataset
    from llava.data.openfly import OpenFlyActionDataset, OpenFlyRLDSActionDataset

    dataset = DATASETS_LEGACY[name]
    dataset_type = dataset.dataset_type
    if dataset_type == "torch":
        dataset_cls = LazySupervisedDataset
    elif dataset_type == "envdrop":
        dataset_cls = LazyEnvDropDataset
    elif dataset_type == "vlnce":
        dataset_cls = LazyVLNCEDataset
    elif dataset_type == "openfly":
        dataset_cls = OpenFlyActionDataset
    elif dataset_type == "openfly_rlds":
        dataset_cls = OpenFlyRLDSActionDataset
    else:
        raise NotImplementedError(f"{dataset_type} is not supported.")

    data_args.meta_path = getattr(dataset, "meta_path", None)
    data_args.caption_choice = getattr(dataset, "caption_choice", None)
    data_args.caption_choice_2 = getattr(dataset, "caption_choice_2", None)
    data_args.start_idx = getattr(dataset, "start_idx", None)
    data_args.end_idx = getattr(dataset, "end_idx", None)

    return dataset_cls(
        tokenizer=tokenizer,
        data_path=dataset.data_path,
        image_folder=getattr(dataset, "image_path"),
        data_args=data_args,
        training_args=training_args,
    )
