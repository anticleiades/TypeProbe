from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Subset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

V2_DIR = ROOT_DIR / "dataset"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from dataset.metadata import (  # noqa: E402
    FIM_TOKENS_BY_MODEL,
    Python_fncMap,
    get_type_list,
    java_gen_example,
    prober_meta_null_type,
    py_gen_example,
    type_to_class,
)

from dataset import function_bank_java as java_bank


def _load_parquet_columns(path: Path) -> Dict[str, List]:
    table = pq.read_table(path)
    return table.to_pydict()


def _row_from_columns(columns: Dict[str, List], idx: int) -> Dict[str, object]:
    return {name: values[idx] for name, values in columns.items()}


def _maybe_type_tag(type_name: str, has_tag: bool) -> str:
    return type_name if has_tag else ""


_TYPE_TO_TOKEN = {
    "int": "INT",
    "str": "STR",
    "bool": "BOOL",
    "float": "FLOAT",
    "list[int]": "LIST_INT",
    "list[str]": "LIST_STR",
    "list[bool]": "LIST_BOOL",
    "list[float]": "LIST_FLOAT",
}


def _func_list_name(in_type: str, out_type: str) -> str:
    try:
        return f"{_TYPE_TO_TOKEN[in_type]}_{_TYPE_TO_TOKEN[out_type]}_FUNCS"
    except KeyError as exc:
        raise KeyError(f"Unsupported type for function list: {exc}") from exc


def _build_prompt_from_row(row: Dict, model_name: str, *, use_java: bool) -> str:
    if use_java:
        func1_list = getattr(
            java_bank,
            _func_list_name(row["func1InputType"], row["func1OutType"]),
        )
        func2_list = getattr(
            java_bank,
            _func_list_name(row["func2InputType"], row["func2OutType"]),
        )
    else:
        func1_list = Python_fncMap[(row["func1InputType"], row["func1OutType"])]
        func2_list = Python_fncMap[(row["func2InputType"], row["func2OutType"])]

    func1_in_type = _maybe_type_tag(row["func1InputType"], row["func1InHasTypeTag"])
    func2_in_type = _maybe_type_tag(row["func2InputType"], row["func2InHasTypeTag"])
    func1_out_type = _maybe_type_tag(row["func1OutType"], row["func1OutHasTypeTag"])
    func2_out_type = _maybe_type_tag(row["func2OutType"], row["func2OutHasTypeTag"])

    a_type = _maybe_type_tag(row["aVarExpectedType"], row["aHasTypeTag"])
    b_type = _maybe_type_tag(row["b_expectedType"], row["bHasTypeTag"])
    if b_type == prober_meta_null_type:
        if use_java:
            raise ValueError("Cannot use null type with Java.")
        b_type = ""

    gen_example = java_gen_example if use_java else py_gen_example
    return gen_example(
        row["func1Name"],
        row["func2Name"],
        row["func1BodyIndex"],
        row["func2BodyIndex"],
        func1_list,
        func2_list,
        func1_out_type,
        func2_out_type,
        func1_in_type,
        func2_in_type,
        row["aVarValue"],
        a_type,
        b_type,
        row["f1ArgID"],
        row["f2ArgID"],
        row["aID"],
        row["bID"],
        model_name,
    )


class V2PromptDataset(Dataset):
    def __init__(
        self, parquet_path: Path, *, minimal: bool, model_name: str, use_java: bool
    ) -> None:
        self.parquet_path = parquet_path
        self.minimal = minimal
        self.model_name = model_name
        self.use_java = use_java
        self.columns = _load_parquet_columns(parquet_path)
        self.length = len(next(iter(self.columns.values()))) if self.columns else 0
        self.nil_class = len(get_type_list(self.minimal))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = _row_from_columns(self.columns, idx)
        prompt = _build_prompt_from_row(row, self.model_name, use_java=self.use_java)

        task0 = int(row["expectedFunctionIDX"])
        task1 = type_to_class(row["aVarExpectedType"], self.minimal)
        if row["b_expectedType"] == prober_meta_null_type:
            task2 = self.nil_class
        else:
            task2 = type_to_class(row["b_expectedType"], self.minimal)

        labels = torch.tensor([task0, task1, task2], dtype=torch.long)
        return {"prompt": prompt, "labels": labels}

    def get_row(self, idx: int) -> Dict[str, object]:
        return _row_from_columns(self.columns, idx)


class CachedActivationDataset(Dataset):
    def __init__(self, acts_path: Path, labels_path: Path) -> None:
        self.acts = np.load(acts_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        if self.acts.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"Cached acts rows ({self.acts.shape[0]}) "
                f"do not match labels rows ({self.labels.shape[0]})."
            )

    @property
    def in_features(self) -> int:
        return int(self.acts.shape[1])

    def __len__(self) -> int:
        return int(self.acts.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {"feats": self.acts[idx], "labels": self.labels[idx]}


class RandomTargetsDataset(Dataset):
    def __init__(self, dataset: Dataset, random_labels: np.ndarray) -> None:
        self.dataset = dataset
        self.random_labels = random_labels

    @property
    def in_features(self) -> int:
        return self.dataset.in_features

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.dataset[idx]
        labels = item["labels"]
        replacement = self.random_labels[idx]
        if isinstance(labels, torch.Tensor):
            item = dict(item)
            item["labels"] = torch.as_tensor(replacement, dtype=labels.dtype)
        else:
            item = dict(item)
            item["labels"] = replacement.astype(labels.dtype, copy=False)
        return item


def _normalize_prompt_for_validation(prompt: str, expected_func_name: str, model_name: str) -> str:
    if model_name not in FIM_TOKENS_BY_MODEL:
        raise ValueError(
            f"Unknown model '{model_name}' for FIM tokens. "
            "Add it to FIM_TOKENS_BY_MODEL."
        )
    fim = FIM_TOKENS_BY_MODEL[model_name]
    prompt = prompt.replace(f"f_{fim['suffix']}", f"f_{expected_func_name}")
    for tag in (fim["prefix"], fim["middle"], fim["suffix"]):
        prompt = prompt.replace(tag, "")
    return prompt


def _type_matches(value: object, type_name: str) -> bool:
    if type_name == "int":
        return type(value) is int
    if type_name == "str":
        return type(value) is str
    if type_name == "bool":
        return type(value) is bool
    if type_name == "float":
        return type(value) is float
    if type_name == "list[int]":
        return isinstance(value, list) and all(type(v) is int for v in value)
    if type_name == "list[str]":
        return isinstance(value, list) and all(type(v) is str for v in value)
    if type_name == "list[bool]":
        return isinstance(value, list) and all(type(v) is bool for v in value)
    if type_name == "list[float]":
        return isinstance(value, list) and all(type(v) is float for v in value)
    return False


def collate_batch(batch: List[Dict[str, object]]) -> Tuple[List[str], torch.Tensor]:
    prompts = [item["prompt"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    return prompts, labels


def collate_batch_cached(
    batch: List[Dict[str, np.ndarray]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = torch.stack([torch.from_numpy(item["feats"]) for item in batch], dim=0)
    labels = torch.stack([torch.from_numpy(item["labels"]) for item in batch], dim=0)
    return feats, labels


def _load_meta(path: Path) -> Dict[str, int]:
    table = pq.read_table(path)
    if table.num_rows != 1:
        raise ValueError(f"Expected 1 row in meta parquet, got {table.num_rows}")
    row = {name: table[name][0].as_py() for name in table.column_names}
    return {
        "nClassesTask0": int(row["nClassesTask0"]),
        "nClassesTask1": int(row["nClassesTask1"]),
        "nClassesTask2": int(row["nClassesTask2"]),
    }


def _infer_minimal(n_classes_task1: int) -> bool:
    return n_classes_task1 == len(get_type_list(True))


def _java_tagged_indices(dataset: V2PromptDataset) -> List[int]:
    cols = dataset.columns
    n_samples = dataset.length
    indices: List[int] = []
    for idx in range(n_samples):
        required = [
            cols["func1InHasTypeTag"][idx],
            cols["func2InHasTypeTag"][idx],
            cols["func1OutHasTypeTag"][idx],
            cols["func2OutHasTypeTag"][idx],
            cols["aHasTypeTag"][idx],
        ]
        if cols["b_expectedType"][idx] != prober_meta_null_type:
            required.append(cols["bHasTypeTag"][idx])
        if all(required):
            indices.append(idx)
    return indices


def _make_random_labels(*, n_samples: int, class_counts: List[int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.integers(0, n_cls, size=n_samples) for n_cls in class_counts],
        axis=1,
    ).astype(np.int64)


def get_dataset_path(dataset: Dataset) -> Optional[Path]:
    if isinstance(dataset, RandomTargetsDataset):
        return get_dataset_path(dataset.dataset)
    if hasattr(dataset, "parquet_path"):
        return dataset.parquet_path
    if isinstance(dataset, Subset):
        return get_dataset_path(dataset.dataset)
    return None


class GenericParquetPromptDataset(Dataset):
    def __init__(
        self,
        parquet_path: Path,
        *,
        prompt_col: str = "prompt",
        label_cols: Optional[List[str]] = None,
        labels_col: Optional[str] = None,
        infer_label_cols=None,
    ) -> None:
        self.parquet_path = parquet_path
        self.prompt_col = prompt_col
        self.label_cols = label_cols
        self.labels_col = labels_col
        self.columns = _load_parquet_columns(parquet_path)
        self.length = len(next(iter(self.columns.values()))) if self.columns else 0

        if self.prompt_col not in self.columns:
            raise ValueError(f"Missing prompt column '{self.prompt_col}' in {parquet_path}")

        if self.labels_col is None and self.label_cols is None:
            if "labels" in self.columns:
                self.labels_col = "labels"
            elif infer_label_cols is not None:
                self.label_cols = infer_label_cols(list(self.columns.keys()))

        if self.labels_col is None and self.label_cols is None:
            raise ValueError(
                "No label columns found. Provide --labels-col or --label-cols."
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = _row_from_columns(self.columns, idx)
        prompt = row[self.prompt_col]

        if self.labels_col is not None:
            labels_val = row[self.labels_col]
            if isinstance(labels_val, (list, tuple, np.ndarray)):
                labels = torch.tensor(labels_val, dtype=torch.long)
            else:
                labels = torch.tensor([labels_val], dtype=torch.long)
        else:
            labels = torch.tensor([row[col] for col in self.label_cols], dtype=torch.long)

        return {"prompt": prompt, "labels": labels}
