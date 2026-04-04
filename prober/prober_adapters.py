from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


import pyarrow.parquet as pq
from torch.utils.data import Dataset, Subset

from prober_data import (
    GenericParquetPromptDataset,
    V2PromptDataset,
    _infer_minimal,
    _java_tagged_indices,
    _load_meta,
)


@dataclass
class TaskMeta:
    class_counts: List[int]
    minimal: Optional[bool] = None


def _parse_class_counts(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _load_class_counts_from_json(path: Path) -> List[int]:
    import json

    data = json.loads(path.read_text())
    if "class_counts" in data:
        return [int(x) for x in data["class_counts"]]
    if "nClassesTask0" in data:
        counts = []
        idx = 0
        while True:
            key = f"nClassesTask{idx}"
            if key not in data:
                break
            counts.append(int(data[key]))
            idx += 1
        if counts:
            return counts
    raise ValueError(f"Unrecognized meta JSON format: {path}")


def _load_class_counts_from_parquet(path: Path) -> List[int]:
    table = pq.read_table(path)
    if table.num_rows != 1:
        raise ValueError(f"Expected 1 row in meta parquet, got {table.num_rows}")
    cols = table.column_names
    if "class_counts" in cols:
        value = table["class_counts"][0].as_py()
        if isinstance(value, (list, tuple)):
            return [int(x) for x in value]
    if "nClassesTask0" in cols:
        idx = 0
        counts = []
        while True:
            key = f"nClassesTask{idx}"
            if key not in cols:
                break
            counts.append(int(table[key][0].as_py()))
            idx += 1
        if counts:
            return counts
    raise ValueError(f"Unrecognized meta parquet format: {path}")


def _load_task_meta(path: Path, *, class_counts_override: Optional[str] = None) -> TaskMeta:
    if class_counts_override:
        return TaskMeta(class_counts=_parse_class_counts(class_counts_override))
    if path.suffix.lower() == ".json":
        return TaskMeta(class_counts=_load_class_counts_from_json(path))
    return TaskMeta(class_counts=_load_class_counts_from_parquet(path))


def _parse_label_cols(label_cols: Optional[str]) -> Optional[List[str]]:
    if label_cols is None:
        return None
    cols = [c.strip() for c in label_cols.split(",") if c.strip()]
    return cols or None


def _infer_label_cols(columns: List[str]) -> Optional[List[str]]:
    label_cols = [c for c in columns if c.startswith("label_")]
    if not label_cols:
        return None
    def key_fn(name: str) -> int:
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return 0
    return sorted(label_cols, key=key_fn)


class BaseAdapter:
    name = "base"

    def load_meta(self, meta_path: Path, args) -> TaskMeta:
        return _load_task_meta(meta_path, class_counts_override=args.class_counts)

    def build_dataset(self, args, meta: TaskMeta) -> Dataset:
        raise NotImplementedError

    def apply_java_filter(self, dataset: Dataset, args) -> Dataset:
        return dataset


class V2Adapter(BaseAdapter):
    name = "v2"

    def load_meta(self, meta_path: Path, args) -> TaskMeta:
        meta = _load_meta(meta_path)
        minimal = _infer_minimal(meta["nClassesTask1"])
        class_counts = [
            meta["nClassesTask0"],
            meta["nClassesTask1"],
            meta["nClassesTask2"],
        ]
        return TaskMeta(class_counts=class_counts, minimal=minimal)

    def build_dataset(self, args, meta: TaskMeta) -> Dataset:
        if meta.minimal is None:
            raise ValueError("V2 adapter requires minimal flag in metadata.")
        return V2PromptDataset(
            args.dataset, minimal=meta.minimal, model_name=args.model, use_java=args.java
        )

    def apply_java_filter(self, dataset: Dataset, args) -> Dataset:
        if not args.java:
            return dataset
        eligible_indices = _java_tagged_indices(dataset)
        if not eligible_indices:
            raise ValueError("No rows with all required type tags for --java.")
        return Subset(dataset, eligible_indices)


class ParquetAdapter(BaseAdapter):
    name = "parquet"

    def build_dataset(self, args, meta: TaskMeta) -> Dataset:
        label_cols = _parse_label_cols(args.label_cols)
        labels_col = args.labels_col
        return GenericParquetPromptDataset(
            args.dataset,
            prompt_col=args.prompt_col,
            label_cols=label_cols,
            labels_col=labels_col,
            infer_label_cols=_infer_label_cols,
        )


_ADAPTERS = {
    "v2": V2Adapter,
    "parquet": ParquetAdapter,
}


def get_adapter(name: str) -> BaseAdapter:
    key = name.lower()
    if key not in _ADAPTERS:
        raise ValueError(f"Unknown adapter: {name}. Available: {sorted(_ADAPTERS)}")
    return _ADAPTERS[key]()
