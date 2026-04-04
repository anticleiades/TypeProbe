from __future__ import annotations

from typing import Any, Dict, Optional, List

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from dataset.parquet_stream import ParquetStreamConfig, record_fingerprint
except ModuleNotFoundError:
    from parquet_stream import ParquetStreamConfig, record_fingerprint


def normalize_record_strict(rec: Dict[str, Any], schema: pa.Schema) -> Dict[str, Any]:
    """
    Normalize record to schema fields only and enforce strict type matches.
    """
    out: Dict[str, Any] = {}
    def _list_elem_ok(elem, elem_type: pa.DataType) -> bool:
        if pa.types.is_boolean(elem_type):
            return isinstance(elem, bool)
        if pa.types.is_integer(elem_type):
            return isinstance(elem, int) and not isinstance(elem, bool)
        if pa.types.is_floating(elem_type):
            return isinstance(elem, float)
        if pa.types.is_string(elem_type) or pa.types.is_large_string(elem_type):
            return isinstance(elem, str)
        if pa.types.is_struct(elem_type):
            return isinstance(elem, dict)
        if pa.types.is_list(elem_type) or pa.types.is_large_list(elem_type):
            return isinstance(elem, list)
        return True

    for field in schema:
        name = field.name
        typ = field.type
        v = rec.get(name, None)

        if v is None:
            out[name] = None
            continue

        if pa.types.is_boolean(typ):
            if isinstance(v, bool):
                out[name] = v
                continue
            raise TypeError(
                f"Field '{name}' expects bool but got {type(v).__name__}: {v!r}"
            )

        if pa.types.is_integer(typ):
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(
                    f"Field '{name}' expects int but got {type(v).__name__}: {v!r}"
                )
            out[name] = v
            continue

        if pa.types.is_floating(typ):
            if not isinstance(v, float):
                raise TypeError(
                    f"Field '{name}' expects float but got {type(v).__name__}: {v!r}"
                )
            out[name] = v
            continue

        if pa.types.is_string(typ) or pa.types.is_large_string(typ):
            if not isinstance(v, str):
                raise TypeError(
                    f"Field '{name}' expects str but got {type(v).__name__}: {v!r}"
                )
            out[name] = v
            continue

        if pa.types.is_struct(typ):
            if not isinstance(v, dict):
                raise TypeError(
                    f"Field '{name}' expects dict but got {type(v).__name__}: {v!r}"
                )
            out[name] = v
            continue

        if pa.types.is_list(typ) or pa.types.is_large_list(typ):
            if not isinstance(v, list):
                raise TypeError(
                    f"Field '{name}' expects list but got {type(v).__name__}: {v!r}"
                )
            elem_type = typ.value_type
            for i, elem in enumerate(v):
                if elem is None:
                    raise TypeError(
                        f"Field '{name}' list element {i} is None; expected {elem_type}"
                    )
                if not _list_elem_ok(elem, elem_type):
                    raise TypeError(
                        f"Field '{name}' list element {i} expects {elem_type} but got {type(elem).__name__}: {elem!r}"
                    )
            out[name] = v
            continue

        out[name] = v

    return out


class ParquetStreamWriterV2:
    def __init__(self, cfg: ParquetStreamConfig):
        self.cfg = cfg
        self.cfg.out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.cfg.overwrite and self.cfg.out_path.exists():
            self.cfg.out_path.unlink()

        self._writer = pq.ParquetWriter(
            str(self.cfg.out_path),
            schema=self.cfg.schema,
            compression=self.cfg.compression,
            use_dictionary=self.cfg.use_dictionary,
        )
        self._buf: List[Dict[str, Any]] = []
        self._seen: Optional[set[str]] = set() if self.cfg.dedupe else None
        self.written = 0
        self.skipped_dupes = 0

    def add(self, rec: Dict[str, Any]) -> None:
        rec2 = normalize_record_strict(rec, self.cfg.schema)

        if self._seen is not None:
            fp = record_fingerprint(rec2, self.cfg.dedupe_keys)
            if fp in self._seen:
                self.skipped_dupes += 1
                return
            self._seen.add(fp)

        self._buf.append(rec2)
        if len(self._buf) >= self.cfg.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        table = pa.Table.from_pylist(self._buf, schema=self.cfg.schema)
        self._writer.write_table(table)
        self.written += len(self._buf)
        self._buf.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()
