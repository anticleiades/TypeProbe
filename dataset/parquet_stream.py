from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq



def _normalize_and_coerce_record(rec: Dict[str, Any], schema: pa.Schema) -> Dict[str, Any]:
    """
    1) Normalize: keep ONLY schema fields; fill missing with None.
    2) Coerce: cast values to match schema types where reasonable.

    Important: Never stringify None into "None".
    """
    out: Dict[str, Any] = {}

    for field in schema:
        name = field.name
        typ = field.type
        v = rec.get(name, None)

        if v is None:
            out[name] = None
            continue

        # ints (your schema uses int64, but keep generic)
        if pa.types.is_integer(typ):
            if isinstance(v, bool):
                out[name] = int(v)
            elif isinstance(v, int):
                out[name] = v
            elif isinstance(v, str):
                try:
                    out[name] = int(v)
                except ValueError:
                    out[name] = None
            else:
                out[name] = None
            continue

        # strings
        if pa.types.is_string(typ) or pa.types.is_large_string(typ):
            out[name] = v if isinstance(v, str) else str(v)
            continue

        # fallback
        out[name] = v

    return out


# ----------------------------
# Dedupe
# ----------------------------

def record_fingerprint(rec: Dict[str, Any], keys: Sequence[str]) -> str:
    """
    Stable hash over selected keys to dedupe.
    """
    h = hashlib.blake2b(digest_size=16)
    for k in keys:
        v = rec.get(k, None)
        h.update(k.encode("utf-8"))
        h.update(b"=")
        h.update(("" if v is None else str(v)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


@dataclass
class ParquetStreamConfig:
    out_path: Path
    schema: pa.Schema
    batch_size: int = 10_000
    compression: str = "zstd"
    use_dictionary: bool = True

    # If True, dedupe within this one run (in-memory set).
    dedupe: bool = False
    dedupe_keys: Tuple[str, ...] = ("task", "variant", "a_type", "index", "f1_body", "f2_body", "prompt")

    # If True, delete existing out_path first (safer during iteration)
    overwrite: bool = True


class ParquetStreamWriter:
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
        rec2 = _normalize_and_coerce_record(rec, self.cfg.schema)

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


def write_records_to_parquet(
    records: Iterable[Dict[str, Any]],
    out_path: str | Path,
    *,
    schema: pa.Schema = None,
    batch_size: int = 10_000,
    compression: str = "zstd",
    use_dictionary: bool = True,
    dedupe: bool = False,
    dedupe_keys: Tuple[str, ...] = ("task", "variant", "a_type", "index", "f1_body", "f2_body", "prompt"),
    overwrite: bool = True,
) -> Tuple[int, int]:
    cfg = ParquetStreamConfig(
        out_path=Path(out_path),
        schema=schema,
        batch_size=batch_size,
        compression=compression,
        use_dictionary=use_dictionary,
        dedupe=dedupe,
        dedupe_keys=dedupe_keys,
        overwrite=overwrite,
    )
    w = ParquetStreamWriter(cfg)
    try:
        for r in records:
            w.add(r)
    finally:
        w.close()
    return w.written, w.skipped_dupes