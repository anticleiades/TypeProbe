import argparse
import ast
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow.parquet as pq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prober.prober_data import _build_prompt_from_row
from dataset.metadata import FIM_TOKENS_BY_MODEL

try:
    from tqdm import tqdm
except ModuleNotFoundError as exc:
    raise SystemExit("tqdm is required. Install it and re-run.") from exc

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

V2_DIR = ROOT_DIR / "dataset" / "v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))


def _row_from_columns(columns: Dict[str, List], idx: int) -> Dict[str, object]:
    return {name: values[idx] for name, values in columns.items()}


def _normalize_python_prompt(prompt: str, expected_func: str, model_name: str) -> str:
    fim = FIM_TOKENS_BY_MODEL.get(model_name)
    if fim is None:
        fim = next(iter(FIM_TOKENS_BY_MODEL.values()))

    # Same FIM replacement logic, but for Python
    prompt = prompt.replace(f"f_{fim['suffix']}", f"f_{expected_func}")
    for tag in (fim["prefix"], fim["middle"], fim["suffix"]):
        prompt = prompt.replace(tag, "")
    return prompt


_GLOBAL_COLUMNS = None


def _init_worker(dataset_path: Path) -> None:
    """Loads the dataset into memory for each isolated process, avoiding IPC overhead."""
    global _GLOBAL_COLUMNS
    _GLOBAL_COLUMNS = pq.read_table(dataset_path).to_pydict()


def _worker_task_python(indices: List[int], model_name: str) -> Dict[str, object]:
    """Task executed in parallel. Validates Python syntax purely in-memory."""
    oks = 0
    failures = []

    for idx in indices:
        row = _row_from_columns(_GLOBAL_COLUMNS, idx)
        # Note: use_java=False for Python
        prompt = _build_prompt_from_row(row, model_name, use_java=False)
        expected_idx = int(row["expectedFunctionIDX"])
        expected_func = row["func1Name"] if expected_idx == 0 else row["func2Name"]
        code = _normalize_python_prompt(prompt, expected_func, model_name)

        try:
            # Check Python syntax in-memory without executing the code
            ast.parse(code)
            oks += 1
        except SyntaxError as e:
            # Catch standard syntax errors
            failures.append({"idx": idx, "stderr": f"SyntaxError: {e.msg} at line {e.lineno}"})
        except Exception as e:
            # Catch any other unexpected exceptions
            failures.append({"idx": idx, "stderr": f"Exception: {str(e)}"})

    return {"oks": oks, "failures": failures}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--model", type=str, default="bigcode/santacoder")
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1)),
        help="Number of parallel workers.",
    )
    ap.add_argument(
        "--out-bad-indices",
        type=Path,
        default=None,
        help="Path to save the indices of failed validations as a text file.",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Number of samples to validate per worker batch. Defaults to dataset_size / workers.",
    )
    args = ap.parse_args()

    print("[1/3] Loading dataset...")
    # We only need to read the table length here; workers will load the full dataset
    table = pq.read_table(args.dataset)
    n_samples = table.num_rows

    if n_samples == 0:
        raise SystemExit("Dataset is empty; nothing to compile.")

    # All samples are eligible for Python validation
    eligible = list(range(n_samples))
    total_eligible = n_samples

    print(f"      Found {total_eligible} eligible samples (all rows).")

    workers = args.workers

    if args.chunk_size is not None and args.chunk_size > 0:
        chunk_size = args.chunk_size
    else:
        chunk_size = max(1, total_eligible // workers)

    chunks = [eligible[i: i + chunk_size] for i in range(0, total_eligible, chunk_size)]

    print(f"[2/3] Spawning {len(chunks)} batches across {workers} Python processes (chunk_size={chunk_size})...")
    failures = []
    total_oks = 0

    # Launch full parallel Multi-Processing for Python validation
    with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(args.dataset,),
    ) as ex:
        futures = [
            ex.submit(_worker_task_python, chunk, args.model)
            for chunk in chunks
        ]

        for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="AST Parsing",
                unit="chunk",
                smoothing=0.0
        ):
            res = fut.result()
            total_oks += res["oks"]
            failures.extend(res["failures"])

    failures.sort(key=lambda x: x["idx"])

    print("\n[3/3] Printing errors:")
    for fail in failures:
        print(f"  [fail] idx={fail['idx']} stderr={fail['stderr']}")

    print(f"\n[SUMMARY] valid={total_oks} failed={len(failures)} out of {total_eligible}")

    if args.out_bad_indices and failures:
        args.out_bad_indices.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_bad_indices, "w") as f:
            for fail in failures:
                f.write(f"{fail['idx']}\n")
        print(f"[INFO] Saved {len(failures)} bad indices to {args.out_bad_indices}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
