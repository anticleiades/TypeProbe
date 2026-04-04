import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow.parquet as pq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prober.prober_data import _build_prompt_from_row
from dataset.metadata import FIM_TOKENS_BY_MODEL, prober_meta_null_type

try:
    from tqdm import tqdm
except ModuleNotFoundError as exc:
    raise SystemExit(
        "tqdm is required for progress bars. Install it in your environment and re-run."
    ) from exc

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

V2_DIR = ROOT_DIR / "dataset" / "v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

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
    return f"{_TYPE_TO_TOKEN[in_type]}_{_TYPE_TO_TOKEN[out_type]}_FUNCS"


def _row_from_columns(columns: Dict[str, List], idx: int) -> Dict[str, object]:
    return {name: values[idx] for name, values in columns.items()}


def _maybe_type_tag(type_name: str, has_tag: bool) -> str:
    return type_name if has_tag else ""


def _normalize_java_prompt(prompt: str, expected_func: str, model_name: str) -> str:
    fim = FIM_TOKENS_BY_MODEL.get(model_name)
    if fim is None:
        fim = next(iter(FIM_TOKENS_BY_MODEL.values()))
    prompt = prompt.replace(f"f_{fim['suffix']}", f"f_{expected_func}")
    for tag in (fim["prefix"], fim["middle"], fim["suffix"]):
        prompt = prompt.replace(tag, "")
    return prompt



_GLOBAL_COLUMNS = None


def _init_worker(dataset_path: Path) -> None:
    """Loads the dataset into memory for each isolated process, avoiding IPC overhead."""
    global _GLOBAL_COLUMNS
    _GLOBAL_COLUMNS = pq.read_table(dataset_path).to_pydict()


def _worker_task(chunk_id: int, indices: List[int], model_name: str, tmp_dir_str: str) -> Dict[str, object]:
    """Task executed in parallel. Creates Java files and invokes the optimized BatchCompiler."""
    chunk_dir = Path(tmp_dir_str) / str(chunk_id)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    filelist_path = chunk_dir / "filelist.txt"

    # 1. Generate prompts and perform massive writes to RAM disk
    with open(filelist_path, "w", encoding="utf-8") as flist:
        for idx in indices:
            row = _row_from_columns(_GLOBAL_COLUMNS, idx)
            prompt = _build_prompt_from_row(row, model_name, use_java=True)
            expected_idx = int(row["expectedFunctionIDX"])
            expected_func = row["func1Name"] if expected_idx == 0 else row["func2Name"]
            code = _normalize_java_prompt(prompt, expected_func, model_name)

            src_path = chunk_dir / f"{idx}.java"
            src_path.write_text(code, encoding="utf-8")
            flist.write(f"{src_path.absolute()}\n")

    # compile files everything in the same JVM, to reduce the overhead
    cmd = ["java", "-cp", tmp_dir_str, "BatchCompiler", str(filelist_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    oks = 0
    failures = []

    # Decode the standard output from our daemon
    for line in result.stdout.splitlines():
        if line.startswith("OK"):
            oks += 1
        elif line.startswith("FAIL"):
            parts = line.split("\t", 2)
            if len(parts) >= 3:
                idx_str = parts[1].replace(".java", "")
                failures.append({"idx": int(idx_str), "stderr": parts[2]})

    # Protection against catastrophic compiler crashes (e.g., Java OutOfMemory)
    if result.returncode != 0 and oks == 0 and not failures:
        print(f"[FATAL ERROR Worker {chunk_id}] {result.stderr}", file=sys.stderr)

    return {"oks": oks, "failures": failures}


# We ignore name clashes deriving from type erasure, since, in order to assess the ability of the model to dinstinguish between type (semantics), we DO NOT want to consider List<String> as the same as List<Integer>.
# https://docs.oracle.com/javase/tutorial/java/generics/erasure.html
# We purposefully "hardcode" this Java program here to avoid any "mess" related to the shoddy way Python can mess with the working directory.
JAVA_BATCH_COMPILER = """
import javax.tools.*;
import java.io.*;
import java.util.*;

public class BatchCompiler {
    public static void main(String[] args) throws Exception {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) {
            System.err.println("JDK not found. Ensure you are using a JDK, not a JRE.");
            System.exit(1);
        }

        StandardJavaFileManager fileManager = compiler.getStandardFileManager(null, null, null);

        try (BufferedReader br = new BufferedReader(new FileReader(args[0]))) {
            String line;
            while ((line = br.readLine()) != null) {
                File file = new File(line);
                Iterable<? extends JavaFileObject> units = fileManager.getJavaFileObjects(file);

                DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
                List<String> options = Arrays.asList("-d", file.getParentFile().getAbsolutePath());

                // We compile collecting the diagnostics manually
                compiler.getTask(null, fileManager, diagnostics, options, null, units).call();

                boolean hasRealError = false;
                StringBuilder errorMsg = new StringBuilder();

                for (Diagnostic<? extends JavaFileObject> diagnostic : diagnostics.getDiagnostics()) {
                    if (diagnostic.getKind() == Diagnostic.Kind.ERROR) {
                        String msg = diagnostic.getMessage(Locale.ENGLISH);
                        // IGNORE the specific type-erasure name clash error
                        if (msg != null && msg.contains("have the same erasure")) {
                            continue;
                        }

                        hasRealError = true;
                        errorMsg.append(msg).append(" | ");
                    }
                }

                if (hasRealError) {
                    System.out.println("FAIL\\t" + file.getName() + "\\t" + errorMsg.toString().replace("\\n", " "));
                } else {
                    System.out.println("OK\\t" + file.getName());
                }
            }
        }
    }
}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--model", type=str, default="bigcode/santacoder")
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Number of samples to compile per worker batch. If not set, it splits the dataset equally among workers.",
    )
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
        help="Path to save the indices of failed compilations as a text file.",
    )
    args = ap.parse_args()

    if shutil.which("javac") is None:
        raise SystemExit("javac not found; install JDK to run compile checks.")

    columns = pq.read_table(args.dataset).to_pydict()
    n_samples = len(next(iter(columns.values()))) if columns else 0
    if n_samples == 0:
        raise SystemExit("Dataset is empty; nothing to compile.")

    print("[1/3] Filtering eligible rows...")

    f1_in = columns["func1InHasTypeTag"]
    f2_in = columns["func2InHasTypeTag"]
    f1_out = columns["func1OutHasTypeTag"]
    f2_out = columns["func2OutHasTypeTag"]
    a_tag = columns["aHasTypeTag"]
    b_tag = columns["bHasTypeTag"]
    b_exp = columns["b_expectedType"]

    eligible = []
    for idx in range(n_samples):
        req = f1_in[idx] and f2_in[idx] and f1_out[idx] and f2_out[idx] and a_tag[idx]
        if b_exp[idx] != prober_meta_null_type:
            req = req and b_tag[idx]
        if req:
            eligible.append(idx)

    total_eligible = len(eligible)
    if total_eligible == 0:
        raise SystemExit("No rows with all required type tags.")

    print(f"      Found {total_eligible} eligible samples.")

    # Initialize RAM memory (/dev/shm on Linux) for zero I/O latency
    base_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
    workers = args.workers

    with tempfile.TemporaryDirectory(prefix="prober_javac_", dir=base_dir) as tmpdir:
        # Create and compile the BatchCompiler daemon in RAM
        compiler_path = Path(tmpdir) / "BatchCompiler.java"
        compiler_path.write_text(JAVA_BATCH_COMPILER)
        subprocess.run(["javac", str(compiler_path)], check=True)

        if args.chunk_size is not None and args.chunk_size > 0:
            chunk_size = args.chunk_size
        else:
            # Default: split equally among workers
            chunk_size = max(1, total_eligible // workers)

        chunks = [eligible[i: i + chunk_size] for i in range(0, total_eligible, chunk_size)]

        print(f"[2/3] Spawning {len(chunks)} batches across {workers} JVM processes (chunk_size={chunk_size})...")

        failures = []
        total_oks = 0

        with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(args.dataset,),
        ) as ex:
            futures = [
                ex.submit(_worker_task, i, chunk, args.model, tmpdir)
                for i, chunk in enumerate(chunks)
            ]

            for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Batched Javac",
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

        print(f"\n[SUMMARY] compiled={total_oks} failed={len(failures)} out of {total_eligible}")

        # Save bad indices to a file if requested, so we can exclude them later from the dataset and get only valid code examples!
        if args.out_bad_indices and failures:
            args.out_bad_indices.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_bad_indices, "w") as f:
                for fail in failures:
                    f.write(f"{fail['idx']}\n")
            print(f"[INFO] Saved {len(failures)} bad indices to {args.out_bad_indices}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
