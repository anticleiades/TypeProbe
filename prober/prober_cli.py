import argparse
import os
import re
from pathlib import Path
from typing import Optional

PAPER_TITLE = "TypeProbe: Recovering Type Representations from Hidden States of Pre-trained Code Models."

HELP_NEW = """
Directory Layout (per model/activation/dataset triple):
-------------------------------------------------------
  <work-dir>/<model_slug>_<act>_<dataset_tag>/
    caches/{py,java}/
    probers/{py,java,control_py,control_java}/results/

Evaluation Directory:
----------------------------
  Every run (training or eval-only) writes the result in JSON format:
    <work-dir>/results/<model_slug>/<eval_tag>/results.json
  so all evaluations can be collected from a single root.

Typical eval_tag (auto-inferred unless overridden with --eval-tag):
  - prober_pyTag_test_pyTag
  - prober_pyTag_test_advPyTag
  - prober_java_test_pyTag
  - prober_control_pyTag_test_pyTag

Metadata:
---------
  If --metadata is omitted, the tool defaults to `<dataset>.meta.parquet` 
  (matching the dataset stem with a .meta.parquet suffix).

Examples (REFER TO THE README AND TO THE PAPER FOR THE ACTUAL CONFIGURATION NEEDED TO REPLICATE THE RESULTS):
-----------------
  # 1. Cache activations for all layers on a given dataset
  python prober/prober.py \\
    --work-dir runs/work \\
    --dataset cleanOut/tagged.parquet \\
    --act resid_post \\
    --cache-acts

  # 2. Train probes from cached activations (Python prompts)
  python prober/prober.py \\
    --work-dir runs/work \\
    --dataset cleanOut/tagged.parquet \\
    --act resid_post \\
    --use-cache

  # 3. Cache activations for Java prompts
  python prober/prober.py \\
    --work-dir runs/work \\
    --dataset cleanOut/tagged.parquet \\
    --act resid_post \\
    --java \\
    --cache-acts

  # 4. Change pooling strategy over sequence tokens
  python prober/prober.py \\
    --work-dir runs/work \\
    --dataset cleanOut/tagged.parquet \\
    --act resid_post \\
    --pool fim \\
    --use-cache

  # 5. Eval-only on an existing prober directory
  python prober/prober.py \\
    --work-dir runs/work \\
    --dataset cleanOut/tagged.parquet \\
    --act resid_post \\
    --use-cache \\
    --eval-only \\
    --eval-dir runs/work/santacoder_resid_post_tagged/probers/py/results

  # 6. Eval-only on a single saved probe .pt file
  python prober/prober.py \\
    --work-dir runs/work \\
    --dataset cleanOut/tagged.parquet \\
    --act resid_post \\
    --use-cache \\
    --eval-only \\
    --probe-path path/to/prober_layer13_best.pt \\
    --eval-tag custom_eval_tag

Notes on Eval-Only Mode:
------------------------
  - You must provide exactly ONE source: either --eval-dir or --probe-path.
  - If both are given, --probe-path takes precedence.
  - The --eval-tag argument is mandatory and determines the final output folder name.
"""

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT_DIR / "dataset" / "outV2"


def _slugify_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _resolve_path_arg(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or (os.sep in value) or ("/" in value):
        return path
    return base_dir / value


def _derive_meta_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(f"{dataset_path.stem}.meta.parquet")


def _infer_eval_tag(
    dataset_path: Path,
    use_java: bool,
    control_task: bool) -> str:
    name = dataset_path.stem.lower()

    if "adv" in name:
        if "untagged" in name and not use_java:
            train_tag = "adv_pyUnt"
        elif not use_java:
            train_tag = "adv_pyTag"
        else:
            train_tag = "adv_java"
    else:
        if use_java:
            train_tag = "java"
        elif "untagged" in name:
            train_tag = "pyUnt"
        else:
            train_tag = "pyTag"

    if control_task:
        return f"prober_control_{train_tag}_test_control_{train_tag}"

    return f"prober_{train_tag}_test_{train_tag}"


def _model_slug_from_name(model_name: str, explicit: Optional[str]) -> str:
    if explicit:
        return _slugify_component(explicit)
    return _slugify_component(model_name.split("/")[-1]).lower()


def _work_root_dir(*, work_dir: Path, model_slug: str, act_name: str, dataset_tag: str) -> Path:
    act_tag = _slugify_component(act_name)
    return work_dir / f"{model_slug}_{act_tag}_{dataset_tag}"


def _extract_eval_dirs(work_dir: Path) -> Optional[tuple[Path, Path]]:
    parts = work_dir.parts
    if "probers" not in parts:
        return None
    idx = len(parts) - 1 - parts[::-1].index("probers")
    root_dir = Path(*parts[:idx])
    if len(parts) == idx + 1:
        return None
    if len(parts) >= idx + 3 and parts[idx + 2] == "results":
        eval_dir = Path(*parts[: idx + 3])
        return root_dir, eval_dir
    eval_dir = Path(*parts[: idx + 2]) / "results"
    return root_dir, eval_dir


def _resolve_run_paths(args: argparse.Namespace) -> argparse.Namespace:
    dataset_dir = args.dataset_dir
    dataset_path = _resolve_path_arg(args.dataset, dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if args.metadata is None:
        meta_path = _derive_meta_path(dataset_path)
    else:
        meta_path = _resolve_path_arg(args.metadata, dataset_dir)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    model_slug = _model_slug_from_name(args.model, args.model_slug)
    dataset_tag = _slugify_component(dataset_path.stem)
    work_dir = Path(args.work_dir).expanduser()

    internal_root = _work_root_dir(
        work_dir=work_dir,
        model_slug=model_slug,
        act_name=args.act,
        dataset_tag=dataset_tag,
    )

    lang = "java" if args.java else "py"
    prober_group = f"control_{lang}" if args.control_task else lang
    cache_dir = internal_root / "caches" / lang

    if args.eval_only:
        # STRICT SOURCE: Must provide directory or single probe
        if (args.eval_dir is None) == (args.probe_path is None):
            raise ValueError("EVAL_ONLY requires exactly one source: --eval-dir or --probe-path.")

        # STRICT TAG: Required for the flat folder name
        if args.eval_tag is None:
            raise ValueError("--eval-tag is mandatory in eval-only mode.")

        eval_dir = Path(args.eval_dir).expanduser() if args.eval_dir else None
        eval_tag = args.eval_tag
        save_dir = internal_root / "probers" / prober_group / "results"
        if args.control_task:
            raise ValueError("--control-task is not allowed in eval-only mode.")
    else:
        # TRAINING: Probers are saved in the structured internal tree
        if args.cache_acts:
            save_dir = cache_dir
        else:
            save_dir = internal_root / "probers" / prober_group / "results"

        eval_dir = save_dir

        if args.eval_tag is not None:
            raise ValueError("--eval-tag is not allowed in training mode.")

        eval_tag = _infer_eval_tag(dataset_path, use_java=args.java, control_task=args.control_task)

    # Target: ${workdir}/results/${model}/${eval_tag}
    eval_out_dir = work_dir / "results" / model_slug / eval_tag
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    # All final evaluation artifacts go here
    if args.acc_json is None:
        args.acc_json = eval_out_dir / "results.json"

    if args.plot_path is None:
        plot_dir = eval_out_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        args.plot_path = plot_dir / "accuracy_per_layer.png"

    args.dataset = dataset_path
    args.metadata = meta_path
    args.model_slug = model_slug
    args.eval_tag = eval_tag
    args.eval_out_dir = eval_out_dir
    args.work_dir = internal_root
    args.cache_dir = cache_dir
    args.save_dir = save_dir
    args.eval_dir = eval_dir

    return args


def _add_args() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=PAPER_TITLE,
        epilog=HELP_NEW,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Root directory for caches/probers output.",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset parquet path or filename.",
    )
    ap.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Metadata parquet path or filename (default: <dataset>.meta.parquet).",
    )
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Base dir used when dataset/metadata are passed as filenames.",
    )
    ap.add_argument(
        "--model-slug",
        type=str,
        default=None,
        help="Optional short slug used in work-dir naming.",
    )
    ap.add_argument("--model", type=str, default="bigcode/santacoder")
    ap.add_argument("--act", type=str, default=None, help="Activation name.")
    ap.add_argument(
        "--pool",
        type=str,
        default="fim",
        choices=["avg", "last", "fim"],
        help="Pooling strategy over sequence tokens.",
    )
    ap.add_argument(
        "--adapter",
        type=str,
        default="v2",
        help="Dataset adapter: v2 (default) or parquet.",
    )
    ap.add_argument(
        "--prompt-col",
        type=str,
        default="prompt",
        help="Prompt column name for parquet adapter.",
    )
    ap.add_argument(
        "--labels-col",
        type=str,
        default=None,
        help="Single column containing labels (list) for parquet adapter.",
    )
    ap.add_argument(
        "--label-cols",
        type=str,
        default=None,
        help="Comma-separated label columns for parquet adapter.",
    )
    ap.add_argument(
        "--class-counts",
        type=str,
        default=None,
        help="Override task class counts (comma-separated).",
    )
    ap.add_argument(
        "--eval-tag",
        type=str,
        default=None,
        help="Override eval output folder name.",
    )
    ap.add_argument("--layer", type=int, default=0, help="Layer index for activation name.")
    ap.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Path to save the accuracy-per-layer plot.",
    )
    ap.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Alias for --plot-path.",
    )
    ap.add_argument("--layer-start", type=int, default=0, help="Start layer for plotting.")
    ap.add_argument("--layer-end", type=int, default=None, help="End layer (inclusive) for plotting.")
    ap.add_argument("--k-folds", type=int, default=4)
    ap.add_argument(
        "--disable-kfold",
        action="store_true",
        help="Use a single train/val/test split instead of k-fold cross-validation.",
    )
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="L2 regularization strength (passed to Adam as weight_decay).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--repro",
        action="store_true",
        help="Enable reproducible settings (deterministic CUDA, pinned model revision, cache checks).",
    )
    ap.add_argument(
        "--model-revision",
        type=str,
        default=None,
        help="Optional model revision/commit hash to pin weights.",
    )
    ap.add_argument(
        "--cache-acts",
        action="store_true",
        help="Cache pooled activations for each layer and exit.",
    )
    ap.add_argument(
        "--use-cache",
        action="store_true",
        help="Train probe using cached activations instead of running the model.",
    )
    ap.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Overwrite existing cached activations/labels.",
    )
    ap.add_argument(
        "--dump-acts",
        action="store_true",
        help="Print available activation names from a cached forward pass and exit.",
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only evaluate a saved probe (requires --probe-path).",
    )
    ap.add_argument(
        "--probe-path",
        type=Path,
        default=None,
        help="Path to a saved probe .pt file for evaluation.",
    )
    ap.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Directory of prober_layer*_best.pt files to evaluate and plot.",
    )
    ap.add_argument(
        "--control-task",
        action="store_true",
        help="Implements Control Task (Hewitt et al.). Replace labels with random targets (still uses cached activations if enabled).",
    )
    ap.add_argument(
        "--causality",
        action="store_true",
        help="Report task2 accuracy conditioned on task0/1 correctness (test set).",
    )
    ap.add_argument(
        "--java",
        action="store_true",
        help="Generate Java prompts (including activation caching).",
    )
    ap.add_argument(
        "--acc-json",
        type=Path,
        default=None,
        help="Optional path to write per-layer per-task accuracies as JSON.",
    )
    return ap


def parse_args() -> argparse.Namespace:
    parser = _add_args()
    args = parser.parse_args()
    args.legacy = False
    if args.act is None:
        args.act = "resid_post"
    return _resolve_run_paths(args)

