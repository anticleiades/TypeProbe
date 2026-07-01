#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib as mpl

mpl.rcParams.update(
    {
        "font.size": 22,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.titlesize": 28,
    }
)

import matplotlib.pyplot as plt
import numpy as np


NICE_NAMES = {
    "java": "Java",
    "pytag": "pyTag",
    "pyunt": "pyUnt",
    "adv_java": "Java (Adversarial)",
    "adv_pytag": "pyTag (Adversarial)",
    "adv_pyunt": "pyUnt (Adversarial)",
}

MODEL_MAP = {
    "codellama-7b-hf": "CodeLlama (7B)",
    "santacoder": "SantaCoder (1.1B)",
}


def get_model_name(model_id: str) -> str:
    if model_id in MODEL_MAP:
        return MODEL_MAP[model_id]
    raise ValueError(f"Unknown model ID: {model_id}")


def clean_name(name: str) -> str:
    return NICE_NAMES.get(name.lower(), name)


def normalize_lang_name(name: str) -> str:
    """
    Paranoia-mode normalization :)
    """
    n = name.lower()
    if n in ("java",):
        return "java"
    if n in ("pytag", "py_tag", "tagged", "pymix", "py_mix"):
        return "pyTag"
    if n in ("pyunt", "py_untagged", "untagged", "pyuntag"):
        return "pyUnt"
    if n.startswith("adv_"):
        return normalize_lang_name(n.replace("adv_", ""))
    return name


def normalize_test_name(name: str) -> str:
    """
    Paranoia-mode normalization :)
    """
    raw = name.lower()
    is_adv = "adv_" in raw
    n_clean = raw.replace("adv_", "").replace("control_", "")

    base = normalize_lang_name(n_clean)
    base = base.lower()  # "java", "pytag", "pyunt"

    if is_adv:
        return f"adv_{base}"
    return base


def extract_series_from_json(data: dict, task: str):
    if not isinstance(data, dict):
        return [], []

    layers = []
    values = []
    for key, task_dict in data.items():
        if not isinstance(key, str) or not key.startswith("layer_"):
            continue
        if not isinstance(task_dict, dict):
            continue
        if task not in task_dict:
            continue
        try:
            layer_idx = int(key.split("_")[1])
        except (IndexError, ValueError):
            continue
        layers.append(layer_idx)
        values.append(task_dict[task])

    if not layers:
        return [], []

    order = sorted(range(len(layers)), key=lambda i: layers[i])
    layers = [layers[i] for i in order]
    values = [float(values[i]) for i in order]
    return values, layers


PRETTY_PRINT_TASK = {
    "task0": "Task 0: Function Prediction",
    "task1": "Task 1: Argument Type Prediction",
    "task2": "Task 2: Application Result Prediction",
}

# --------- LaTeX stuff  -----------------------------------------------------
def write_title(task: str, is_selectivity: bool = False) -> str:
    pretty_print_task = PRETTY_PRINT_TASK[task]

    # Task 0 (50% baseline): 75% accuracy = 25 selectivity
    # Others (12.5% baseline): 75% accuracy = 63 selectivity
    is_task0 = task.endswith("sk0")
    bold_thrsh = "25" if is_task0 else "63"
    italic_thrsh = (
        "20" if is_task0 else "30"
    )  # Lowered for Task 0 to maintain hierarchy

    if is_selectivity:
        if task == "task2":
            title = (
                f"\\textbf{{{pretty_print_task}.}} "
                f"Values report peak selectivity $\\mathcal{{S}}$ ($\\times 100$) and peak layer ($L$). "
                f"Formatting and thresholds follows Table~\\ref{{tab:task1-results}}."
            )
        elif task == "task1":
            title = (
                f"\\textbf{{{pretty_print_task}.}} "
                f"Values report peak selectivity $\\mathcal{{S}}$ ($\\times 100$) and peak layer ($L$). "
                f"Bold indicates values corresponding to at least $75\\%$ raw accuracy ($S \\ge 63$); italics indicate non-negligible signals ($S \\ge 30$)."
            )
        else:
            title = (
                f"\\textbf{{{pretty_print_task}.}} "
                f"Values report peak selectivity $\\mathcal{{S}}$ ($\\times 100$) and peak layer ($L$). "
                f"Bold indicates values corresponding to at least $75\\%$ raw accuracy ($S \\ge 25$); italics indicate non-negligible signals ($S \\ge 20$)."
            )
    else:
        title = (
            f"\\textbf{{{pretty_print_task}.}} "
            f"Values represent \\emph{{raw accuracy}} ($\\times 100$) and the peak layer $(L)$. "
            f"Accuracies $\\ge 75$ are in \\textbf{{bold}}."
        )

    return title



def human_readable_train_domain(dmn: str) -> str:
    names = {
        "java": "Java",
        "pytag": "pyTag",
        "pyunt": "pyUnt",
        "adv_java": "Java (Adversarial)",
        "adv_pytag": "pyTag (Adversarial)",
        "adv_pyunt": "pyUnt (Adversarial)",
    }
    dmn = dmn.lower()
    if dmn in names:
        return names[dmn]
    raise ValueError(f"Unknown train domain: {dmn}")


def task_human_readable(task: str) -> str:
    return task.replace("task", "Task ")


def parse_flat_resultsV2(work_dir: Path):
    """
    Reads:
      work_dir/results/<model>/<eval_tag>/results.json

    Outputs:
      full_data[task][model][train_domain][test_key] = (series, layers)
    """
    full_data = {
        "task0": defaultdict(lambda: defaultdict(dict)),
        "task1": defaultdict(lambda: defaultdict(dict)),
        "task2": defaultdict(lambda: defaultdict(dict)),
    }

    results_root = work_dir / "results"
    if not results_root.exists():
        print(f"[WARN] results directory not found: {results_root}")
        return full_data

    for model_dir in results_root.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        for eval_dir in model_dir.iterdir():
            if not eval_dir.is_dir():
                continue

            eval_tag = eval_dir.name
            if not eval_tag.startswith("prober_") or "_test_" not in eval_tag:
                continue

            parts = eval_tag.split("_test_")
            if len(parts) != 2:
                print(f"[WARN] malformed eval_tag: {eval_tag}")
                continue

            train_domain = parts[0].replace("prober_", "")
            test_ds = parts[1]
            test_key = normalize_test_name(test_ds)

            result_file = eval_dir / "results.json"
            if not result_file.exists():
                print(f"[WARN] missing results.json: {result_file}")
                continue

            try:
                data = json.loads(result_file.read_text())
            except Exception as e:
                print(f"[WARN] error reading {result_file}: {e}")
                continue

            if not isinstance(data, dict):
                print(f"[WARN] JSON root is not a dict in {result_file}")
                continue

            for task in ["task0", "task1", "task2"]:
                try:
                    series, layers = extract_series_from_json(data, task)
                    full_data[task][model][train_domain][test_key] = (series, layers)

                except Exception as e:
                    print(f"[WARN] error extracting {task} from {result_file}: {e}")
                    continue

    return full_data


def getselectivitydict(fulldata: dict, peak_layers: dict) -> dict:
    selectivity_dict = {
        "task0": defaultdict(lambda: defaultdict(dict)),
        "task1": defaultdict(lambda: defaultdict(dict)),
        "task2": defaultdict(lambda: defaultdict(dict)),
    }
    for key_task, task_data in fulldata.items():
        for key_model_data, model_data in task_data.items():
            non_ctl_keys = [
                k for k in model_data.keys() if not k.startswith("control_")
            ]
            for train_non_ctl_key in non_ctl_keys:
                control_train_non_ctl_key = f"control_{train_non_ctl_key}"

                try:
                    id_peak_layer = peak_layers[key_task][key_model_data][
                        train_non_ctl_key
                    ]
                except KeyError:
                    print(
                        f"[WARN] Nessun peak layer per {key_task} - {key_model_data} - {train_non_ctl_key}. Salto."
                    )
                    continue

                non_adv_keys_eval = [
                    k
                    for k in model_data[train_non_ctl_key].keys()
                    if not "adv" in k.lower()
                ]
                for key_eval_on in non_adv_keys_eval:
                    try:
                        raw_acc = model_data[train_non_ctl_key][key_eval_on][0][
                            id_peak_layer
                        ]
                        baseline = model_data[control_train_non_ctl_key][key_eval_on][
                            0
                        ][id_peak_layer]
                        sel_value = raw_acc - baseline
                    except (IndexError, KeyError):
                        continue

                    adv_key_eval_on = f"adv_{key_eval_on}"
                    if adv_key_eval_on in model_data[train_non_ctl_key]:
                        try:
                            adv_raw_acc = model_data[train_non_ctl_key][
                                adv_key_eval_on
                            ][0][id_peak_layer]
                            adv_baseline = model_data[control_train_non_ctl_key][
                                adv_key_eval_on
                            ][0][id_peak_layer]
                            adv_sel = adv_raw_acc - adv_baseline
                        except (IndexError, KeyError):
                            adv_sel = None
                    else:
                        adv_sel = None

                    delta_sel = (sel_value - adv_sel) if adv_sel is not None else None

                    selectivity_dict[key_task][key_model_data][train_non_ctl_key][
                        key_eval_on
                    ] = (id_peak_layer, sel_value, adv_sel, delta_sel)

    return selectivity_dict


def getrawaccdata(fulldata: dict, peak_layers: dict) -> dict:
    rawacc_dict = {
        "task0": defaultdict(lambda: defaultdict(dict)),
        "task1": defaultdict(lambda: defaultdict(dict)),
        "task2": defaultdict(lambda: defaultdict(dict)),
    }
    for key_task, task_data in fulldata.items():
        for key_model_data, model_data in task_data.items():
            non_ctl_keys = [
                k for k in model_data.keys() if not k.startswith("control_")
            ]
            for train_non_ctl_key in non_ctl_keys:
                try:
                    id_peak_layer = peak_layers[key_task][key_model_data][
                        train_non_ctl_key
                    ]
                except KeyError:
                    continue
                non_adv_keys_eval = [
                    k
                    for k in model_data[train_non_ctl_key].keys()
                    if not "adv" in k.lower()
                ]
                for key_eval_on in non_adv_keys_eval:
                    try:
                        acc_value = model_data[train_non_ctl_key][key_eval_on][0][
                            id_peak_layer
                        ]
                    except (IndexError, KeyError):
                        continue

                    adv_key_eval_on = f"adv_{key_eval_on}"
                    if adv_key_eval_on in model_data[train_non_ctl_key]:
                        try:
                            adv_acc = model_data[train_non_ctl_key][adv_key_eval_on][0][
                                id_peak_layer
                            ]
                        except (IndexError, KeyError):
                            adv_acc = None
                    else:
                        adv_acc = None

                    delta_acc = (acc_value - adv_acc) if adv_acc is not None else None

                    rawacc_dict[key_task][key_model_data][train_non_ctl_key][
                        key_eval_on
                    ] = (id_peak_layer, acc_value, adv_acc, delta_acc)

    return rawacc_dict


def write_tex_tableV2(
    path: Path,
    task: str,
    results_dict: dict,
    column_order,
    is_selectivity: bool = False,
    clamp: bool = True,
    enable_adv_train_domain: bool = False,
) -> None:
    """
    results_dict is either selectivity_dict or raw_acc_dict (output from your V2 functions).
    Structure: results_dict[task][model][train_domain][test_domain] = (peak_layer, val_std, val_adv, delta_val)
    If any data is missing -> 'X'.
    """
    title = write_title(task, is_selectivity)

    with path.open("w") as f:
        f.write("\\begin{table*}[t]\n")
        f.write("    \\centering\n")
        f.write(f"    \\caption{{{title}}}\n")
        if is_selectivity:
            f.write(f"    \\label{{tab:{task}-results}}\n")
        else:
            f.write(f"    \\label{{tab:{task}-resultsRaw}}\n")
        f.write("    \\small\n")
        f.write("    \\setlength{\\tabcolsep}{4pt}\n")
        f.write("    \\resizebox{\\textwidth}{!}{\n")
        f.write("    \\begin{tabular}{@{} ll ccc ccc ccc @{} }\n")
        f.write("        \\toprule\n")
        f.write(
            "        & & \\multicolumn{3}{c}{\\textbf{Eval on Standard}} & "
            "\\multicolumn{3}{c}{\\textbf{Eval on Adversarial}} & "
            "\\multicolumn{3}{c}{\\textbf{Absolute Drop (\\boldmath$\\Delta$)}} \\\\\n"
        )
        f.write(
            "        \\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}\n"
        )
        f.write(
            "        \\textbf{Model} & \\textbf{Train} & "
            "\\textbf{Java} & \\textbf{pyTag} & \\textbf{pyUnt} & "
            "\\textbf{Java} & \\textbf{pyTag} & \\textbf{pyUnt} & "
            "\\textbf{Java} & \\textbf{pyTag} & \\textbf{pyUnt} \\\\\n"
        )

        if task not in results_dict or not results_dict[task]:
            f.write("        \\midrule\n")
            f.write("        \\multicolumn{11}{c}{No results} \\\\\n")
            f.write("        \\bottomrule\n")
            f.write("    \\end{tabular}}\n")
            f.write("\\end{table*}\n")
            return

        task_block = results_dict[task]
        sorted_models = sorted(task_block.keys())

        # Split domains into Standard and Adversarial
        train_domains = [False, True] if enable_adv_train_domain else [False]
        for is_adv_train in train_domains:
            if enable_adv_train_domain:
                f.write("        \\midrule\n")
                if is_adv_train:
                    f.write(
                        "        \\multicolumn{11}{c}{\\textit{\\textbf{Trained on Adversarial Partitions}}} \\\\\n"
                    )
                else:
                    f.write(
                        "        \\multicolumn{11}{c}{\\textit{\\textbf{Trained on Standard Partitions}}} \\\\\n"
                    )
            f.write("        \\midrule\n")

            for mi, model in enumerate(sorted_models):
                model_data = task_block.get(model, {})
                if not isinstance(model_data, dict):
                    continue

                trains = sorted(model_data.keys())
                if not trains:
                    continue

                # Filter trains for this block (V2 already stripped 'control_')
                block_trains = [
                    t for t in trains if (t.lower().startswith("adv_")) == is_adv_train
                ]

                if not block_trains:
                    continue

                for ti, train_domain in enumerate(block_trains):
                    pretty_model = get_model_name(model)
                    if ti == 0:
                        # For first row of model, split into two lines if needed
                        if "\\n" in pretty_model:
                            parts = pretty_model.split("\\n")
                            model_label = f"\\textbf{{{parts[0]}}}"
                            # model_sublabel = parts[1]
                        else:
                            model_label = f"\\textbf{{{pretty_model}}}"
                    else:
                        model_label = ""
                        if ti == 1 and "\\n" in pretty_model:
                            parts = pretty_model.split("\\n")
                            model_label = parts[1]

                    td_lower = train_domain.lower()
                    base_train = normalize_lang_name(td_lower)

                    # Capitalize for the table
                    if base_train == "java":
                        base_train = "Java"
                    elif base_train == "pytag":
                        base_train = "PyTag"
                    elif base_train == "pyunt":
                        base_train = "PyUnt"

                    train_label = base_train

                    # Pre-calculate cell formatting thresholds
                    if is_selectivity:
                        bold_threshold = 25 if task == "task0" else 63
                        italic_threshold = 20 if task == "task0" else 30
                    else:
                        threshold = 75

                    # We will collect the row cells in three groups, then concatenate
                    std_cells = []
                    adv_cells = []
                    delta_cells = []

                    # In V2, column_order passed from main should just be ["java", "pytag", "pyunt"]
                    # because the eval 'adv_' are extracted automatically alongside the standard eval!
                    eval_order = ["java", "pytag", "pyunt"]

                    for col in eval_order:
                        train_data = model_data.get(train_domain, {})
                        if not isinstance(train_data, dict):
                            std_cells.append("X")
                            adv_cells.append("X")
                            delta_cells.append("X")
                            continue

                        entry = train_data.get(col)
                        if (
                            entry is None
                            or not isinstance(entry, tuple)
                            or len(entry) != 4
                        ):
                            std_cells.append("X")
                            adv_cells.append("X")
                            delta_cells.append("X")
                            continue

                        layer_idx, val_std, val_adv, delta_val = entry

                        if clamp:
                            val_std = max(0.0, float(val_std))
                            val_adv = max(0.0, float(val_adv))
                            # Recompute delta dynamically if clamped
                            delta_val = val_std - val_adv

                        try:
                            # Convert to *100
                            v_std = round(float(val_std) * 100, 0)
                            v_adv = round(float(val_adv) * 100, 0)
                            d_val = round(float(delta_val) * 100, 0)

                            layer_str = f"{int(layer_idx):02d}"

                            # ----- 1) Format Standard Cell (includes layer) -----
                            v_str_std = f"{int(v_std):02d}"
                            if is_selectivity:
                                is_bold = v_std >= bold_threshold
                                is_italic = v_std >= italic_threshold
                                if is_bold and is_italic:
                                    v_str_std = f"\\textbf{{\\textit{{{v_str_std}}}}}"
                                elif is_bold:
                                    v_str_std = f"\\textbf{{{v_str_std}}}"
                                elif is_italic:
                                    v_str_std = f"\\textit{{{v_str_std}}}"
                                std_cells.append(f"{v_str_std} ({layer_str})")
                            else:
                                if v_std >= threshold:
                                    std_cells.append(
                                        f"\\textbf{{{int(v_std):02d}}} ({layer_str})"
                                    )
                                else:
                                    std_cells.append(f"{int(v_std):02d} ({layer_str})")

                            # ----- 2) Format Adversarial Cell (NO LAYER) -----
                            v_str_adv = f"{int(v_adv):02d}"
                            if is_selectivity:
                                is_bold = v_adv >= bold_threshold
                                is_italic = v_adv >= italic_threshold
                                if is_bold and is_italic:
                                    v_str_adv = f"\\textbf{{\\textit{{{v_str_adv}}}}}"
                                elif is_bold:
                                    v_str_adv = f"\\textbf{{{v_str_adv}}}"
                                elif is_italic:
                                    v_str_adv = f"\\textit{{{v_str_adv}}}"
                                adv_cells.append(f"{v_str_adv}")
                            else:
                                if v_adv >= threshold:
                                    adv_cells.append(f"\\textbf{{{int(v_adv):02d}}}")
                                else:
                                    adv_cells.append(f"{int(v_adv):02d}")

                            # ----- 3) Format Delta Cell -----
                            delta_cells.append(f"{int(d_val):02d}")

                        except (ValueError, TypeError):
                            std_cells.append("X")
                            adv_cells.append("X")
                            delta_cells.append("X")

                    # Reconstruct row_cells order: [Model, Train, Std1, Std2, Std3, Adv1, Adv2, Adv3, Del1, Del2, Del3]
                    row_cells = (
                        [model_label, train_label] + std_cells + adv_cells + delta_cells
                    )
                    f.write("        " + " & ".join(row_cells) + " \\\\\n")

                if mi < len(sorted_models) - 1:
                    f.write("        \\midrule\n")

        f.write("        \\bottomrule\n")
        f.write("    \\end{tabular}}\n")
        f.write("\\end{table*}\n")


def plot_per_proberV3(
    args_out: Path, fulldata: dict, model: str, enable_adv_train_domain: bool
) -> None:
    sample_task = None
    for t in ["task1", "task2"]:
        if t in fulldata and model in fulldata[t]:
            sample_task = t
            break

    if sample_task is None:
        print(f"[WARN] model {model} not found in task1/task2 data")
        return

    model_data_sample = fulldata[sample_task][model]

    train_domains = sorted(
        [t for t in model_data_sample.keys() if not t.startswith("control_")]
    )
    if not enable_adv_train_domain:
        train_domains = [t for t in train_domains if not t.lower().startswith("adv_")]

    if not train_domains:
        print(f"[WARN] no valid train domains for model {model}")
        return

    eval_order = ["java", "pytag", "pyunt", "adv_java", "adv_pytag", "adv_pyunt"]

    task_styles = {
        "task1": {
            "color": "#e41a1c",
            "linestyle": "-",
            "marker": "s",
            "markersize": 6,
            "markeredgecolor": "black",
            "markeredgewidth": 0.6,
        },
        "task2": {
            "color": "#4daf4a",
            "linestyle": "-",
            "marker": "D",
            "markersize": 6,
            "markeredgecolor": "black",
            "markeredgewidth": 0.6,
        },
    }

    for train_domain in train_domains:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
        axes = np.array(axes)

        for idx, test_dom in enumerate(eval_order):
            r, c = idx // 3, idx % 3
            ax = axes[r, c]

            for task in ["task1", "task2"]:
                if task not in fulldata or model not in fulldata[task]:
                    continue

                model_data = fulldata[task][model]
                if train_domain not in model_data:
                    continue

                train_data = model_data[train_domain]
                control_key = f"control_{train_domain}"
                control_data = model_data.get(control_key)

                entry = train_data.get(test_dom, ([], []))
                actual_series = (
                    entry[0] if (isinstance(entry, tuple) and len(entry) >= 1) else []
                )

                if isinstance(control_data, dict):
                    centry = control_data.get(test_dom, ([], []))
                    control_series = (
                        centry[0]
                        if (isinstance(centry, tuple) and len(centry) >= 1)
                        else []
                    )
                else:
                    control_series = []

                n = max(len(actual_series), len(control_series))
                if n == 0:
                    continue

                if len(actual_series) != n:
                    actual_series = list(actual_series) + [0.0] * (
                        n - len(actual_series)
                    )
                if len(control_series) != n:
                    control_series = list(control_series) + [0.0] * (
                        n - len(control_series)
                    )

                try:
                    aarr = np.array(actual_series, dtype=float)
                    carr = np.array(control_series, dtype=float)
                    sel = 100.0*np.maximum(0.0, aarr - carr)
                    x_vals = np.arange(n)

                    style = task_styles[task]
                    ax.plot(
                        x_vals,
                        sel,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=3.2,
                        marker=style["marker"],
                        markersize=7,
                        markerfacecolor="white",
                        markeredgecolor="black",
                        markeredgewidth=0.6,
                        label=f"{task_human_readable(task)}",
                    )
                except Exception as e:
                    print(
                        f"[WARN] selectivity plot error {model}:{train_domain}->{test_dom} [{task}] : {e}"
                    )

            ax.set_title(
                f"Evaluated on {clean_name(test_dom)}", fontsize=20, fontweight="bold"
            )
            ax.set_yticks([-12.5, 0, 25, 50, 75])
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
            ax.set_ylim(-12.5, 87.5)
            ax.tick_params(axis="both", labelsize=16)

            if r == 1:
                ax.set_xlabel("Layer", fontsize=18, fontweight="bold")
            if c == 0:
                ax.set_ylabel("Selectivity ($\\times 100$)", fontsize=18, fontweight="bold")

        handles, labels = [], []
        for ax in fig.axes:
            h_list, l_list = ax.get_legend_handles_labels()
            for h, lab in zip(h_list, l_list):
                if lab not in labels:
                    labels.append(lab)
                    handles.append(h)

        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=2,
                fontsize=20,
                frameon=False,
                bbox_to_anchor=(0.5, 0.00),
            )

        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.15, wspace=0.18, hspace=0.30)

        safe_train = train_domain.replace("control_", "").replace("_", "")
        png_path = args_out / f"{model}_prober_{safe_train}_task1_task2_selectivity.png"
        pdf_path = args_out / f"{model}_prober_{safe_train}_task1_task2_selectivity.pdf"

        try:
            plt.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.savefig(pdf_path, bbox_inches="tight")
            print(f"[OK] Generated -> {png_path}")
            print(f"[OK] Generated -> {pdf_path}")
        except Exception as e:
            print(f"[ERR] Error saving plots for {model}/{train_domain}: {e}")

        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("paper_output"))
    ap.add_argument(
        "--no-plots", default=False, action="store_true", help="Do not generate plots"
    )
    ap.add_argument(
        "--train-on-adv",
        default=False,
        action="store_true",
        help="Train regime: on adversarial data",
    )
    ap.add_argument(
        "--no-clamp",
        default=True,
        action="store_true",
        help="Do not clamp the selectivity threshold to 0",
    )
    ap.add_argument(
        "--peaks-file",
        type=Path,
        default=Path("peak_layers.json"),
        help="JSON file generated by extract_peak_layers.py",
    )
    args = ap.parse_args()
    if not args.peaks_file.exists():
        print(f"[ERR] {args.peaks_file} not foun")
        return
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[ERR] Error creating output directory: {e}")
        return

    with open(args.peaks_file, "r") as f:
        peak_layers = json.load(f)

    full_data = parse_flat_resultsV2(args.work_dir)

    sel_result = getselectivitydict(full_data, peak_layers)
    acc_result = getrawaccdata(full_data, peak_layers)

    column_order = ["java", "pytag", "pyunt", "adv_java", "adv_pytag", "adv_pyunt"]

    for task in ["task0", "task1", "task2"]:
        try:
            tex_raw = args.output_dir / f"table_{task}_raw.tex"
            write_tex_tableV2(
                tex_raw,
                task,
                acc_result,
                column_order,
                is_selectivity=False,
                enable_adv_train_domain=args.train_on_adv,
            )
            print(f"[OK] Generated -> {tex_raw}")
        except Exception as e:
            print(f"[ERR] Error generating raw table for {task}: {e}")

        try:
            tex_sel = args.output_dir / f"table_{task}_selectivity.tex"
            write_tex_tableV2(
                tex_sel,
                task,
                sel_result,
                column_order,
                is_selectivity=True,
                clamp=not args.no_clamp,
                enable_adv_train_domain=args.train_on_adv,
            )
            print(f"[OK] Generated -> {tex_sel}")
        except Exception as e:
            print(f"[ERR] Error generating selectivity table for {task}: {e}")

    if args.no_plots:
        print("skipping per-prober plots")
        return
    all_models = set()
    for task in ["task0", "task1", "task2"]:
        all_models |= set(full_data[task].keys())

    print(f"Generating per-prober plots for {len(all_models)} models")
    for model in sorted(all_models):
        try:
            plot_per_proberV3(
                args.output_dir,
                full_data,
                model,
                enable_adv_train_domain=args.train_on_adv,
            )
        except Exception as e:
            print(
                f"[ERR] Error generating per-prober plots for model {model}, {task}: {e}"
            )


if __name__ == "__main__":
    main()
