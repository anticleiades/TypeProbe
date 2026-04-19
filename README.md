# TypeProbe: Recovering Type Representations from Hidden States of Pre-trained Code Models
This repository contains all the necessary code to reproduce the experiments and produce the tables and figures in the paper.

## Abstract
  State-of-the-art code models achieve impressive performance, yet the extent to which they internally keep track of types remains poorly understood. To investigate the internal logic of these models, we probe the residual streams of pretrained code models for internal type representation. We curate a dataset of Java and Python code examples and execute a series of cross-language transfer experiments by training linear probes on one language and inferring argument-type and result-type in the other, upon function application. Our results from these suggest that SantaCoder develops a latent, language-agnostic manifold for type semantics. We find that this internal representation is largely robust to both lexical variation and cross-language syntactic differences. In contrast, CodeLlama displays higher sensitivity to language-specific syntax and lexical cues. To the best of our knowledge, prior works on interpretability of code models have not directly targeted formal type semantics or cross-lingual type representations, leaving open the question of whether models internalize a language-agnostic notion of types.


## Environment and Dependencies

> **⚠️ Important Notice Regarding Bash Scripts:**  
> All provided `.sh` convenience wrapper scripts (such as `./cacheAll.sh`, `./trainAll.sh`, and those in `slurmV4/`) rely on a Slurm-based cluster environment. You should adapt those scripts to your own partition names, modules, and filesystem layout before running them.

Create the main Conda environment with:

```bash
conda env create -f environment.yml
conda activate TypeProbe
```

If `environment.yml` fails to solve on your machine due to platform-specific constraints, use the fallback:

```bash
conda create -n TypeProbe python=3.9.21 pip
conda activate TypeProbe
pip install -r requirementsFallback.txt
```
### Optional Java Toolchain

To validate Java prompts, a working JDK installation with the `javac` compiler available on your `PATH` is required. This is only needed if you want to run `genAndValidateDatasetAll.py` and perform local compilation checks for Java examples. We provide pre-generated datasets, so installing Java is not strictly necessary unless you want to regenerate and revalidate the Java prompts locally.

## Hardware Requirements

The experiments in this repository were originally executed on a single **NVIDIA H100 GPU (80GB VRAM)**. However, a single **NVIDIA A100 GPU (40GB/80GB)** is also fully sufficient. 

- **Extraction & Training:** Feature extraction for 7B-parameter models (e.g., Code Llama) and subsequent probe training require significant memory. Lower-VRAM setups can be accommodated by carefully balancing the batch size.
- **Data Generation:** The dataset generation and validation pipeline relies entirely on the CPU. While it can be executed on a standard workstation, validating all the datasets heavily benefits from multiple CPU cores and ample system RAM (32GB+ recommended).

## Probing

Unless otherwise specified, all probing experiments use the following hyperparameters.

- **Optimizer:** Adam
- **Learning rate:** $10^{-3}$
- **Weight decay:** $0.1$
- **Epochs**: 30
- **Batch size (probe training):** 2048
- **Batch size (activation extraction):** 256 or 512 (adjustable on lower-VRAM setups)
- **Cross-validation:** 4-fold, reporting peak layer-wise performance. 
- **Probe model:** single linear layer trained on `resid_post` activations, one per task (task0, task 1, task 2).

For reproducibility, all experiments are run with (`--repro`):
- It enables deterministic runs by fixing random seeds, configuring deterministic backends, checking dataset hashes, and enforcing a fixed experimental configuration.  
- It also performs consistency checks: it will abort if you try to cache activations and train probes in a different environment than the one used originally, or if you attempt to train a probe on cached activations produced from a different dataset (to avoid label–activation mismatches).


## Data Generation

Generate and validate the dataset variants with:

```bash
python genAndValidateDatasetAll.py
```

This script programmatically generates Java and Python examples under specific naming policies (random identifiers, and with adversarial). It validates them via compilation (`javac` for Java, `ast.parse` for Python) and writes intermediate Parquet files plus `.meta.parquet` metadata.
- Raw generated datasets are saved in `dataset/outV2/`
- Cleaned/validated datasets are written to `cleanOut/`


> ⚠️ Important Notice: Resource Tuning
> 
> The script is optimized for high-performance hardware (we used a **192-core machine with 128GB RAM**). By default, it spawns one worker per CPU core and processes batches of 4 samples at a time.
> 
> If you are running this on a standard laptop or a machine with limited memory, you **must** tune the following arguments to avoid **Out of Memory (OOM)** errors or system freezes:
> *   `--workers`: Limits the number of parallel processes.
> *   `--chunk-size`: Controls how many examples each worker handles in a single batch.

*Note: Execution takes a significant amount of time due to exhaustive compilation checks.*

For convenience, we also provide pre-generated versions of all dataset variants in `dataset/outV2/` and `cleanOut/`, so you can directly use them without re-running the generation and validation pipeline. These are the same we used in our paper.


## Activation Caching

A convenience wrapper is provided:

```bash
./cacheAll.sh workDir datasetDir
```
*Note: replace `workDir` with the desired directory and `datasetDir` with the directory containing the validated generated datasets.*

This calls `prober/prober.py` with `--cache-acts`. The script prevents running cache creation and evaluation simultaneously. The cache directories are structured as: `runs/work/<model>/<act>/<datasetTag>/caches/`.

For a manual caching run:
```bash
python prober/prober.py \
  --work-dir WORK_DIR \
  --dataset DATASET_PATH \
  --act resid_post \
  --cache-acts \
  --model MODEL \
  --metadata METADATA_PATH \
  --batch-size 256 \
  --pool fim \
  --repro \
  --model-revision main
```

Where:
- `--model MODEL`: the Hugging Face model identifier. Valid values are: `bigcode/santacoder`, and  `codellama/CodeLlama-7b-hf`.
- `--dataset DATASET_PATH`: the path to the Parquet dataset file (e.g., `/path/to/tagged.parquet`).
- `--metadata METADATA_PATH`: the path to the accompanying metadata file (e.g., `/path/to/tagged.meta.parquet`).
- `--work-dir WORK_DIR`: the base directory for saving caches and results (e.g., `runs/work`).

*Note: Append `--java` to caching activations for Java examples.*


## Probe Training

After caching, train linear probes using the cached activations:

```bash
./trainProbeAll.sh workDir datasetDir
./controlTask.sh workDir datasetDir
```

*Note: replace `workDir` with the desired directory and `datasetDir` with the directory containing the validated generated datasets.*


Internally, this invokes `prober/prober.py` using `--use-cache`. To train control baselines (randomized labels), it passes `--control-task`. Probe artifacts and summary `results.json` files are stored under `workDir/<model>/<act>/<datasetTag>/probers/<lang>/results/`.

## Cross-Lingual Evaluation

Once the probes are trained, you can evaluate them on out-of-domain partitions (e.g., zero-shot transfer from Java to Python, or standard to adversarial) to test for type representations shared across languages:

**Local execution:**
```bash
python prober/prober_crossEval.py \
  --work-dir /path/to/work_dir_root \
  --datasets-dir /path/to/datasets_dir \
  [--dry-run]
```

**Slurm execution:**
```bash
python slurmCrossEval.py \
  --work-dir /path/to/work_dir_root \
  --datasets-dir /path/to/datasets_dir \
  [--dry-run]
```

Under the hood, these scripts iterate over the saved probes and execute `prober/prober.py` with `--eval-only` and `--eval-dir`. All evaluation JSON outputs are generated inside the respective model work directories.

## Reproducing Paper Results

To summarize, to fully reproduce the paper tables and figures:

- Generate datasets (`python genAndValidateDatasetAll.py`).
- Cache activations for the required models (`./cacheAll.sh`).
- Train probes (`./trainProbeAll.sh`).
- Train control probes (`./controlTask.sh`).
- Execute cross-evaluation to generate all `results.json` files (`python prober/prober_crossEval.py`).
- Run the reporting script to parse flat results, extract layer-wise metrics, and compile final artifacts:

```bash
python genPaperV2.py \
  --work-dir /path/to/work_dir_root \
  --output-dir paper_output/
```

You can also generate the final artifacts directly from the `results.json` files we provide:
```bash
python genPaperV2.py \
  --work-dir paperData \
  --output-dir paperData/figAndTables
```

This script parses all `results.json` files in `<work-dir>/results`, and outputs:
- Per-model accuracy and selectivity plots (PDF & PNG).
- Raw accuracy and selectivity LaTeX tables containing optimal layer choices for `task0`, `task1`, and `task2`.

## Acknowledgements

We utilize the [SantaCoder](https://huggingface.co) and [CodeLlama](https://huggingface.co) models for our experiments and would like to thank the authors for their valuable contributions to the community.

Our implementation relies on the following libraries, for which we wish to thank their authors:
* [PyTorch](https://github.com/pytorch/pytorch): for tensor operations and model development.
* [Hugging Face Transformers](https://github.com/huggingface/transformers): for model loading and inference.
* [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens): for activation extraction.

```
[{Partially} omitted] (due to double-blind review constraints).
```
## Citation

If you use this repository or the resulting datasets in academic work, please cite the accompanying paper.

```text
[Omissis] (due to double-blind review constraints).
```