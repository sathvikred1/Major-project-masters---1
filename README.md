# Machine Learning-Based Fault Detection and Reliability in Internet of Things Sensor Networks

Master of Science in Computer Science research project, CSCI 7200.

## Project Overview

Faulty sensor measurements can reduce the usefulness of Internet of Things monitoring systems. This project compares machine-learning classifiers for identifying faulty wireless sensor observations. Classification performance, false alarms, missed faults, and execution cost provide an initial view of detection performance and its reliability implications. They do not directly measure network uptime or sensor lifetime.

## Research Question

“How do commonly used machine-learning models compare for sensor fault detection when evaluated using both classification performance and reliability-related measures?”

## Dataset

The public [Dataset for Wireless Sensor Network Fault Detection](https://github.com/tmoulahi/Dataset-for-WSN-fault-detection) supplies injected faults in temperature and humidity observations. The source describes outdoor multi-hop TelosB sensor data derived from a 2010 collection, with each prepared vector covering three successive time steps. The twelve worksheet variables are:

`T01, H01, T02, H02, T11, H11, T12, H12, T21, H21, T22, H22`.

The first digit denotes the time group (`t0`, `t1`, `t2`); the second distinguishes the two measurements of each quantity within that group, following the worksheet headers. No node identifiers or timestamps are supplied.

The initial selection is **`Gain.rar` → `Gain/beta=2/gain302.xlsx`**, paired by row with **`Gain/beta=2/ygain302.xlsx`**, at source revision **`a36f90b55b30ba3552747430fb508735025ca80b`**. The feature matrix has **4,688 rows × 12 columns**. Its worksheet has 4,690 rows including two header rows; the separate label workbook has 4,688 rows × 1 column with no header. The combined analytical observation has twelve inputs and one target.

This is **binary classification**, with documented source labels **`1 = normal`** and **`-1 = faulty`**. The selected faults are **gain faults only**. The full source also distributes offset, stuck-at, and out-of-bounds archives; those are not additional classes in this experiment. No spike or data-loss faults are asserted.

The nominal **30% fault** variant contains **3,280 normal (69.9659%)** and **1,408 faulty (30.0341%)** observations. It was selected before training to retain substantial counts of both classes without a balanced artificial 50/50 scenario or computational subsampling. `beta=2` is the first listed beta directory. The repository does not define beta's physical meaning; this work records the directory without assigning an unsupported interpretation.

Inspection found twelve `float64` inputs, `int64` labels, no missing or infinite values, and no duplicate complete feature rows. See [the complete inspection](results/dataset_inspection.json), [source inventory](results/dataset_inventory.csv), and [source audit](results/source_audit.json). The source README describes 60 datasets; the archive audit records actual files and a mismatched filename rather than assuming they all pair cleanly. Several thousand normal vectors are shared between fault families, so naive concatenation would duplicate normal data. The initial experiment uses one documented binary pair.

See [data/README.md](data/README.md) for provenance, checksums, download instructions, and licensing limitations. Raw archives and workbooks are excluded from Git.

## Models

- **Random Forest:** primary baseline, 100 trees, `random_state=42`, `n_jobs=1`, other scikit-learn defaults. Tree-based classification motivates this baseline; no external performance numbers are used as our results.
- **Support Vector Machine:** `SVC(kernel="rbf", C=1.0, gamma="scale")`. The RBF kernel provides a conventional nonlinear baseline using standardized features; probabilities and hyperparameter searches are disabled.
- **Multilayer Perceptron:** hidden layers `(32, 16)`, ReLU, Adam, `max_iter=1000`, `random_state=42`, default training-loss stopping tolerance `1e-4`, no early-stopping validation partition. Features are standardized. Actual iterations, loss curve, warnings and parameters are saved.

All three models use the same observations and evaluation. None receives tuning on the test set. Full classifier parameters are in [model_details.json](results/model_details.json).

## Methodology

Dataset inspection → preprocessing policy → shared chronological train/test split → training-only pipeline fitting → held-out prediction → evaluation → comparison.

**Reason for temporal separation:** all 4,687 adjacent row pairs share eight measurements: a row's `t1,t2` values are the next row's `t0,t1` values. A random stratified split would share constituent sensor readings across partitions. Therefore the experiment uses the first nominal 80% for training and the last 20% for testing, purging two training-boundary rows to remove direct window overlap.

- Training: zero-based rows **0–3747**, **3,748 observations** (2,669 normal; 1,079 faulty).
- Purged: rows **3748–3749**, **2 observations**.
- Testing: rows **3750–4687**, **938 observations** (611 normal; 327 faulty).

The nominal boundary is 3,750/938 before purging. This is a chronological holdout, so stratification and random shuffling are intentionally inapplicable. Seed 42 controls stochastic estimators. Both partitions contain both classes. All data are used except the two embargo rows; there is no computational sampling. [Split assignments](results/split_assignments.csv) and [split metadata](results/split_metadata.json) make the partition explicit. No identical complete vectors occur across it.

All inputs are numeric sensor variables, so no categorical encoding or identifier removal is needed. Infinite values are converted to missing values, and a median imputer is fitted **only on training data** inside each `Pipeline`. The observed dataset needs no imputation. SVM and MLP additionally fit `StandardScaler` on training data; RF uses unscaled inputs. Labels remain `1/-1`, with **`-1` explicitly designated the positive fault class**.

Large finite sensor values are retained: injected abnormalities are the detection signal, so physically implausible humidity readings are not clipped away as preprocessing errors.

## Evaluation Metrics

Let TP be detected faults, TN accepted normal observations, FP falsely flagged normal observations, and FN missed faults.

| Measure | Interpretation |
| --- | --- |
| Accuracy | `(TP+TN)/N`: overall correct fraction |
| Precision | `TP/(TP+FP)`: fraction of fault alarms that are correct |
| Recall | `TP/(TP+FN)`: fraction of faults detected |
| F1 score | Harmonic mean of fault precision and recall |
| False positive rate | `FP/(FP+TN)`: normal observations falsely flagged |
| False negative rate | `FN/(FN+TP)`: faulty observations missed |
| Training time | Wall-clock seconds for full training pipeline fit |
| Prediction time | Wall-clock seconds for full pipeline prediction of all 938 test rows |

Complete reports include per-class scores, macro averages and weighted averages. Confusion matrix rows are actual classes; columns are predicted classes, ordered normal then gain fault. Timing uses `perf_counter`, includes preprocessing, and excludes downloads, file reads and plotting. RF and BLAS/OpenMP thread limits are one. Each timing is a single measurement on the recorded host; small time differences should not be generalized to embedded devices.

## Initial Experimental Results

All scores below are fractions, not percentages; times are seconds. This block is regenerated from measured outputs on every run.

<!-- RESULTS:START -->
| Model | Accuracy | Precision | Recall | F1 Score | False Positive Rate | False Negative Rate | Training Time (seconds) | Prediction Time (seconds) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.998934 | 0.996951 | 1.000000 | 0.998473 | 0.001637 | 0.000000 | 0.317040 | 0.007681 |
| Support Vector Machine | 0.997868 | 0.993921 | 1.000000 | 0.996951 | 0.003273 | 0.000000 | 0.011074 | 0.002948 |
| Multilayer Perceptron | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.350107 | 0.001491 |
<!-- RESULTS:END -->

See [comparison CSV](results/model_comparison.csv), [automatically generated interpretation](results/summary.txt), [classification reports](results/classification_reports), and [figures](results/figures). Saved predictions permit independent metric verification without retraining.

## Key Findings

<!-- FINDINGS:START -->
- Highest accuracy: Multilayer Perceptron. Highest fault recall: Random Forest, Support Vector Machine, Multilayer Perceptron; lowest missed-fault rate: Random Forest, Support Vector Machine, Multilayer Perceptron.
- Lowest false-alarm rate: Multilayer Perceptron. Highest fault F1: Multilayer Perceptron.
- The accuracy winner also attains the lowest false-negative rate.
- Fastest training: Support Vector Machine; fastest prediction: Multilayer Perceptron. Slowest/fastest ratios: 31.61x training and 5.15x prediction in this single run.
- These are gain-fault versus normal results only; no per-fault-type difficulty comparison is supported. Timing is host dependent and not an embedded-device benchmark.
<!-- FINDINGS:END -->

## Repository Structure

```text
README.md
requirements.txt              # Exact versions of the five direct dependencies
data/README.md                # Provenance and retrieval instructions
data/raw/                     # Ignored downloads/extracted workbooks
src/download_data.py          # Pinned, checksum-verified retrieval
src/inspect_dataset.py        # Cross-version/family inventory; no training
src/experiment.py             # Reproducible experiment and generated interpretation
src/validate_results.py       # Independently check saved predictions and outputs
results/
  model_comparison.csv
  summary.txt
  dataset_inventory.csv
  source_audit.json
  dataset_inspection.json
  split_assignments.csv
  split_metadata.json
  test_predictions.csv
  model_details.json
  mlp_loss_curve.csv
  environment.json
  validation.json
  reproducibility_check.json  # Two full runs compared; timings may vary
  classification_reports/     # Text and JSON, all three models
  figures/                    # Class distribution, 3 confusion matrices, 8 comparisons
```

## Running the Experiment

The recorded run uses Python 3.14.3. Use Python 3.14 for the pinned package versions, and a RAR-capable extractor (`bsdtar`, `unrar`, or `7z`). macOS supplies `bsdtar`; on Linux install a RAR-capable libarchive tool, unrar or 7-Zip. On Windows install 7-Zip and add `7z` to PATH, or extract the archives manually into `data/raw/` using the paths in the data README.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/download_data.py
python src/experiment.py
python src/validate_results.py
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1` (use `python -m venv .venv` to create it). Windows Command Prompt activation: `.venv\Scripts\activate.bat`.

To repeat the complete archive inspection as well:

```bash
python src/download_data.py --all
python src/inspect_dataset.py
```

Scripts resolve paths relative to the repository. Re-running the experiment regenerates its outputs and updates only the marked result/finding blocks in this README. Training times will change between runs. Downloads are cached and verified against pinned hashes.

To repeat the full run and verify identical predictions and non-timing metrics against the current outputs, run `python src/validate_results.py --rerun`. The saved experiment was executed twice and passed this check.

## Reproducibility

Random seed = **42**; one shared chronological split; consistent preprocessing policy; the same evaluation procedure for all models; no test-based tuning. Fixed source revision and workbook SHA-256 hashes guard against silent data changes. The environment file records Python, package versions, platform and thread policy. The complete environment is described rather than introducing unnecessary direct dependencies. Numerical libraries and hardware can still affect exact outputs and timings.

## Current Research Status

This is an **INITIAL BASELINE EXPERIMENT**. It evaluates one injected gain-fault scenario, one beta directory and one later time segment. Purging prevents direct shared-window overlap but cannot remove all temporal dependence or demonstrate generalization to unseen devices. There are no supplied node IDs/timestamps for such a split. Class prevalence differs over time. No confidence intervals, cross-validation or real deployment reliability claims are made.

Next steps are repeated purged time-block evaluation, uncertainty estimates, other fault percentages and beta directories, separate evaluation of offset/stuck-at/out-of-bounds faults, and target-device resource measurements. A broader multiclass study would first require defensible reconciliation of overlapping normal observations and fault labels.

## Source References

1. Suthaharan, S., Alzahrani, M., Rajasegarar, S., Leckie, C., and Palaniswami, M. “Labelled Data Collection for Anomaly Detection in Wireless Sensor Networks.” ISSNIP, 2010. Referenced by the dataset documentation for the original measurements.
2. Zidi, S., Moulahi, T., and Alaya, B. “Fault detection in Wireless Sensor Networks through SVM classifier.” IEEE Sensors Journal, 18(1), January 2018. Attribution requested by the dataset authors. This project does not copy reported paper scores.
3. [Pinned dataset documentation](https://github.com/tmoulahi/Dataset-for-WSN-fault-detection/blob/a36f90b55b30ba3552747430fb508735025ca80b/README.md).
