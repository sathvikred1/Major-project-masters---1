"""Run the initial, temporally separated binary sensor-fault baseline."""
import os

# Limit BLAS parallelism consistently for this small computational-cost baseline.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".mplconfig" / "cache"))

import importlib.metadata
import json
import math
import platform
from time import perf_counter
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, precision_recall_fscore_support)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from download_data import REVISION, sha256

SEED = 42
DATA = ROOT / "data" / "raw" / "Gain" / "beta=2"
FEATURE_FILE = DATA / "gain302.xlsx"
LABEL_FILE = DATA / "ygain302.xlsx"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
LABELS = [1, -1]  # Confusion matrix order: normal, faulty; faulty is the positive class.
CLASS_NAMES = ["Normal (1)", "Gain fault (-1)"]
EXPECTED_HASHES = {
    "gain302.xlsx": "90ad926d50595beae7e463fcc849f0a4e968de793aa033d5e25e9560f44581ac",
    "ygain302.xlsx": "d77d3351013cf1f4a7e98793b9f3bf907f59fec609fb715651c1ad1a8c431aef",
}
METRIC_PLOTS = {
    "Accuracy": "accuracy", "Precision": "precision", "Recall": "recall",
    "F1 Score": "f1", "False Positive Rate": "false_positive_rate",
    "False Negative Rate": "false_negative_rate",
    "Training Time (seconds)": "training_time", "Prediction Time (seconds)": "prediction_time",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_and_inspect():
    for path in (FEATURE_FILE, LABEL_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run python src/download_data.py first.")
        if sha256(path) != EXPECTED_HASHES[path.name]:
            raise ValueError(f"Source checksum mismatch for {path}; refusing an undocumented data change.")
    raw = pd.read_excel(FEATURE_FILE, header=None)
    # Row 1 groups time steps; row 2 contains the twelve original variable names.
    x = pd.read_excel(FEATURE_FILE, header=1)
    label_frame = pd.read_excel(LABEL_FILE, header=None)
    if label_frame.shape[1] != 1 or len(x) != len(label_frame):
        raise ValueError("Expected one aligned, headerless label column per feature observation.")
    y = label_frame.iloc[:, 0]
    if set(y.unique()) != {-1, 1} or y.isna().any():
        raise ValueError("Expected documented binary labels -1 (fault) and 1 (normal).")
    if x.shape[1] != 12 or not all(pd.api.types.is_numeric_dtype(dtype) for dtype in x.dtypes):
        raise ValueError("Expected twelve numeric sensor features. Inspect changed source before proceeding.")
    y = y.astype(int).rename("y")
    distribution = {str(label): {"name": name, "count": int((y == label).sum()),
                                 "percent": float(100 * (y == label).mean())}
                    for label, name in zip(LABELS, CLASS_NAMES)}
    audit = {
        "source_revision": REVISION,
        "selected_feature_file": str(FEATURE_FILE.relative_to(ROOT)),
        "selected_label_file": str(LABEL_FILE.relative_to(ROOT)),
        "file_sha256": EXPECTED_HASHES,
        "nominal_fault_percent": 30, "beta_directory": "beta=2",
        "selection_reason": "Moderate fault prevalence with substantial counts in both classes; first listed beta value, selected before fitting. No subsampling.",
        "feature_worksheet_shape_including_headers": list(raw.shape),
        "feature_matrix_shape": list(x.shape), "label_matrix_shape": list(label_frame.shape),
        "observations": len(x), "input_features": x.shape[1],
        "feature_names": list(x.columns), "target": "y (separate headerless label workbook, column A)",
        "unique_target_values": sorted(map(int, y.unique())), "class_distribution": distribution,
        "missing_values_by_feature": x.isna().sum().astype(int).to_dict(),
        "missing_target_values": int(y.isna().sum()),
        "infinite_values_by_feature": dict(zip(x.columns, map(int, np.isinf(x.to_numpy()).sum(axis=0)))),
        "duplicate_feature_rows": int(x.duplicated().sum()),
        "duplicate_feature_target_rows": int(x.assign(y=y).duplicated().sum()),
        "feature_dtypes": x.dtypes.astype(str).to_dict(), "target_dtype": str(y.dtype),
        "feature_ranges": x.agg(["min", "max"]).to_dict(),
        "identifier_columns": [], "categorical_columns": [],
        "adjacent_rows_sharing_eight_time_window_features": int(np.all(
            x.iloc[:-1, 4:12].to_numpy() == x.iloc[1:, 0:8].to_numpy(), axis=1).sum()),
    }
    write_json(RESULTS / "dataset_inspection.json", audit)
    print("Dataset inspection (before any training):", flush=True)
    print(json.dumps(audit, indent=2), flush=True)
    # No observed missing/infinite values; this policy also handles them within the pipeline on reruns.
    return x.replace([np.inf, -np.inf], np.nan), y, audit


def create_split(x, y):
    # A random row split leaks overlapping three-step windows. Purge two rows.
    test_start = len(x) - math.ceil(0.20 * len(x))
    train_indices = np.arange(test_start - 2)
    test_indices = np.arange(test_start, len(x))
    if len(np.unique(y.iloc[train_indices])) != 2 or len(np.unique(y.iloc[test_indices])) != 2:
        raise ValueError("Temporal partitions must contain both classes.")
    assert train_indices.max() + 2 < test_indices.min()
    assert not set(map(tuple, x.iloc[train_indices].to_numpy())).intersection(
        map(tuple, x.iloc[test_indices].to_numpy()))
    assignment = np.full(len(x), "purged", dtype=object)
    assignment[train_indices] = "train"
    assignment[test_indices] = "test"
    pd.DataFrame({"observation_index_zero_based": np.arange(len(x)), "partition": assignment}).to_csv(
        RESULTS / "split_assignments.csv", index=False)
    metadata = {
        "method": "chronological nominal 80/20 holdout with two training-boundary rows purged",
        "reason": "All adjacent rows share eight measurements; random stratification would share source time steps.",
        "train_observations": len(train_indices), "test_observations": len(test_indices),
        "purged_indices": list(map(int, np.flatnonzero(assignment == "purged"))),
        "train_index_range_inclusive": [int(train_indices.min()), int(train_indices.max())],
        "test_index_range_inclusive": [int(test_indices.min()), int(test_indices.max())],
        "train_class_counts": {str(k): int(v) for k, v in y.iloc[train_indices].value_counts().items()},
        "test_class_counts": {str(k): int(v) for k, v in y.iloc[test_indices].value_counts().items()},
        "identical_complete_vectors_shared_between_partitions": 0,
        "seed": SEED, "split_is_randomized": False,
    }
    write_json(RESULTS / "split_metadata.json", metadata)
    return train_indices, test_indices, metadata


def build_models():
    def pipeline(classifier, scale=False):
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        return Pipeline(steps + [("classifier", classifier)])
    return {
        "Random Forest": pipeline(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)),
        "Support Vector Machine": pipeline(SVC(kernel="rbf", C=1.0, gamma="scale"), scale=True),
        "Multilayer Perceptron": pipeline(MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            max_iter=1000, random_state=SEED, early_stopping=False), scale=True),
    }


def evaluate_models(x, y, train, test):
    rows, details = [], {}
    predictions = pd.DataFrame({"observation_index_zero_based": test, "actual_y": y.iloc[test].to_numpy()})
    models = build_models()
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            start = perf_counter()
            model.fit(x.iloc[train], y.iloc[train])
            training_seconds = perf_counter() - start
        start = perf_counter()
        predicted = model.predict(x.iloc[test])
        prediction_seconds = perf_counter() - start
        precision, recall, f1, _ = precision_recall_fscore_support(
            y.iloc[test], predicted, average="binary", pos_label=-1, zero_division=0)
        matrix = confusion_matrix(y.iloc[test], predicted, labels=LABELS)
        tn, fp, fn, tp = map(int, matrix.ravel())
        row = {"Model": name, "Accuracy": accuracy_score(y.iloc[test], predicted),
               "Precision": precision, "Recall": recall, "F1 Score": f1,
               "False Positive Rate": fp / (fp + tn), "False Negative Rate": fn / (fn + tp),
               "Training Time (seconds)": training_seconds, "Prediction Time (seconds)": prediction_seconds}
        rows.append(row)
        slug = name.lower().replace(" ", "_")
        predictions[slug] = predicted
        report_args = dict(labels=LABELS, target_names=CLASS_NAMES, zero_division=0)
        report = classification_report(y.iloc[test], predicted, **report_args, output_dict=True)
        write_json(RESULTS / "classification_reports" / f"{slug}.json", report)
        (RESULTS / "classification_reports" / f"{slug}.txt").write_text(
            classification_report(y.iloc[test], predicted, **report_args, digits=6), encoding="utf-8")
        classifier = model.named_steps["classifier"]
        warning_text = [f"{w.category.__name__}: {w.message}" for w in captured]
        for message in warning_text:
            print(f"{name}: {message}", flush=True)
        details[name] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
                         "confusion_matrix_rows_actual_columns_predicted": matrix.tolist(),
                         "warnings": warning_text, "classifier_parameters": classifier.get_params()}
        if isinstance(classifier, MLPClassifier):
            details[name].update({"iterations": int(classifier.n_iter_),
                                  "final_training_loss": float(classifier.loss_),
                                  "stopping_rule": "training loss improvement below tol=0.0001 for n_iter_no_change=10; max_iter=1000"})
            pd.DataFrame({"iteration": np.arange(1, classifier.n_iter_ + 1),
                          "training_loss": classifier.loss_curve_}).to_csv(RESULTS / "mlp_loss_curve.csv", index=False)
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{name}\nChronological held-out sensor observations")
        fig.tight_layout()
        fig.savefig(FIGURES / f"{slug}_confusion_matrix.png", dpi=180)
        plt.close(fig)
        print(f"Completed {name}: accuracy={row['Accuracy']:.6f}, FP={fp}, FN={fn}", flush=True)
    predictions.to_csv(RESULTS / "test_predictions.csv", index=False)
    write_json(RESULTS / "model_details.json", details)
    return pd.DataFrame(rows), details


def plot_class_distribution(audit):
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(7, 5))
    classes = list(audit["class_distribution"].values())
    bars = ax.bar([c["name"] for c in classes], [c["count"] for c in classes], color=["#386cb0", "#e08214"])
    ax.bar_label(bars, labels=[f"{c['count']:,} ({c['percent']:.2f}%)" for c in classes], padding=5)
    ax.set(title="Selected gain-fault dataset: class distribution", xlabel="Documented class", ylabel="Observations")
    ax.set_ylim(0, max(c["count"] for c in classes) * 1.18)
    fig.tight_layout()
    fig.savefig(FIGURES / "class_distribution.png", dpi=180)
    plt.close(fig)


def make_comparison_figures(table):
    for metric, filename in METRIC_PLOTS.items():
        fig, ax = plt.subplots(figsize=(8, 5.5))
        bars = ax.bar(table["Model"], table[metric], color=["#386cb0", "#e08214", "#3b8c6e"])
        is_time = "seconds" in metric
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in table[metric]], padding=5)
        ax.set(title=f"Baseline model comparison: {metric}", xlabel="Model",
               ylabel="Seconds (full partition)" if is_time else f"{metric} (fraction)")
        ax.set_ylim(0, max(float(table[metric].max()) * 1.2, 0.001) if is_time else 1.08)
        ax.tick_params(axis="x", labelsize=10)
        fig.tight_layout()
        fig.savefig(FIGURES / f"{filename}_comparison.png", dpi=180)
        plt.close(fig)


def best_names(table, metric, minimize=False):
    value = table[metric].min() if minimize else table[metric].max()
    return ", ".join(table.loc[table[metric] == value, "Model"])


def markdown_table(table):
    # Avoid a tabulate dependency just for Markdown output.
    columns = list(table.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]) if c == "Model" else f"{row[c]:.6f}" for c in columns) + " |")
    return "\n".join(lines)


def interpret(table, details):
    accuracy_best = best_names(table, "Accuracy")
    recall_best = best_names(table, "Recall")
    fpr_best = best_names(table, "False Positive Rate", minimize=True)
    fnr_best = best_names(table, "False Negative Rate", minimize=True)
    observations = [
        f"Highest accuracy: {accuracy_best}. Highest fault recall: {recall_best}; lowest missed-fault rate: {fnr_best}.",
        f"Lowest false-alarm rate: {fpr_best}. Highest fault F1: {best_names(table, 'F1 Score')}.",
        f"The accuracy winner {'also attains' if set(accuracy_best.split(', ')) & set(fnr_best.split(', ')) else 'does not attain'} the lowest false-negative rate.",
    ]
    for name in table["Model"]:
        d = details[name]
        observations.append(f"{name}: {d['fp']} normal observations falsely flagged; {d['fn']} gain-fault observations missed; {d['tp']} faults detected and {d['tn']} normal observations accepted.")
    train_ratio = table["Training Time (seconds)"].max() / table["Training Time (seconds)"].min()
    prediction_ratio = table["Prediction Time (seconds)"].max() / table["Prediction Time (seconds)"].min()
    observations.append(f"Fastest training: {best_names(table, 'Training Time (seconds)', True)}; fastest prediction: {best_names(table, 'Prediction Time (seconds)', True)}. Slowest/fastest ratios: {train_ratio:.2f}x training and {prediction_ratio:.2f}x prediction in this single run.")
    observations.append("These are gain-fault versus normal results only; no per-fault-type difficulty comparison is supported. Timing is host dependent and not an embedded-device benchmark.")
    return observations


def save_summary_and_readme(table, audit, split, details):
    observations = interpret(table, details)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "processor": platform.processor(), "seed": SEED,
                   "thread_policy": "RF n_jobs=1; BLAS/OpenMP environment thread limits=1",
                   "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    write_json(RESULTS / "environment.json", environment)
    lines = [
        "INITIAL BASELINE EXPERIMENT — measured results", "",
        f"Dataset: {audit['selected_feature_file']}", f"Labels: {audit['selected_label_file']}",
        f"Source revision: {REVISION}",
        "Learning task: binary; source 1=normal and -1=gain fault. Faulty (-1) is the positive class.",
        "No concatenation: fault archives share normal observations. No spike/data-loss classes are present in this selection.",
        f"Samples: {audit['observations']}; features: {audit['input_features']} ({', '.join(audit['feature_names'])}).",
        "Selection: nominal 30% faults, beta=2; enough of both classes without subsampling, selected before fitting.",
        f"Class distribution: {json.dumps(audit['class_distribution'])}",
        f"Inspection: {sum(audit['missing_values_by_feature'].values())} missing feature values, {sum(audit['infinite_values_by_feature'].values())} infinities, {audit['duplicate_feature_rows']} duplicate complete feature vectors; feature types {sorted(set(audit['feature_dtypes'].values()))}, target {audit['target_dtype']}; no identifier or categorical inputs.",
        f"Split: {json.dumps(split)}",
        "Preprocessing: replace infinities with missing; training-only median imputation in all pipelines; training-only StandardScaler for SVM and MLP. RF unscaled. No resampling or tuning.",
        "Large finite values are retained because injected sensor abnormalities are the signal being detected; no physical-range clipping is applied.",
        "Models: RF 100 trees, seed 42, n_jobs=1; SVC RBF, C=1, gamma=scale; MLP (32,16), ReLU/Adam, seed 42, max_iter=1000, no early stopping validation split.",
        "All other classifier parameters are saved in model_details.json. RBF provides a conventional nonlinear SVM baseline.",
        "Precision/recall/F1 apply to -1; FPR=FP/(FP+TN), FNR=FN/(FN+TP). Reports include per-class, macro and weighted averages.",
        "Timing: one perf_counter measurement of full pipeline.fit and pipeline.predict on the same partitions, excluding file I/O and plotting. Includes preprocessing; no timing repetitions.",
        "", markdown_table(table), "", "Measured interpretation:", *[f"- {item}" for item in observations], "",
        f"MLP optimization: {details['Multilayer Perceptron'].get('iterations')} iterations; final training loss {details['Multilayer Perceptron'].get('final_training_loss'):.8f}.",
        f"Training warnings: {json.dumps({name: d['warnings'] for name, d in details.items()})}",
        "", "Research limits and next steps:",
        "This is an initial baseline on one synthetic gain-fault scenario and one chronological holdout, not a completed reliability study.",
        "Purging prevents direct sliding-window boundary overlap; nearby time blocks may still be correlated. No node IDs or timestamps are supplied for deployment-level separation.",
        "Class prevalence differs over time; the holdout tests a later segment rather than an IID random sample. No test-based tuning was performed.",
        "Fault labels and patterns are synthetic. Detection metrics are reliability-related proxies, not measured uptime, lifetime, or field failure probabilities.",
        "Next: repeat across purged time blocks, fault rates, beta directories and other fault types; estimate uncertainty and benchmark on target hardware.",
    ]
    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    readme = ROOT / "README.md"
    content = readme.read_text(encoding="utf-8")
    for marker, replacement in [("RESULTS", markdown_table(table)),
                                 ("FINDINGS", "\n".join(f"- {s}" for s in (observations[0], observations[1], observations[2], observations[-2], observations[-1])))]:
        start, end = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
        if content.count(start) != 1 or content.count(end) != 1:
            raise ValueError(f"README must contain one {marker} marker pair for automatic measured results.")
        before, rest = content.split(start)
        _, after = rest.split(end)
        content = before + start + "\n" + replacement + "\n" + end + after
    readme.write_text(content, encoding="utf-8")


def main():
    for path in (RESULTS, FIGURES, RESULTS / "classification_reports"):
        path.mkdir(parents=True, exist_ok=True)
    x, y, audit = load_and_inspect()
    # Save class distribution before fitting anything.
    plot_class_distribution(audit)
    train, test, split = create_split(x, y)
    table, details = evaluate_models(x, y, train, test)
    table.to_csv(RESULTS / "model_comparison.csv", index=False)
    make_comparison_figures(table)
    save_summary_and_readme(table, audit, split, details)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\nSaved measured outputs to {RESULTS}")


if __name__ == "__main__":
    main()
