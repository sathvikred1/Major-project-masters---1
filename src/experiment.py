"""Compare three classifiers separately on four sensor-fault datasets."""
import os
from pathlib import Path

# Use one thread so repeated runs use the same numerical settings.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".mplconfig" / "cache"))

import argparse
import importlib.metadata
import json
import math
import platform
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from download_data import DATASETS, REVISION, sha256

SEED = 42
RESULTS = ROOT / "results" / "fault_comparison"
RAW = ROOT / "data" / "raw"
FEATURE_NAMES = ["T01", "H01", "T02", "H02", "T11", "H11", "T12", "H12", "T21", "H21", "T22", "H22"]
LABELS = [1, -1]  # Matrix order: normal, fault. Fault is positive.
CLASS_NAMES = ["Normal (1)", "Fault (-1)"]
METRICS = ["Fault", "Model", "Accuracy", "Precision", "Recall", "F1 Score", "False Positive Rate", "False Negative Rate"]
SOURCE = f"https://github.com/tmoulahi/Dataset-for-WSN-fault-detection/blob/{REVISION}/README.md"


def load_and_inspect(fault):
    feature_name, label_name, feature_hash, label_hash = DATASETS[fault]
    feature_file, label_file = RAW / feature_name, RAW / label_name
    hashes = {feature_name: feature_hash, label_name: label_hash}
    for path, digest in ((feature_file, feature_hash), (label_file, label_hash)):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run python src/download_data.py first.")
        if sha256(path) != digest:
            raise ValueError(f"Checksum mismatch: {path}")
    # The feature workbook has two header rows; the labels have none.
    x = pd.read_excel(feature_file, header=1)
    labels = pd.read_excel(label_file, header=None)
    if labels.shape != (len(x), 1):
        raise ValueError("Expected one label for every feature row.")
    y = labels.iloc[:, 0].rename("y")
    if y.isna().any() or set(y.unique()) != {-1, 1}:
        raise ValueError("Expected labels 1 (normal) and -1 (fault).")
    if list(x.columns) != FEATURE_NAMES or not all(pd.api.types.is_numeric_dtype(dtype) for dtype in x.dtypes):
        raise ValueError("Expected the twelve temperature and humidity columns.")
    missing = int(x.isna().sum().sum())
    infinite = int(np.isinf(x.to_numpy()).sum())
    if missing or infinite:
        raise ValueError(f"Check the source data: {missing} missing values and {infinite} infinities.")
    shared = int(np.all(x.iloc[:-1, 4:12].to_numpy() == x.iloc[1:, 0:8].to_numpy(), axis=1).sum())
    audit = {
        "source_documentation": SOURCE,
        "source_revision": REVISION,
        "fault": fault, "nominal_fault_percent": 30,
        "feature_file": str(feature_file.relative_to(ROOT)),
        "label_file": str(label_file.relative_to(ROOT)),
        "sha256": hashes,
        "observations": len(x), "features": list(x.columns),
        "labels": {"1": "normal", "-1": "fault (positive class)"},
        "class_counts": {str(label): int((y == label).sum()) for label in LABELS},
        "missing_values": missing, "infinite_values": infinite,
        "adjacent_rows_sharing_eight_measurements": shared,
    }
    print(f"{fault}: {len(x):,} rows, {x.shape[1]} features; "
          f"normal={audit['class_counts']['1']:,}, faulty={audit['class_counts']['-1']:,}.")
    return x, y.astype(int), audit


def create_split(x, y):
    # Each row covers three time steps. Skip two rows before the test block.
    test_start = len(x) - math.ceil(0.20 * len(x))
    train = np.arange(test_start - 2)
    test = np.arange(test_start, len(x))
    if len(y.iloc[train].unique()) != 2 or len(y.iloc[test].unique()) != 2:
        raise ValueError("Both partitions must contain normal and faulty readings.")
    if set(map(tuple, x.iloc[train].to_numpy())).intersection(map(tuple, x.iloc[test].to_numpy())):
        raise ValueError("Identical feature rows occur in both partitions.")
    split = {
        "method": "chronological 80/20 split with two rows removed before testing",
        "train_rows": [int(train[0]), int(train[-1])],
        "excluded_rows": [test_start - 2, test_start - 1],
        "test_rows": [int(test[0]), int(test[-1])],
        "train_count": len(train), "test_count": len(test),
        "train_class_counts": {str(label): int((y.iloc[train] == label).sum()) for label in LABELS},
        "test_class_counts": {str(label): int((y.iloc[test] == label).sum()) for label in LABELS},
        "normal_T01_range": {
            "train": list(map(float, x.iloc[train].loc[y.iloc[train].eq(1), "T01"].agg(["min", "max"]))),
            "test": list(map(float, x.iloc[test].loc[y.iloc[test].eq(1), "T01"].agg(["min", "max"]))),
        },
    }
    return train, test, split


def build_models():
    def pipeline(classifier, scale=False):
        # Retain the original preprocessing; the checked data need no imputation.
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        return Pipeline(steps + [("classifier", classifier)])
    return {
        "Random Forest": pipeline(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)),
        "Support Vector Machine": pipeline(SVC(kernel="rbf", C=1.0, gamma="scale"), scale=True),
        "Multilayer Perceptron": pipeline(MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=1000, random_state=SEED, early_stopping=False), scale=True),
    }


def evaluate_models(fault, x, y, train, test):
    rows, details = [], {}
    predictions = pd.DataFrame({"Fault": fault, "observation_index_zero_based": test, "actual_y": y.iloc[test].to_numpy()})
    for name, model in build_models().items():
        # Every model and its preprocessing see only the same training rows.
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(x.iloc[train], y.iloc[train])
        predicted = model.predict(x.iloc[test])
        precision, recall, f1, _ = precision_recall_fscore_support(
            y.iloc[test], predicted, average="binary", pos_label=-1, zero_division=0)
        matrix = confusion_matrix(y.iloc[test], predicted, labels=LABELS)
        tn, fp, fn, tp = map(int, matrix.ravel())
        rows.append({"Fault": fault, "Model": name, "Accuracy": accuracy_score(y.iloc[test], predicted),
                     "Precision": precision, "Recall": recall, "F1 Score": f1,
                     "False Positive Rate": fp / (fp + tn), "False Negative Rate": fn / (fn + tp)})
        slug = name.lower().replace(" ", "_")
        predictions[slug] = predicted
        warning_text = [f"{w.category.__name__}: {w.message}" for w in captured]
        for message in warning_text:
            print(f"{name}: {message}")
        details[name] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
                         "confusion_matrix": matrix.tolist(), "warnings": warning_text}
        print(f"{name}: accuracy={rows[-1]['Accuracy']:.6f}, false alarms={fp}, missed faults={fn}")
    return pd.DataFrame(rows, columns=METRICS), details, predictions


def plot_comparison(table):
    faults, models = list(DATASETS), list(build_models())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, metric in zip(axes.flat, ["Accuracy", "Precision", "Recall", "F1 Score"]):
        positions, width = np.arange(len(faults)), 0.24
        for index, model in enumerate(models):
            values = table.loc[table.Model.eq(model)].set_index("Fault").loc[faults, metric]
            ax.bar(positions + (index - 1) * width, values, width, label=model)
        ax.set_xticks(positions, faults)
        ax.set(title=metric, ylabel="Score (fraction)", ylim=(0, 1.05))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Normal vs fault: separate tests at the nominal 30% fault rate")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(RESULTS / "model_comparison.png", dpi=180)
    plt.close(fig)


def plot_matrices(details):
    for model in build_models():
        fig, axes = plt.subplots(2, 2, figsize=(9, 8))
        for ax, (fault, saved) in zip(axes.flat, details.items()):
            matrix = np.array(saved["models"][model]["confusion_matrix"])
            ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
                ax=ax, cmap="Blues", colorbar=False, values_format="d")
            ax.set_title(fault)
        fig.suptitle(model)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        slug = model.lower().replace(" ", "_")
        fig.savefig(RESULTS / f"{slug}_confusion_matrix.png", dpi=180)
        plt.close(fig)


def markdown_table(table):
    lines = ["| " + " | ".join(table.columns) + " |", "| " + " | ".join(["---"] * len(table.columns)) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]) if c in {"Fault", "Model"} else
                                        f"{int(row[c]):,}" if pd.api.types.is_integer_dtype(table[c]) else f"{row[c]:.6f}"
                                        for c in table.columns) + " |")
    return "\n".join(lines)


def write_report(table, details):
    first = next(iter(details.values()))
    audit, split = first["dataset"], first["split"]
    files = "\n".join(f"- {fault}: `{spec[0]}` and `{spec[1]}`." for fault, spec in DATASETS.items())
    distribution = pd.DataFrame([
        {"Fault": fault, "Observations": saved["dataset"]["observations"],
         "Normal": saved["dataset"]["class_counts"]["1"], "Faulty": saved["dataset"]["class_counts"]["-1"]}
        for fault, saved in details.items()
    ])
    dataset = (
        f"The data come from the [Dataset for Wireless Sensor Network Fault Detection]({SOURCE}) "
        "by Zidi, Moulahi, and Alaya. Four nominal 30% fault datasets were selected: gain, offset, stuck-at, "
        "and out-of-bounds. Gain and offset use the beta=2 directory. The other two archives have no beta directories.\n\n"
        f"{files}\n\n{markdown_table(distribution)}\n\n"
        "Each observation contains two temperature and two humidity readings at three consecutive time steps, "
        f"giving twelve features: {', '.join(audit['features'])}. "
        "The source labels are 1 for normal and -1 for faulty. All four files have matching label sequences, "
        "and no missing or infinite values. These are altered versions of the same underlying sensor series, "
        "not four independent sensor networks."
    )
    method = (
        "Each fault type was tested separately as normal versus faulty. The files were not pooled into one dataset. "
        "Every model was trained again for each fault type using the same settings.\n\n"
        "Rows stayed in their original order. Adjacent observations share measurements, so a random split would "
        f"allow some readings to appear in training and testing. Each test used {split['train_count']:,} training rows "
        f"(zero-based rows {split['train_rows'][0]} to {split['train_rows'][1]}), skipped rows "
        f"{split['excluded_rows'][0]} and {split['excluded_rows'][1]}, and tested on the last {split['test_count']:,} rows "
        f"({split['test_rows'][0]} to {split['test_rows'][1]}). The test set contained "
        f"{split['test_class_counts']['1']} normal and {split['test_class_counts']['-1']} faulty observations "
        "for each fault type. All twelve model/fault comparisons used these same row positions.\n\n"
        "Random Forest used 100 trees. SVM used a radial basis function kernel with C=1 and gamma=scale. "
        "MLP used two hidden layers of 32 and 16 neurons and a maximum of 1,000 iterations. "
        "The random state was 42 for Random Forest and MLP. SVM and MLP inputs were standardized using only "
        "the training data; Random Forest used unscaled inputs. The existing median imputer stayed in the "
        "pipelines, although none of the selected files needed imputation. No parameters were tuned.\n\n"
        "Fault (-1) was the positive class. A false positive is a normal observation flagged as faulty; "
        "a false negative is a missed fault. Precision, recall, and F1 refer to faults. "
        "False positive rate is FP/(FP+TN), and false negative rate is FN/(FN+TP)."
    )
    observations = []
    for fault in DATASETS:
        subset = table.loc[table.Fault.eq(fault)]
        best_f1 = subset["F1 Score"].max()
        winners = ", ".join(subset.loc[subset["F1 Score"].eq(best_f1), "Model"])
        errors = "; ".join(f"{name}: {saved['fp']} false alarm{'s' if saved['fp'] != 1 else ''}, {saved['fn']} missed faults"
                           for name, saved in details[fault]["models"].items())
        observations.append(f"- **{fault}:** accuracy ranged from {subset.Accuracy.min():.2%} to "
                            f"{subset.Accuracy.max():.2%}. Highest fault F1: {winners}. {errors}.")
    observations = "\n".join(observations)
    ranges = " ".join(
        f"{model} accuracy ranged from {group.Accuracy.min():.2%} to {group.Accuracy.max():.2%} across the four fault types."
        for model, group in table.groupby("Model", sort=False)
    )
    offset = details["Offset"]
    diagnostics = (
        f"On offset faults, Random Forest flagged all {offset['split']['test_count']} test observations as faulty, "
        f"including all {offset['models']['Random Forest']['fp']} normal observations. "
        f"SVM missed {offset['models']['Support Vector Machine']['fn']} faults. "
        f"MLP had the highest offset accuracy, but still raised {offset['models']['Multilayer Perceptron']['fp']} "
        "false alarms. All three models performed poorly on this offset test segment."
    )
    normal_range = offset["split"]["normal_T01_range"]
    if normal_range["test"][1] < normal_range["train"][0]:
        diagnostics += (
            f" A check of the first temperature feature, T01, found that normal test values "
            f"({normal_range['test'][0]:.2f} to {normal_range['test'][1]:.2f}) were below its normal training range "
            f"({normal_range['train'][0]:.2f} to {normal_range['train'][1]:.2f}). "
            "This change in the later segment may contribute to the errors; it does not establish their sole cause."
        )
    limits = (
        "These results use one fault percentage, one selected version of each fault, and one chronological split. "
        "The four datasets share the same underlying sensor readings, so they do not provide independent "
        "evidence from four networks. The two-row gap removes direct window overlap at the split boundary, "
        "but nearby observations may still be related. The source has no device identifiers for an unseen-device test. "
        "The results describe detection on these files, not network uptime or performance in a deployment. "
        "The same fault percentage does not mean equal fault severity across categories. "
        "A single split does not establish a reliable overall model ranking or statistical significance."
    )
    next_steps = (
        "Compare other fault percentages and beta directories using a selection fixed before training. "
        "Repeat the evaluation across separate time blocks, then test on other sensor devices where suitable data "
        "are available. Keep track of missed faults as well as overall accuracy."
    )
    comparison = markdown_table(table)
    summary = (
        "# Sensor fault comparison\n\n"
        "Random Forest, SVM, and MLP were tested separately on gain, offset, stuck-at, and out-of-bounds faults "
        "at the nominal 30% fault rate. Each test used the same chronological row positions and a two-row gap.\n\n"
        f"{comparison}\n\nScores are fractions; fault is the positive class.\n\n{observations}\n\n{diagnostics}\n\n"
        "See [the report](../../reports/sensor_fault_detection_report.md) for the method and limitations.\n"
    )
    (RESULTS / "experiment_summary.md").write_text(summary, encoding="utf-8")
    image_base = "../results/fault_comparison"
    matrices = "\n\n".join(
        f"![{name} confusion matrices]({image_base}/{name.lower().replace(' ', '_')}_confusion_matrix.png)"
        for name in build_models()
    )
    report = (
        "# Sensor Fault Detection Experiment\n\n"
        "## Objective\n\nCompare Random Forest, Support Vector Machine (SVM), and Multilayer Perceptron (MLP) "
        "on gain, offset, stuck-at, and out-of-bounds faults. Each test asks whether a sensor observation is "
        "normal or faulty. The aim is to see how the models' errors change across fault types.\n\n"
        f"## Dataset\n\n{dataset}\n\n## Method\n\n{method}\n\n"
        f"## Results\n\nScores are fractions; fault is the positive class.\n\n{comparison}\n\n"
        f"![Model comparison]({image_base}/model_comparison.png)\n\n"
        f"## Observations\n\n{observations}\n\n{ranges}\n\n{diagnostics}\n\n"
        "The differences apply to the selected files and settings. High scores on an easier fault type "
        "do not show that a model will detect other faults equally well.\n\n"
        f"## Confusion matrices\n\nActual classes are in rows and predicted classes are in columns.\n\n{matrices}\n\n"
        f"## Limitations\n\n{limits}\n\n## Next steps\n\n{next_steps}\n"
    )
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "sensor_fault_detection_report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", action="store_true", help=argparse.SUPPRESS)
    parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    tables, predictions, details = [], [], {}
    reference_y, reference_train, reference_test = None, None, None
    # Inspect every selected file before any fitting.
    datasets = {fault: load_and_inspect(fault) for fault in DATASETS}
    for fault, (x, y, audit) in datasets.items():
        train, test, split = create_split(x, y)
        if reference_y is None:
            reference_y, reference_train, reference_test = y, train, test
        else:
            if not (np.array_equal(reference_y, y) and np.array_equal(reference_train, train)
                    and np.array_equal(reference_test, test)):
                raise ValueError("Expected matching labels and row positions across the four selected datasets.")
        table, models, predicted = evaluate_models(fault, x, y, train, test)
        tables.append(table)
        predictions.append(predicted)
        details[fault] = {"dataset": audit, "split": split, "models": models}
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(RESULTS / "model_comparison.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(RESULTS / "test_predictions.csv", index=False)
    plot_comparison(table)
    plot_matrices(details)
    write_report(table, details)
    record = {"selection": "Four fixed nominal 30% datasets; beta=2 for gain and offset; no beta choice for the other two.",
              "datasets": details, "seed": SEED, "python": platform.python_version(),
              "packages": {name: importlib.metadata.version(name) for name in
                           ("numpy", "pandas", "scikit-learn", "matplotlib", "openpyxl")}}
    (RESULTS / "experiment_details.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Results saved to {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
