"""Compare three classifiers on normal and gain-fault sensor readings."""
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

from download_data import REVISION, sha256

SEED = 42
RESULTS = ROOT / "results" / "initial_experiment"
DATA = ROOT / "data" / "raw" / "Gain" / "beta=2"
FEATURE_FILE = DATA / "gain302.xlsx"
LABEL_FILE = DATA / "ygain302.xlsx"
FEATURE_NAMES = ["T01", "H01", "T02", "H02", "T11", "H11", "T12", "H12", "T21", "H21", "T22", "H22"]
LABELS = [1, -1]  # Matrix order: normal, gain fault. Gain fault is positive.
CLASS_NAMES = ["Normal (1)", "Gain fault (-1)"]
METRICS = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "False Positive Rate", "False Negative Rate"]
EXPECTED_HASHES = {
    "gain302.xlsx": "90ad926d50595beae7e463fcc849f0a4e968de793aa033d5e25e9560f44581ac",
    "ygain302.xlsx": "d77d3351013cf1f4a7e98793b9f3bf907f59fec609fb715651c1ad1a8c431aef",
}
SOURCE = f"https://github.com/tmoulahi/Dataset-for-WSN-fault-detection/blob/{REVISION}/README.md"


def load_and_inspect():
    for path in (FEATURE_FILE, LABEL_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run python src/download_data.py first.")
        if sha256(path) != EXPECTED_HASHES[path.name]:
            raise ValueError(f"Checksum mismatch: {path}")
    # The feature workbook has two header rows; the labels have none.
    x = pd.read_excel(FEATURE_FILE, header=1)
    labels = pd.read_excel(LABEL_FILE, header=None)
    if labels.shape != (len(x), 1):
        raise ValueError("Expected one label for every feature row.")
    y = labels.iloc[:, 0].rename("y")
    if y.isna().any() or set(y.unique()) != {-1, 1}:
        raise ValueError("Expected labels 1 (normal) and -1 (gain fault).")
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
        "feature_file": str(FEATURE_FILE.relative_to(ROOT)),
        "label_file": str(LABEL_FILE.relative_to(ROOT)),
        "sha256": EXPECTED_HASHES,
        "observations": len(x), "features": list(x.columns),
        "labels": {"1": "normal", "-1": "gain fault (positive class)"},
        "class_counts": {str(label): int((y == label).sum()) for label in LABELS},
        "missing_values": missing, "infinite_values": infinite,
        "adjacent_rows_sharing_eight_measurements": shared,
    }
    print(f"Loaded {len(x):,} rows and {x.shape[1]} features; "
          f"normal={audit['class_counts']['1']:,}, gain fault={audit['class_counts']['-1']:,}.")
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


def evaluate_models(x, y, train, test):
    rows, details = [], {}
    predictions = pd.DataFrame({"observation_index_zero_based": test, "actual_y": y.iloc[test].to_numpy()})
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
        rows.append({"Model": name, "Accuracy": accuracy_score(y.iloc[test], predicted),
                     "Precision": precision, "Recall": recall, "F1 Score": f1,
                     "False Positive Rate": fp / (fp + tn), "False Negative Rate": fn / (fn + tp)})
        slug = name.lower().replace(" ", "_")
        predictions[slug] = predicted
        warning_text = [f"{w.category.__name__}: {w.message}" for w in captured]
        for message in warning_text:
            print(f"{name}: {message}")
        details[name] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
                         "confusion_matrix": matrix.tolist(), "warnings": warning_text}
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(RESULTS / f"{slug}_confusion_matrix.png", dpi=180)
        plt.close(fig)
        print(f"{name}: accuracy={rows[-1]['Accuracy']:.6f}, false alarms={fp}, missed faults={fn}")
    predictions.to_csv(RESULTS / "test_predictions.csv", index=False)
    return pd.DataFrame(rows, columns=METRICS), details


def plot_comparison(table):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    positions = np.arange(len(table))
    width = 0.19
    for index, metric in enumerate(["Accuracy", "Precision", "Recall", "F1 Score"]):
        bars = ax.bar(positions + (index - 1.5) * width, table[metric], width, label=metric)
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in table[metric]], rotation=90, padding=3, fontsize=8)
    ax.set_xticks(positions, table["Model"])
    ax.set(title="Normal and gain-fault classification", ylabel="Score (fraction)", ylim=(0, 1.20))
    ax.legend(loc="lower center", ncol=4, framealpha=1)
    fig.tight_layout()
    fig.savefig(RESULTS / "model_comparison.png", dpi=180)
    plt.close(fig)


def markdown_table(table):
    lines = ["| " + " | ".join(table.columns) + " |", "| " + " | ".join(["---"] * len(table.columns)) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]) if c == "Model" else f"{row[c]:.6f}" for c in table.columns) + " |")
    return "\n".join(lines)


def write_report(table, audit, split, details):
    counts = audit["class_counts"]
    dataset = (
        f"The data come from the [wireless sensor dataset]({SOURCE}) published by Zidi, Moulahi, and Alaya. "
        "The selected files are `Gain/beta=2/gain302.xlsx` and `Gain/beta=2/ygain302.xlsx`. "
        f"They contain {audit['observations']:,} observations: {counts['1']:,} normal and {counts['-1']:,} gain faults. "
        "This is the nominal 30% fault dataset.\n\n"
        "Each observation contains two temperature and two humidity readings at three consecutive time steps, "
        f"giving twelve features: {', '.join(audit['features'])}. "
        "The labels are 1 for normal and -1 for gain fault. The rows match between the two files, "
        "and there are no missing or infinite values."
    )
    method = (
        "The rows were kept in their original order. Neighbouring rows share measurements, so randomly "
        "splitting them would allow some readings to appear in both training and testing. "
        f"The first {split['train_count']:,} rows were used for training, rows "
        f"{split['excluded_rows'][0]} and {split['excluded_rows'][1]} were skipped, and the last "
        f"{split['test_count']:,} rows were used for testing. Row numbers start at zero. "
        "All three models used the same split.\n\n"
        "Random Forest used 100 trees. SVM used an RBF kernel with C=1 and gamma=scale. "
        "MLP used two hidden layers with 32 and 16 neurons and a maximum of 1,000 iterations. "
        "The random state was 42 for Random Forest and MLP. SVM and MLP inputs were standardized "
        "using only the training data. Random Forest used unscaled inputs. "
        "Each pipeline includes a median imputer, although these files have no missing values. "
        "No parameters were tuned.\n\n"
        "Gain fault was treated as the positive class. A false positive is a normal reading flagged as faulty; "
        "a false negative is a fault that the model missed."
    )
    observations = "\n".join(
        f"- {name}: {d['fp']} false alarm{'s' if d['fp'] != 1 else ''}, {d['fn']} missed faults."
        for name, d in details.items()
    )
    best = ", ".join(table.loc[table.Accuracy.eq(table.Accuracy.max()), "Model"])
    discussion = (
        f"{best} had the highest accuracy on this test set. "
        "The high scores suggest that the selected gain faults are easy for these models to separate "
        "from normal readings. They do not show how the models would perform on other faults or on a different sensor network."
    )
    limits = (
        "Only one gain-fault dataset, one fault percentage, and one training/test split were tested. "
        "The experiment does not cover unseen devices or other fault types. The gap removes shared measurements "
        "at the split boundary, but nearby observations may still be related. "
        "Classification accuracy alone does not measure network reliability."
    )
    next_steps = (
        "Test offset, stuck-at, and out-of-bounds faults, then compare different fault percentages. "
        "Further work should examine false alarms and missed faults across separate time blocks and sensor devices "
        "where suitable data are available."
    )
    comparison = markdown_table(table)
    summary = (
        "# Gain-fault experiment\n\n"
        f"Compared Random Forest, SVM, and MLP on {audit['observations']:,} sensor observations "
        f"({counts['1']:,} normal, {counts['-1']:,} gain faults). "
        f"Used {split['train_count']:,} training rows and {split['test_count']:,} test rows, with a two-row gap.\n\n"
        f"{comparison}\n\nScores are fractions; gain fault is the positive class.\n\n"
        f"{observations}\n\n{discussion}\n\n"
        "See [the report](../../reports/initial_experiment_report.md) for the method and limitations.\n"
    )
    (RESULTS / "experiment_summary.md").write_text(summary, encoding="utf-8")
    image_base = "../results/initial_experiment"
    matrices = "\n\n".join(
        f"![{name} confusion matrix]({image_base}/{name.lower().replace(' ', '_')}_confusion_matrix.png)"
        for name in details
    )
    report = (
        "# Sensor Fault Detection Experiment\n\n"
        "## Objective\n\nCompare Random Forest, SVM, and MLP to see how well they distinguish normal "
        "sensor readings from gain faults.\n\n"
        f"## Dataset\n\n{dataset}\n\n## Method\n\n{method}\n\n"
        f"## Results\n\nScores are fractions; gain fault is the positive class.\n\n{comparison}\n\n"
        f"{observations}\n\n![Model comparison]({image_base}/model_comparison.png)\n\n"
        f"The confusion matrices show actual classes in rows and predicted classes in columns.\n\n{matrices}\n\n"
        f"## Observations\n\n{discussion}\n\n## Limitations\n\n{limits}\n\n## Next steps\n\n{next_steps}\n"
    )
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "initial_experiment_report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Accept the previous command while using one output location for every run.
    parser.add_argument("--initial", action="store_true", help=argparse.SUPPRESS)
    parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    x, y, audit = load_and_inspect()
    train, test, split = create_split(x, y)
    table, models = evaluate_models(x, y, train, test)
    table.to_csv(RESULTS / "model_comparison.csv", index=False)
    plot_comparison(table)
    write_report(table, audit, split, models)
    details = {"dataset": audit, "split": split, "models": models, "seed": SEED,
               "python": platform.python_version(),
               "packages": {name: importlib.metadata.version(name) for name in
                            ("numpy", "pandas", "scikit-learn", "matplotlib", "openpyxl")}}
    (RESULTS / "experiment_details.json").write_text(json.dumps(details, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Results saved to {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
