"""Independently recompute saved binary metrics and check required artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from experiment import ROOT, RESULTS, FIGURES, METRIC_PLOTS, markdown_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true", help="Run the whole experiment again and verify identical predictions and non-timing metrics.")
    args = parser.parse_args()
    if args.rerun:
        first = pd.read_csv(RESULTS / "model_comparison.csv")
        first_hash = hashlib.sha256((RESULTS / "test_predictions.csv").read_bytes()).hexdigest()
        subprocess.run([sys.executable, str(ROOT / "src" / "experiment.py")], check=True)
        second = pd.read_csv(RESULTS / "model_comparison.csv")
        second_hash = hashlib.sha256((RESULTS / "test_predictions.csv").read_bytes()).hexdigest()
        metric_columns = [c for c in first.columns if c != "Model" and "seconds" not in c]
        assert first["Model"].tolist() == second["Model"].tolist()
        assert first_hash == second_hash, "Predictions changed between identical seeded runs"
        assert np.allclose(first[metric_columns], second[metric_columns], rtol=0, atol=1e-12)
        check = {"status": "passed", "complete_runs_compared": 2,
                 "first_predictions_sha256": first_hash, "second_predictions_sha256": second_hash,
                 "non_timing_metrics_identical": True, "timings_excluded_from_equality_check": True}
        (RESULTS / "reproducibility_check.json").write_text(json.dumps(check, indent=2) + "\n")
    table = pd.read_csv(RESULTS / "model_comparison.csv")
    predictions = pd.read_csv(RESULTS / "test_predictions.csv")
    assignments = pd.read_csv(RESULTS / "split_assignments.csv")
    details = json.loads((RESULTS / "model_details.json").read_text())
    metadata = json.loads((RESULTS / "split_metadata.json").read_text())
    inspection = json.loads((RESULTS / "dataset_inspection.json").read_text())
    assert len(table) == 3 and table["Model"].nunique() == 3
    assert table.select_dtypes("number").notna().all().all()
    assert np.isfinite(table.select_dtypes("number").to_numpy()).all()
    assert len(assignments) == inspection["observations"]
    train = assignments.loc[assignments.partition.eq("train"), "observation_index_zero_based"]
    test = assignments.loc[assignments.partition.eq("test"), "observation_index_zero_based"]
    assert train.max() + 2 < test.min(), "Shared time steps across split boundary"
    assert len(train) == metadata["train_observations"]
    assert len(test) == metadata["test_observations"] == len(predictions)
    assert np.array_equal(test.to_numpy(), predictions["observation_index_zero_based"].to_numpy())
    actual = predictions["actual_y"].to_numpy()
    normal = actual == 1
    faulty = actual == -1
    assert normal.sum() + faulty.sum() == len(actual)
    assert normal.sum() == metadata["test_class_counts"]["1"]
    assert faulty.sum() == metadata["test_class_counts"]["-1"]
    verified_counts = {}
    required_figures = ["class_distribution.png"] + [f"{slug}_comparison.png" for slug in METRIC_PLOTS.values()]
    for _, row in table.iterrows():
        name = row["Model"]
        slug = name.lower().replace(" ", "_")
        predicted = predictions[slug].to_numpy()
        assert set(predicted).issubset({-1, 1})
        tn = int(np.sum(normal & (predicted == 1)))
        fp = int(np.sum(normal & (predicted == -1)))
        fn = int(np.sum(faulty & (predicted == 1)))
        tp = int(np.sum(faulty & (predicted == -1)))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        recomputed = {"Accuracy": (tp + tn) / len(actual), "Precision": precision,
                      "Recall": recall, "F1 Score": f1, "False Positive Rate": fp / (fp + tn),
                      "False Negative Rate": fn / (fn + tp)}
        for metric, value in recomputed.items():
            assert np.isclose(row[metric], value, rtol=0, atol=1e-12), (name, metric)
        assert row["Training Time (seconds)"] > 0 and row["Prediction Time (seconds)"] > 0
        counts = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
        assert all(details[name][key] == value for key, value in counts.items())
        report = json.loads((RESULTS / "classification_reports" / f"{slug}.json").read_text())
        assert np.isclose(report["Gain fault (-1)"]["recall"], recall)
        assert (RESULTS / "classification_reports" / f"{slug}.txt").stat().st_size > 100
        assert not any("ConvergenceWarning" in message for message in details[name]["warnings"]), "Investigate convergence before accepting results"
        required_figures.append(f"{slug}_confusion_matrix.png")
        verified_counts[name] = counts
    for name in required_figures:
        path = FIGURES / name
        assert path.stat().st_size > 1000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", path
    readme = (ROOT / "README.md").read_text()
    summary = (RESULTS / "summary.txt").read_text()
    assert markdown_table(table) in readme and markdown_table(table) in summary
    assert "INITIAL BASELINE EXPERIMENT" in summary
    result = {"status": "passed", "independent_metrics_verified": 18,
              "required_figures_verified": len(required_figures),
              "test_predictions_verified": len(predictions), "confusion_counts": verified_counts,
              "readme_and_summary_match_csv": True, "no_convergence_warnings": True,
              "split_boundary_purge_verified": True}
    (RESULTS / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
