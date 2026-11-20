"""Check saved predictions, metrics, figures, and report against the source data."""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from experiment import ROOT, RESULTS, METRICS, build_models, create_split, load_and_inspect, markdown_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true", help="Repeat the experiment and compare predictions and scores.")
    parser.add_argument("--initial", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rerun:
        first_predictions = (RESULTS / "test_predictions.csv").read_bytes()
        first_table = pd.read_csv(RESULTS / "model_comparison.csv")
        subprocess.run([sys.executable, str(ROOT / "src" / "experiment.py")], check=True)
        assert first_predictions == (RESULTS / "test_predictions.csv").read_bytes(), "Predictions changed on rerun."
        pd.testing.assert_frame_equal(first_table, pd.read_csv(RESULTS / "model_comparison.csv"), check_exact=True)
        print("Repeated run: predictions and scores match.")

    x, y, audit = load_and_inspect()
    train, test, split = create_split(x, y)
    details = json.loads((RESULTS / "experiment_details.json").read_text())
    assert details["dataset"] == audit and details["split"] == split
    assert train[-1] + 3 == test[0]
    assert len(train) + len(test) + 2 == len(x)
    assert audit["adjacent_rows_sharing_eight_measurements"] == len(x) - 1
    table = pd.read_csv(RESULTS / "model_comparison.csv")
    predictions = pd.read_csv(RESULTS / "test_predictions.csv")
    assert table.columns.tolist() == METRICS
    assert table.Model.tolist() == list(build_models()) == list(details["models"])
    assert np.array_equal(predictions["observation_index_zero_based"], test)
    assert np.array_equal(predictions["actual_y"], y.iloc[test])
    assert np.isfinite(table.iloc[:, 1:].to_numpy()).all()
    actual = predictions.actual_y.to_numpy()
    figures = ["model_comparison.png"]
    for _, row in table.iterrows():
        name = row["Model"]
        slug = name.lower().replace(" ", "_")
        predicted = predictions[slug].to_numpy()
        assert set(predicted).issubset({1, -1})
        tn = int(np.sum((actual == 1) & (predicted == 1)))
        fp = int(np.sum((actual == 1) & (predicted == -1)))
        fn = int(np.sum((actual == -1) & (predicted == 1)))
        tp = int(np.sum((actual == -1) & (predicted == -1)))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        calculated = {"Accuracy": (tp + tn) / len(actual), "Precision": precision,
                      "Recall": recall, "F1 Score": f1,
                      "False Positive Rate": fp / (fp + tn), "False Negative Rate": fn / (fn + tp)}
        for metric, value in calculated.items():
            assert np.isclose(row[metric], value, rtol=0, atol=1e-12), (name, metric)
        saved = details["models"][name]
        assert all(saved[key] == value for key, value in {"tn": tn, "fp": fp, "fn": fn, "tp": tp}.items())
        assert saved["confusion_matrix"] == [[tn, fp], [fn, tp]]
        assert not any("ConvergenceWarning" in message for message in saved["warnings"]), name
        figures.append(f"{slug}_confusion_matrix.png")
    report = (ROOT / "reports" / "initial_experiment_report.md").read_text()
    summary = (RESULTS / "experiment_summary.md").read_text()
    assert markdown_table(table) in report and markdown_table(table) in summary
    for filename in figures:
        path = RESULTS / filename
        assert path.stat().st_size > 1000 and path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert f"../results/initial_experiment/{filename}" in report
    print(f"Passed: source data, two-row gap, {len(predictions)} predictions, 18 metrics, four figures, and report tables.")


if __name__ == "__main__":
    main()
