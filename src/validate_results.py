"""Check the four-fault comparison against source data and saved predictions."""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from experiment import ROOT, RESULTS, DATASETS, METRICS, build_models, create_split, load_and_inspect, markdown_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true", help="Repeat all twelve fits and compare predictions and scores.")
    parser.add_argument("--initial", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rerun:
        first_predictions = (RESULTS / "test_predictions.csv").read_bytes()
        first_table = pd.read_csv(RESULTS / "model_comparison.csv")
        subprocess.run([sys.executable, str(ROOT / "src" / "experiment.py")], check=True)
        assert first_predictions == (RESULTS / "test_predictions.csv").read_bytes(), "Predictions changed on rerun."
        pd.testing.assert_frame_equal(first_table, pd.read_csv(RESULTS / "model_comparison.csv"), check_exact=True)
        print("Repeated run: all four datasets have identical predictions and scores.")

    details = json.loads((RESULTS / "experiment_details.json").read_text())["datasets"]
    table = pd.read_csv(RESULTS / "model_comparison.csv")
    predictions = pd.read_csv(RESULTS / "test_predictions.csv")
    models = list(build_models())
    assert table.columns.tolist() == METRICS
    assert list(details) == list(DATASETS) == table.Fault.unique().tolist() == predictions.Fault.unique().tolist()
    assert len(table) == len(DATASETS) * len(models)
    assert not predictions.duplicated(["Fault", "observation_index_zero_based"]).any()
    assert np.isfinite(table.iloc[:, 2:].to_numpy()).all()
    reference_actual, reference_train, reference_test = None, None, None
    verified = 0
    for fault in DATASETS:
        x, y, audit = load_and_inspect(fault)
        train, test, split = create_split(x, y)
        saved = details[fault]
        assert saved["dataset"] == audit and saved["split"] == split
        assert train[-1] + 3 == test[0] and len(train) + len(test) + 2 == len(x)
        assert audit["adjacent_rows_sharing_eight_measurements"] == len(x) - 1
        subset = table.loc[table.Fault.eq(fault)]
        predicted_rows = predictions.loc[predictions.Fault.eq(fault)]
        assert subset.Model.tolist() == list(saved["models"]) == models
        assert np.array_equal(predicted_rows["observation_index_zero_based"], test)
        assert np.array_equal(predicted_rows["actual_y"], y.iloc[test])
        actual = predicted_rows.actual_y.to_numpy()
        if reference_actual is None:
            reference_actual, reference_train, reference_test = actual, train, test
        else:
            assert np.array_equal(reference_actual, actual)
            assert np.array_equal(reference_train, train) and np.array_equal(reference_test, test)
        for _, row in subset.iterrows():
            name = row["Model"]
            predicted = predicted_rows[name.lower().replace(" ", "_")].to_numpy()
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
                assert np.isclose(row[metric], value, rtol=0, atol=1e-12), (fault, name, metric)
                verified += 1
            counts = saved["models"][name]
            assert all(counts[key] == value for key, value in {"tn": tn, "fp": fp, "fn": fn, "tp": tp}.items())
            assert counts["confusion_matrix"] == [[tn, fp], [fn, tp]]
            assert not any("ConvergenceWarning" in message for message in counts["warnings"]), (fault, name)

    report = (ROOT / "reports" / "sensor_fault_detection_report.md").read_text()
    summary = (RESULTS / "experiment_summary.md").read_text()
    assert markdown_table(table) in report and markdown_table(table) in summary
    figures = ["model_comparison.png"] + [f"{name.lower().replace(' ', '_')}_confusion_matrix.png" for name in models]
    for filename in figures:
        path = RESULTS / filename
        assert path.stat().st_size > 1000 and path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert f"../results/fault_comparison/{filename}" in report
    print(f"Passed: four source pairs, matching splits, {len(predictions)} predictions, {verified} metrics, "
          "12 confusion matrices, four figure files, and report tables.")


if __name__ == "__main__":
    main()
