# Machine Learning-Based Fault Detection and Reliability Analysis in IoT Sensor Networks

This experiment compares Random Forest, Support Vector Machine, and Multilayer Perceptron on four sensor faults: gain, offset, stuck-at, and out-of-bounds. Each fault type is tested separately as normal versus faulty.

## Data and method

The [Dataset for Wireless Sensor Network Fault Detection](https://github.com/tmoulahi/Dataset-for-WSN-fault-detection) by Zidi, Moulahi, and Alaya provides temperature and humidity readings with injected faults. The experiment uses four fixed nominal 30% fault files. Gain and offset use beta=2; the other archives have no beta directories. See [data/README.md](data/README.md) for the exact files.

Each selected pair has 4,688 observations and twelve features: 3,280 normal and 1,408 faulty. These are versions of the same sensor series, not four independent networks. The files are evaluated separately rather than pooled.

All twelve model/fault comparisons use the first 3,748 rows for training and the last 938 for testing. Two rows are skipped at the boundary because neighbouring observations share measurements. SVM and MLP inputs are standardized using only training data. The model settings are fixed, with no tuning.

## Run

Use Python 3.14 and a RAR extractor: `bsdtar` on macOS, or a RAR-capable `unrar` or `7z` on other systems.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/download_data.py
python src/experiment.py
python src/validate_results.py
```

On Windows, activate with `.venv\Scripts\activate` instead. The downloader verifies hashes, extracts only the eight selected workbooks, and reuses them on later runs. Downloaded archives are temporary. Raw data stay out of Git.

To repeat all twelve fits and check that predictions and scores match:

```bash
python src/validate_results.py --rerun
```

## Results

Read the [report](reports/sensor_fault_detection_report.md), [PDF](reports/sensor_fault_detection_report.pdf), or [short summary](results/fault_comparison/experiment_summary.md). The PDF is a formatted copy of the recorded results; running the experiment updates the Markdown report and result files.

The three models performed well on gain, stuck-at, and out-of-bounds faults. Offset was much harder on the later test segment, with accuracy from 34.86% to 52.13% and many false alarms. The report discusses the errors and a change in the input range over time.

One reason for the high scores is visible in the selected test data: normal `H22` values range from 71.01 to 73.56, while gain faults range from 138.56 to 142.20, stuck-at faults are fixed at 80, and out-of-bounds faults are fixed at 110. These clear patterns make those three cases easier to detect. Offset values (71.28 to 73.10) overlap the normal range. This describes these particular files, not the difficulty of every fault of these types.

`results/fault_comparison/` contains eight files:

- `model_comparison.csv` and `model_comparison.png`: all twelve comparisons and a chart.
- Three confusion matrix sheets, one per model, each showing all four fault types.
- `experiment_summary.md`: scores and a short interpretation.
- `test_predictions.csv`: labels and predictions, identified by fault type and source row.
- `experiment_details.json`: source hashes, dataset checks, splits, error counts, and package versions.

This comparison replaces the earlier gain-only experiment. Earlier results remain in Git history. One fault percentage and one chronological split cannot establish a general ranking or statistically significant model differences.
