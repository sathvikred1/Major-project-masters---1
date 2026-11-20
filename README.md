# Machine Learning-Based Fault Detection and Reliability Analysis in IoT Sensor Networks

This project compares Random Forest, Support Vector Machine, and Multilayer Perceptron for detecting faulty sensor readings. The current experiment uses temperature and humidity data with injected gain faults.

## Data and method

The data come from the [Dataset for Wireless Sensor Network Fault Detection](https://github.com/tmoulahi/Dataset-for-WSN-fault-detection) by Zidi, Moulahi, and Alaya. The selected pair is `Gain/beta=2/gain302.xlsx` and `ygain302.xlsx`: 4,688 observations with twelve features, including 3,280 normal readings and 1,408 gain faults. Labels are `1 = normal` and `-1 = gain fault`.

The rows stay in their original order. Training uses the first 3,748 rows and testing uses the last 938, with two rows skipped between them because neighbouring observations share measurements. SVM and MLP inputs are standardized using only the training data. All three models use the same split, with no parameter tuning.

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

On Windows, activate with `.venv\Scripts\activate` instead. The downloader checks file hashes and keeps raw data out of Git. See [data/README.md](data/README.md) for the source and file format.

To repeat the experiment and check that predictions and scores match:

```bash
python src/validate_results.py --rerun
```

## Results

Read the [experiment report](reports/initial_experiment_report.md) for the method, results, and limitations. The [short summary](results/initial_experiment/experiment_summary.md) gives the scores and error counts.

Everything from a run is saved in `results/initial_experiment/`:

- `model_comparison.csv` and `model_comparison.png`: the six metrics and a comparison chart.
- Three confusion matrix images, one for each model.
- `experiment_summary.md`: a short explanation of the results.
- `test_predictions.csv`: the actual labels and each model's predictions.
- `experiment_details.json`: dataset checks, split boundaries, error counts, and package versions.

Running the experiment replaces these outputs and updates the report. There is one set of results; earlier versions remain in Git history.

The current results cover one gain-fault dataset and one split. Further work will compare other faults and fault percentages, with closer attention to false alarms and missed faults.
