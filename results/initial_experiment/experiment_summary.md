# Gain-fault experiment

Compared Random Forest, SVM, and MLP on 4,688 sensor observations (3,280 normal, 1,408 gain faults). Used 3,748 training rows and 938 test rows, with a two-row gap.

| Model | Accuracy | Precision | Recall | F1 Score | False Positive Rate | False Negative Rate |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.998934 | 0.996951 | 1.000000 | 0.998473 | 0.001637 | 0.000000 |
| Support Vector Machine | 0.997868 | 0.993921 | 1.000000 | 0.996951 | 0.003273 | 0.000000 |
| Multilayer Perceptron | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |

Scores are fractions; gain fault is the positive class.

- Random Forest: 1 false alarm, 0 missed faults.
- Support Vector Machine: 2 false alarms, 0 missed faults.
- Multilayer Perceptron: 0 false alarms, 0 missed faults.

Multilayer Perceptron had the highest accuracy on this test set. The high scores suggest that the selected gain faults are easy for these models to separate from normal readings. They do not show how the models would perform on other faults or on a different sensor network.

See [the report](../../reports/initial_experiment_report.md) for the method and limitations.
