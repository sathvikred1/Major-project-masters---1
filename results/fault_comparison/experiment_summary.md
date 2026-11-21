# Sensor fault comparison

Random Forest, SVM, and MLP were tested separately on gain, offset, stuck-at, and out-of-bounds faults at the nominal 30% fault rate. Each test used the same chronological row positions and a two-row gap.

| Fault | Model | Accuracy | Precision | Recall | F1 Score | False Positive Rate | False Negative Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gain | Random Forest | 0.998934 | 0.996951 | 1.000000 | 0.998473 | 0.001637 | 0.000000 |
| Gain | Support Vector Machine | 0.997868 | 0.993921 | 1.000000 | 0.996951 | 0.003273 | 0.000000 |
| Gain | Multilayer Perceptron | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| Offset | Random Forest | 0.348614 | 0.348614 | 1.000000 | 0.516996 | 1.000000 | 0.000000 |
| Offset | Support Vector Machine | 0.440299 | 0.233871 | 0.266055 | 0.248927 | 0.466448 | 0.733945 |
| Offset | Multilayer Perceptron | 0.521322 | 0.421392 | 1.000000 | 0.592928 | 0.734861 | 0.000000 |
| Stuck-at | Random Forest | 0.998934 | 0.996951 | 1.000000 | 0.998473 | 0.001637 | 0.000000 |
| Stuck-at | Support Vector Machine | 0.998934 | 0.996951 | 1.000000 | 0.998473 | 0.001637 | 0.000000 |
| Stuck-at | Multilayer Perceptron | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| Out-of-bounds | Random Forest | 0.998934 | 0.996951 | 1.000000 | 0.998473 | 0.001637 | 0.000000 |
| Out-of-bounds | Support Vector Machine | 0.997868 | 0.993921 | 1.000000 | 0.996951 | 0.003273 | 0.000000 |
| Out-of-bounds | Multilayer Perceptron | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |

Scores are fractions; fault is the positive class.

- **Gain:** accuracy ranged from 99.79% to 100.00%.
- **Offset:** accuracy ranged from 34.86% to 52.13%.
- **Stuck-at:** accuracy ranged from 99.89% to 100.00%.
- **Out-of-bounds:** accuracy ranged from 99.79% to 100.00%.

The selected gain, stuck-at, and out-of-bounds faults leave clear patterns in the sensor inputs. For example, H22 is a humidity feature. Its normal and faulty values in the test set were:

| Fault | Normal H22 | Faulty H22 |
| --- | --- | --- |
| Gain | 71.01 to 73.56 | 138.56 to 142.20 |
| Offset | 71.01 to 73.56 | 71.28 to 73.10 |
| Stuck-at | 71.01 to 73.56 | 80.00 (constant) |
| Out-of-bounds | 71.01 to 73.56 | 110.00 (constant) |

Gain, stuck-at, and out-of-bounds faults do not overlap the normal H22 range in these test files. These distinct patterns help explain why different models can all achieve very high accuracy. Offset values overlap the normal range, making that case less straightforward. This range check describes the held-out readings; it was not used to choose model settings or input features. Faults with values closer to normal may be harder to detect.

On offset faults, Random Forest flagged all 938 test observations as faulty, including all 611 normal observations. SVM missed 240 faults. MLP had the highest offset accuracy, but still raised 449 false alarms. All three models performed poorly on this offset test segment. A check of the first temperature feature, T01, found that normal test values (26.29 to 26.79) were below its normal training range (26.96 to 48.24). This change in the later segment may contribute to the errors; it does not establish their sole cause.

See [the report](../../reports/sensor_fault_detection_report.md) for the method and limitations.
