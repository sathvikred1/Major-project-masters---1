# Dataset provenance and local organization

**Dataset for Wireless Sensor Network Fault Detection** by the repository maintained by tmoulahi:
<https://github.com/tmoulahi/Dataset-for-WSN-fault-detection>.

Pinned source revision: `a36f90b55b30ba3552747430fb508735025ca80b`.

The source describes temperature/humidity measurements from outdoor multi-hop TelosB sensors, assembled into three-time-step vectors, with injected gain, offset, stuck-at and out-of-bounds faults in separate archives. Its labels are binary: `1` normal, `-1` faulty. This experiment selects **gain faults only**, nominal **30%**, directory **beta=2**. The observed percentage is **30.0341%** (1,408/4,688); 3,280 observations are normal. The beta parameter's physical meaning is not defined in the repository README.

Selected files:

```text
data/raw/Gain/beta=2/gain302.xlsx    # 4,688 observations × 12 sensor features
data/raw/Gain/beta=2/ygain302.xlsx   # 4,688 labels × 1; no header
```

`gain302.xlsx` contains two header rows: time groups followed by `T01 H01 T02 H02 T11 H11 T12 H12 T21 H21 T22 H22`. Its worksheet is therefore 4,690 × 12 including headers. Read with `pandas.read_excel(..., header=1)`. Read its label workbook with `header=None` and align by row, without sorting either file.

This moderate-prevalence variant supplies enough of both classes for the initial comparison without sampling. It was chosen before model training. Multiple archives share normal vectors, so they were not concatenated. Other variants and anomalies are recorded in `results/dataset_inventory.csv` and `results/source_audit.json`.

## Download and extract

From the repository root, after installing requirements:

```bash
python src/download_data.py
```

The downloader uses the pinned revision, verifies SHA-256, and extracts using an installed `bsdtar`, `unrar` or `7z` supporting RAR. For the full inspection, run `python src/download_data.py --all` followed by `python src/inspect_dataset.py`. Archives total approximately 17.2 MB; only the 7.2 MB Gain archive is needed for the experiment.

Manual fallback: download [Gain.rar at the pinned revision](https://raw.githubusercontent.com/tmoulahi/Dataset-for-WSN-fault-detection/a36f90b55b30ba3552747430fb508735025ca80b/Gain.rar), extract into `data/raw/` retaining the archive's `Gain/beta=2/` directories. Then run `python src/experiment.py`. On macOS the equivalent extraction command is `bsdtar -xf data/raw/Gain.rar -C data/raw`. Windows users can use 7-Zip's extraction interface or `7z x data/raw/Gain.rar -odata/raw`.

| File | SHA-256 |
| --- | --- |
| Gain.rar | `59737c45290f22ee527fbdc2ecfec35bf8ff62b5446bb318e31b9a4f5385b191` |
| gain302.xlsx | `90ad926d50595beae7e463fcc849f0a4e968de793aa033d5e25e9560f44581ac` |
| ygain302.xlsx | `d77d3351013cf1f4a7e98793b9f3bf907f59fec609fb715651c1ad1a8c431aef` |

## Attribution and redistribution

The inspected source root has no explicit license file. Public availability alone does not establish unrestricted redistribution rights. Raw archives/workbooks are therefore downloaded locally and excluded by `.gitignore`; the repository saves code, derived evaluation outputs and provenance. Consult the source authors for reuse terms.

The dataset README requests attribution to Salah Zidi, Tarek Moulahi and Bechir Alaya, “Fault detection in Wireless Sensor Networks through SVM classifier,” IEEE Sensors Journal, Vol. 18, No. 1, January 2018. It attributes the original measurements to Shan Suthaharan et al., “Labelled Data Collection for Anomaly Detection in Wireless Sensor Networks,” ISSNIP 2010. Full references are in the project README.
