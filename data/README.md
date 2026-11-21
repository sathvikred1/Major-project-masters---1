# Dataset

Source: [Dataset for Wireless Sensor Network Fault Detection](https://github.com/tmoulahi/Dataset-for-WSN-fault-detection), revision `a36f90b55b30ba3552747430fb508735025ca80b`.

Run `python src/download_data.py` to retrieve the eight selected workbooks into `data/raw/`. The downloader uses temporary archives, extracts only the required pairs, and reuses verified workbooks on later runs. These nominal 30% fault files were fixed before training:

| Fault | Feature file | Label file |
| --- | --- | --- |
| Gain | `Gain/beta=2/gain302.xlsx` | `Gain/beta=2/ygain302.xlsx` |
| Offset | `Offset/beta=2/offset302.xlsx` | `Offset/beta=2/y302.xlsx` |
| Stuck-at | `Stuck-at/stuck302.xlsx` | `Stuck-at/ystuck302.xlsx` |
| Out-of-bounds | `Out of bound/out302.xlsx` | `Out of bound/yout302.xlsx` |

Gain and offset use beta=2. The other two archives have no beta subdirectories. The source README does not define beta's physical meaning, and equal fault percentages do not imply equal fault severity.

The feature workbooks have two header rows and are read with `header=1`. Label workbooks have no header and are read with `header=None`. Rows must stay in their supplied order. Each pair contains 4,688 observations with twelve inputs, 3,280 normal labels (1), and 1,408 faulty labels (-1). No missing or infinite input values were found.

The four datasets are altered versions of the same sensor series and share the same label sequence. They are evaluated separately; they are not concatenated or described as independent networks.

Archive and workbook checksums, along with the selected paths, are fixed in `src/download_data.py` and shared by the experiment. The recorded hashes and inspected contents are saved in `results/fault_comparison/experiment_details.json`.

If automatic extraction is unavailable, extract the archives into `data/raw/` with a RAR-capable tool, retaining their directory names. Ignore Excel lock files beginning with `~$`.

The source has no explicit license file, so raw data are downloaded locally and excluded from Git. The authors request citation of Salah Zidi, Tarek Moulahi, and Bechir Alaya, “Fault detection in Wireless Sensor Networks through SVM classifier,” IEEE Sensors Journal, 18(1), 2018. Their documentation attributes the original measurements to Suthaharan et al., “Labelled Data Collection for Anomaly Detection in Wireless Sensor Networks,” ISSNIP 2010.
