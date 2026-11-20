# Dataset

Source: [Dataset for Wireless Sensor Network Fault Detection](https://github.com/tmoulahi/Dataset-for-WSN-fault-detection), revision `a36f90b55b30ba3552747430fb508735025ca80b`.

Run `python src/download_data.py` to download and extract `Gain.rar` into `data/raw/`. The experiment reads only these two files:

- `Gain/beta=2/gain302.xlsx`: temperature and humidity inputs. Read with `header=1` because the worksheet has two header rows.
- `Gain/beta=2/ygain302.xlsx`: one label per observation, without a header. Read with `header=None`.

The pair contains 4,688 observations and twelve features. The source defines `1` as normal and `-1` as faulty. For this pair, 3,280 observations are normal and 1,408 have gain faults. Rows must stay in their supplied order.

The download and experiment scripts verify these SHA-256 hashes:

| File | SHA-256 |
| --- | --- |
| Gain.rar | `59737c45290f22ee527fbdc2ecfec35bf8ff62b5446bb318e31b9a4f5385b191` |
| gain302.xlsx | `90ad926d50595beae7e463fcc849f0a4e968de793aa033d5e25e9560f44581ac` |
| ygain302.xlsx | `d77d3351013cf1f4a7e98793b9f3bf907f59fec609fb715651c1ad1a8c431aef` |

If automatic extraction is unavailable, extract the archive into `data/raw/` with a RAR-capable tool, retaining its directory names.

The source has no explicit license file, so raw data are downloaded locally and excluded from Git. The authors request citation of Salah Zidi, Tarek Moulahi, and Bechir Alaya, “Fault detection in Wireless Sensor Networks through SVM classifier,” IEEE Sensors Journal, 18(1), 2018. Their documentation attributes the original sensor measurements to Suthaharan et al., “Labelled Data Collection for Anomaly Detection in Wireless Sensor Networks,” ISSNIP 2010.
