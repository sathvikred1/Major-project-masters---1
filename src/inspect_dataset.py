"""Audit all available workbook pairs before choosing a baseline; no model fitting."""
import json
from pathlib import Path
import re

import pandas as pd

from download_data import RAW, ROOT, REVISION, sha256


def main():
    records = []
    family_normals = {}
    for path in sorted(RAW.rglob("*.xlsx")):
        if path.name.startswith(("y", "~$")):
            continue
        suffix = re.search(r"(\d+)\.xlsx$", path.name).group(1)
        label_name = ("y" if path.parent.parent.name == "Offset" else
                      "y" + re.sub(r"\d+\.xlsx$", "", path.name)) + suffix + ".xlsx"
        label_path = path.with_name(label_name)
        if not label_path.exists():
            records.append({"feature_file": str(path.relative_to(RAW)), "status": "no matching label filename"})
            continue
        features = pd.read_excel(path, header=1)
        labels = pd.read_excel(label_path, header=None)
        counts = labels.iloc[:, 0].value_counts().to_dict()
        records.append({
            "feature_file": str(path.relative_to(RAW)),
            "label_file": str(label_path.relative_to(RAW)),
            "rows": len(features), "features": features.shape[1],
            "label_rows": len(labels), "normal_count": counts.get(1, 0),
            "fault_count": counts.get(-1, 0),
            "fault_percent": 100 * counts.get(-1, 0) / len(labels),
            "missing_feature_values": int(features.isna().sum().sum()),
            "status": "paired", "feature_sha256": sha256(path),
        })
        if suffix == "302" and ("beta=2" in path.parts or "beta=" not in str(path)):
            family = path.relative_to(RAW).parts[0]
            family_normals[family] = features.loc[labels.iloc[:, 0].eq(1)].reset_index(drop=True)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(output / "dataset_inventory.csv", index=False)
    overlaps = {}
    gain = family_normals.get("Gain")
    if gain is not None:
        gain_set = set(map(tuple, gain.to_numpy()))
        for family, normal in family_normals.items():
            overlaps[family] = len(gain_set.intersection(map(tuple, normal.to_numpy())))
    audit = {
        "source_revision": REVISION,
        "feature_workbooks": len(records),
        "paired_workbooks": sum(r["status"] == "paired" for r in records),
        "unpaired": [r for r in records if r["status"] != "paired"],
        "distinct_normal_vectors_shared_with_selected_gain_30_percent": overlaps,
        "notes": [
            "Excel lock files (~$...) excluded.",
            "Gain/beta=6/gain108.xlsx has no matching ygain108.xlsx in that directory; ygain106.xlsx exists. No guessed repair.",
            "Original README describes 60 datasets; actual archive inventory is reported separately.",
            "Shared normal vectors make naive multiclass concatenation inappropriate.",
        ],
    }
    (output / "source_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
