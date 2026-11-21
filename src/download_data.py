"""Retrieve and verify only the eight selected sensor-fault workbooks."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REVISION = "a36f90b55b30ba3552747430fb508735025ca80b"
BASE = f"https://raw.githubusercontent.com/tmoulahi/Dataset-for-WSN-fault-detection/{REVISION}/"
ARCHIVES = {
    "Gain.rar": "59737c45290f22ee527fbdc2ecfec35bf8ff62b5446bb318e31b9a4f5385b191",
    "Offset.rar": "9cc0ffd12e797e27bd9c0b66b9e127148f32c28d3519bedeb565428fa705c1b7",
    "Stuck-at.rar": "7dc6857998201b3a627ce05a04bc14b3f730dd220fe301749269eed0120c6384",
    "Out of bound.rar": "3393f95dffa3b3d10694b2b6f8620a9e04cc9f78d24391fc208be0b0e72fda52",
}


# Fix the 30% selections before fitting. Gain and offset use beta=2;
# the other two source archives do not have beta subdirectories.
DATASETS = {
    "Gain": ("Gain/beta=2/gain302.xlsx", "Gain/beta=2/ygain302.xlsx",
             "90ad926d50595beae7e463fcc849f0a4e968de793aa033d5e25e9560f44581ac",
             "d77d3351013cf1f4a7e98793b9f3bf907f59fec609fb715651c1ad1a8c431aef"),
    "Offset": ("Offset/beta=2/offset302.xlsx", "Offset/beta=2/y302.xlsx",
               "be4eff486e66079fcac7bc86f12a5457b6ec898e8ec1de349e50639176190a07",
               "d93f24dea25ca2eb2235ca58ec7701796c7bc350ee210be1975288b9c7a122d3"),
    "Stuck-at": ("Stuck-at/stuck302.xlsx", "Stuck-at/ystuck302.xlsx",
                 "f16cd49271a54169547742ac55715680255dc8a9a591643b4939777f14301811",
                 "d3dcd0ae802cb68731f677a5abd11f3fa5d398c316319b6a4598ab8ad8603e0d"),
    "Out-of-bounds": ("Out of bound/out302.xlsx", "Out of bound/yout302.xlsx",
                      "6259fb0964780eea35de446400a02a6102423d452cb5ec7cb867a8e38e26a16b",
                      "24f681376f7370124dce8a66643583b4795b8223c7fa20234f88429b040f29ca"),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(archive, members):
    if shutil.which("bsdtar"):
        command = ["bsdtar", "-xf", str(archive), "-C", str(RAW), *members]
    elif shutil.which("unrar"):
        command = ["unrar", "x", "-o+", str(archive), *members, str(RAW) + "/"]
    elif shutil.which("7z"):
        command = ["7z", "x", "-y", str(archive), f"-o{RAW}", *members]
    else:
        raise RuntimeError("Install bsdtar, unrar, or 7z with RAR support, or extract manually into data/raw/.")
    subprocess.run(command, check=True, capture_output=True, text=True)


def verify_workbooks(files):
    for name, digest in files.items():
        path = RAW / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Missing or changed workbook: {path}. Remove the affected pair and download again.")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, digest in ARCHIVES.items():
        spec = next(pair for pair in DATASETS.values() if pair[0].split("/")[0] == name[:-4])
        files = {spec[0]: spec[2], spec[1]: spec[3]}
        if all((RAW / member).is_file() for member in files):
            verify_workbooks(files)
            print(f"Verified cached workbooks for {name[:-4]}")
            continue
        # Download archives temporarily and extract only the two selected files.
        with tempfile.TemporaryDirectory(prefix="sensor-data-") as directory:
            cached = RAW / name
            path = cached if cached.exists() else Path(directory) / name
            if not path.exists():
                print(f"Downloading {name} at revision {REVISION}", flush=True)
                try:
                    with urlopen(BASE + quote(name), timeout=90) as response:
                        path.write_bytes(response.read())
                except OSError as error:
                    raise RuntimeError(f"Download failed for {name}: {error}") from error
            if sha256(path) != digest:
                raise ValueError(f"Checksum mismatch: {path}.")
            extract(path, list(files))
            verify_workbooks(files)
            print(f"Verified and extracted the selected pair from {name}")
    readme = RAW / "SOURCE_README.md"
    if not readme.exists():
        with urlopen(BASE + "README.md", timeout=90) as response:
            readme.write_bytes(response.read())


if __name__ == "__main__":
    main()
