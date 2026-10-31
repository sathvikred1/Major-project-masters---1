"""Download pinned public archives; keep raw data out of Git."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REVISION = "a36f90b55b30ba3552747430fb508735025ca80b"
BASE = f"https://raw.githubusercontent.com/tmoulahi/Dataset-for-WSN-fault-detection/{REVISION}/"
ARCHIVES = {
    "Gain.rar": "59737c45290f22ee527fbdc2ecfec35bf8ff62b5446bb318e31b9a4f5385b191",
    "Offset.rar": "9cc0ffd12e797e27bd9c0b66b9e127148f32c28d3519bedeb565428fa705c1b7",
    "Out of bound.rar": "3393f95dffa3b3d10694b2b6f8620a9e04cc9f78d24391fc208be0b0e72fda52",
    "Stuck-at.rar": "7dc6857998201b3a627ce05a04bc14b3f730dd220fe301749269eed0120c6384",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(archive):
    if shutil.which("bsdtar"):
        command = ["bsdtar", "-xf", str(archive), "-C", str(RAW)]
    elif shutil.which("unrar"):
        command = ["unrar", "x", "-o+", str(archive), str(RAW) + "/"]
    elif shutil.which("7z"):
        command = ["7z", "x", "-y", str(archive), f"-o{RAW}"]
    else:
        raise RuntimeError("Install bsdtar, unrar, or 7z with RAR support, or extract manually into data/raw/.")
    subprocess.run(command, check=True, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Also download other fault families for the inventory audit.")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for name, digest in ARCHIVES.items():
        if not args.all and name != "Gain.rar":
            continue
        path = RAW / name
        if not path.exists():
            print(f"Downloading {name} at revision {REVISION}", flush=True)
            try:
                with urlopen(BASE + quote(name), timeout=90) as response:
                    path.write_bytes(response.read())
            except OSError as error:
                raise RuntimeError(f"Download failed for {name}: {error}") from error
        if sha256(path) != digest:
            raise ValueError(f"Checksum mismatch: {path}. Remove this file and download again.")
        extract(path)
        print(f"Verified and extracted {name}")
    readme = RAW / "SOURCE_README.md"
    if not readme.exists():
        with urlopen(BASE + "README.md", timeout=90) as response:
            readme.write_bytes(response.read())


if __name__ == "__main__":
    main()
