"""
Download VisDrone MOT Task-4 Validation Set.

Primary:   Google Drive (official assignment link)
Fallback:  gdown --fuzzy (bypasses some quota restrictions)

If both fail (Google Drive quota exceeded), download manually:
  https://drive.google.com/file/d/1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu/view
and place the zip at data/raw/VisDrone2019-MOT-val.zip, then re-run.

Usage: python scripts/download_dataset.py [--dest data/raw]
"""
import argparse
import os
import sys
import zipfile

GDRIVE_FILE_ID = "1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
GDRIVE_FUZZY_URL = f"https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view"


def _try_gdown(zip_path: str) -> bool:
    try:
        import gdown
        print("  Trying gdown ...")
        # Try with fuzzy=True if supported (newer gdown), fall back without it
        import inspect
        kw = {"fuzzy": True} if "fuzzy" in inspect.signature(gdown.download).parameters else {}
        url = GDRIVE_FUZZY_URL if kw else GDRIVE_URL
        gdown.download(url, zip_path, quiet=False, **kw)
        return os.path.exists(zip_path) and os.path.getsize(zip_path) > 1_000_000
    except Exception as e:
        print(f"  gdown failed: {e}")
    return False


def download(dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "VisDrone2019-MOT-val.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading VisDrone MOT val set -> {zip_path}")
        ok = _try_gdown(zip_path)

        if not ok:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            print()
            print("=" * 65)
            print("  Google Drive quota exceeded -- manual download required.")
            print()
            print("  1. Open in browser:")
            print(f"     {GDRIVE_FUZZY_URL}")
            print()
            print("  2. Place the downloaded zip at:")
            print(f"     {os.path.abspath(zip_path)}")
            print()
            print("  3. Re-run this script.")
            print("=" * 65)
            sys.exit(1)
    else:
        print(f"Zip already exists at {zip_path}, skipping download.")

    extract_dir = os.path.join(dest, "VisDrone2019-MOT-val")
    if not os.path.exists(extract_dir):
        print(f"Extracting to {extract_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        print("Done.")
    else:
        print(f"Already extracted at {extract_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/raw", help="Destination directory")
    args = parser.parse_args()
    download(args.dest)
