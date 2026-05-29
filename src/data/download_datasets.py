from datasets import load_dataset
import os
import subprocess
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def hf_dataset_exists(path: Path) -> bool:
    return path.exists() and any((path / name).exists() for name in ["dataset_info.json", "state.json", "data"])


def git_repo_exists(path: Path) -> bool:
    return path.exists() and (path / ".git").exists()


def clone_or_update_repo(repo_url: str, dest: Path):
    if git_repo_exists(dest):
        print(f"{dest} already exists. Pulling latest changes...")
        subprocess.run(["git", "-C", str(dest), "pull"], check=True)
    elif dest.exists() and any(dest.iterdir()):
        print(f"{dest} exists and is non-empty, but is not a git repo. Skipping for safety.")
    else:
        print(f"Cloning into {dest}...")
        subprocess.run(["git", "clone", repo_url, str(dest)], check=True)


# ---------------------------------------------------
# EkaCare Medical ASR
# ---------------------------------------------------
eka_path = RAW_DIR / "ekacare"
print("Downloading EkaCare dataset...")
if hf_dataset_exists(eka_path):
    print("EkaCare already downloaded. Skipping.")
else:
    eka = load_dataset("ekacare/eka-medical-asr-evaluation-dataset")
    eka.save_to_disk(str(eka_path))
    print("EkaCare saved.")

# ---------------------------------------------------
# ACI-BENCH
# ---------------------------------------------------
aci_path = RAW_DIR / "aci_bench"
print("Downloading ACI-BENCH...")
clone_or_update_repo("https://github.com/wyim/aci-bench.git", aci_path)
print("ACI-BENCH ready.")

# ---------------------------------------------------
# IndicVoices Hindi
# ---------------------------------------------------
indic_path = RAW_DIR / "indicvoices_hi"

print("Downloading IndicVoices Hindi subset...")

if hf_dataset_exists(indic_path):
    print("IndicVoices Hindi already downloaded. Skipping.")

else:
    indic = load_dataset(
        "ai4bharat/indicvoices_r",
        "Hindi",
        split="train"
    )

    indic.save_to_disk(str(indic_path))

    print("IndicVoices Hindi saved.")

print("All datasets processed successfully.")