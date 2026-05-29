from datasets import load_from_disk
from pathlib import Path
import json
import os

RAW_DIR = Path("data/raw")

# EkaCare
print("\n========== EkaCare ==========")
eka = load_from_disk(str(RAW_DIR / "ekacare"))
print(eka)

print("\nColumns:")
print(eka["test"].column_names)

print("\nSample:")
eka_meta = eka["test"].remove_columns(["audio"])
eka_sample = eka_meta[0]
print(eka_sample)

# ACI-BENCH
print("\n========== ACI-BENCH ==========")
aci_json_path = RAW_DIR / "aci_bench/data/challenge_data_json"
files = sorted(os.listdir(aci_json_path))
print(f"\nTotal files: {len(files)}")
sample_file = files[0]
print(f"\nSample file: {sample_file}")
with open(os.path.join(aci_json_path, sample_file)) as f:
    sample = json.load(f)
print("\nTop-level keys:")
print(sample.keys())
records = sample.get("data", [])
print(f"\nNumber of records in 'data': {len(records)}")
if records:
    print("\nFirst record:")
    print(json.dumps(records[0], indent=2)[:3000])

print("\n========== IndicVoices Hindi ==========")
indic = load_from_disk(str(RAW_DIR / "indicvoices_hi"))
print(indic)

print("\nColumns:")
print(indic.column_names)

print("\nSample:")
indic_meta = indic.remove_columns(["audio"]) if "audio" in indic.column_names else indic
print(indic_meta[0])