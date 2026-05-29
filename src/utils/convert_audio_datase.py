from datasets import load_from_disk, Dataset, DatasetDict
from pathlib import Path

INPUT_PATH = "data/processed/asr_train"
OUTPUT_PATH = "data/processed/asr_train_paths"

ROOT = Path.cwd()

print("Loading dataset ...")

dataset = load_from_disk(INPUT_PATH)

new_splits = {}

for split in dataset.keys():

    print(f"\nProcessing split: {split}")

    ds = dataset[split]

    table = ds.data

    audio_col = table.column("audio")
    text_col = table.column("text")
    source_col = table.column("source")

    rows = []

    for i in range(len(ds)):

        audio_obj = audio_col[i].as_py()

        path = audio_obj.get("path")

        if path is None:
            continue

        path = Path(path)

        filename = path.name

        source = source_col[i].as_py()

        # =====================================================
        # Rebuild correct paths
        # =====================================================

        if source == "synthetic_cs":

            full_path = (
                ROOT
                / "data/processed/synthetic_cs/audio"
                / filename
            )

        elif source == "ekacare":

            # adjust if needed after find command
            full_path = (
                ROOT
                / "data/raw/ekacare_audio"
                / filename
            )

        elif source == "indicvoices":

            # adjust if needed after find command
            full_path = (
                ROOT
                / "data/raw/indicvoices_audio"
                / filename
            )

        else:
            continue

        rows.append({
            "audio_path": str(full_path),
            "text": text_col[i].as_py(),
            "source": source,
        })

    new_splits[split] = Dataset.from_list(rows)

final_ds = DatasetDict(new_splits)

print("\nSaving converted dataset ...")

final_ds.save_to_disk(OUTPUT_PATH)

print(f"\nSaved successfully → {OUTPUT_PATH}")