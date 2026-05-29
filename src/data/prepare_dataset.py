"""
Optimized Dataset Preparation Pipeline
--------------------------------------
- EkaCare ASR
- IndicVoices Hindi
- Synthetic Hindi-English Code-Switched Medical Speech

Features:
- multiprocessing
- checkpointing
- resumable generation
- retry/backoff
- deterministic synthetic text
- per-source train/val/test splits
- lazy audio loading
"""

import json
import os
import random
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_from_disk,
)
from gtts import gTTS
from pydub import AudioSegment
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================

os.environ["HF_DATASETS_CACHE"] = "/workspace/hf_cache"

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000

MAX_WORKERS = 8
CHECKPOINT_EVERY = 50
MAX_RETRIES = 5

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# =========================================================
# 1. EkaCare
# =========================================================


def load_ekacare(max_duration=30.0):

    print("\n[1/3] Loading EkaCare ...")

    ds = load_from_disk(str(RAW_DIR / "ekacare"))["test"]

    def is_valid(ex):
        return (
            ex["audio_language"] == "en"
            and ex["text"] is not None
            and len(ex["text"].strip()) > 5
            and ex["duration"] <= max_duration
        )

    ds = ds.filter(
        is_valid,
        num_proc=16,
        desc="Filtering EkaCare"
    )

    ds = ds.select_columns(["audio", "text"])

    ds = ds.cast_column(
        "audio",
        Audio(sampling_rate=SAMPLE_RATE)
    )

    print(f"   EkaCare samples: {len(ds)}")

    return ds


# =========================================================
# 2. IndicVoices Hindi
# =========================================================

HEALTH_KEYWORDS = [
    "health",
    "medical",
    "hospital",
    "doctor",
    "medicine",
    "swasthya",
    "dawai",
]


def load_indicvoices(
    max_samples=5000,
    min_snr=20.0
):

    print("\n[2/3] Loading IndicVoices Hindi ...")

    ds = load_from_disk(str(RAW_DIR / "indicvoices_hi"))

    def is_valid(ex):

        try:
            return (
                ex["snr"] is not None
                and float(ex["snr"]) >= min_snr
                and ex["text"] is not None
                and len(ex["text"].strip()) > 5
            )

        except:
            return False

    ds = ds.filter(
        is_valid,
        num_proc=16,
        desc="Filtering IndicVoices"
    )

    def is_health(ex):

        task = (ex.get("task_name") or "").lower()

        return any(
            k in task
            for k in HEALTH_KEYWORDS
        )

    health_ds = ds.filter(
        is_health,
        num_proc=16,
        desc="Health filter"
    )

    general_ds = ds.filter(
        lambda ex: not is_health(ex),
        num_proc=16,
        desc="General filter"
    )

    print(f"   Health samples: {len(health_ds)}")

    remaining = max_samples - len(health_ds)

    if remaining > 0:

        general_ds = general_ds.shuffle(seed=SEED)

        general_ds = general_ds.select(
            range(min(remaining, len(general_ds)))
        )

        ds = concatenate_datasets([
            health_ds,
            general_ds
        ])

    else:

        ds = health_ds.select(range(max_samples))

    ds = ds.shuffle(seed=SEED)

    ds = ds.select_columns(["audio", "text"])

    ds = ds.cast_column(
        "audio",
        Audio(sampling_rate=SAMPLE_RATE)
    )

    print(f"   IndicVoices samples used: {len(ds)}")

    return ds


# =========================================================
# 3. Synthetic CS Generation
# =========================================================

CLINICAL_SWAPS = {
    "बुखार": "fever",
    "दर्द": "pain",
    "खांसी": "cough",
    "उल्टी": "vomiting",
    "सिरदर्द": "headache",
    "थकान": "fatigue",
    "छाती": "chest",
    "पेट": "abdomen",
    "पैरासिटामोल": "Paracetamol",
    "रक्तचाप": "blood pressure",
    "डायबिटीज": "diabetes",
    "इन्फेक्शन": "infection",
}

HINDI_TEMPLATES = [
    "मरीज़ को {symptom} की शिकायत है।",
    "आपको {symptom} कब से हो रहा है?",
    "रोज़ाना {drug} की एक गोली लें।",
    "{bodypart} में {symptom} हो रहा है।",
    "इसके लिए {drug} लेना होगा।",
    "आपको {symptom} कितने दिन से है?",
]

TEMPLATE_VARS = {
    "symptom": [
        "बुखार",
        "दर्द",
        "खांसी",
        "सिरदर्द",
        "उल्टी",
        "थकान",
    ],
    "drug": [
        "पैरासिटामोल",
        "दवाई",
        "कैप्सूल",
    ],
    "bodypart": [
        "पेट",
        "छाती",
    ],
}


def fill_template(template):

    text = template

    for var, values in TEMPLATE_VARS.items():

        placeholder = "{" + var + "}"

        if placeholder in text:

            text = text.replace(
                placeholder,
                random.choice(values),
                1
            )

    cs_text = text

    keys = [
        k for k in CLINICAL_SWAPS
        if k in cs_text
    ]

    if keys:

        n_swaps = min(
            random.randint(1, 2),
            len(keys)
        )

        chosen = random.sample(keys, n_swaps)

        for k in chosen:
            cs_text = cs_text.replace(
                k,
                CLINICAL_SWAPS[k]
            )

    return cs_text


def pre_generate_texts(n_samples):

    texts = []

    for i in range(n_samples):

        template = random.choice(HINDI_TEMPLATES)

        text = fill_template(template)

        texts.append({
            "id": i,
            "text": text
        })

    return texts


def process_single_sample(args):

    sample, output_dir = args

    i = sample["id"]
    text = sample["text"]

    audio_dir = output_dir / "audio"

    audio_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    wav_path = audio_dir / f"cs_{i:05d}.wav"

    if wav_path.exists():

        return {
            "id": i,
            "audio": str(wav_path),
            "text": text,
            "status": "skipped"
        }

    mp3_path = audio_dir / f"cs_{i:05d}.mp3"

    try:

        success = False

        for retry in range(MAX_RETRIES):

            try:

                tts = gTTS(
                    text=text,
                    lang="hi",
                    slow=False
                )

                tts.save(str(mp3_path))

                success = True
                break

            except Exception:

                sleep_time = (
                    (2 ** retry)
                    + random.uniform(0.5, 2.0)
                )

                time.sleep(sleep_time)

        if not success:
            raise RuntimeError(
                "TTS failed after retries"
            )

        audio = AudioSegment.from_mp3(
            str(mp3_path)
        )

        audio = (
            audio
            .set_frame_rate(SAMPLE_RATE)
            .set_channels(1)
        )

        audio.export(
            str(wav_path),
            format="wav"
        )

        if mp3_path.exists():
            mp3_path.unlink()

        return {
            "id": i,
            "audio": str(wav_path),
            "text": text,
            "status": "success"
        }

    except Exception as e:

        return {
            "id": i,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def generate_synthetic_cs_data(
    n_samples=1000,
):

    print(
        f"\n[3/3] Generating "
        f"{n_samples} synthetic samples ..."
    )

    output_dir = PROCESSED_DIR / "synthetic_cs"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        output_dir / "metadata.jsonl"
    )

    completed = set()

    if checkpoint_path.exists():

        with open(checkpoint_path, "r") as f:

            for line in f:

                try:
                    obj = json.loads(line)
                    completed.add(obj["id"])
                except:
                    pass

        print(
            f"   Resuming from "
            f"{len(completed)} completed samples"
        )

    samples = pre_generate_texts(n_samples)

    tasks = [
        (s, output_dir)
        for s in samples
        if s["id"] not in completed
    ]

    success = 0
    failed = 0

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                process_single_sample,
                task
            )
            for task in tasks
        ]

        for idx, future in enumerate(
            tqdm(
                as_completed(futures),
                total=len(futures)
            )
        ):

            try:

                result = future.result(
                    timeout=60
                )

            except Exception as e:

                failed += 1

                if failed <= 10:
                    print(f"\nWorker timeout/error: {e}")

                continue

            if result["status"] in [
                "success",
                "skipped"
            ]:

                success += 1

                with open(
                    checkpoint_path,
                    "a"
                ) as f:

                    f.write(
                        json.dumps(result)
                        + "\n"
                    )

            else:

                failed += 1

                if failed <= 10:

                    print(
                        f"\nFailed sample "
                        f"{result['id']}"
                    )

                    print(result["error"])

            if idx % CHECKPOINT_EVERY == 0:

                print(
                    f"\nProgress | "
                    f"success={success} "
                    f"failed={failed}"
                )

    print("\nSynthetic generation complete")

    final_records = []

    with open(checkpoint_path, "r") as f:

        for line in f:

            obj = json.loads(line)

            if obj["status"] in [
                "success",
                "skipped"
            ]:

                final_records.append({
                    "audio": obj["audio"],
                    "text": obj["text"],
                    "source": "synthetic_cs",
                })

    ds = Dataset.from_list(final_records)

    ds = ds.cast_column(
        "audio",
        Audio(sampling_rate=SAMPLE_RATE)
    )

    save_path = output_dir / "dataset"

    ds.save_to_disk(str(save_path))

    print(f"   Saved synthetic dataset")

    return ds


# =========================================================
# 4. Splits
# =========================================================

def split_source_dataset(ds):

    temp = ds.train_test_split(
        test_size=0.2,
        seed=SEED
    )

    val_test = temp["test"].train_test_split(
        test_size=0.5,
        seed=SEED
    )

    return {
        "train": temp["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    }


def build_splits(
    eka_ds,
    indic_ds,
    synth_ds
):

    print("\n[4/4] Building splits ...")

    eka_ds = eka_ds.map(
        lambda _: {"source": "ekacare"}
    )

    indic_ds = indic_ds.map(
        lambda _: {"source": "indicvoices"}
    )

    synth_ds = synth_ds.map(
        lambda _: {"source": "synthetic_cs"}
    )

    eka_split = split_source_dataset(eka_ds)
    indic_split = split_source_dataset(indic_ds)
    synth_split = split_source_dataset(synth_ds)

    train_ds = concatenate_datasets([
        eka_split["train"],
        indic_split["train"],
        synth_split["train"],
    ]).shuffle(seed=SEED)

    val_ds = concatenate_datasets([
        eka_split["validation"],
        indic_split["validation"],
        synth_split["validation"],
    ]).shuffle(seed=SEED)

    test_ds = concatenate_datasets([
        eka_split["test"],
        indic_split["test"],
        synth_split["test"],
    ]).shuffle(seed=SEED)

    splits = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    print(
        f"Train={len(train_ds)} | "
        f"Val={len(val_ds)} | "
        f"Test={len(test_ds)}"
    )

    counts = Counter(train_ds["source"])

    for src, count in counts.items():
        print(f"   {src}: {count}")

    save_path = PROCESSED_DIR / "asr_train"

    splits.save_to_disk(str(save_path))

    print(f"Saved splits → {save_path}")

    return splits


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    eka_ds = load_ekacare()

    indic_ds = load_indicvoices(
        max_samples=8000
    )

    synth_ds = generate_synthetic_cs_data(
        n_samples=2000
    )

    splits = build_splits(
        eka_ds,
        indic_ds,
        synth_ds
    )

    print("\n✓ Dataset preparation complete")