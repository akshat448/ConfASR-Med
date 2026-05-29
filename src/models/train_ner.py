"""
Script 5: Clinical NER with IndicBERT
Fine-tunes ai4bharat/IndicBERTv2-MLM-Sam-TLM for token classification.
Entity types: DRUG, SYMPTOM, DIAGNOSIS, DURATION, DOSAGE, BODY_PART, O

Since no large Hindi clinical NER dataset exists publicly, we:
  1. Use EkaCare's medical_entities field as silver labels (English)
  2. Generate a synthetic Hindi-English NER dataset using known entities
  3. Fine-tune IndicBERT on this combined data

Outputs:
  - checkpoints/indicbert_ner/
  - results/stage2_ner.json  (F1 per entity type)
"""

import json
import random
import re
from pathlib import Path
from typing import List

import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

RESULTS_DIR    = Path("results")
PROCESSED_DIR  = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints/indicbert_ner")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-Sam-TLM"
MAX_LEN    = 128

# BIO label scheme
LABEL_LIST = [
    "O",
    "B-DRUG",     "I-DRUG",
    "B-SYMPTOM",  "I-SYMPTOM",
    "B-DIAGNOSIS","I-DIAGNOSIS",
    "B-DURATION", "I-DURATION",
    "B-DOSAGE",   "I-DOSAGE",
    "B-BODY_PART","I-BODY_PART",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)


# ─────────────────────────────────────────────
# 1. Build NER dataset from EkaCare silver labels
# ─────────────────────────────────────────────

# EkaCare medical_entities format:
# [["adequate rest", "advices", [[11, 24]]]]
# We map EkaCare entity types to our schema:
EKACARE_TYPE_MAP = {
    "drug":        "DRUG",
    "drugs":       "DRUG",
    "medicine":    "DRUG",
    "symptom":     "SYMPTOM",
    "symptoms":    "SYMPTOM",
    "complaint":   "SYMPTOM",
    "disease":     "DIAGNOSIS",
    "diagnosis":   "DIAGNOSIS",
    "condition":   "DIAGNOSIS",
    "duration":    "DURATION",
    "dosage":      "DOSAGE",
    "dose":        "DOSAGE",
    "body_part":   "BODY_PART",
    "body part":   "BODY_PART",
    "anatomy":     "BODY_PART",
}


def ekacare_to_ner_sample(text: str, entities_json: str):
    """
    Convert one EkaCare record to BIO-tagged token list.
    Returns (tokens, labels) or None if parsing fails.
    """
    try:
        entities = json.loads(entities_json) if isinstance(entities_json, str) else entities_json
    except Exception:
        return None

    if not entities or not text:
        return None

    tokens = text.split()
    labels = ["O"] * len(tokens)

    # Character offset → token index mapping
    char_to_tok = {}
    char_pos = 0
    for tok_idx, tok in enumerate(tokens):
        for c in range(len(tok)):
            char_to_tok[char_pos + c] = tok_idx
        char_pos += len(tok) + 1  # +1 for space

    for ent in entities:
        if len(ent) < 3:
            continue
        ent_text, ent_type_raw, spans = ent[0], ent[1], ent[2]
        ent_type = EKACARE_TYPE_MAP.get(ent_type_raw.lower(), None)
        if ent_type is None:
            continue
        for span in spans:
            if len(span) != 2:
                continue
            start_char, end_char = span
            start_tok = char_to_tok.get(start_char)
            end_tok   = char_to_tok.get(end_char - 1)
            if start_tok is None or end_tok is None:
                continue
            labels[start_tok] = f"B-{ent_type}"
            for i in range(start_tok + 1, end_tok + 1):
                if i < len(labels):
                    labels[i] = f"I-{ent_type}"

    return {"tokens": tokens, "ner_tags": [LABEL2ID[l] for l in labels]}


# ─────────────────────────────────────────────
# 2. Synthetic Hindi-English NER sentences
# ─────────────────────────────────────────────

SYNTH_TEMPLATES_NER = [
    # (sentence_template, entity_annotations)
    ("Patient has fever since 3 days",
     [("fever","SYMPTOM"), ("3 days","DURATION")]),
    ("Doctor prescribed Paracetamol 500mg twice daily",
     [("Paracetamol","DRUG"), ("500mg","DOSAGE"), ("twice daily","DURATION")]),
    ("Mujhe bukhar aur headache hai",
     [("bukhar","SYMPTOM"), ("headache","SYMPTOM")]),
    ("Crocin lene ke baad dard kam hua",
     [("Crocin","DRUG"), ("dard","SYMPTOM")]),
    ("Pet mein dard 2 din se hai",
     [("Pet","BODY_PART"), ("dard","SYMPTOM"), ("2 din","DURATION")]),
    ("Blood pressure 140/90 hai",
     [("Blood pressure","DIAGNOSIS")]),
    ("Amoxicillin 250mg teen baar lena hai",
     [("Amoxicillin","DRUG"), ("250mg","DOSAGE"), ("teen baar","DURATION")]),
    ("Patient ko diabetes aur hypertension dono hain",
     [("diabetes","DIAGNOSIS"), ("hypertension","DIAGNOSIS")]),
    ("Seene mein dard aur saans lene mein taklif",
     [("Seene","BODY_PART"), ("dard","SYMPTOM"), ("saans","SYMPTOM")]),
    ("Metformin 500 subah shaam lena hai",
     [("Metformin","DRUG"), ("500","DOSAGE"), ("subah shaam","DURATION")]),
    ("Infection ke liye antibiotic diya",
     [("Infection","DIAGNOSIS"), ("antibiotic","DRUG")]),
    ("Sar mein dard raat se hai",
     [("Sar","BODY_PART"), ("dard","SYMPTOM"), ("raat se","DURATION")]),
    ("Patient ko 5 din ki dawai di",
     [("5 din","DURATION"), ("dawai","DRUG")]),
    ("Throat pain and difficulty swallowing",
     [("Throat","BODY_PART"), ("pain","SYMPTOM"), ("difficulty swallowing","SYMPTOM")]),
    ("Ek tablet Ibuprofen 400mg dard ke liye",
     [("tablet","DOSAGE"), ("Ibuprofen","DRUG"), ("400mg","DOSAGE"), ("dard","SYMPTOM")]),
]


def build_synth_ner_sample(template_text: str, entity_spans: list):
    """Convert template + entity list → BIO tokens."""
    tokens = template_text.split()
    labels = ["O"] * len(tokens)

    for ent_text, ent_type in entity_spans:
        ent_tokens = ent_text.split()
        # Find contiguous match in tokens
        for i in range(len(tokens) - len(ent_tokens) + 1):
            if [t.lower() for t in tokens[i:i+len(ent_tokens)]] == \
               [t.lower() for t in ent_tokens]:
                labels[i] = f"B-{ent_type}"
                for j in range(1, len(ent_tokens)):
                    labels[i+j] = f"I-{ent_type}"
                break

    return {"tokens": tokens, "ner_tags": [LABEL2ID[l] for l in labels]}


def build_ner_dataset():
    print("[1/3] Building NER dataset ...")

    samples = []

    # EkaCare silver labels
    raw_eka = load_from_disk("data/raw/ekacare")["test"]
    eka_count = 0
    for ex in raw_eka:
        if ex["medical_entities"] and ex["text"]:
            s = ekacare_to_ner_sample(ex["text"], ex["medical_entities"])
            if s and any(t != LABEL2ID["O"] for t in s["ner_tags"]):
                samples.append(s)
                eka_count += 1
    print(f"   EkaCare silver samples: {eka_count}")

    # Synthetic Hindi-English samples (repeat templates with variations)
    synth_count = 0
    for _ in range(50):  # 50x augmentation
        for tmpl_text, entity_spans in SYNTH_TEMPLATES_NER:
            s = build_synth_ner_sample(tmpl_text, entity_spans)
            samples.append(s)
            synth_count += 1
    print(f"   Synthetic NER samples: {synth_count}")
    print(f"   Total NER samples: {len(samples)}")

    random.shuffle(samples)
    n = len(samples)
    n_val  = int(n * 0.15)
    n_test = int(n * 0.10)

    ds = DatasetDict({
        "train":      Dataset.from_list(samples[n_val+n_test:]),
        "validation": Dataset.from_list(samples[:n_val]),
        "test":       Dataset.from_list(samples[n_val:n_val+n_test]),
    })
    ds.save_to_disk(str(PROCESSED_DIR / "ner_data"))
    print(f"   Saved → {PROCESSED_DIR / 'ner_data'}")
    return ds


# ─────────────────────────────────────────────
# 3. Tokenize + align labels
# ─────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        max_length=MAX_LEN,
        truncation=True,
        padding=False,
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids     = tokenized.word_ids(batch_index=i)
        label_ids    = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(labels[word_id])
            else:
                # For continuation sub-tokens: use I- version
                lbl = labels[word_id]
                if LABEL_LIST[lbl].startswith("B-"):
                    lbl = LABEL2ID["I-" + LABEL_LIST[lbl][2:]]
                label_ids.append(lbl)
            prev_word_id = word_id
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


# ─────────────────────────────────────────────
# 4. Metrics (seqeval)
# ─────────────────────────────────────────────

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    
    # FIX: Ensure we handle tuple predictions safely
    if isinstance(predictions, tuple):
        predictions = predictions[0]
        
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [LABEL_LIST[pred] for pred, lbl in zip(preds, lbls) if lbl != -100]
        for preds, lbls in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[lbl] for pred, lbl in zip(preds, lbls) if lbl != -100]
        for preds, lbls in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": round(results["overall_precision"], 4),
        "recall":    round(results["overall_recall"],    4),
        "f1":        round(results["overall_f1"],        4),
        "accuracy":  round(results["overall_accuracy"],  4),
        # Per-entity F1
        **{
            f"{ent}_f1": round(results[ent]["f1"], 4)
            for ent in ["DRUG","SYMPTOM","DIAGNOSIS","DURATION","DOSAGE","BODY_PART"]
            if ent in results
        }
    }


# ─────────────────────────────────────────────
# 5. Train
# ─────────────────────────────────────────────

def train_ner(ner_ds: DatasetDict):
    print("\n[2/3] Fine-tuning IndicBERT for NER ...")

    tokenized_ds = ner_ds.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=["tokens", "ner_tags"],
        desc="Tokenizing NER data",
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        num_train_epochs=15, # Increased epochs for Early Stopping
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True, # FIX: Switched to BFloat16 for stability
        eval_strategy="epoch", # FIX: Updated from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        report_to=["tensorboard"],
        save_total_limit=2,
        save_safetensors=False, # FIX: Prevent contiguous memory crashes when saving
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)], # FIX: Added Early Stopping
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(str(CHECKPOINT_DIR))

    # Test set evaluation
    test_results = trainer.evaluate(tokenized_ds["test"])
    print("\n   ── NER Test Results ──────────────────────")
    for k, v in test_results.items():
        if not k.startswith("eval_runtime"):
            print(f"   {k}: {v}")

    log_history = trainer.state.log_history
    eval_logs   = [l for l in log_history if "eval_f1" in l]

    results = {
        "test_overall_f1":   test_results.get("eval_f1"),
        "test_per_entity":   {
            k.replace("eval_","").replace("_f1",""): v
            for k, v in test_results.items() if "_f1" in k and k != "eval_f1"
        },
        "eval_f1_curve": [
            {"epoch": l.get("epoch"), "f1": l["eval_f1"]} for l in eval_logs
        ],
    }
    with open(RESULTS_DIR / "stage2_ner.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n   Model saved → {CHECKPOINT_DIR}")
    print(f"   Results → {RESULTS_DIR / 'stage2_ner.json'}")
    return trainer


# ─────────────────────────────────────────────
# 6. Inference helper (used by Stage 3)
# ─────────────────────────────────────────────

def run_ner_on_text(text: str, ner_model=None, ner_tokenizer=None) -> list:
    """
    Run NER on a plain text string.
    Returns list of {word, entity, score} dicts.
    """
    from transformers import pipeline as hf_pipeline

    # FIX: Always use the explicitly passed model and tokenizer
    if ner_model is None or ner_tokenizer is None:
        raise ValueError("You must pass the live ner_model and ner_tokenizer objects to avoid fast tokenizer decode bugs.")

    pipe = hf_pipeline(
        "token-classification",
        model=ner_model,
        tokenizer=ner_tokenizer,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
    )

    results = pipe(text)
    return [
        {
            "word":   r["word"],
            "entity": r["entity_group"],
            "score":  round(r["score"], 4),
            "start":  r["start"],
            "end":    r["end"],
        }
        for r in results
    ]

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    ner_ds  = build_ner_dataset()
    trainer = train_ner(ner_ds)

    # Quick sanity check
    print("\n[3/3] Sanity check on example ...")
    example = "Patient ko fever aur headache hai. Doctor ne Paracetamol 500mg diya."
    entities = run_ner_on_text(example, ner_model=trainer.model, ner_tokenizer=tokenizer)
    # entities = run_ner_on_text(example)
    print(f"   Input: {example}")
    print(f"   Entities: {entities}")

    print("\n✓ Stage 2 NER training complete.")