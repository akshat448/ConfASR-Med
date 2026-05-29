"""
ConfASR-Med: Stage 2 Pipeline (MIND - Corrector)
------------------------------------------------
- Ingests Stage 1 confidence outputs
- Fine-tunes google/mt5-small as a seq2seq corrector using BFloat16
- Implements Early Stopping and Validation
- Runs Blind vs Confidence-Guided Ablation
"""

import json
import random
import re
from pathlib import Path

import evaluate
import nltk
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    MT5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Ensure NLTK punkt is downloaded for ROUGE score computation
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

RESULTS_DIR    = Path("results")
PROCESSED_DIR  = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints/mt5_corrector")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "google/mt5-small"
CONFIDENCE_THRESHOLD = 0.6     # tokens below this → [UNC]
MAX_INPUT_LEN  = 256
MAX_TARGET_LEN = 256

# =========================================================
# 1. Build corrector training data
# =========================================================

def tag_with_confidence(text: str, chunks: list, threshold: float = CONFIDENCE_THRESHOLD) -> str:
    if not chunks:
        return f"[UNC] {text}"

    tagged_parts = []
    for chunk in chunks:
        chunk_text  = chunk["text"].strip()
        chunk_score = chunk.get("score", 0.5)
        tag = "[UNC]" if chunk_score < threshold else "[CRT]"
        tagged_parts.append(f"{tag} {chunk_text}")

    return " ".join(tagged_parts)

def build_corrector_data(confidence_path: Path, val_ratio: float = 0.15):
    print("\n[1/3] Building corrector training data ...")

    with open(confidence_path, encoding="utf-8") as f:
        records = json.load(f)

    pairs = []

    # Source 1: Real ASR errors 
    real_errors = 0
    for rec in records:
        hyp    = rec["hypothesis"]
        ref    = rec["reference"]
        chunks = rec.get("chunks", [])

        if hyp.strip().lower() == ref.strip().lower():
            continue   # Perfect prediction

        tagged_input = tag_with_confidence(hyp, chunks, CONFIDENCE_THRESHOLD)
        pairs.append({"input_text": tagged_input, "target_text": ref})
        real_errors += 1

    print(f"   Real ASR error pairs extracted: {real_errors}")

    # Source 2: Simulated medical errors to boost dataset robustness
    COMMON_CONFUSIONS = {
        "Paracetamol": ["Paracetemol", "Paracetamole", "Paracetmol"],
        "fever":       ["fevar", "feever", "fiver"],
        "headache":    ["hadache", "headake", "hed ache"],
        "cough":       ["caugh", "cogh", "cof"],
        "diabetes":    ["diabetis", "diabeties", "diebetes"],
        "hypertension":["hypertansion", "hypertenshun", "hyper tension"],
        "lisinopril":  ["lisinoprl", "lisonopril", "lisinopil"],
        "metformin":   ["metfromin", "metformine", "metfornin"],
        "infection":   ["infecton", "infecshun", "infectoin"],
        "prescription":["prescripshun", "prescrption", "perscription"],
        "dosage":      ["dosege", "dosaje", "doseage"],
        "symptoms":    ["symtoms", "symptomes", "simptoms"],
        "diagnosis":   ["diagnosus", "diagonsis", "diagnosos"],
        "blood pressure":["blod pressure", "blood presure", "blood pressur"],
        "tablet":      ["tabelt", "tablat", "tablit"],
        "injection":   ["injecshun", "injekshun", "injecton"],
    }

    all_refs = [rec["reference"] for rec in records if rec["reference"]]
    sim_errors = 0
    for ref in random.sample(all_refs, min(len(all_refs), 3000)):
        corrupted = ref
        swapped   = False
        for correct, errors in COMMON_CONFUSIONS.items():
            if correct.lower() in corrupted.lower():
                corrupted = re.sub(
                    re.escape(correct),
                    random.choice(errors),
                    corrupted,
                    flags=re.IGNORECASE
                )
                swapped = True

        if swapped:
            words = corrupted.split()
            tagged_words = []
            for w in words:
                is_error = any(e.lower() in w.lower() for errors in COMMON_CONFUSIONS.values() for e in errors)
                tag = "[UNC]" if is_error else "[CRT]"
                tagged_words.append(f"{tag} {w}")
            tagged_input = " ".join(tagged_words)
            pairs.append({"input_text": tagged_input, "target_text": ref})
            sim_errors += 1

    print(f"   Simulated error pairs generated: {sim_errors}")
    print(f"   Total training pairs: {len(pairs)}")

    random.shuffle(pairs)
    n_val = int(len(pairs) * val_ratio)
    train_ds = Dataset.from_list(pairs[n_val:])
    val_ds   = Dataset.from_list(pairs[:n_val])

    splits = DatasetDict({"train": train_ds, "validation": val_ds})
    return splits

# =========================================================
# 2. Preprocess & Metrics
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_tokens(["[UNC]", "[CRT]"])
PREFIX = "correct medical transcript: "

def preprocess(examples):
    inputs  = [PREFIX + t for t in examples["input_text"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LEN, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_text"], max_length=MAX_TARGET_LEN, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

metric_rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Sometimes generate returns a tuple, grab the actual prediction tensor
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # FIX: Replace -100 padding tokens in BOTH predictions and labels
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {
        "rouge1": round(result["rouge1"], 4),
        "rouge2": round(result["rouge2"], 4),
        "rougeL": round(result["rougeL"], 4),
    }

# =========================================================
# 3. Train
# =========================================================

def train_corrector(splits: DatasetDict):
    print("\n[2/3] Fine-tuning mT5-small corrector ...")

    tokenized = splits.map(preprocess, batched=True, remove_columns=splits["train"].column_names, desc="Tokenizing")

    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer)) 

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    # FIX: Replaced fp16=True with bf16=True for mT5 stability on Ampere GPUs
    args = Seq2SeqTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-4,
        num_train_epochs=10, # Increased epochs because early stopping will catch it
        warmup_steps=100,
        bf16=True, # Critical for mT5 on A100
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        logging_steps=20,
        save_total_limit=2,
        save_safetensors=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # Prevent overfitting
    )

    trainer.train()
    trainer.save_model(str(CHECKPOINT_DIR / "best_model"))
    tokenizer.save_pretrained(str(CHECKPOINT_DIR / "best_model"))

    # Log metrics
    eval_logs = [l for l in trainer.state.log_history if "eval_rougeL" in l]
    results = {
        "best_rougeL": max(l["eval_rougeL"] for l in eval_logs) if eval_logs else None,
    }
    with open(RESULTS_DIR / "stage2_corrector.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"   Model saved → {CHECKPOINT_DIR / 'best_model'}")
    return trainer

# =========================================================
# 4. Ablation Study
# =========================================================

def run_ablation(splits: DatasetDict):
    print("\n[3/3] Running corrector ablation ...")
    from transformers import pipeline as hf_pipeline

    corrector = hf_pipeline(
        "text2text-generation",
        model=str(CHECKPOINT_DIR / "best_model"),
        tokenizer=str(CHECKPOINT_DIR / "best_model"),
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=MAX_TARGET_LEN,
    )

    val_data = splits["validation"]
    references    = val_data["target_text"]
    noisy_inputs  = val_data["input_text"]

    def strip_tags(text):
        return re.sub(r"\[(UNC|CRT)\]\s*", "", text).strip()

    blind_inputs = [PREFIX + strip_tags(t) for t in noisy_inputs]
    conf_inputs  = [PREFIX + t for t in noisy_inputs]
    no_corr      = [strip_tags(t) for t in noisy_inputs]

    print("   Running blind corrector ...")
    blind_preds = [r["generated_text"] for r in corrector(blind_inputs, batch_size=32)]

    print("   Running confidence-guided corrector ...")
    conf_preds  = [r["generated_text"] for r in corrector(conf_inputs, batch_size=32)]

    rouge = evaluate.load("rouge")
    def score(preds):
        return rouge.compute(predictions=preds, references=references, use_stemmer=True)

    r_none  = score(no_corr)
    r_blind = score(blind_preds)
    r_conf  = score(conf_preds)

    ablation = {
        "no_correction":         {k: round(v, 4) for k, v in r_none.items()},
        "blind_mt5":             {k: round(v, 4) for k, v in r_blind.items()},
        "confidence_guided_mt5": {k: round(v, 4) for k, v in r_conf.items()},
    }
    with open(RESULTS_DIR / "stage2_ablation.json", "w") as f:
        json.dump(ablation, f, indent=2)

    print("\n   ── Ablation Results ──────────────────────────")
    for name, scores in ablation.items():
        print(f"   {name:<30} ROUGE-L: {scores['rougeL']:.4f}")
    
    return ablation

# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    confidence_path = RESULTS_DIR / "stage1_confidence_outputs.json"
    if not confidence_path.exists():
        raise FileNotFoundError(f"{confidence_path} not found. Run Stage 1 first.")

    splits  = build_corrector_data(confidence_path)
    trainer = train_corrector(splits)
    ablation = run_ablation(splits)

    print("\n✓ Stage 2 corrector training & ablation complete. Check results/stage2_ablation.json")