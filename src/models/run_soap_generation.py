"""
Script 6: Stage 3 — SOAP Note Generation + Evaluation
- Parses ACI-BENCH doctor-patient conversations
- Runs the full pipeline (corrector → NER [Manual Bypass] → SOAP generator) on each
- Evaluates against ACI-BENCH reference notes using ROUGE + BERTScore
- Saves all results and a qualitative example for the report

Outputs:
  - results/stage3_soap.json         (ROUGE/BERTScore per SOAP section)
  - results/qualitative_example.json (one full example for the report figure)
  - results/stage3_all_outputs.json  (all predictions for analysis)
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline as hf_pipeline,
)

RESULTS_DIR   = Path("results")
RAW_DIR       = Path("data/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NER_CKPT       = Path("checkpoints/indicbert_ner")
CORRECTOR_CKPT = Path("checkpoints/mt5_corrector/best_model")

CONFIDENCE_THRESHOLD = 0.6
USE_GROQ = True   # Set False to use local Qwen2.5-7B
GROQ_MODEL = "llama-3.3-70b-versatile"  # Upgraded to a stronger reasoning model

# BIO label scheme for Manual NER Bypass
LABEL_LIST = [
    "O", "B-DRUG", "I-DRUG", "B-SYMPTOM", "I-SYMPTOM",
    "B-DIAGNOSIS", "I-DIAGNOSIS", "B-DURATION", "I-DURATION",
    "B-DOSAGE", "I-DOSAGE", "B-BODY_PART", "I-BODY_PART",
]
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

# ─────────────────────────────────────────────
# 1. Parse ACI-BENCH
# ─────────────────────────────────────────────

def parse_aci_bench(aci_dir: Path):
    print("[1/5] Parsing ACI-BENCH ...")
    json_dir = aci_dir / "data/challenge_data_json"
    records  = []

    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith(".json"):
            continue
        with open(json_dir / fname, encoding="utf-8") as f:
            data = json.load(f)

        for item in data.get("data", []):
            src = item.get("src", "")
            tgt = item.get("tgt", "")
            if not src or not tgt:
                continue
            records.append({"conversation": src, "reference_note": tgt})

    print(f"   Loaded {len(records)} ACI-BENCH records")
    return records

# ─────────────────────────────────────────────
# 2. Load Stage 2 models (with NER Bypass)
# ─────────────────────────────────────────────

def load_stage2_models():
    print("[2/5] Loading Stage 2 models ...")
    device = 0 if torch.cuda.is_available() else -1

    corrector_pipe = hf_pipeline(
        "text2text-generation",
        model=str(CORRECTOR_CKPT),
        tokenizer=str(CORRECTOR_CKPT),
        device=device,
        max_new_tokens=256,
    )

    # FIX: Load RAW model and tokenizer to bypass HF Pipeline aggregation crash
    ner_tokenizer = AutoTokenizer.from_pretrained(str(NER_CKPT))
    ner_model = AutoModelForTokenClassification.from_pretrained(str(NER_CKPT))
    if torch.cuda.is_available():
        ner_model = ner_model.to("cuda")

    print("   ✓ Stage 2 models loaded")
    return corrector_pipe, ner_model, ner_tokenizer

# ─────────────────────────────────────────────
# 3. SOAP section extractor
# ─────────────────────────────────────────────

SOAP_PATTERN = re.compile(
    r"(SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN)[:\s]*(.*?)(?=(?:SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN)[:\s]|$)",
    re.DOTALL | re.IGNORECASE,
)

def extract_soap_sections(note: str) -> dict:
    sections = {"S": "", "O": "", "A": "", "P": ""}
    map_ = {"SUBJECTIVE": "S", "OBJECTIVE": "O", "ASSESSMENT": "A", "PLAN": "P"}

    for match in SOAP_PATTERN.finditer(note):
        key  = match.group(1).upper()
        text = match.group(2).strip()
        if key in map_:
            sections[map_[key]] = text

    return sections

# ─────────────────────────────────────────────
# 4. Apply Stage 2 (with Manual NER)
# ─────────────────────────────────────────────

def extract_entities_manual(text: str, ner_model, ner_tokenizer) -> list:
    """Bypasses HF Pipeline aggregation bug by extracting directly from logits"""
    inputs = ner_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(ner_model.device)
    
    with torch.no_grad():
        outputs = ner_model(**inputs)
    
    logits = outputs.logits[0]
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    entities = []
    current_ent = None
    current_words = []

    for token, pred in zip(tokens, preds):
        if token in ["[CLS]", "[SEP]", "<pad>"]: continue
        label = ID2LABEL[pred]
        clean_token = token.replace(' ', ' ') # Handle ALBERT subword space

        if label == "O":
            if current_ent:
                entities.append({"word": "".join(current_words).strip(), "entity": current_ent, "score": 0.99})
                current_ent = None
                current_words = []
        elif label.startswith("B-"):
            if current_ent:
                entities.append({"word": "".join(current_words).strip(), "entity": current_ent, "score": 0.99})
            current_ent = label[2:]
            current_words = [clean_token]
        elif label.startswith("I-"):
            if current_ent == label[2:]:
                current_words.append(clean_token)

    if current_ent:
        entities.append({"word": "".join(current_words).strip(), "entity": current_ent, "score": 0.99})

    # Deduplicate
    unique_entities = []
    seen = set()
    for e in entities:
        if e["word"] and e["word"] not in seen:
            unique_entities.append(e)
            seen.add(e["word"])
            
    return unique_entities

def apply_stage2(text: str, corrector_pipe, ner_model, ner_tokenizer) -> dict:
    UNCERTAIN_MEDICAL = [
        "lisinopril","metformin","amlodipine","atorvastatin","hydrochlorothiazide",
        "prednisone","albuterol","omeprazole","sertraline","losartan",
    ]
    words = text.split()
    tagged = []
    for w in words:
        clean = re.sub(r"[^a-zA-Z]", "", w).lower()
        tag = "[UNC]" if clean in UNCERTAIN_MEDICAL else "[CRT]"
        tagged.append(f"{tag} {w}")
    tagged_text = " ".join(tagged)

    # Corrector
    corrected = corrector_pipe(
        "correct medical transcript: " + tagged_text,
        max_new_tokens=256,
    )[0]["generated_text"]

    # NER (Manual Bypass)
    entities = extract_entities_manual(corrected, ner_model, ner_tokenizer)

    return {
        "original":  text,
        "tagged":    tagged_text,
        "corrected": corrected,
        "entities":  entities,
    }

# ─────────────────────────────────────────────
# 5. SOAP generation
# ─────────────────────────────────────────────

def build_soap_prompt(conversation: str, entities: list) -> str:
    ent_summary = ""
    for etype in ["SYMPTOM", "DRUG", "DIAGNOSIS", "DOSAGE", "DURATION", "BODY_PART"]:
        items = [e["word"] for e in entities if e["entity"] == etype]
        if items:
            ent_summary += f"  {etype}: {', '.join(set(items))}\n"

    prompt = f"""You are a clinical documentation assistant. Given the following doctor-patient conversation and extracted medical entities, generate a concise SOAP note.

CONVERSATION:
{conversation[:2000]}

EXTRACTED ENTITIES:
{ent_summary if ent_summary else "  (none extracted)"}

Generate the SOAP note in this exact format:
SUBJECTIVE: [Patient's complaints, symptoms, history as stated]
OBJECTIVE: [Physician observations, vitals, exam findings]
ASSESSMENT: [Diagnosis or clinical impression]
PLAN: [Treatment plan, medications, follow-up]

SOAP NOTE:"""
    return prompt

def generate_soap_groq(prompt: str) -> Optional[str]:
    try:
        from groq import Groq
        client = Groq() 
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"   Groq error: {e}")
        return None

# ─────────────────────────────────────────────
# 6. Evaluate
# ─────────────────────────────────────────────

def evaluate_soap(predictions: list, references: list) -> dict:
    print("[4/5] Evaluating SOAP outputs ...")
    rouge   = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    results = {}
    sections = ["S", "O", "A", "P"]
    section_names = {"S": "Subjective", "O": "Objective", "A": "Assessment", "P": "Plan"}

    all_preds = []
    all_refs  = []

    for sec in sections:
        preds = [p.get(sec, "") for p in predictions]
        refs  = [r.get(sec, "") for r in references]

        valid = [(p, r) for p, r in zip(preds, refs) if p.strip() and r.strip()]
        if not valid:
            continue
        v_preds, v_refs = zip(*valid)

        r = rouge.compute(predictions=list(v_preds), references=list(v_refs), use_stemmer=True)
        bs = bertscore.compute(predictions=list(v_preds), references=list(v_refs), lang="en")

        results[section_names[sec]] = {
            "rouge1": round(r["rouge1"], 4),
            "rouge2": round(r["rouge2"], 4),
            "rougeL": round(r["rougeL"], 4),
            "bertscore_f1": round(np.mean(bs["f1"]), 4),
            "n_samples": len(valid),
        }
        all_preds.extend(v_preds)
        all_refs.extend(v_refs)

    if all_preds:
        r_all  = rouge.compute(predictions=all_preds, references=all_refs, use_stemmer=True)
        bs_all = bertscore.compute(predictions=all_preds, references=all_refs, lang="en")
        results["Overall"] = {
            "rouge1": round(r_all["rouge1"], 4),
            "rouge2": round(r_all["rouge2"], 4),
            "rougeL": round(r_all["rougeL"], 4),
            "bertscore_f1": round(np.mean(bs_all["f1"]), 4),
            "n_samples": len(all_preds),
        }

    return results

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    records = parse_aci_bench(RAW_DIR / "aci_bench")

    corrector_pipe, ner_model, ner_tokenizer = load_stage2_models()

    all_outputs    = []
    pred_sections  = []
    ref_sections   = []

    print(f"[3/5] Running pipeline on ACI-BENCH records ...")
    for i, rec in enumerate(tqdm(records[:100])):  # cap at 100 for speed
        conv  = rec["conversation"]
        ref   = rec["reference_note"]

        stage2 = apply_stage2(conv, corrector_pipe, ner_model, ner_tokenizer)
        prompt = build_soap_prompt(stage2["corrected"], stage2["entities"])

        if USE_GROQ:
            soap_text = generate_soap_groq(prompt)
            time.sleep(0.5)  # gentle rate limit
        else:
            print("Local LLM not loaded in this script bypass version.")
            continue

        if soap_text is None:
            continue

        pred_soap = extract_soap_sections(soap_text)
        ref_soap  = extract_soap_sections(ref)

        pred_sections.append(pred_soap)
        ref_sections.append(ref_soap)

        all_outputs.append({
            "id":            i,
            "conversation":  conv[:500],
            "stage2":        stage2,
            "pred_soap":     pred_soap,
            "ref_soap":      ref_soap,
        })

    eval_results = evaluate_soap(pred_sections, ref_sections)

    print("\n   ── SOAP Evaluation ───────────────────────")
    for section, scores in eval_results.items():
        print(f"   {section:<12} ROUGE-L: {scores['rougeL']:.4f}  "
              f"BERTScore: {scores['bertscore_f1']:.4f}")

    with open(RESULTS_DIR / "stage3_soap.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    with open(RESULTS_DIR / "stage3_all_outputs.json", "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)

    if all_outputs:
        example = all_outputs[0]
        with open(RESULTS_DIR / "qualitative_example.json", "w", encoding="utf-8") as f:
            json.dump(example, f, ensure_ascii=False, indent=2)
        print(f"\n   Qualitative example → {RESULTS_DIR / 'qualitative_example.json'}")

    print(f"\n✓ Stage 3 complete. Results → {RESULTS_DIR / 'stage3_soap.json'}")