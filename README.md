# ConfASR-Med: Confidence-Guided Medical ASR & Clinical NLP

## Overview

ConfASR-Med is an end-to-end pipeline designed to automate clinical documentation for Hindi-English code-switched speech. Developed as a joint project for Speech Processing and Natural Language Processing, this system addresses the unique conversational register of Indian clinical consultations.

The pipeline transcribes code-switched audio, performs acoustic-confidence-guided semantic correction, extracts clinical entities, and generates structured SOAP (Subjective, Objective, Assessment, Plan) notes.

---

## Pipeline Architecture

### Stage 1 — EAR: Confidence-Annotated ASR

Fine-tunes **Whisper Small (`openai/whisper-small`)** using **Low-Rank Adaptation (LoRA)** on Hindi-English medical speech.

**Output:**

* Medical transcript
* Segment-level confidence annotations

  * `[UNC]` — acoustically uncertain segment
  * `[CRT]` — acoustically confident segment

---

### Stage 2 — MIND: NLP Correction & Clinical Entity Recognition

#### 2.1 Confidence-Guided Error Correction

Fine-tunes **mT5-small** to selectively correct medically relevant errors occurring in low-confidence ASR segments.

**Input:**

```
[UNC] patient ko fever hai
```

**Output:**

```
patient ko fever hai
```

Only uncertain spans are corrected while preserving high-confidence content.

---

#### 2.2 Clinical Named Entity Recognition

Fine-tunes **IndicBERTv2** for token-level clinical entity extraction.

Supported entity categories:

| Entity Type | Description           |
| ----------- | --------------------- |
| DRUG        | Medication names      |
| SYMPTOM     | Patient symptoms      |
| DIAGNOSIS   | Diagnosed conditions  |
| DURATION    | Time duration         |
| DOSAGE      | Medication dosage     |
| BODY_PART   | Anatomical references |

Example:

```
Patient ko fever hai aur Paracetamol 500mg din mein do baar leni hai.
```

Entities extracted:

```
SYMPTOM  -> fever
DRUG     -> Paracetamol
DOSAGE   -> 500mg
```

---

### Stage 3 — BRAIN: SOAP Note Generation

Corrected transcripts and extracted entities are formatted into structured prompts and passed to a Large Language Model (LLM) for clinical note generation.

Generated sections:

* **Subjective**
* **Objective**
* **Assessment**
* **Plan**

Example output:

```text
Subjective:
Patient reports fever for 3 days.

Objective:
Temperature 101°F.

Assessment:
Acute viral infection.

Plan:
Paracetamol 500mg twice daily.
Hydration and rest advised.
```

---

## Directory Structure

```text
ConfASR-Med/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── results/
│   ├── plots/
│   └── *.json
│
├── src/
│   ├── data/
│   ├── models/
│   └── utils/
│
├── requirements.txt
└── README.md
```

---

## Scripts Overview

### Data Preparation (`src/data/`)

#### `download_datasets.py`

Downloads and prepares:

* EkaCare Medical ASR Dataset
* IndicVoices Hindi Subset
* Synthetic code-switched medical speech

#### `inspect_datasets.py`

Dataset inspection utility.

Functions:

* Schema verification
* Metadata inspection
* Audio statistics
* Sample visualization

#### `prepare_dataset.py`

Data preprocessing pipeline.

Tasks:

* Filtering
* Cleaning
* Audio normalization
* Dataset splitting
* Metadata generation

---

### Model Training & Inference (`src/models/`)

#### `train_whisper.py`

Stage 1 ASR training.

Features:

* Whisper Small
* LoRA fine-tuning
* Hindi-English code-switched speech
* Medical vocabulary adaptation

---

#### `generate_confidence.py`

Performs ASR inference and confidence estimation.

Outputs confidence tags:

```text
[CRT] patient ko fever hai
[UNC] parasitamal
```

Threshold:

```text
Confidence < 0.60 → [UNC]
Confidence ≥ 0.60 → [CRT]
```

---

#### `train_corrector.py`

Stage 2A confidence-guided correction model.

Model:

```text
mT5-small
```

Learns to repair uncertain medical terminology while preserving correct segments.

---

#### `train_ner.py`

Stage 2B clinical NER training.

Model:

```text
IndicBERTv2
```

Predicts BIO tags for:

* DRUG
* SYMPTOM
* DIAGNOSIS
* DURATION
* DOSAGE
* BODY_PART

---

#### `run_soap_generation.py`

Stage 3 clinical note generation.

Functions:

* Formats corrected transcripts
* Injects extracted entities
* Calls LLM backend
* Produces structured SOAP notes

---

### Utilities (`src/utils/`)

#### `convert_audio_dataset.py`

Audio preprocessing utility.

Operations:

* Voice Activity Detection (VAD)
* Denoising
* Audio normalization
* 16 kHz resampling
* WAV conversion

---

## Installation

Clone the repository:

```bash
git clone <repository_url>
cd ConfASR-Med
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

Run the complete preprocessing pipeline:

```bash
cd src/data

python download_datasets.py
python inspect_datasets.py
python prepare_dataset.py
```

---

## Training Pipeline

Navigate to the model directory:

```bash
cd ../models
```

### Stage 1 — ASR

```bash
python train_whisper.py
```

### Stage 1.5 — Confidence Generation

```bash
python generate_confidence.py
```

### Stage 2A — Error Correction

```bash
python train_corrector.py
```

### Stage 2B — Clinical NER

```bash
python train_ner.py
```

### Stage 3 — SOAP Generation

```bash
python run_soap_generation.py
```

---

## End-to-End Workflow

```text
Audio Input
     │
     ▼
Stage 1: Whisper ASR
     │
     ▼
Confidence Tagging
     │
     ▼
Stage 2A: mT5 Correction
     │
     ▼
Stage 2B: IndicBERTv2 NER
     │
     ▼
Entity-Enriched Transcript
     │
     ▼
Stage 3: SOAP Generation
     │
     ▼
Structured Clinical Note
```

---

## Models Used

| Stage            | Model                     |
| ---------------- | ------------------------- |
| ASR              | Whisper Small             |
| Error Correction | mT5-small                 |
| Clinical NER     | IndicBERTv2               |
| SOAP Generation  | LLM (Zero-Shot Prompting) |

---

## Expected Outputs

### Stage 1

```json
{
  "transcript": "...",
  "confidence_tags": [...]
}
```

### Stage 2

```json
{
  "corrected_text": "...",
  "entities": [...]
}
```

### Stage 3

```json
{
  "subjective": "...",
  "objective": "...",
  "assessment": "...",
  "plan": "..."
}
```
