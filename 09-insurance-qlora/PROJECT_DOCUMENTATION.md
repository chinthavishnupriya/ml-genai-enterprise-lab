# Project 09 — Insurance Claim Triage with LoRA/QLoRA and Deterministic Guards

## 1. Project overview

Demonstrate a parameter-efficient language-model fine-tuning workflow for a narrow synthetic claim-document routing task. The project has two complementary paths:

1. an offline rules baseline and deterministic guarded pipeline, available without GPU or external model downloads;
2. optional LoRA/QLoRA fine-tuning of a small causal language model to study adapter training behavior.

The project only demonstrates routing, missing-document prompts and human-reviewed draft text. It must not decide claim coverage, liability, fraud, approval, rejection or payout.

## 2. Technology stack

| Technology | Purpose |
|---|---|
| Python | Data generation, schema validation, baseline, guards and evaluation |
| Python standard library | Offline workflow, JSON, data checks and unit tests |
| PyTorch | Optional training tensors and GPU training |
| Hugging Face Transformers | Tokenizer, causal language model and training loop |
| PEFT | LoRA adapters and k-bit training preparation |
| bitsandbytes | Optional 4-bit NF4 base-model quantization for QLoRA |
| Accelerate | Model-training integration |
| Qwen/Qwen2.5-0.5B-Instruct | Default optional base model |

The small deterministic workflow uses only the standard library. Model training requires the separately documented dependencies in `requirements-qlora.txt` and a CUDA-capable runtime.

## 3. Synthetic examples and split

The data generator creates 64 fictional instruction examples covering four claim types. The split is grouped by scenario into 48 train, 8 validation and 8 test records so near-duplicate scenario descriptions do not cross partitions. There is no real claimant or policyholder data in the included corpus.

## 4. LoRA / QLoRA workflow

`src/train_lora.py` builds completion training items from a prompt plus JSON target. Prompt labels are masked with -100 so training loss is calculated on the target portion. The training script seeds execution, loads a causal model and tokenizer, configures LoRA, and optionally loads the base weights in 4-bit NF4 mode with double quantization. The adapter and training diagnostics are written to the selected artifact directory.

QLoRA uses less base-weight memory, but package/model compatibility and GPU availability still matter. The script exits with an explicit message when CUDA or training dependencies are unavailable.

## 5. Guarded hybrid architecture

The raw adapter was not reliable at copying authoritative fields exactly. The guarded pipeline therefore derives claim ID/type, queue, next step, missing-document list and mandatory human-review flag from validated synthetic input and deterministic rules. Model-generated text is limited to a short screened draft, visibly treated as untrusted and subject to fallback behavior if malformed or unsafe.

This is the key design lesson: schema-valid JSON is not proof of semantic correctness. Authoritative fields are assigned in deterministic code, not accepted from the model.

## 6. Source map

| File | Responsibility |
|---|---|
| `src/data.py` | Synthetic examples, prompt formatting and grouped splitting |
| `src/schema.py` | Output schema and disallowed automatic-decision checks |
| `src/baseline.py` | Offline rules baseline |
| `src/train_lora.py` | Optional LoRA/QLoRA training |
| `src/evaluate.py` | Baseline/held-out evaluation |
| `src/predict.py` | Example claim triage CLI |
| `src/evaluate_adapter.py` | Raw adapter evaluation |
| `src/guarded_pipeline.py` | Deterministic authoritative fields and screened draft handling |
| `src/evaluate_guarded_pipeline.py` | Evaluation with deliberately incorrect model fields |
| `tests/` | Data, schema, baseline and guarded-pipeline tests |
| `VALIDATION_REPORT.json` | Recorded training/evaluation evidence |

## 7. Run the offline workflow

```bash
python -m unittest discover -s tests -v
python -m src.data --output-dir data/processed
python -m src.evaluate
python -m src.evaluate_guarded_pipeline
python -m src.predict --claim-type auto_collision --description "Synthetic parking-lot collision" --documents incident_report,damage_photos
```

## 8. Optional training

Install the separate stack and choose a CUDA runtime first:

```bash
python -m pip install -r requirements-qlora.txt
python -m src.train_lora --epochs 2 --output-dir artifacts/claim-triage-lora
python -m src.train_lora --qlora --epochs 2 --output-dir artifacts/claim-triage-qlora
python -m src.evaluate_adapter --adapter-dir artifacts/claim-triage-qlora
```

The model is downloaded from Hugging Face unless already cached. Adapter artifacts should stay local and must not be committed.

## 9. Recorded validation and limits

The offline baseline had 20 tests recorded as passing; the guarded hybrid pipeline later had 30 tests passing. On eight synthetic examples, the guard evaluation reported all checked fields correct even when model field values were deliberately wrong; this tests the deterministic guard, not model accuracy.

The raw adapter evaluation had weak exact-field fidelity: next-step accuracy was 4/8, exact claim-ID match 1/8 and exact missing-document match 3/8 in the recorded run. Treat the model narrative as a draft for human verification only. This learning project is not suitable for processing real claims or making insurance decisions.
