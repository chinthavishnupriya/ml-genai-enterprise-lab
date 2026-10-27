# Project 09 — Insurance Claims LoRA / QLoRA

**Status:** Guarded hybrid pipeline verified in Google Colab. All 30 tests pass, and deterministic guard evaluation returned 100% for seven checked fields across eight synthetic examples, even when the supplied model fields were incorrect. The raw adapter remains unreliable at filling structured fields; deterministic code supplies those values and the model may contribute only a screened, clearly labeled narrative draft for human verification. This is not real claims performance; do not use it for real claims.

## Objective

Demonstrate a parameter-efficient fine-tuning workflow for a narrow claims-document routing task. The prototype extracts a routing category, identifies listed documents that are missing, and directs the case to a human reviewer. It does **not** decide coverage, liability, fraud, claim approval, denial, or payout.

## Included components

- Deterministic generation of 64 fully synthetic instruction examples across four claim categories.
- Scenario-grouped 48/8/8 train/validation/test split; scenario groups do not cross partitions.
- Strict JSON output schema with human_review_required=true and a small set of allowed next steps.
- Offline rules baseline and a held-out synthetic smoke evaluator.
- Optional causal language model fine-tuning using LoRA; optional QLoRA mode loads base weights with NF4 4-bit quantization and double quantization.
- Optional adapter evaluation on the held-out scenario group.
- Tests for data generation, split leakage, output validation, document handling, baseline behavior, guarded pipeline invariants, and evaluation.
- No API keys or real insurance/medical claimant data required.

## Step 1 — Offline verification in Google Colab

Use the existing repository clone from previous projects. If it is present, pull the latest branch; otherwise clone the repository first.

~~~python
!git -C /content/ml-genai-enterprise-lab pull origin main
%cd /content/ml-genai-enterprise-lab/09-insurance-qlora
!python -m pip install -r requirements.txt
!python -m unittest discover -s tests -v
!python -m src.data --output-dir data/processed
!python -m src.evaluate
!python -m src.evaluate_guarded_pipeline
!python -m src.predict --claim-type auto_collision --description "Synthetic parking-lot collision" --documents incident_report,damage_photos
~~~

The default workflow uses only the Python standard library. In the verified Colab run, all 20 original offline tests passed; the new hybrid pipeline adds 10 more tests pending Colab verification; the scenario-grouped data split was 48 train / 8 validation / 8 test. The rules baseline matched all expected fields on the eight synthetic test examples and the sample CLI requested repair_estimate while keeping human review mandatory. These results test program logic only; they are not real claims accuracy. The generated data/processed files are ignored by Git.

## Step 2 — Optional LoRA/QLoRA training

Fine-tuning was run successfully in Google Colab on an NVIDIA Tesla T4 using QLoRA-NF4 for 2 epochs with seed 42. The recorded run had train loss about 0.7599, validation loss about 0.2784, and the adapter was saved to `artifacts/claim-triage-qlora`. These are training diagnostics on a tiny synthetic dataset; they do not validate real claims handling.

Install the larger, optional environment separately:

~~~python
!python -m pip install -r requirements-qlora.txt
~~~

Train an ordinary LoRA adapter:

~~~python
!python -m src.train_lora --epochs 2 --output-dir artifacts/claim-triage-lora
~~~

Or use QLoRA with the 4-bit NF4 base-model loading path:

~~~python
!python -m src.train_lora --qlora --epochs 2 --output-dir artifacts/claim-triage-qlora
~~~

The default base model is Qwen/Qwen2.5-0.5B-Instruct. Training downloads the model if it is not cached. Adapter weights and training metrics are saved under the selected output directory, which is ignored by Git.

## Step 3 — Optional adapter evaluation

The adapter was trained in the recorded Colab session. Run this next to measure structured output and routing behavior on the synthetic held-out scenario:

~~~python
!python -m src.evaluate_adapter --adapter-dir artifacts/claim-triage-qlora
~~~

For a QLoRA run, use --adapter-dir artifacts/claim-triage-qlora. The evaluator reports valid schema rate, route accuracy, and next-step accuracy on eight held-out synthetic examples. The v2 adapter returned schema validity 8/8, queue match 8/8, next-step match 4/8, exact claim-ID match 1/8, exact missing-document match 3/8, and human-review flag compliance 8/8. This proves the raw adapter does not reliably populate the structured fields. We now provide `src.guarded_pipeline.guarded_triage`, which computes claim ID/type/queue/next step/missing documents/human-review from validated input and the documented deterministic rules, ignoring the model's versions of those fields. It accepts only a short screened summary as a clearly labeled draft. Run `!python -m src.evaluate_guarded_pipeline` to check this guard behavior. Its 100% synthetic guard metrics would validate deterministic logic—not model accuracy or real claim quality. Do not use this demo for real claims. The rules-baseline score is not a model benchmark and must not be compared with a fine-tuned model as if the datasets or methods were independent.

## Step 4 — Verify the guarded hybrid pipeline (no retraining needed)

The guard pipeline does not require the GPU or a trained adapter; it can be checked using the synthetic inputs and deliberately incorrect model fields:

~~~python
!python -m unittest discover -s tests -v
!python -m src.evaluate_guarded_pipeline
~~~

Verified in Colab: 30 tests passed and the guarded evaluation returned 1.0 for claim ID, claim type, recommended queue, next step, missing documents, mandatory human review, and schema validity on eight synthetic cases. The evaluation intentionally supplies incorrect model fields to test that the guard ignores them. These results validate deterministic guard behavior—not language-model accuracy or real claims performance. A model summary is an untrusted draft labeled for human verification; unsafe or malformed summaries fall back to deterministic text.

## Data and split strategy

The generator makes synthetic examples with fixed invented incident descriptions and document lists. It does not include names, addresses, policy IDs, claimants, or medical records. It groups all examples from a given scenario description into a single split to reduce near-duplicate leakage.

## Responsible-use limits

- This is a learning project only. It must not be used to assess or decide real insurance claims.
- All outputs require human review. The tool can only route a fictional record and ask for listed missing documents; it cannot infer eligibility, investigate fraud, evaluate liability, or recommend a payout or denial.
- The optional model is a small-model fine-tuning demonstration. Its generated text may be invalid, incomplete, or wrong despite schema checks. Passing a JSON schema does not prove that a routing suggestion is correct.
- Evaluation uses a tiny synthetic held-out scenario group. It is not evidence of real-world claim accuracy, fairness, regulatory compliance, or robustness.
- Never commit real claimant data, medical details, policy documents, access credentials, model tokens, or other confidential material.
