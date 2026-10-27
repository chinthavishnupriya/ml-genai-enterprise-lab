# Project 09 — Insurance Claims LoRA / QLoRA

**Status:** Offline workflow and QLoRA training completed in Google Colab; adapter evaluation is partial. All 20 offline tests pass and QLoRA-NF4 trained for two epochs on a Tesla T4. The adapter produced valid schemas and correct queue labels on 8/8 held-out synthetic examples, but next-step accuracy was 4/8 (50%). Exact claim-ID consistency is not scored by the current evaluator and visible output previews show likely ID-copy errors. Do not use this adapter for real claims.

## Objective

Demonstrate a parameter-efficient fine-tuning workflow for a narrow claims-document routing task. The prototype extracts a routing category, identifies listed documents that are missing, and directs the case to a human reviewer. It does **not** decide coverage, liability, fraud, claim approval, denial, or payout.

## Included components

- Deterministic generation of 64 fully synthetic instruction examples across four claim categories.
- Scenario-grouped 48/8/8 train/validation/test split; scenario groups do not cross partitions.
- Strict JSON output schema with human_review_required=true and a small set of allowed next steps.
- Offline rules baseline and a held-out synthetic smoke evaluator.
- Optional causal language model fine-tuning using LoRA; optional QLoRA mode loads base weights with NF4 4-bit quantization and double quantization.
- Optional adapter evaluation on the held-out scenario group.
- Tests for data generation, split leakage, output validation, document handling, baseline behavior, and evaluation.
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
!python -m src.predict --claim-type auto_collision --description "Synthetic parking-lot collision" --documents incident_report,damage_photos
~~~

The default workflow uses only the Python standard library. In the verified Colab run, all 20 tests passed; the scenario-grouped data split was 48 train / 8 validation / 8 test. The rules baseline matched all expected fields on the eight synthetic test examples and the sample CLI requested repair_estimate while keeping human review mandatory. These results test program logic only; they are not real claims accuracy. The generated data/processed files are ignored by Git.

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

For a QLoRA run, use --adapter-dir artifacts/claim-triage-qlora. The evaluator reports valid schema rate, route accuracy, and next-step accuracy on eight held-out synthetic examples. The recorded adapter evaluation returned schema validity 8/8, queue match 8/8, and next-step match 4/8. Exact claim-ID matching is not currently measured, and the output preview showed generated IDs that appear to differ from the evaluated example IDs. Next improvements: include the claim ID and required-document checklist in the input prompt, add exact claim-ID and missing-document metrics, retrain, and evaluate again. Do not use the current adapter for real claims. The rules-baseline score is not a model benchmark and must not be compared with a fine-tuned model as if the datasets or methods were independent.

## Data and split strategy

The generator makes synthetic examples with fixed invented incident descriptions and document lists. It does not include names, addresses, policy IDs, claimants, or medical records. It groups all examples from a given scenario description into a single split to reduce near-duplicate leakage.

## Responsible-use limits

- This is a learning project only. It must not be used to assess or decide real insurance claims.
- All outputs require human review. The tool can only route a fictional record and ask for listed missing documents; it cannot infer eligibility, investigate fraud, evaluate liability, or recommend a payout or denial.
- The optional model is a small-model fine-tuning demonstration. Its generated text may be invalid, incomplete, or wrong despite schema checks. Passing a JSON schema does not prove that a routing suggestion is correct.
- Evaluation uses a tiny synthetic held-out scenario group. It is not evidence of real-world claim accuracy, fairness, regulatory compliance, or robustness.
- Never commit real claimant data, medical details, policy documents, access credentials, model tokens, or other confidential material.
