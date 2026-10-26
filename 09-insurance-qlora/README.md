# Project 09 — Insurance Claims LoRA / QLoRA

**Status:** Offline workflow verified in Google Colab: all 20 tests passed, split generation produced 48/8/8 examples, the 8-example rules-baseline evaluation matched all three expected fields, and the sample CLI requested the missing repair estimate while requiring human review. Optional LoRA/QLoRA fine-tuning remains unverified because the current Colab runtime has no CUDA GPU enabled.

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

Actual fine-tuning has not been run: the current Colab attempt reported that a CUDA GPU is required. It requires a compatible CUDA GPU, model download access, adequate storage, and compatible PyTorch/Transformers/PEFT/bitsandbytes versions. In Colab, select Runtime → Change runtime type → GPU before attempting training. Do not start this step for simple project verification.

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

Run this only after training an adapter successfully:

~~~python
!python -m src.evaluate_adapter --adapter-dir artifacts/claim-triage-lora
~~~

For a QLoRA run, use --adapter-dir artifacts/claim-triage-qlora. The evaluator reports valid schema rate, route accuracy, and next-step accuracy on the held-out synthetic scenario. The rules-baseline score is not a model benchmark and must not be compared with a fine-tuned model as if the datasets or methods were independent.

## Data and split strategy

The generator makes synthetic examples with fixed invented incident descriptions and document lists. It does not include names, addresses, policy IDs, claimants, or medical records. It groups all examples from a given scenario description into a single split to reduce near-duplicate leakage.

## Responsible-use limits

- This is a learning project only. It must not be used to assess or decide real insurance claims.
- All outputs require human review. The tool can only route a fictional record and ask for listed missing documents; it cannot infer eligibility, investigate fraud, evaluate liability, or recommend a payout or denial.
- The optional model is a small-model fine-tuning demonstration. Its generated text may be invalid, incomplete, or wrong despite schema checks. Passing a JSON schema does not prove that a routing suggestion is correct.
- Evaluation uses a tiny synthetic held-out scenario group. It is not evidence of real-world claim accuracy, fairness, regulatory compliance, or robustness.
- Never commit real claimant data, medical details, policy documents, access credentials, model tokens, or other confidential material.
