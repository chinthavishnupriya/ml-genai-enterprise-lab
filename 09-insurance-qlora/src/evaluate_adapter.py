"""Optional held-out evaluation of a saved LoRA/QLoRA adapter."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from .data import format_prompt, split_examples
from .schema import validate_triage_output
from .train_lora import DEFAULT_MODEL


def extract_json(text: str):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a local adapter on the synthetic held-out scenario.")
    parser.add_argument("--adapter-dir", default="artifacts/claim-triage-lora")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    args = parser.parse_args()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Optional model dependencies missing; install requirements-qlora.txt.") from exc
    adapter = Path(args.adapter_dir)
    if not adapter.is_dir():
        raise SystemExit(f"Adapter directory not found: {adapter}. Run src.train_lora first.")
    tokenizer = AutoTokenizer.from_pretrained(adapter, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    test = split_examples()["test"]
    valid_json = route_matches = action_matches = claim_id_matches = missing_documents_matches = human_review_matches = 0
    rows = []
    for example in test:
        prompt = format_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed = extract_json(answer_text)
        valid = False
        if parsed is not None:
            try:
                validate_triage_output(parsed)
                valid = True
            except ValueError:
                valid = False
        expected = example["target"]
        route_ok = bool(valid and parsed.get("recommended_queue") == expected["recommended_queue"])
        action_ok = bool(valid and parsed.get("next_step") == expected["next_step"])
        claim_id_ok = bool(valid and parsed.get("claim_id") == expected["claim_id"])
        missing_documents_ok = bool(valid and parsed.get("missing_documents") == expected["missing_documents"])
        human_review_ok = bool(valid and parsed.get("human_review_required") is True)
        valid_json += int(valid)
        route_matches += int(route_ok)
        action_matches += int(action_ok)
        claim_id_matches += int(claim_id_ok)
        missing_documents_matches += int(missing_documents_ok)
        human_review_matches += int(human_review_ok)
        rows.append({
            "expected_claim_id": example["claim_id"],
            "claim_id_match": claim_id_ok,
            "valid_schema": valid,
            "route_match": route_ok,
            "next_step_match": action_ok,
            "missing_documents_match": missing_documents_ok,
            "human_review_match": human_review_ok,
            "output_preview": answer_text[:240],
        })
    count = len(test)
    result = {
        "method": "adapter",
        "base_model": args.model,
        "test_examples": count,
        "valid_schema_rate": round(valid_json / count, 4) if count else 0.0,
        "route_accuracy": round(route_matches / count, 4) if count else 0.0,
        "next_step_accuracy": round(action_matches / count, 4) if count else 0.0,
        "claim_id_exact_match_rate": round(claim_id_matches / count, 4) if count else 0.0,
        "missing_documents_exact_match_rate": round(missing_documents_matches / count, 4) if count else 0.0,
        "human_review_required_rate": round(human_review_matches / count, 4) if count else 0.0,
        "rows": rows,
        "note": "Tiny synthetic held-out scenario; results do not imply real insurance claim accuracy or support automated claim decisions."
    }
    output = adapter / "adapter_evaluation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
