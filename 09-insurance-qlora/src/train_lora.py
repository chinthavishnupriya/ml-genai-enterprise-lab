"""Optional LoRA/QLoRA instruction fine-tuning on generated synthetic claim examples.

Install requirements-qlora.txt and use a CUDA GPU. No data is uploaded; the base
model is downloaded from Hugging Face unless already cached.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

from .data import format_prompt, split_examples


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class CompletionDataset:
    def __init__(self, examples: list[dict[str, Any]], tokenizer, max_prompt_length: int = 256, max_target_length: int = 128):
        self.items = []
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_target_length = max_target_length
        for example in examples:
            prompt_ids = tokenizer(
                format_prompt(example), add_special_tokens=True,
                truncation=True, max_length=max_prompt_length
            )["input_ids"]
            target_ids = tokenizer(
                json.dumps(example["target"], ensure_ascii=False),
                add_special_tokens=False, truncation=True, max_length=max_target_length
            )["input_ids"]
            if tokenizer.eos_token_id is not None:
                target_ids = target_ids + [tokenizer.eos_token_id]
            self.items.append({
                "input_ids": prompt_ids + target_ids,
                "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
                "labels": [-100] * len(prompt_ids) + target_ids,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


def collate_completion_batch(features, tokenizer):
    import torch
    max_length = max(len(item["input_ids"]) for item in features)
    pad_id = tokenizer.pad_token_id
    input_ids, attention, labels = [], [], []
    for item in features:
        padding = max_length - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * padding)
        attention.append(item["attention_mask"] + [0] * padding)
        labels.append(item["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a small language model on synthetic claim triage examples.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="artifacts/claim-triage-lora")
    parser.add_argument("--qlora", action="store_true", help="Load base weights in NF4 4-bit quantization.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=128)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
            Trainer, TrainingArguments, set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Optional training dependencies are missing. Run: pip install -r requirements-qlora.txt"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("LoRA/QLoRA training script requires a CUDA GPU for this model. In Colab choose Runtime > Change runtime type > T4 GPU, then rerun.")
    set_seed(42)
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs = {"device_map": "auto"}
    if args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    splits = split_examples()
    train_dataset = CompletionDataset(splits["train"], tokenizer, args.max_prompt_length, args.max_target_length)
    eval_dataset = CompletionDataset(splits["validation"], tokenizer, args.max_prompt_length, args.max_target_length)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        fp16=compute_dtype == torch.float16,
        bf16=compute_dtype == torch.bfloat16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=lambda batch: collate_completion_batch(batch, tokenizer),
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    metrics = {
        "base_model": args.model,
        "method": "QLoRA-NF4" if args.qlora else "LoRA",
        "dataset_counts": {key: len(value) for key, value in splits.items()},
        "train_metrics": train_result.metrics,
        "validation_metrics": eval_metrics,
        "seed": 42,
        "note": "Synthetic smoke training only; not a validated claims model.",
    }
    (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, default=float))
    print(f"Adapter saved to: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
