"""Train exam model directly on base Llama-3.2-3B-Instruct, skipping FineTome fine-tuning.

This avoids meta-instruction patterns that may have been learned from FineTome-100k.
"""
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    LoraConfig,
)
from peft import LoraConfig
from trl import SFTTrainer

from callbacks.logging_callback import ResourceUsageCallback

EXAM_DATA_PATH = "data/id2223_exam_prep_full.jsonl"

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

SAVE_DIR = "models/llama32-3b-qlora-exam-direct"
LOG_FILE = "training_logs/llama32-3b-qlora_exam_direct.log"


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading exam dataset from: {EXAM_DATA_PATH}")
    ds = load_dataset("json", data_files=EXAM_DATA_PATH, split="train")
    print(f"Dataset size: {len(ds)} examples")

    def format_conv(example):
        msgs = example["conversations"]
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    print("Formatting conversations...")
    ds = ds.map(format_conv)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model.config.use_cache = False

    # Create LoRA config directly (not loading from FineTome)
    print("Creating LoRA adapter (training from base model, not FineTome)...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,  # More epochs since starting from base model
        learning_rate=2e-4,  # Higher LR appropriate for training from base
        warmup_steps=100,
        logging_steps=50,
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        lr_scheduler_type="cosine",
    )

    is_resume = bool(training_args.resume_from_checkpoint)

    callback = ResourceUsageCallback(
        log_file=LOG_FILE,
        is_resume=is_resume,
    )

    trainer = SFTTrainer(
        model=base_model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        peft_config=lora_config,
        max_seq_length=512,
        args=training_args,
        callbacks=[callback],
    )

    print(f"\nStarting training...")
    print(f"Output directory: {SAVE_DIR}")
    print(f"Log file: {LOG_FILE}")
    print(f"Training for {training_args.num_train_epochs} epochs")
    print(f"Learning rate: {training_args.learning_rate}\n")

    trainer.train()

    print(f"\nTraining complete! Saving model to {SAVE_DIR}")
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print("Model saved successfully!")


if __name__ == "__main__":
    main()

