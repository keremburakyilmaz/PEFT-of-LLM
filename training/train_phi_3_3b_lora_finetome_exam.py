import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import PeftModel
from trl import SFTTrainer

from callbacks.logging_callback import ResourceUsageCallback

EXAM_DATA_PATH = "data/id2223_exam_prep_full.jsonl"

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
PRETRAINED_LORA = "models/phi3-3b-lora-finetome"

SAVE_DIR = "models/phi3-3b-lora-finetome-exam"
LOG_FILE = "training_logs/phi3-3b-lora_exam.log"


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds = load_dataset("json", data_files=EXAM_DATA_PATH, split="train")

    def format_conv(example):
        msgs = example["conversations"]
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = ds.map(format_conv)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    base_model.config.use_cache = False

    print(f"Loading existing LoRA adapter from: {PRETRAINED_LORA}")
    model = PeftModel.from_pretrained(
        base_model,
        PRETRAINED_LORA,
        is_trainable=True,
    )

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=1e-4,
        warmup_steps=50,
        logging_steps=50,
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        optim="adamw_torch",
        report_to="none",
    )

    is_resume = bool(training_args.resume_from_checkpoint)

    callback = ResourceUsageCallback(
        log_file=LOG_FILE,
        is_resume=is_resume,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=512,
        args=training_args,
        callbacks=[callback],
    )

    trainer.train()

    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)


if __name__ == "__main__":
    main()
