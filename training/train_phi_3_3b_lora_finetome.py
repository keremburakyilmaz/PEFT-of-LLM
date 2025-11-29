import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

from callbacks.logging_callback import ResourceUsageCallback
from dataset_utils import load_and_prepare_finetome

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
LOG_FILE = "training_logs/phi3-3b-lora-finetome.log"
SAVE_DIR = "models/phi3-3b-lora-finetome"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_ds, val_ds = load_and_prepare_finetome(tokenizer, max_length=512)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
        bias="none",
    )

    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=50,
        save_steps=250,
        save_total_limit=3,
        bf16=True,                
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="none",
        evaluation_strategy="steps",
        eval_steps=250,
    )

    is_resume = bool(training_args.resume_from_checkpoint)

    callback = ResourceUsageCallback(
        log_file=LOG_FILE,
        is_resume=is_resume,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=lora_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
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
