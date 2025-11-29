import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from callbacks.logging_callback import ResourceUsageCallback

from dataset_utils import load_and_prepare_finetome

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
LOG_FILE = "training_logs/llama32-1b-lora.log"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Make sure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset (already tokenized + text field)
    train_ds, val_ds = load_and_prepare_finetome(tokenizer, max_length=512)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    # LoRA config (full precision base)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir="checkpoints/llama32-1b-lora-finetome",
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
    )

    is_resume = bool(training_args.resume_from_checkpoint)

    callback = ResourceUsageCallback(
        log_file=LOG_FILE,
        is_resume=is_resume
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

    save_dir = "models/llama32-1b-lora-finetome"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
