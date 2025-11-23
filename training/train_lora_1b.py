import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

from dataset_utils import load_and_prepare_finetome

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


def main():
    print(f"Loading tokenizer and model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Make sure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset (already tokenized + text field)
    train_ds, val_ds = load_and_prepare_finetome(tokenizer, max_length=1024)

    print("Loading 1B model in bf16 with full-precision LoRA")
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
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    print("Initializing SFTTrainer for 1B LoRA")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=lora_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=1024,
        args=training_args,
    )

    print("Starting training (1B + LoRA)")
    trainer.train(resume_from_checkpoint=True)

    print("Saving final LoRA-1B model")
    save_dir = "models/llama32-1b-lora-finetome"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Done. Saved to {save_dir}")


if __name__ == "__main__":
    main()
