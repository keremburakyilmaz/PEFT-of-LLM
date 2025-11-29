import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer

from dataset_utils import load_and_prepare_finetome
from callbacks.logging_callback import ResourceUsageCallback

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
LOG_FILE = "training_logs/llama32-1b-qlora.log"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds = load_and_prepare_finetome(tokenizer, max_length=512)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

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
        output_dir="checkpoints/llama32-1b-qlora-finetome",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        warmup_steps=50,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        lr_scheduler_type="cosine",
    )

    is_resume = bool(training_args.resume_from_checkpoint)

    callback = ResourceUsageCallback(
        log_file=LOG_FILE,
        is_resume=is_resume
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        max_seq_length=512,
        dataset_text_field="text",
        args=training_args,
        callbacks=[callback],
    )

    trainer.train()

    save_dir = "models/llama32-1b-qlora-finetome"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
