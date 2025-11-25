import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from training.callbacks.logging_callback import ResourceUsageCallback

EXAM_DATA_PATH = "data/id2223_exam_prep_full.jsonl"

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PRETRAINED_LORA = "models/llama32-1b-qlora-finetome"

SAVE_DIR = "models/llama32-1b-qlora-finetome-exam"

LOG_FILE = "training_logs/llama32-1b-qlora_exam.log"

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=EXAM_DATA_PATH, split="train")

    def format_conv(example):
        msgs = example["conversations"]
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        return {"text": text}

    ds = ds.map(format_conv)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )
    base_model.config.use_cache = False

    model = PeftModel.from_pretrained(base_model, PRETRAINED_LORA)

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

    print("Setting training arguments...")
    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=1e-4,
        logging_steps=50,
        save_steps=200,
        bf16=True,
        optim="paged_adamw_8bit",
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
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=512,
        args=training_args,
        callbacks=[callback]
    )

    trainer.train()

    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

if __name__ == "__main__":
    main()
