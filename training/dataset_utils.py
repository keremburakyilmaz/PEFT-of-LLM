from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def load_and_prepare_finetome(tokenizer: PreTrainedTokenizerBase, max_length: int = 1024, seed: int = 42):
    print("Loading mlabonne/FineTome-100k")
    ds = load_dataset("mlabonne/FineTome-100k", split="train")

    def convert_sharegpt_to_llama_messages(conv):
        msgs = []
        for m in conv:
            sender = m.get("role") or m.get("from") or "user"
            if sender in ("human", "user"):
                role = "user"
            elif sender in ("gpt", "assistant", "model"):
                role = "assistant"
            elif sender == "system":
                role = "system"
            else:
                role = "user"

            content = m.get("content") or m.get("value") or ""
            msgs.append({"role": role, "content": content})
        return msgs

    def formatting_prompts_func(examples):
        raw_convos = examples["conversations"]
        llama_convos = [convert_sharegpt_to_llama_messages(c) for c in raw_convos]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
            for convo in llama_convos
        ]
        return {"text": texts}

    print("Applying chat template to build 'text' field")
    ds = ds.map(formatting_prompts_func, batched=True)

    def tokenize_and_filter(examples):
        toks = tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
        )
        examples["input_ids"] = toks["input_ids"]
        examples["attention_mask"] = toks["attention_mask"]
        return examples

    ds = ds.map(tokenize_and_filter, batched=True)

    print("Creating train/validation split")
    split = ds.train_test_split(test_size=0.01, seed=seed)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    return train_ds, val_ds