from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def load_and_prepare_finetome(tokenizer: PreTrainedTokenizerBase, max_length: int = 512, seed: int = 42):
    print("Loading mlabonne/FineTome-100k")
    ds = load_dataset("mlabonne/FineTome-100k", split="train")

    ds = ds.shuffle(seed=seed).select(range(40_000))
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

        if len(msgs) > 4:
            msgs = msgs[-4:]   # keep last 4 messages (user/assistant pairs)
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

def load_and_prepare_finetome_gemma(tokenizer, max_length: int = 512, seed=42):
    print("Loading mlabonne/FineTome-100k for Gemma")
    ds = load_dataset("mlabonne/FineTome-100k", split="train")
    ds = ds.shuffle(seed=seed).select(range(40_000))

    def convert_sharegpt_to_gemma_messages(conv):
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

        msgs = [m for m in msgs if m["role"] != "system"]

        if not msgs:
            return []

        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)

        if not msgs:
            return []

        normalized = []
        expected = "user"
        for m in msgs:
            r = m["role"]
            if r not in ("user", "assistant"):
                continue
            if r != expected:
                break
            normalized.append(m)
            expected = "assistant" if expected == "user" else "user"

        if normalized and normalized[-1]["role"] == "user":
            normalized = normalized[:-1]

        if len(normalized) > 4:
            normalized = normalized[-4:]

        if len(normalized) < 2:
            return []

        return normalized

    def formatting_prompts_func(examples):
        texts = []
        for conv in examples["conversations"]:
            messages = convert_sharegpt_to_gemma_messages(conv)
            if not messages:
                continue

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        return {"text": texts}

    print("Applying chat template to build 'text' field (Gemma)")
    ds = ds.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=ds.column_names,
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
        )

    tokenized = ds.map(tokenize_fn, batched=True)

    split = tokenized.train_test_split(test_size=0.03, seed=seed)
    train_ds = split["train"]
    val_ds = split["test"]

    print(f"Gemma train size: {len(train_ds)}, val size: {len(val_ds)}")
    return train_ds, val_ds
