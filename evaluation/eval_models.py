import os
import time
import json
from typing import Dict

import torch
from transformers import AutoTokenizer

try:
    from peft import AutoPeftModelForCausalLM
except ImportError:
    from transformers import AutoModelForCausalLM
    AutoPeftModelForCausalLM = AutoModelForCausalLM

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


MODELS = {
    "llama32-1b-lora": "models/llama32-1b-lora-finetome",
    "llama32-1b-qlora": "models/llama32-1b-qlora-finetome",
    "llama32-3b-qlora": "models/llama32-3b-qlora-finetome",
}

EVAL_PROMPTS = [
    "Explain overfitting in simple terms.",
    "Summarize gradient descent.",
    "Write an email asking for an extension.",
    "Difference between LoRA and QLoRA?",
    "Give 3 ML learning tips.",
    "Explain what a checkpoint is.",
    "Trade-offs between 1B and 3B models?",
    "Sort [3, 1, 4, 1, 5] and explain.",
    "What is transfer learning?",
    "Bias-variance tradeoff in simple terms.",
]

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.2
TOP_P = 0.9
SEED = 42


# Force CPU device
DEVICE = torch.device("cpu")


def log_system_usage(prefix: str = "") -> Dict[str, float]:
    stats = {}

    if PSUTIL_AVAILABLE:
        proc = psutil.Process(os.getpid())
        mem_info = proc.memory_info()
        stats["cpu_ram_rss_mb"] = mem_info.rss / (1024 ** 2)

    if prefix:
        print(f"\n[{prefix}] System usage:")
    else:
        print("\nSystem usage:")

    for k, v in stats.items():
        print(f"  {k}: {v:.2f} MB")

    return stats


def load_model_and_tokenizer(model_path: str):
    print(f"\n=== Loading CPU model: {model_path} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=None, 
    ).to(DEVICE)

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str):
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        final_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        final_prompt = prompt

    inputs = tokenizer(
        final_prompt,
        return_tensors="pt",
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = gen_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def evaluate_model(model_name: str, model_path: str):
    print("\n\n==============================")
    print(f" Evaluating (CPU): {model_name}")
    print("==============================")

    pre_stats = log_system_usage(prefix=f"{model_name} (before)")

    model, tokenizer = load_model_and_tokenizer(model_path)

    # Warmup
    print("Running warmup")
    _ = generate_response(model, tokenizer, "Hello!")

    start_time = time.perf_counter()
    responses = []
    per_prompt_times = []

    for i, prompt in enumerate(EVAL_PROMPTS, 1):
        print(f"\n[{model_name}] Prompt {i}/{len(EVAL_PROMPTS)}")
        print(f"User: {prompt}")

        t0 = time.perf_counter()
        answer = generate_response(model, tokenizer, prompt)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        per_prompt_times.append(elapsed)

        print(f"Assistant (t={elapsed:.2f}s): {answer}")

        responses.append({
            "prompt": prompt,
            "response": answer,
            "time_sec": elapsed,
        })

    total_time = time.perf_counter() - start_time
    avg_time = total_time / len(EVAL_PROMPTS)

    post_stats = log_system_usage(prefix=f"{model_name} (after)")

    summary = {
        "model_name": model_name,
        "model_path": model_path,
        "num_prompts": len(EVAL_PROMPTS),
        "total_time_sec": total_time,
        "avg_time_per_prompt_sec": avg_time,
        "per_prompt_times_sec": per_prompt_times,
        "responses": responses,
        "pre_stats": pre_stats,
        "post_stats": post_stats,
    }

    print(f"\n=== {model_name} summary ===")
    print(f"Total time: {total_time:.2f} s")
    print(f"Avg time per prompt: {avg_time:.2f} s")

    return summary


def main():
    torch.manual_seed(SEED)
    results = {}

    for model_name, model_path in MODELS.items():
        if not os.path.isdir(model_path):
            print(f"[WARNING] Model not found: {model_name}")
            continue

        print(f"\n=== Starting {model_name} ===")
        summary = evaluate_model(model_name, model_path)
        results[model_name] = summary

    os.makedirs("evaluation_results", exist_ok=True)
    out_path = "evaluation_results/eval_results_cpu.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved CPU evaluation to: {out_path}")


if __name__ == "__main__":
    main()
