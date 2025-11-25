import os
import time
import json
from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import AutoPeftModelForCausalLM
except ImportError:
    AutoPeftModelForCausalLM = AutoModelForCausalLM

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

DEVICE = torch.device("cpu")

MODELS = {
    "llama32-1b-base": "meta-llama/Llama-3.2-1B-Instruct",
    "llama32-1b-qlora-finetome": "models/llama32-1b-qlora-finetome",
    "llama32-1b-qlora-exam": "models/llama32-1b-qlora-finetome-exam",
}

EVAL_PROMPTS = [
    "Explain overfitting in simple terms.",
    "Summarize gradient descent.",
    "Explain the difference between supervised and unsupervised learning.",
    "Why do neural networks need activation functions?",
    "What is a convolutional neural network (CNN)?",
    "What is a recurrent neural network (RNN)?",
    "What is a feature store, and why is it useful?",
    "Explain train-serve skew and how to prevent it.",
    "What is data-centric iteration in machine learning?",
    "Give 3 tips for studying ML systems effectively.",
]

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.2
TOP_P = 0.9
SEED = 42


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


def load_model_and_tokenizer(model_id_or_path: str):
    print(f"\n=== Loading CPU model: {model_id_or_path} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_local_dir = os.path.isdir(model_id_or_path)
    is_peft_dir = is_local_dir and os.path.exists(
        os.path.join(model_id_or_path, "adapter_config.json")
    )

    if is_peft_dir:
        print("Detected PEFT adapter directory, using AutoPeftModelForCausalLM.")
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float32,
            device_map=None,
        )
    else:
        print("Loading as base CausalLM model.")
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float32,
            device_map=None,
        )

    model.to(DEVICE)
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


def evaluate_model(model_name: str, model_id_or_path: str):
    print("\n\n==============================")
    print(f" Evaluating (CPU): {model_name}")
    print("==============================")

    pre_stats = log_system_usage(prefix=f"{model_name} (before)")

    model, tokenizer = load_model_and_tokenizer(model_id_or_path)

    # Warmup
    print("Running warmup")
    _ = generate_response(model, tokenizer, "Hello! Just a quick warmup prompt.")

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
        "model_path": model_id_or_path,
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
        print(f"\n=== Starting {model_name} ===")
        summary = evaluate_model(model_name, model_path)
        results[model_name] = summary

    os.makedirs("evaluation_results", exist_ok=True)
    out_path = "evaluation_results/eval_results_domain.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved CPU evaluation to: {out_path}")


if __name__ == "__main__":
    main()
