import os
import time
import json
from typing import Dict, List

import torch
from transformers import AutoTokenizer

try:
    from peft import AutoPeftModelForCausalLM
except ImportError:
    from transformers import AutoModelForCausalLM
    AutoPeftModelForCausalLM = AutoModelForCausalLM

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

MODEL_CONFIGS = [
    {
        "name": "llama32-1b-lora-finetome",
        "adapter_dir": "models/llama32-1b-lora-finetome",
        "gguf_path": "gguf/llama32-1b-lora-finetome-q4_k_m.gguf",
    },
    {
        "name": "llama32-1b-qlora-finetome",
        "adapter_dir": "models/llama32-1b-qlora-finetome",
        "gguf_path": "gguf/llama32-1b-qlora-finetome-q4_k_m.gguf",
    },
    {
        "name": "llama32-3b-qlora-finetome",
        "adapter_dir": "models/llama32-3b-qlora-finetome",
        "gguf_path": "gguf/llama32-3b-qlora-finetome-q4_k_m.gguf",
    },
]

EVAL_PROMPTS: List[str] = [
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

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.2
TOP_P = 0.9
SEED = 42

OUT_JSON = "evaluation_results/eval_pytorch_vs_gguf.json"

DEVICE = torch.device("cpu")

def log_system_usage(prefix: str = "") -> Dict[str, float]:
    stats = {}

    if PSUTIL_AVAILABLE:
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        stats["cpu_ram_rss_mb"] = mem.rss / (1024 ** 2)

    if prefix:
        print(f"\n[{prefix}] System usage:")
    else:
        print("\nSystem usage:")

    for k, v in stats.items():
        print(f"  {k}: {v:.2f} MB")

    return stats

def load_pytorch_model_and_tokenizer(adapter_dir: str):
    print(f"\n=== [PyTorch] Loading adapter model from: {adapter_dir} ===")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        torch_dtype=torch.float32,
        device_map=None,
    ).to(DEVICE)

    model.eval()
    return model, tokenizer

def generate_pytorch(model, tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful chatbot."},
            {"role": "user", "content": prompt},
        ]
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


def evaluate_pytorch_model(model_name: str, adapter_dir: str) -> Dict:
    print("\n" + "=" * 50)
    print(f"[PyTorch] Evaluating model: {model_name}")
    print("=" * 50)

    pre_stats = log_system_usage(prefix=f"{model_name} [PyTorch] (before)")

    model, tokenizer = load_pytorch_model_and_tokenizer(adapter_dir)

    print("[PyTorch] Warmup")
    _ = generate_pytorch(model, tokenizer, "Hello, just a warmup.")

    start_time = time.perf_counter()
    responses = []
    per_prompt_times = []

    for i, prompt in enumerate(EVAL_PROMPTS, 1):
        print(f"\n[PyTorch:{model_name}] Prompt {i}/{len(EVAL_PROMPTS)}")
        print(f"User: {prompt}")

        t0 = time.perf_counter()
        answer = generate_pytorch(model, tokenizer, prompt)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        per_prompt_times.append(elapsed)

        print(f"Assistant (t={elapsed:.2f}s): {answer[:300]}...")

        responses.append({
            "prompt": prompt,
            "response": answer,
            "time_sec": elapsed,
        })

    total_time = time.perf_counter() - start_time
    avg_time = total_time / len(EVAL_PROMPTS)

    post_stats = log_system_usage(prefix=f"{model_name} [PyTorch] (after)")

    summary = {
        "backend": "pytorch_peft",
        "model_name": model_name,
        "adapter_dir": adapter_dir,
        "num_prompts": len(EVAL_PROMPTS),
        "total_time_sec": total_time,
        "avg_time_per_prompt_sec": avg_time,
        "per_prompt_times_sec": per_prompt_times,
        "responses": responses,
        "pre_stats": pre_stats,
        "post_stats": post_stats,
    }

    print(f"\n[PyTorch:{model_name}] Total time: {total_time:.2f} s")
    print(f"[PyTorch:{model_name}] Avg per prompt: {avg_time:.2f} s")

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary

def load_gguf_model(gguf_path: str):
    if not LLAMACPP_AVAILABLE:
        raise RuntimeError("llama-cpp-python is not installed. Install via `pip install llama-cpp-python`.")

    print(f"\n=== [GGUF] Loading GGUF model: {gguf_path} ===")
    llm = Llama(
        model_path=gguf_path,
        n_ctx=4096,
        n_gpu_layers=0,
        seed=SEED,
        logits_all=False,
        embedding=False,
    )
    return llm


def generate_gguf(llm, prompt: str) -> str:
    completion = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful chatbot.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return completion["choices"][0]["message"]["content"].strip()


def evaluate_gguf_model(model_name: str, gguf_path: str) -> Dict:
    print("\n" + "=" * 50)
    print(f"[GGUF] Evaluating model: {model_name}")
    print("=" * 50)

    pre_stats = log_system_usage(prefix=f"{model_name} [GGUF] (before)")

    if not os.path.isfile(gguf_path):
        print(f"[GGUF:{model_name}] GGUF file not found: {gguf_path}")
        return {
            "backend": "gguf",
            "model_name": model_name,
            "gguf_path": gguf_path,
            "error": "GGUF file not found",
        }

    llm = load_gguf_model(gguf_path)

    print("[GGUF] Warmup")
    _ = generate_gguf(llm, "Hello, just a warmup.")

    start_time = time.perf_counter()
    responses = []
    per_prompt_times = []

    for i, prompt in enumerate(EVAL_PROMPTS, 1):
        print(f"\n[GGUF:{model_name}] Prompt {i}/{len(EVAL_PROMPTS)}")
        print(f"User: {prompt}")

        t0 = time.perf_counter()
        answer = generate_gguf(llm, prompt)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        per_prompt_times.append(elapsed)

        print(f"Assistant (t={elapsed:.2f}s): {answer[:300]}...")

        responses.append({
            "prompt": prompt,
            "response": answer,
            "time_sec": elapsed,
        })

    total_time = time.perf_counter() - start_time
    avg_time = total_time / len(EVAL_PROMPTS)

    post_stats = log_system_usage(prefix=f"{model_name} [GGUF] (after)")

    summary = {
        "backend": "gguf_llama_cpp",
        "model_name": model_name,
        "gguf_path": gguf_path,
        "num_prompts": len(EVAL_PROMPTS),
        "total_time_sec": total_time,
        "avg_time_per_prompt_sec": avg_time,
        "per_prompt_times_sec": per_prompt_times,
        "responses": responses,
        "pre_stats": pre_stats,
        "post_stats": post_stats,
    }

    print(f"\n[GGUF:{model_name}] Total time: {total_time:.2f} s")
    print(f"[GGUF:{model_name}] Avg per prompt: {avg_time:.2f} s")

    return summary

def main():
    torch.manual_seed(SEED)
    os.makedirs("evaluation_results", exist_ok=True)

    all_results: Dict[str, Dict] = {}

    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        adapter_dir = cfg["adapter_dir"]
        gguf_path = cfg["gguf_path"]

        print("\n" + "#" * 70)
        print(f"Model: {name}")
        print("#" * 70)

        model_results = {}

        if os.path.isdir(adapter_dir):
            pyt_result = evaluate_pytorch_model(name, adapter_dir)
            model_results["pytorch"] = pyt_result
        else:
            print(f"[WARN] Adapter dir not found, skipping PyTorch eval: {adapter_dir}")
            model_results["pytorch"] = {
                "error": f"adapter_dir_not_found: {adapter_dir}"
            }

        if LLAMACPP_AVAILABLE:
            gguf_result = evaluate_gguf_model(name, gguf_path)
            model_results["gguf"] = gguf_result
        else:
            print("[WARN] llama-cpp-python not installed, skipping GGUF eval.")
            model_results["gguf"] = {
                "error": "llama_cpp_not_installed"
            }

        all_results[name] = model_results

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved combined PyTorch vs GGUF evaluation to: {OUT_JSON}")


if __name__ == "__main__":
    main()
