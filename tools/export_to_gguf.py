import os
import subprocess

LLAMACPP_PATH = r"C:\\Users\\kbura\\llama.cpp\\build\\bin\\Release"
QUANTIZE_BIN = os.path.join(LLAMACPP_PATH, "llama-quantize.exe")

BASE_MODELS = {
    "llama32-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama32-3b": "meta-llama/Llama-3.2-3B-Instruct",
}

TRAINED_MODELS = {
    "llama32-1b-lora": "models/llama32-1b-lora-finetome",
    "llama32-1b-qlora": "models/llama32-1b-qlora-finetome",
    "llama32-3b-qlora": "models/llama32-3b-qlora-finetome",
}

OUT_DIR = "gguf_export"
os.makedirs(OUT_DIR, exist_ok=True)

def merge_lora(base_model, adapter_model, output_model):
    subprocess.run([
        "python",
        "-m", "peft.merge_lora_weights",
        "--base_model", base_model,
        "--adapter_model", adapter_model,
        "--output_model", output_model
    ], check=True)

def convert_to_gguf(hf_model_path, out_gguf_path):
    print("Converting to GGUF...")
    subprocess.run([
        QUANTIZE_BIN,
        hf_model_path,
        out_gguf_path,
        "Q4_K_M"
    ], check=True)

def main():
    for name, trained_path in TRAINED_MODELS.items():
        print(f"\n=== Converting {name} ===")

        base_key = "llama32-1b" if "1b" in name else "llama32-3b"
        base_model = BASE_MODELS[base_key]

        merged_dir = os.path.join(OUT_DIR, f"{name}_merged")
        gguf_file = os.path.join(OUT_DIR, f"{name}.gguf")

        print("Merging LoRA adapter with base model")
        merge_lora(base_model, trained_path, merged_dir)

        print("Exporting to GGUF")
        convert_to_gguf(merged_dir, gguf_file)

        print(f"Done: {gguf_file}")

if __name__ == "__main__":
    main()
