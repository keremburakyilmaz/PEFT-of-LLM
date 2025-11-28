import os
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

LLAMACPP_BIN = r"C:\\Users\\kbura\\llama.cpp\\build\\bin\\Release"

CONVERT_HF_TO_GGUF = r"C:\\Users\\kbura\\llama.cpp\\convert_hf_to_gguf.py"
QUANTIZE_BIN = os.path.join(LLAMACPP_BIN, "llama-quantize.exe")

BASE_MODELS = {
    "llama32-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama32-3b": "meta-llama/Llama-3.2-3B-Instruct",
}

TRAINED_MODELS = {
    "llama32-1b-lora": "models/llama32-1b-lora-finetome",
    "llama32-1b-qlora": "models/llama32-1b-qlora-finetome",
    "llama32-3b-qlora": "models/llama32-3b-qlora-finetome",
    "llama32-1b-qlora-exam": "models/llama32-1b-qlora-finetome-exam",
}

OUT_DIR = "gguf_export"
os.makedirs(OUT_DIR, exist_ok=True)

def merge_adapter_into_base(base_model_id: str, adapter_dir: str, merged_out_dir: str):
    os.makedirs(merged_out_dir, exist_ok=True)

    print(f"  - Loading base HF model: {base_model_id}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    print(f"  - Loading adapter from: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base, adapter_dir)

    print("  - Merging adapter weights into base (merge_and_unload)")
    merged = peft_model.merge_and_unload()

    print(f"  - Saving merged HF model to: {merged_out_dir}")
    merged.save_pretrained(merged_out_dir)

    print("  - Saving tokenizer + config to merged directory")
    config = AutoConfig.from_pretrained(base_model_id)
    config.save_pretrained(merged_out_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.save_pretrained(merged_out_dir)

def convert_merged_to_gguf_f16(merged_dir: str, gguf_f16_path: str):
    print("  - Converting merged HF model -> GGUF (f16)")
    cmd = [
        "python",
        CONVERT_HF_TO_GGUF,
        merged_dir,
        "--outfile",
        gguf_f16_path,
        "--outtype",
        "f16",
    ]
    print("    Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def quantize_gguf(input_gguf: str, output_gguf: str, quant_type: str):
    print(f"  - Quantizing GGUF: {quant_type}")
    cmd = [
        QUANTIZE_BIN,
        input_gguf,
        output_gguf,
        quant_type,
    ]
    print("    Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    for name, adapter_path in TRAINED_MODELS.items():
        print(f"\n=== Converting {name} ===")

        if "1b" in name:
            base_key = "llama32-1b"
            quant_type = "Q8_0"
        else:
            base_key = "llama32-3b"
            quant_type = "Q4_K_M"

        base_model_id = BASE_MODELS[base_key]

        merged_dir = os.path.join(OUT_DIR, f"{name}_merged_hf")
        gguf_f16 = os.path.join(OUT_DIR, f"{name}_f16.gguf")
        gguf_quant = os.path.join(OUT_DIR, f"{name}_{quant_type.lower()}.gguf")

        if not os.path.isdir(merged_dir):
            print("  - Merging base + adapter")
            merge_adapter_into_base(base_model_id, adapter_path, merged_dir)
        else:
            print(f"  - Skipping merge, already exists: {merged_dir}")

        if not os.path.isfile(gguf_f16):
            convert_merged_to_gguf_f16(merged_dir, gguf_f16)
        else:
            print(f"  - Skipping HF->GGUF f16, already exists: {gguf_f16}")

        if not os.path.isfile(gguf_quant):
            quantize_gguf(gguf_f16, gguf_quant, quant_type)
        else:
            print(f"  - Skipping quantization, already exists: {gguf_quant}")

        print(f"  -> Done: {gguf_quant}")


if __name__ == "__main__":
    main()
