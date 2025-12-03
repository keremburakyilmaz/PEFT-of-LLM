# Parameter-Efficient Fine-Tuning (PEFT) of Large Language Models

**Course:** ID2223 - Scalable Machine Learning and Deep Learning  
**Lab Assignment 2**  
**Author:** Kerem Burak Yilmaz

## Abstract

This project implements and evaluates Parameter-Efficient Fine-Tuning (PEFT) methods, specifically LoRA and QLoRA, on Llama 3.2 models for domain-specific instruction following. I fine-tuned models on both [general instruction data](https://huggingface.co/datasets/mlabonne/FineTome-100k) and domain-specific [exam preparation data](https://huggingface.co/datasets/kevembuvak/id2223_exam_prep), comparing different model sizes (1B vs 3B parameters) and quantization strategies. The final models are exported to GGUF format and deployed via a Gradio web interface on Huggingface space [kevembuvak/iris](https://huggingface.co/spaces/kevembuvak/iris) which is an exam preparation study chatbot.

## 1. Introduction

### 1.1 Objectives
- Implement LoRA and QLoRA fine-tuning on Llama 3.2 models
- Compare full-precision LoRA vs 4-bit quantized QLoRA
- Evaluate the impact of model size (1B vs 3B parameters)
- Fine-tune models on domain-specific data (ID2223 exam preparation)
- Export models to GGUF format for efficient CPU inference
- Deploy models via a web interface for practical use

### 1.2 Motivation
Parameter-efficient fine-tuning enables adapting large language models to specific tasks without full fine-tuning, reducing computational requirements and memory footprint. This is particularly important for:
- Resource-constrained environments
- Domain-specific applications
- Rapid iteration and experimentation

## 2. Methodology

### 2.1 Base Models
- **Llama 3.2 1B Instruct** (`meta-llama/Llama-3.2-1B-Instruct`)
- **Llama 3.2 3B Instruct** (`meta-llama/Llama-3.2-3B-Instruct`)

### 2.2 Datasets

#### Stage 1: General Instruction Tuning
- **FineTome-100k** (`mlabonne/FineTome-100k`)
  - 40.000 examples sampled for training
  - General-purpose instruction-following dataset
  - Format: ShareGPT conversation format

#### Stage 2: Domain-Specific Fine-Tuning
- **ID2223 Exam Prep Dataset** (`data/id2223_exam_prep_full.jsonl`)
  - 6,521 examples
  - Domain: KTH ID2223 course content
  - Format: JSONL with conversations (user/assistant pairs)
  - **Note:** Dataset was created with the help of AI.

### 2.3 Training Configuration

#### LoRA Configuration
```python
LoraConfig(
    r=16,                   # Rank
    lora_alpha=32,          # Scaling factor
    lora_dropout=0.05,      # Dropout
    target_modules=[        # Modules to apply LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM",
)
```

#### Training Hyperparameters

**Stage 1 (FineTome-100k):**
- Epochs: 2
- Learning rate: 2e-4
- Batch size: 4
- Gradient accumulation: 4-8 steps
- Optimizer: `paged_adamw_8bit` (QLoRA) or `adamw_torch` (LoRA)
- Learning rate scheduler: Cosine
- Max sequence length: 512 tokens

**Stage 2 (Exam Data):**
- Epochs: 2-3
- Learning rate: 1e-4 to 2e-4
- Batch size: 4
- Gradient accumulation: 4 steps
- Optimizer: `paged_adamw_8bit`
- Learning rate scheduler: Cosine

#### QLoRA Quantization
- Quantization: 4-bit NF4
- Compute dtype: bfloat16
- Double quantization: Enabled
- Quantization type: NF4

### 2.4 Models Trained

| Model | Base | Method | Stage 1 | Stage 2 | Checkpoints |
|-------|------|--------|---------|---------|-------------|
| llama32-1b-lora-finetome | 1B | LoRA (FP16) | FineTome | - | checkpoint-4500, 4750, 4950 |
| llama32-1b-qlora-finetome | 1B | QLoRA (4-bit) | FineTome | - | checkpoint-4000, 4500, 4950 |
| llama32-1b-qlora-finetome-exam | 1B | QLoRA (4-bit) | FineTome | Exam | checkpoint-4000, 4500, 4950 |
| llama32-3b-qlora-finetome | 3B | QLoRA (4-bit) | FineTome | - | checkpoint-4000, 4500, 4950 |
| llama32-3b-qlora-finetome-exam | 3B | QLoRA (4-bit) | FineTome | Exam | checkpoint-4000, 4500, 4950 |
| llama32-3b-qlora-exam | 3B | QLoRA (4-bit) | - | Exam (direct) | checkpoint-600, 800, 1000, 1200 |

### 2.5 Training Infrastructure
- Framework: PyTorch, Transformers, PEFT, TRL
- Hardware:
  - **Processor**: Intel Core Ultra 9 185H (16 cores, 22 threads, up to 5.1 GHz)
  - **Graphics**: NVIDIA GeForce RTX 4090 Laptop GPU
    - 16GB GDDR6 VRAM
    - Maximum TGP: 115W (95W base + 20W Dynamic Boost)
  - **Memory**: 32GB LPDDR5X-7467 MHz
- Monitoring: Custom ResourceUsageCallback tracking GPU memory, CPU RAM, and training metrics
- Logging: Training logs saved to `training_logs/`

## 3. Evaluation

### 3.1 Evaluation Setup

Three evaluation scripts were created:

1. **`eval_models.py`** - Compares different PEFT methods (LoRA vs QLoRA)
   - Models evaluated: 
      - `llama32-1b-lora-finetome`
      - `llama32-1b-qlora-finetome`
      - `llama32-3b-qlora-finetome`
2. **`eval_domain.py`** - Evaluates domain-specific performance on ID2223 topics
   - Models tested: 
      - `meta-llama/Llama-3.2-3B-Instruct` 
      - `llama32-3b-qlora-finetome`
      - `llama32-3b-qlora-finetome-exam`
      - `llama32-3b-qlora-exam`
3. **`eval_pytorch_vs_gguf.py`** - Compares PyTorch vs GGUF inference performance
   - Models tested: 
      - `llama32-1b-lora-finetome`
      - `llama32-1b-qlora-finetome`
      - `llama32-3b-qlora-finetome`

### 3.2 Evaluation Metrics

- **Response Quality**: Manual inspection of generated responses
- **Inference Speed**: Time per prompt (seconds)
- **Memory Usage**: GPU/CPU RAM consumption
- **Domain Relevance**: Relevance to ID2223 course topics

### 3.3 Evaluation Prompts

**General Evaluation (eval_models.py, eval_pytorch_vs_gguf.py):**
- "Explain overfitting in simple terms."
- "Summarize gradient descent."
- "Write an email asking for an extension."
- "Difference between LoRA and QLoRA?"
- "Give 3 ML learning tips."
- "Explain what a checkpoint is."
- "Trade-offs between 1B and 3B models?"
- "Sort [3, 1, 4, 1, 5] and explain."
- "What is transfer learning?"
- "Bias-variance tradeoff in simple terms."

**Domain-Specific Evaluation (eval_domain.py):**
- "Explain overfitting in simple terms."
- "Summarize gradient descent."
- "Explain the difference between supervised and unsupervised learning."
- "Why do neural networks need activation functions?"
- "What is a convolutional neural network (CNN)?"
- "What is a recurrent neural network (RNN)?"
- "What is a feature store, and why is it useful?"
- "Explain train-serve skew and how to prevent it."
- "What is data-centric iteration in machine learning?"
- "Give 3 tips for studying ML systems effectively."

### 3.4 Results

#### Training Metrics

**llama32-1b-lora-finetome:**
- Total steps: 4,950
- Final loss: ~0.74
- Training time: 2.18 hours
- GPU memory: ~2.5 GB allocated

**llama32-1b-qlora-finetome:**
- Total steps: 4,950
- Final loss: ~0.74
- Training time: 2.81 hours
- GPU memory: ~1.05 GB allocated

**llama32-3b-qlora-finetome:**
- Total steps: 4,950
- Final loss: ~0.61
- Training time: 6.45 hours
- GPU memory: ~2.3 GB allocated

**llama32-1b-qlora-finetome-exam:**
- Total steps: 814
- Final loss: ~0.09
- Training time: 0.36 hours
- GPU memory: ~1.05 GB allocated

**llama32-3b-qlora-finetome-exam:**
- Total steps: 814
- Final loss: ~0.08
- Training time: 0.53 hours
- GPU memory: ~2.3 GB allocated

**llama32-3b-qlora-exam (direct):**
- Total steps: 1,221
- Final loss: ~0.08
- Training time: 0.64 hours
- GPU memory: ~2.3 GB allocated

#### Inference Performance

Results stored in:
- `evaluation_results/eval_results_models.json`
- `evaluation_results/eval_results_domain.json`
- `evaluation_results/eval_pytorch_vs_gguf.json`

#### eval_models.py - LoRA vs QLoRA Comparison

**Performance Summary:**
- **llama32-1b-lora**: Avg time: 83.2s/prompt, CPU RAM: 5.7GB
- **llama32-1b-qlora**: Avg time: 79.8s/prompt, CPU RAM: 5.5GB  
- **llama32-3b-qlora**: Avg time: 173.9s/prompt, CPU RAM: 12.5GB

**Key Findings:**
- **Memory Efficiency**: QLoRA (1B) uses similar memory to LoRA (1B) despite 4-bit quantization, likely due to CPU inference overhead
- **Speed**: QLoRA (1B) is ~4% faster than LoRA (1B) on average, but both are slow on CPU (80-83s per prompt)
- **Model Size Impact**: 3B model is ~2.2x slower than 1B models, with significantly higher memory usage (12.5GB vs 5.5GB)
- **Response Quality**: All models show similar instruction-following capabilities, but some confusion on domain-specific terms (e.g., LoRA/QLoRA interpreted as wireless communication technologies rather than PEFT methods)

#### eval_domain.py - Domain-Specific Performance

**Performance Summary:**
- **llama32-3b-base**: Avg time: 54.8s/prompt, CPU RAM: 12.8GB
- **llama32-3b-qlora-finetome**: Avg time: 144.8s/prompt, CPU RAM: 12.8GB
- **llama32-3b-qlora-finetome-exam**: Avg time: 118.4s/prompt, CPU RAM: 12.5GB
- **llama32-3b-qlora-exam**: Avg time: 95.7s/prompt, CPU RAM: 12.6GB

**Key Findings:**
- **Base Model Performance**: Base model is fastest (54.8s) but shows poor domain knowledge (e.g., interprets "train-serve skew" as tennis terminology)
- **Fine-Tuning Impact**: Fine-tuned models show improved domain understanding:
  - **finetome-exam**: Best domain knowledge (correctly explains feature stores, train-serve skew, data-centric iteration)
  - **exam-only**: Good domain knowledge with faster inference (95.7s vs 118.4s)
  - **finetome-only**: Moderate domain knowledge, slower inference
- **Domain Accuracy**: Models fine-tuned on exam data correctly understand ID2223-specific concepts, while base and finetome-only models misinterpret domain terms
- **Inference Speed Trade-off**: Domain-specific models are slower than base model but provide accurate domain knowledge

#### eval_pytorch_vs_gguf.py - Inference Backend Comparison

**Performance Summary (1B Models):**
- **PyTorch (LoRA)**: Avg time: 36.8s/prompt, CPU RAM: 5.5GB
- **GGUF (LoRA)**: Avg time: 6.4s/prompt, CPU RAM: 6.9GB
- **PyTorch (QLoRA)**: Avg time: 32.5s/prompt, CPU RAM: 5.5GB
- **GGUF (QLoRA)**: Avg time: 6.7s/prompt, CPU RAM: 7.0GB

**Performance Summary (3B Model):**
- **PyTorch (QLoRA)**: Avg time: 86.8s/prompt, CPU RAM: 13.1GB
- **GGUF (QLoRA)**: Avg time: 24.4s/prompt, CPU RAM: 15.2GB

**Key Findings:**
- **Speed Advantage**: GGUF is **5.7x faster** for 1B models (6.4s vs 36.8s) and **3.6x faster** for 3B models (24.4s vs 86.8s)
- **Memory Efficiency**: GGUF uses slightly more RAM but provides massive speed improvements, making it ideal for CPU inference
- **Quantization Impact**: QLoRA models show similar performance to LoRA in both backends, confirming quantization doesn't significantly degrade inference speed
- **Production Readiness**: GGUF format is clearly superior for CPU-based deployment, offering near real-time inference (6-24s) compared to PyTorch (33-87s)

## 4. Model Export and Deployment

### 4.1 GGUF Export Pipeline

The `tools/export_to_gguf.py` script:
1. Merges LoRA adapters into base model
2. Converts to GGUF format (F16)
3. Quantizes to various formats (Q8_0, Q4_K_M, etc.)

**Exported Models:**
- `llama32-1b-lora-finetome_q8_0.gguf`
- `llama32-1b-qlora-finetome_q8_0.gguf`
- `llama32-3b-qlora-finetome_q4_k_m.gguf`
- `llama32-1b-qlora-finetome-exam_q8_0.gguf`
- `llama32-3b-qlora-finetome-exam_q4_k_m.gguf`
- `llama32-3b-qlora-exam_q4_k_m.gguf`

### 4.2 Deployment

**Gradio Web Interface** (`iris/app.py`):
- Model selection dropdown (supports multiple models)
- Six interactive tools:
  - Study Chat
  - Quiz Generator
  - Answer Feedback
  - Cheat Sheet Generator
  - Study Planner
  - Mixed Drill (Mini Exam)
- Deployed on HuggingFace Spaces
- Models loaded from HuggingFace repositories

**HuggingFace Spaces:**
- **App Space**: `kevembuvak/iris` - Gradio web interface deployment
- **Model Repositories**:
  - `kevembuvak/llama32-3b-qlora-finetome-exam-gguf` - 3B QLoRA Finetome + Exam model
  - `kevembuvak/llama32-1b-qlora-finetome-exam-gguf` - 1B QLoRA Finetome + Exam model
  - `kevembuvak/llama32-3b-qlora-exam-gguf` - 3B QLoRA Exam (direct) model

## 5. Key Findings and Discussion

### 5.1 LoRA vs QLoRA
- **Memory Efficiency**: QLoRA (4-bit) significantly reduces memory requirements
  - Training: QLoRA uses ~1.05GB GPU memory vs LoRA's ~2.5GB for 1B models (58% reduction)
  - Inference: Similar CPU RAM usage (~5.5GB) for both methods, indicating quantization benefits are more pronounced during training
- **Training Speed**: QLoRA enables training larger models on limited hardware
  - 1B QLoRA training: 2.81 hours vs 1B LoRA: 2.18 hours (29% slower due to quantization overhead)
  - However, QLoRA enables training 3B models on 16GB VRAM, which would be impossible with full-precision LoRA
- **Quality**: Response quality is comparable between LoRA and QLoRA
  - Both methods show similar instruction-following capabilities
  - Both occasionally misinterpret domain-specific terms (e.g., LoRA/QLoRA as wireless communication)
  - QLoRA maintains quality despite 4-bit quantization, confirming the effectiveness of NF4 quantization

### 5.2 Model Size Comparison (1B vs 3B)
- **Performance**: 3B models show better domain understanding and more coherent responses
  - 3B models correctly understand ID2223-specific concepts (feature stores, train-serve skew)
  - 1B models show more confusion on domain terms and sometimes produce less coherent explanations
  - 3B models provide more detailed and structured responses
- **Inference Speed**: 1B models are significantly faster
  - 1B models: ~80-83s per prompt (CPU inference)
  - 3B models: ~174s per prompt (CPU inference) - 2.1x slower
  - GGUF format: 1B models ~6.4s, 3B models ~24.4s (3.8x slower)
- **Memory Requirements**: 3B models require substantially more memory
  - 1B models: ~5.5GB CPU RAM during inference
  - 3B models: ~12.5-13.1GB CPU RAM during inference (2.3x more)
  - Training: 1B uses ~1-2.5GB GPU, 3B uses ~2.3GB GPU (QLoRA makes 3B feasible)

### 5.3 Two-Stage vs Direct Training
- **FineTome + Exam**: Better general instruction following, but may inherit unwanted patterns
  - **Advantages**: Strong general instruction-following capabilities, better structured responses
  - **Disadvantages**: Inherits meta-instruction patterns from FineTome dataset, slower inference (118.4s vs 95.7s)
  - **Best for**: Applications requiring both general and domain-specific knowledge
- **Direct Exam Training**: More focused on domain, avoids meta-instruction leakage
  - **Advantages**: Faster inference (95.7s), better domain accuracy, avoids unwanted meta-instructions
  - **Disadvantages**: May have weaker general instruction-following compared to two-stage training
  - **Best for**: Domain-specific applications where speed and accuracy are priorities
- **Trade-offs**: 
  - **Domain Accuracy**: Both approaches achieve good domain understanding, but finetome-exam shows slightly better structured explanations
  - **Inference Speed**: Direct training is 19% faster (95.7s vs 118.4s)
  - **Training Time**: Direct training requires less total training time (0.64h vs 0.53h + 6.45h = 6.98h)
  - **Recommendation**: For exam preparation chatbot, direct training provides better speed/accuracy trade-off

### 5.4 Data Quality Issues
- **Meta-Instruction Leakage**: Initial dataset contained meta-instructions in user prompts
  - Examples: "Briefly compare", "Relate it to", "Include an example from industry"
  - Models trained on uncleaned data generated these meta-instructions in responses
  - FineTome-100k dataset likely contributed to this pattern
- **Solution**: Implemented cleaning script to remove patterns like "Briefly compare", "Relate it to", etc.
  - Removed meta-instruction prefixes from user prompts in exam dataset
  - Created direct training path (skipping FineTome) to avoid inheriting patterns
  - Enhanced system prompts to explicitly forbid meta-instructions
- **Impact**: Cleaned data reduces unwanted meta-instruction generation in responses
  - Direct exam training models show cleaner responses without meta-instructions
  - Two-stage models still occasionally generate meta-instructions, requiring stronger system prompts
  - Final deployed models use explicit system prompts to prevent meta-instruction leakage

### 5.5 Generation Parameters
- **Temperature**: 0.3 (optimal for factual/educational content)
  - Lower values (0.2-0.3) produce more deterministic, focused responses
  - Higher values increase creativity but reduce factual accuracy
  - 0.3 provides good balance for exam preparation chatbot
- **Repeat Penalty**: 1.15 helps reduce repetitive meta-instruction loops
  - Prevents models from getting stuck in repetitive patterns
  - Particularly important for avoiding meta-instruction repetition
  - Values above 1.1 effectively reduce repetition without overly constraining generation
- **Top-p (Nucleus Sampling)**: 0.9
  - Controls diversity by sampling from top 90% of probability mass
  - Works well with temperature 0.3 for balanced responses
- **Max Tokens**: 512-900 depending on task complexity
  - Study Chat: 512 tokens (concise explanations)
  - Answer Feedback: 700 tokens (detailed feedback)
  - Quiz Generator: 800 tokens (multiple questions + answer key)
  - Cheat Sheet/Study Planner/Mixed Drill: 900 tokens (comprehensive content)

## 6. Challenges and Solutions

### 6.1 Meta-Instruction Leakage
**Problem**: Models generated meta-instructions like "Include an example from industry" in responses.

**Root Cause**: FineTome-100k dataset likely contained similar patterns, and exam fine-tuning wasn't strong enough to override them.

**Solutions Implemented**:
1. Cleaned exam dataset to remove meta-instructions from user prompts
2. Created direct training script (skipping FineTome) for domain-specific models
3. Adjusted generation parameters (temperature, repeat_penalty)
4. Enhanced system prompts to explicitly forbid meta-instructions

### 6.2 Training Stability
- Used gradient accumulation to maintain effective batch size
- Cosine learning rate scheduler for smooth convergence
- Checkpoint saving every 200-500 steps

### 6.3 Memory Constraints
- QLoRA with 4-bit quantization enabled training 3B models on RTX 4090 (16GB VRAM)
- Batch size and gradient accumulation tuned for available memory

## 7. Code Structure

```
PEFT-of-LLM/
├── data/
│   └── id2223_exam_prep_full.jsonl      # Domain-specific training data
├── training/
│   ├── train_llama_3_2_1b_lora_finetome.py
│   ├── train_llama_3_2_1b_qlora_finetome.py
│   ├── train_llama_3_2_3b_qlora_finetome.py
│   ├── train_llama_3_2_1b_qlora_finetome_exam.py
│   ├── train_llama_3_2_3b_qlora_finetome_exam.py
│   ├── train_llama_3_2_3b_qlora_exam.py  # Direct training (no FineTome)
│   ├── train_gemma_2_2b_lora_finetome.py
│   ├── train_phi_3_3b_lora_finetome.py
│   ├── train_phi_3_3b_lora_finetome_exam.py
│   ├── dataset_utils.py                  # Dataset loading utilities
│   └── callbacks/
│       └── logging_callback.py           # Resource usage tracking
├── evaluation/
│   ├── eval_models.py                     # Compare LoRA vs QLoRA
│   ├── eval_domain.py                     # Domain-specific evaluation
│   └── eval_pytorch_vs_gguf.py           # PyTorch vs GGUF comparison
├── evaluation_results/
│   ├── eval_results_models.json
│   ├── eval_results_domain.json
│   └── eval_pytorch_vs_gguf.json
├── tools/
│   └── export_to_gguf.py                 # Model export pipeline
├── models/                                # Trained adapter weights
├── checkpoints/                           # Training checkpoints
├── gguf_export/                           # Exported GGUF models
│   ├── *_f16.gguf                         # F16 GGUF models
│   ├── *_q8_0.gguf / *_q4_k_m.gguf        # Quantized GGUF models
│   └── *_merged_hf/                        # Merged HuggingFace models
├── training_logs/                         # Training logs with metrics
├── iris/                                  # Gradio deployment app
│   ├── app.py
│   ├── README.md
│   └── requirements.txt
├── test_gguf.py                           # Local GGUF testing
└── test_cuda.py                           # CUDA availability test
```

## 8. Usage

### 8.1 Training a Model

**Stage 1 - FineTome Training:**
```bash
python training/train_llama_3_2_3b_qlora_finetome.py
```

**Stage 2 - Exam Fine-Tuning (with FineTome):**
```bash
python training/train_llama_3_2_3b_qlora_finetome_exam.py
```

**Direct Exam Training (no FineTome):**
```bash
python training/train_llama_3_2_3b_qlora_exam.py
```

### 8.2 Evaluation

```bash
# Compare different PEFT methods
python evaluation/eval_models.py

# Evaluate domain-specific performance
python evaluation/eval_domain.py

# Compare PyTorch vs GGUF
python evaluation/eval_pytorch_vs_gguf.py
```

### 8.3 Export to GGUF

```bash
python tools/export_to_gguf.py
```

### 8.4 Local Testing

```bash
python test_gguf.py
```

### 8.5 Deploy Web Interface

```bash
cd iris
python app.py
```

## 9. Future Work

- [ ] 

## 10. Appendix

### 10.1 Training Logs
Training logs with detailed metrics available in `training_logs/`:
- `llama32-1b-lora-finetome.log`
- `llama32-1b-qlora-finetome.log`
- `llama32-3b-qlora-finetome.log`
- `llama32-3b-qlora_exam.log`
- `llama32-3b-qlora-finetome-exam.log`

### 10.2 Evaluation Results
Detailed evaluation results in `evaluation_results/`:
- `eval_results_models.json` - LoRA vs QLoRA comparison
- `eval_results_domain.json` - Domain-specific evaluation
- `eval_pytorch_vs_gguf.json` - Inference format comparison

### 10.3 Model Checkpoints
- Adapter weights: `models/`
- Training checkpoints: `checkpoints/`
- Exported GGUF: `gguf_export/`
