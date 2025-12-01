import os
import sys
from llama_cpp import Llama

DEFAULT_MODEL_PATH = "gguf_export/llama32-3b-qlora-exam_q4_k_m.gguf"

def load_model(model_path: str = None):
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(__file__), model_path)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("\nAvailable models in gguf_export/:")
        gguf_dir = os.path.join(os.path.dirname(__file__), "gguf_export")
        if os.path.exists(gguf_dir):
            for f in os.listdir(gguf_dir):
                if f.endswith('.gguf'):
                    print(f"  - {f}")
        sys.exit(1)
    
    print(f"[Loading model from: {model_path}]")
    
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
        logits_all=False,
        verbose=False,
    )
    print("[Model loaded successfully]\n")
    return llm

def test_query(model: Llama, query: str, system_prompt: str = None):
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": query})
    
    print(f"Query: {query}\n")
    print("Response:")
    print("-" * 60)
    
    response = model.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.2,
        top_p=0.9,
        repeat_penalty=1.5,  # Penalize repetition to reduce meta-instruction loops
    )
    
    output = response["choices"][0]["message"]["content"].strip()
    print(output)
    print("-" * 60)
    return output

if __name__ == "__main__":
    model_path = None
    query_args = []
    
    if len(sys.argv) > 1:
        if sys.argv[1].endswith('.gguf'):
            model_path = sys.argv[1]
            query_args = sys.argv[2:]
        elif sys.argv[1] == '--model' and len(sys.argv) > 2:
            model_path = sys.argv[2]
            query_args = sys.argv[3:]
        else:
            query_args = sys.argv[1:]
    
    model = load_model(model_path)
    
    system_prompt = (
        "You are an expert teaching assistant helping a student prepare for the "
        "ID2223 'Scalable Machine Learning and Deep Learning' exam at KTH. "
        "Explain concepts clearly, step by step, and connect them to real-world "
        "ML systems (pipelines, feature stores, LLM fine-tuning, etc.). "
        "Always keep answers focused, practical, and exam-oriented."
    )
    
    if query_args:
        query = " ".join(query_args)
        test_query(model, query, system_prompt)
    else:
        print("Enter queries (or 'quit' to exit):\n")
        while True:
            query = input("> ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if query:
                test_query(model, query, system_prompt)
                print()
            else:
                print("Please enter a query.")

