import os
import time
import torch
from transformers import TrainerCallback

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ResourceUsageCallback(TrainerCallback):
    def __init__(self, log_file: str, is_resume: bool = False):
        super().__init__()
        self.log_file = log_file
        self.start_time = None

        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        if (not is_resume) and os.path.exists(self.log_file):
            os.remove(self.log_file)

    def log(self, text: str):
        print(text)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

        self.log("=== Training started ===")
        self.log(f"Output dir: {args.output_dir}")
        self.log(f"Batch size: {args.per_device_train_batch_size}")
        self.log(f"Grad accumulate: {args.gradient_accumulation_steps}")
        self.log(f"Epochs: {args.num_train_epochs}")
        self.log(f"LR: {args.learning_rate}")
        self.log(f"BF16: {args.bf16}")
        self.log(f"Optimizer: {args.optim}")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            logs = {}

        step = int(state.global_step) if state.global_step is not None else -1
        loss = logs.get("loss", None)
        lr = logs.get("learning_rate", None)

        loss_str = f"{loss:.4f}" if isinstance(loss, (float, int)) else "N/A"
        lr_str = f"{lr:.6f}" if isinstance(lr, (float, int)) else "N/A"

        # GPU VRAM
        gpu_stats = ""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            gpu_stats = f" | GPU alloc={alloc:.1f}MB, reserved={reserved:.1f}MB"

        # CPU RAM
        cpu_stats = ""
        if PSUTIL_AVAILABLE:
            rss = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
            cpu_stats = f" | CPU RAM={rss:.1f}MB"

        # Timing
        elapsed = time.time() - self.start_time
        steps_per_sec = (state.global_step / elapsed) if elapsed > 0 else 0.0
        time_stats = f" | elapsed={elapsed/60:.1f}min, steps/s={steps_per_sec:.2f}"

        self.log(f"[step {step}] loss={loss_str} lr={lr_str}{gpu_stats}{cpu_stats}{time_stats}")

        return control

    def on_save(self, args, state, control, **kwargs):
        self.log(f"Checkpoint saved at step {state.global_step} → {args.output_dir}")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        self.log("=== Training finished ===")
        self.log(f"Total steps: {state.global_step}")
        self.log(f"Total wall-clock time: {total_time/3600:.2f} hours")
        return control
