import json
import os
from datetime import datetime
from pathlib import Path

class ExperimentLogger:
    def __init__(self, log_dir="./experiment_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "experiments.jsonl"
    
    def log_experiment(self, run_id, hardware, val_bpb, model_size_mb, 
                       train_time_sec, eval_time_sec, config):
        """
        Log a single experiment run
        
        Args:
            run_id: unique identifier for this run
            hardware: "mlx_macbook" | "1xh100" | "8xh100"
            val_bpb: validation bits per byte (lower is better)
            model_size_mb: model size in megabytes
            train_time_sec: training time in seconds
            eval_time_sec: evaluation time in seconds
            config: dict with architecture details
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "hardware": hardware,
            "val_bpb": val_bpb,
            "model_size_mb": model_size_mb,
            "train_time_sec": train_time_sec,
            "eval_time_sec": eval_time_sec,
            "config": config
        }
        
        # Append to JSONL file (one JSON object per line)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        print(f"✓ Logged experiment: {run_id}")
        return entry
    
    def load_experiments(self):
        """Load all experiments from log file"""
        if not self.log_file.exists():
            return []
        
        experiments = []
        with open(self.log_file, 'r') as f:
            for line in f:
                experiments.append(json.loads(line.strip()))
        return experiments
    
    def get_best_runs(self, metric="val_bpb", top_n=5):
        """Get top N runs sorted by metric"""
        experiments = self.load_experiments()
        if not experiments:
            return []
        
        sorted_exps = sorted(experiments, key=lambda x: x[metric])
        return sorted_exps[:top_n]


# Example usage:
if __name__ == "__main__":
    logger = ExperimentLogger()
    
    # Example: Log a run
    logger.log_experiment(
        run_id="mlx_smoke",
        hardware="mlx_macbook",
        val_bpb=1.234,  # Replace with actual value
        model_size_mb=17.0,
        train_time_sec=418,
        eval_time_sec=120,
        config={
            "vocab_size": 1024,
            "architecture": "gpt",
            "context_length": 1024,
            "layers": 9,
            "dim": 512,
            "notes": "baseline test run"
        }
    )
    
    # View all experiments
    all_exps = logger.load_experiments()
    print(f"\nTotal experiments: {len(all_exps)}")
    
    # Get best runs
    best = logger.get_best_runs(metric="val_bpb", top_n=3)
    print("\nTop 3 runs by BPB:")
    for i, exp in enumerate(best, 1):
        print(f"{i}. {exp['run_id']}: {exp['val_bpb']} BPB")

"""
After each training run in respective file:

from experiment_logger import ExperimentLogger

logger = ExperimentLogger()
logger.log_experiment(
    run_id="my_test_1",
    hardware="mlx_macbook",
    val_bpb=1.234,  # Get from training output
    model_size_mb=17.0,
    train_time_sec=418,
    eval_time_sec=120,
    config={
        "vocab_size": 1024,
        "architecture": "gpt",
        "context_length": 1024,
        "notes": "tried reducing layers to 8"
    }
)
"""