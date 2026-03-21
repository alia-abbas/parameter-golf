from experiment_logger import ExperimentLogger

logger = ExperimentLogger()

# Log your mlx_smoke run
logger.log_experiment(
    run_id="mlx_smoke",
    hardware="mlx_macbook",
    val_bpb=2.4101,
    model_size_mb=7.6,
    train_time_sec=418,
    eval_time_sec=0,
    config={
        "vocab_size": 1024,
        "architecture": "gpt",
        "context_length": 1024,
        "layers": 9,
        "dim": 512,
        "heads": 8,
        "kv_heads": 4,
        "notes": "baseline run with default config"
    }
)

print("✓ Logged mlx_smoke run!")