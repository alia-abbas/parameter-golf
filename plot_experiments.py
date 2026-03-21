import json
import matplotlib.pyplot as plt
from pathlib import Path
from experiment_logger import ExperimentLogger

# Competition constraints
MAX_MODEL_SIZE_MB = 16.0
MAX_TRAIN_TIME_MIN = 10.0
THEORETICAL_MIN_BPB = 1.0

def plot_bpb_vs_size():
    """Plot BPB (y-axis) vs Model Size (x-axis)"""
    logger = ExperimentLogger()
    experiments = logger.load_experiments()
    
    if not experiments:
        print("No experiments found!")
        return
    
    # Separate by hardware
    hardware_groups = {}
    for exp in experiments:
        hw = exp['hardware']
        if hw not in hardware_groups:
            hardware_groups[hw] = {'sizes': [], 'bpbs': [], 'labels': [], 'valid': []}
        hardware_groups[hw]['sizes'].append(exp['model_size_mb'])
        hardware_groups[hw]['bpbs'].append(exp['val_bpb'])
        hardware_groups[hw]['labels'].append(exp['run_id'])
        # Check if valid (size <= 16MB, time <= 10min)
        is_valid = (exp['model_size_mb'] <= MAX_MODEL_SIZE_MB and 
                   exp['train_time_sec'] / 60 <= MAX_TRAIN_TIME_MIN)
        hardware_groups[hw]['valid'].append(is_valid)
    
    # Plot
    plt.figure(figsize=(12, 7))
    colors = {'mlx_macbook': 'blue', '1xh100': 'green', '8xh100': 'red'}
    
    for hw, data in hardware_groups.items():
        # Separate valid and invalid runs
        valid_sizes = [s for s, v in zip(data['sizes'], data['valid']) if v]
        valid_bpbs = [b for b, v in zip(data['bpbs'], data['valid']) if v]
        valid_labels = [l for l, v in zip(data['labels'], data['valid']) if v]
        
        invalid_sizes = [s for s, v in zip(data['sizes'], data['valid']) if not v]
        invalid_bpbs = [b for b, v in zip(data['bpbs'], data['valid']) if not v]
        invalid_labels = [l for l, v in zip(data['labels'], data['valid']) if not v]
        
        # Plot valid runs
        if valid_sizes:
            plt.scatter(valid_sizes, valid_bpbs, 
                       label=f'{hw} (valid)', color=colors.get(hw, 'gray'), 
                       s=100, alpha=0.7, marker='o')
            for i, label in enumerate(valid_labels):
                plt.annotate(label, (valid_sizes[i], valid_bpbs[i]),
                            fontsize=8, alpha=0.7)
        
        # Plot invalid runs with X marker
        if invalid_sizes:
            plt.scatter(invalid_sizes, invalid_bpbs,
                       label=f'{hw} (invalid)', color=colors.get(hw, 'gray'),
                       s=100, alpha=0.3, marker='x')
            for i, label in enumerate(invalid_labels):
                plt.annotate(label + ' ✗', (invalid_sizes[i], invalid_bpbs[i]),
                            fontsize=8, alpha=0.5, color='red')
    
    # Add constraint lines
    plt.axvline(x=MAX_MODEL_SIZE_MB, color='red', linestyle='--', 
                linewidth=2, label=f'Max size: {MAX_MODEL_SIZE_MB} MB', alpha=0.7)
    plt.axhline(y=THEORETICAL_MIN_BPB, color='green', linestyle='--',
                linewidth=2, label=f'Theoretical min BPB: {THEORETICAL_MIN_BPB}', alpha=0.7)
    
    # Shade invalid region
    plt.axvspan(MAX_MODEL_SIZE_MB, plt.xlim()[1], alpha=0.1, color='red')
    plt.axhspan(0, THEORETICAL_MIN_BPB, alpha=0.1, color='green')
    
    plt.xlabel('Model Size (MB)', fontsize=12)
    plt.ylabel('Validation BPB (lower is better)', fontsize=12)
    plt.title('Model Size vs Performance (with competition constraints)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_dir = Path("./plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "bpb_vs_size.png", dpi=150)
    print(f"✓ Saved plot: plots/bpb_vs_size.png")
    plt.show()


def plot_bpb_vs_time():
    """Plot BPB (y-axis) vs Training Time (x-axis)"""
    logger = ExperimentLogger()
    experiments = logger.load_experiments()
    
    if not experiments:
        print("No experiments found!")
        return
    
    # Separate by hardware
    hardware_groups = {}
    for exp in experiments:
        hw = exp['hardware']
        if hw not in hardware_groups:
            hardware_groups[hw] = {'times': [], 'bpbs': [], 'labels': [], 'valid': []}
        train_time_min = exp['train_time_sec'] / 60
        hardware_groups[hw]['times'].append(train_time_min)
        hardware_groups[hw]['bpbs'].append(exp['val_bpb'])
        hardware_groups[hw]['labels'].append(exp['run_id'])
        # Check if valid
        is_valid = (exp['model_size_mb'] <= MAX_MODEL_SIZE_MB and 
                   train_time_min <= MAX_TRAIN_TIME_MIN)
        hardware_groups[hw]['valid'].append(is_valid)
    
    # Plot
    plt.figure(figsize=(12, 7))
    colors = {'mlx_macbook': 'blue', '1xh100': 'green', '8xh100': 'red'}
    
    for hw, data in hardware_groups.items():
        # Separate valid and invalid
        valid_times = [t for t, v in zip(data['times'], data['valid']) if v]
        valid_bpbs = [b for b, v in zip(data['bpbs'], data['valid']) if v]
        valid_labels = [l for l, v in zip(data['labels'], data['valid']) if v]
        
        invalid_times = [t for t, v in zip(data['times'], data['valid']) if not v]
        invalid_bpbs = [b for b, v in zip(data['bpbs'], data['valid']) if not v]
        invalid_labels = [l for l, v in zip(data['labels'], data['valid']) if not v]
        
        # Plot valid
        if valid_times:
            plt.scatter(valid_times, valid_bpbs,
                       label=f'{hw} (valid)', color=colors.get(hw, 'gray'),
                       s=100, alpha=0.7, marker='o')
            for i, label in enumerate(valid_labels):
                plt.annotate(label, (valid_times[i], valid_bpbs[i]),
                            fontsize=8, alpha=0.7)
        
        # Plot invalid
        if invalid_times:
            plt.scatter(invalid_times, invalid_bpbs,
                       label=f'{hw} (invalid)', color=colors.get(hw, 'gray'),
                       s=100, alpha=0.3, marker='x')
            for i, label in enumerate(invalid_labels):
                plt.annotate(label + ' ✗', (invalid_times[i], invalid_bpbs[i]),
                            fontsize=8, alpha=0.5, color='red')
    
    # Add constraint lines
    plt.axvline(x=MAX_TRAIN_TIME_MIN, color='red', linestyle='--',
                linewidth=2, label=f'Max time: {MAX_TRAIN_TIME_MIN} min', alpha=0.7)
    plt.axhline(y=THEORETICAL_MIN_BPB, color='green', linestyle='--',
                linewidth=2, label=f'Theoretical min BPB: {THEORETICAL_MIN_BPB}', alpha=0.7)
    
    # Shade invalid region
    plt.axvspan(MAX_TRAIN_TIME_MIN, plt.xlim()[1], alpha=0.1, color='red')
    plt.axhspan(0, THEORETICAL_MIN_BPB, alpha=0.1, color='green')
    
    plt.xlabel('Training Time (minutes)', fontsize=12)
    plt.ylabel('Validation BPB (lower is better)', fontsize=12)
    plt.title('Training Time vs Performance (with competition constraints)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_dir = Path("./plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "bpb_vs_time.png", dpi=150)
    print(f"✓ Saved plot: plots/bpb_vs_time.png")
    plt.show()


def plot_pareto_frontier():
    """Plot the Pareto frontier: best BPB for each model size"""
    logger = ExperimentLogger()
    experiments = logger.load_experiments()
    
    if not experiments:
        print("No experiments found!")
        return
    
    # Filter to valid runs only
    valid_exps = [exp for exp in experiments 
                  if exp['model_size_mb'] <= MAX_MODEL_SIZE_MB 
                  and exp['train_time_sec'] / 60 <= MAX_TRAIN_TIME_MIN]
    
    if not valid_exps:
        print("No valid experiments found (all exceed size or time limits)!")
        return
    
    # Get all points
    sizes = [exp['model_size_mb'] for exp in valid_exps]
    bpbs = [exp['val_bpb'] for exp in valid_exps]
    labels = [exp['run_id'] for exp in valid_exps]
    
    # Find Pareto frontier
    pareto_points = []
    for i, (size, bpb) in enumerate(zip(sizes, bpbs)):
        is_pareto = True
        for j, (other_size, other_bpb) in enumerate(zip(sizes, bpbs)):
            if i != j:
                if other_size <= size and other_bpb < bpb:
                    is_pareto = False
                    break
        if is_pareto:
            pareto_points.append(i)
    
    # Plot
    plt.figure(figsize=(12, 7))
    plt.scatter(sizes, bpbs, s=100, alpha=0.3, label='All valid runs')
    
    pareto_sizes = [sizes[i] for i in pareto_points]
    pareto_bpbs = [bpbs[i] for i in pareto_points]
    pareto_labels = [labels[i] for i in pareto_points]
    
    plt.scatter(pareto_sizes, pareto_bpbs, s=150, color='red', 
               label='Pareto frontier', zorder=5)
    
    # Connect Pareto points
    if len(pareto_points) > 1:
        sorted_pareto = sorted(zip(pareto_sizes, pareto_bpbs))
        plt.plot([p[0] for p in sorted_pareto], [p[1] for p in sorted_pareto],
                'r--', alpha=0.5, linewidth=2)
    
    # Annotate Pareto points
    for i, label in enumerate(pareto_labels):
        plt.annotate(label, (pareto_sizes[i], pareto_bpbs[i]),
                    fontsize=9, fontweight='bold')
    
    # Add constraint lines
    plt.axvline(x=MAX_MODEL_SIZE_MB, color='red', linestyle='--',
                linewidth=2, label=f'Max size: {MAX_MODEL_SIZE_MB} MB', alpha=0.7)
    plt.axhline(y=THEORETICAL_MIN_BPB, color='green', linestyle='--',
                linewidth=2, label=f'Theoretical min BPB: {THEORETICAL_MIN_BPB}', alpha=0.7)
    
    # Shade invalid region
    plt.axvspan(MAX_MODEL_SIZE_MB, plt.xlim()[1], alpha=0.1, color='red')
    plt.axhspan(0, THEORETICAL_MIN_BPB, alpha=0.1, color='green')
    
    plt.xlabel('Model Size (MB)', fontsize=12)
    plt.ylabel('Validation BPB (lower is better)', fontsize=12)
    plt.title('Pareto Frontier: Size-Performance Trade-off (valid runs only)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0, right=MAX_MODEL_SIZE_MB * 1.1)
    plt.tight_layout()
    
    # Save
    output_dir = Path("./plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "pareto_frontier.png", dpi=150)
    print(f"✓ Saved plot: plots/pareto_frontier.png")
    plt.show()


def print_summary():
    """Print a summary of all experiments"""
    logger = ExperimentLogger()
    experiments = logger.load_experiments()
    
    if not experiments:
        print("No experiments found!")
        return
    
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    valid_exps = [exp for exp in experiments 
                  if exp['model_size_mb'] <= MAX_MODEL_SIZE_MB 
                  and exp['train_time_sec'] / 60 <= MAX_TRAIN_TIME_MIN]
    
    print(f"Total experiments: {len(experiments)}")
    print(f"Valid experiments: {len(valid_exps)}")
    print(f"Invalid experiments: {len(experiments) - len(valid_exps)}")
    
    if valid_exps:
        best = min(valid_exps, key=lambda x: x['val_bpb'])
        print(f"\n🏆 BEST VALID RUN:")
        print(f"   Run ID: {best['run_id']}")
        print(f"   BPB: {best['val_bpb']:.4f}")
        print(f"   Size: {best['model_size_mb']:.2f} MB")
        print(f"   Time: {best['train_time_sec']/60:.2f} min")
        print(f"   Architecture: {best['config'].get('architecture', 'N/A')}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print("Generating plots...")
    print_summary()
    plot_bpb_vs_size()
    plot_bpb_vs_time()
    plot_pareto_frontier()
    print("\n✓ All plots generated!")