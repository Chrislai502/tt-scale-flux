import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict

def load_data(file_path):
    """Load the JSON data from the specified file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_metrics(data):
    """Extract all metrics and their values, organized by noise and timestep."""
    metrics_by_noise_timestep = defaultdict(lambda: defaultdict(dict))
    all_metrics = set()
    all_noises = set()
    all_timesteps = set()
    
    for item in data:
        noise, timestep, score_dict = item
        all_noises.add(noise)
        all_timesteps.add(timestep)
        
        for metric, metric_data in score_dict.items():
            if isinstance(metric_data, dict) and 'score' in metric_data:
                all_metrics.add(metric)
                metrics_by_noise_timestep[noise][timestep][metric] = metric_data['score']
    
    return metrics_by_noise_timestep, sorted(all_metrics), sorted(all_noises), sorted(all_timesteps)

def plot_metrics(data, metrics_to_plot, noise_mode, specific_noise=None, output_file=None):
    """
    Plot the specified metrics.
    
    Args:
        data: The parsed data containing metrics by noise and timestep
        metrics_to_plot: List of metrics to plot (or "all")
        noise_mode: Either "average" or "specific"
        specific_noise: If noise_mode is "specific", the noise value to plot
        output_file: Optional file path to save the plot
    """
    metrics_by_noise_timestep, all_metrics, all_noises, all_timesteps = extract_metrics(data)
    
    if metrics_to_plot == ["all"]:
        metrics_to_plot = all_metrics
    
    # Set up the plot grid
    n_metrics = len(metrics_to_plot)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_metrics == 1:
        axes = np.array([axes])  # Ensure axes is always a numpy array
    axes = axes.flatten()
    
    # Create plots for each metric
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        if noise_mode == "average":
            # Calculate average across all noises for each timestep
            avg_scores = {}
            for timestep in all_timesteps:
                scores = [metrics_by_noise_timestep[noise][timestep].get(metric, np.nan) 
                         for noise in all_noises 
                         if timestep in metrics_by_noise_timestep[noise]]
                scores = [s for s in scores if not np.isnan(s)]
                if scores:
                    avg_scores[timestep] = np.mean(scores)
            
            timesteps = sorted(avg_scores.keys())
            scores = [avg_scores[t] for t in timesteps]
            
            # Plot average line and scatter
            ax.plot(timesteps, scores, 'b-', label=f'Average across all noises')
            ax.scatter(timesteps, scores, color='blue', s=50)
            
        elif noise_mode == "specific" and specific_noise in all_noises:
            # Plot data for the specific noise
            noise_data = metrics_by_noise_timestep[specific_noise]
            timesteps = sorted(noise_data.keys())
            scores = [noise_data[t].get(metric, np.nan) for t in timesteps]
            
            # Filter out NaN values
            valid_indices = [i for i, s in enumerate(scores) if not np.isnan(s)]
            valid_timesteps = [timesteps[i] for i in valid_indices]
            valid_scores = [scores[i] for i in valid_indices]
            
            # Plot specific noise line and scatter
            ax.plot(valid_timesteps, valid_scores, 'r-', label=f'Noise: {specific_noise}')
            ax.scatter(valid_timesteps, valid_scores, color='red', s=50)
        
        # Set labels and title
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Score')
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    # Hide any unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot diffusion model metrics.')
    parser.add_argument('json_file', help='Path to the JSON file containing the data')
    parser.add_argument('--metrics', nargs='+', default=['all'], 
                        help='Metrics to plot (space-separated) or "all"')
    parser.add_argument('--noise_mode', choices=['average', 'specific'], default='average',
                        help='Mode for handling multiple noise values')
    parser.add_argument('--specific_noise', help='Specific noise value to plot (required if noise_mode is "specific")')
    parser.add_argument('--output', help='Path to save the output plot (optional)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.noise_mode == 'specific' and args.specific_noise is None:
        parser.error("--specific_noise is required when --noise_mode is 'specific'")
    
    # Load and process the data
    data = load_data(args.json_file)
    
    # Plot the metrics
    plot_metrics(
        data, 
        args.metrics, 
        args.noise_mode, 
        args.specific_noise,
        args.output
    )

if __name__ == "__main__":
    main()