# In utils_io.py
import torch
import json
import os
from datetime import datetime
import pickle

def save_experiment_results(results, config, base_dir='results'):
    """
    Saves experiment results and config to disk.
    Structure: results/{exp_name}/{timestamp}_seed{seed}/
    """
    exp_name = config.get('exp_name', 'default')
    seed = config.get('seed', 0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_path = os.path.join(base_dir, exp_name, f"seed_{seed}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save Config
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        json.dump(config, f, indent=4) # Using JSON for simplicity
        
    # Save Results (Pickle is best for nested dicts with Tensors/Arrays)
    # We convert tensors to lists/floats to be safe, or just use torch.save
    torch.save(results, os.path.join(save_path, 'metrics.pt'))
    
    print(f"Results saved to: {save_path}")
    return save_path

def load_results_from_dir(exp_dir):
    """
    Loads all seeds from a specific experiment directory.
    """
    seeds_data = []
    
    if not os.path.exists(exp_dir):
        print(f"Directory not found: {exp_dir}")
        return []

    for seed_folder in os.listdir(exp_dir):
        full_path = os.path.join(exp_dir, seed_folder)
        if os.path.isdir(full_path):
            metrics_path = os.path.join(full_path, 'metrics.pt')
            if os.path.exists(metrics_path):
                data = torch.load(metrics_path, weights_only=False)
                seeds_data.append(data)
    
    print(f"Loaded {len(seeds_data)} seeds from {exp_dir}")
    return seeds_data