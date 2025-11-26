import argparse
import yaml
import torch
import sys
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

import numpy as np
import itertools
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset

# Import your helper modules
import bitbybit.utils as utils
import utils_io
import utils_monitor
import utils_landscape
import training
import regularizers
import bitbybit.environments as environments 
import bitbybit.networks as networks






# ==========================================
# 1. The Core Experiment Logic
# ==========================================
def run_experiment_loop(config, env, network_init_fn, optimizer_init_fn, scheduler_init_fn, training_mode='cl'):
    
    device = config['device']
    loss_fn = lambda o, t: utils.loss_fn(o, t, config['loss'])
    
    # --- Initialize Model & Optimizer ---
    network = network_init_fn().to(device)
    optimizer = optimizer_init_fn(network)
    scheduler = scheduler_init_fn(optimizer)
    
    # --- Initialize Active Regularizer (Training) ---
    if training_mode == 'regularized':
        active_regularizer = regularizers.get_regularizer(config)
    else:
        # Dummy regularizer for Replay mode
        cfg_copy = config.copy(); cfg_copy['alpha'] = 0.0
        active_regularizer = regularizers.get_regularizer(cfg_copy)

    # --- Initialize Shadow Monitor (Tracking) ---
    shadow_monitor = utils_monitor.ShadowMonitor(config, loss_fn)
    
    replay_buffer = [] 
    results = {} 

    print(f"=== Starting Loop: Mode={training_mode}, Tasks={config['environment_args']['num_tasks']} ===")

    for t in range(config['environment_args']['num_tasks']):
        task_data = env.init_single_task(task_number=t, train=True)
        
        # Create Generator for reproducibility
        g = torch.Generator()
        g.manual_seed(config['seed'])

        task_loader = DataLoader(
            task_data, 
            batch_size=config["batch_size"], 
            shuffle=True, 
            num_workers=4,
            worker_init_fn=utils.seed_worker,
            generator=g
        )
        
        # --- Construct Training Loader ---
        if training_mode == 'replay' and len(replay_buffer) > 0:
            full_data = ConcatDataset(replay_buffer + [task_data])
            train_loader = DataLoader(full_data, batch_size=config["batch_size"], shuffle=True, num_workers=4, worker_init_fn=utils.seed_worker, generator=g)
        else:
            train_loader = task_loader
            
        task_iterator = itertools.cycle(train_loader)
        task_metrics = {'history': [], 'landscape': []}

        # --- Training Loop ---
        num_chunks = config['num_steps'] // config['log_every_n_steps']
        
        # Initial Eval (Task 0 has no past, so skip)
        if t > 0:
            monitor_stats = utils_monitor.evaluate_shadow_monitors(network, shadow_monitor, device)
            past_metrics_all = training.evaluate_on_all_past(network, replay_buffer, loss_fn, device)
            
            acc_reg = np.mean([m['accuracy'] for m in monitor_stats])
            tqdm.write(
                f"  [T{t+1}, PRE] | "
                f"PAST-ALL Acc: {past_metrics_all['mean_accuracy']*100:6.2f}% | "
                f"PAST-REG Acc: {acc_reg*100:6.2f}%"
            )

        for chunk_idx in tqdm(range(num_chunks), desc=f'Task {t} ({training_mode})'):
            
            # A. Train Step
            avg_acc, avg_landscape = training.train_task(
                network, task_iterator, active_regularizer, optimizer, 
                scheduler, loss_fn, config['log_every_n_steps'], device
            )
            
            step = (chunk_idx + 1) * config['log_every_n_steps']
            
            # B. Evaluate & Log
            if t > 0:
                # Monitor Stats (Past Reg samples)
                monitor_stats = utils_monitor.evaluate_shadow_monitors(network, shadow_monitor, device)

                
                task_metrics['history'].append(monitor_stats)
                
                # Landscape Stats
                task_metrics['landscape'].append(avg_landscape)
                
                # Evaluate on ALL past samples (Replay Buffer)
                past_metrics_all = training.evaluate_on_all_past(network, replay_buffer, loss_fn, device)

                acc_reg = np.mean([m['accuracy'] for m in monitor_stats])
                
                tqdm.write(
                    f"  [T{t+1}, Step {step: >4}] | "
                    f"CURR: {avg_acc*100:5.1f}% | "
                    f"ALL: {past_metrics_all['mean_accuracy']*100:5.1f}% | "
                    f"REG: {acc_reg*100:5.1f}% | "
                    f"Sharp: {avg_landscape.get('sharpness', 0):.4f}"
                )
            else:
                tqdm.write(f"  [T{t+1}, Step {step: >4}] | CURR: {avg_acc*100:5.1f}%")

        results[t] = task_metrics

        # --- Post-Task Updates ---
        print(f"  > Task {t} Done. Sampling datasets...")

        # 1. Sample for Replay Buffer
        frac_replay = config.get('replay_frac', 0.1)
        ds_replay = utils.subsample_dataset(task_data, frac_replay)

        # 2. Sample for Regularizer/Monitor
        frac_reg = config.get('reg_frac', 0.05)
        ds_reg = utils.subsample_dataset(task_data, frac_reg)

        # 3. Update Active Regularizer
        if training_mode == 'regularized':
            active_regularizer.update(network, ds_reg, loss_fn, accumulate=config['accumulate'])
            
        # 4. Update Shadow Monitor
        shadow_monitor.update(network, ds_reg)
        
        # 5. Update Replay Buffer
        replay_buffer.append(ds_replay)

        # --- Final Task Eval (Optional but good for summary) ---
        if t > 0:
             past_metrics_all = training.evaluate_on_all_past(network, replay_buffer, loss_fn, device)
             print(f"  > End Task {t+1} Summary: Past-All Accuracy: {past_metrics_all['mean_accuracy']*100:.2f}%")
        
    return results

def make_config_slug(config):
    """
    Creates a unique, deterministic folder name based on important config keys.
    Format: key1_val1-key2_val2-...
    """
    # 1. Define the keys that define a unique 'setting' (excluding seed)
    # You can simply add 'gamma' to this list later.
    keys_to_track = [
        'training_mode', 
        'reg_type', 
        'curvature_type', 
        'accumulate', 
        'ignore_gradient',
        'alpha',
        # 'gamma'  <-- Future proofing: just add this here later!
    ]
    
    parts = []
    for k in keys_to_track:
        if k == 'ignore_gradient': # Special handling for boolean
            if config.get(k, False):
                parts.append(f"no_grad")
            else:
                parts.append(f"with_grad")
            continue

        # Get val, default to 'NA' if missing
        val = config.get(k, 'NA')
        
        # Clean up the string (optional, keeps folders tidy)
        val_str = str(val).replace("taylor-", "") # shorter names
        
        parts.append(f"{k}_{val_str}")
        
    return "-".join(parts)

# ==========================================
# 2. The CLI / Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Run CL Experiment")
    
    # Config File
    parser.add_argument('--config', type=str, default='config_2D_classification.yaml')

    # Overrides
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--reg_type', type=str)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--accumulate', action='store_true', default=False, dest='accumulate')
    
    parser.add_argument('--mode', type=str, default='cl', choices=['sequential', 'regularized', 'replay'])
    parser.add_argument('--output_dir', type=str, default='./results')

    parser.add_argument('--curvature', type=str, default=None, choices=['hessian', 'fisher'],help="Type of curvature matrix to use (hessian=True 2nd derivative, fisher=Gradient outer product)")
    parser.add_argument('--ignore_gradient', action='store_true', 
                        help="If set, the Taylor approximation drops the linear g*delta term.")
    
    args = parser.parse_args()

    # --- Load & Merge Config ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if args.exp_name: config['exp_name'] = args.exp_name
    if args.seed is not None: config['seed'] = args.seed
    if args.reg_type: config['reg_type'] = args.reg_type
    if args.alpha is not None: config['alpha'] = args.alpha
    if args.accumulate is not None: config['accumulate'] = args.accumulate
    if args.mode is not None: config['training_mode'] = args.mode
    if args.curvature is not None: config['curvature_type'] = args.curvature
    if args.ignore_gradient:
            config['ignore_gradient'] = False
    else:
        # Ensure default is set explicitly for clarity in logs
        config['ignore_gradient'] = config.get('ignore_gradient', False)
    
    # Set device
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"--- Running: {config.get('exp_name')} | Mode: {args.mode} | Seed: {config['seed']} ---")

    # --- Setup ---
    utils.seed_everything(config['seed'])

    # --- Environment ---
    env_args = config['environment_args']
    env_args_obj = utils.MyArgs(**env_args)
    
    env, env_name = environments.get_environment_from_name(
        config['environment'], env_args_obj
    )
    config['environment_args']['num_tasks'] = env.number_tasks # Ensure config matches env

    # --- Closures for Re-initialization ---
    def init_net():
        net_args = config['network']
        net_args['num_classes_total'] = env.num_classes
        return networks.get_network_from_name(net_args['name'], **net_args)

    def init_opt(network):
        return utils.setup_optimizer(config['optimizer'], network)
        
    def init_sched(optimizer):
        return utils.setup_scheduler(config['scheduler'], optimizer)

    # --- Run ---
    results = run_experiment_loop(
        config, env, init_net, init_opt, init_sched, 
        training_mode=args.mode
    )

    # --- Save ---
    save_config = config.copy()
    # Generate the folder name dynamically
    slug_name = make_config_slug(config)
    # Structure: results / exp_name / param_string / seed_X
    save_config['exp_name'] = os.path.join(
        save_config.get('exp_name', 'default'), 
        slug_name
    )
    
    utils_io.save_experiment_results(results, save_config, base_dir=args.output_dir)

if __name__ == "__main__":
    main()