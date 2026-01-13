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
def run_experiment_loop_sweep(config, env, network_init_fn, optimizer_init_fn, scheduler_init_fn, training_mode='cl'):
    device = config['device']
    loss_fn = lambda o, t: utils.loss_fn(o, t, config['loss'])
    
    # --- Initialize Model & Optimizer ---
    network = network_init_fn().to(device)
    
    # --- Initialize Regularizer ---
    if training_mode == 'regularized':
        active_regularizer = regularizers.get_regularizer(config)
    else:
        # Dummy regularizer for Sequential/Replay modes (Alpha 0)
        cfg_copy = config.copy(); cfg_copy['alpha'] = 0.0
        active_regularizer = regularizers.get_regularizer(cfg_copy)

    replay_buffer = [] 
    results = {}

    print(f"=== SWEEP START: Mode={training_mode}, Tasks={config['environment_args']['num_tasks']} ===")

    for t in range(config['environment_args']['num_tasks']):

        optimizer = optimizer_init_fn(network)
        scheduler = scheduler_init_fn(optimizer)
        task_data = env.init_single_task(task_number=t, train=True)
        
        g = torch.Generator()
        g.manual_seed(config['seed'])

        task_loader = DataLoader(
            task_data, batch_size=config["batch_size"], shuffle=True, 
            num_workers=4, worker_init_fn=utils.seed_worker, generator=g
        )
        
        # --- Construct Training Loader ---
        if training_mode == 'replay' and len(replay_buffer) > 0:
            full_data = ConcatDataset(replay_buffer + [task_data])
            train_loader = DataLoader(full_data, batch_size=config["batch_size"], shuffle=True, generator=g)
        else:
            train_loader = task_loader
            
        task_iterator = itertools.cycle(train_loader)
        task_metrics = {'performance': []}

        # --- Training Loop ---
        num_chunks = config['num_steps'] // config['log_every_n_steps']
        
        for chunk_idx in tqdm(range(num_chunks), desc=f'Task {t}', leave=False):
            # 1. Train Step
            avg_acc, avg_task_loss, avg_reg_loss = training.train_task(
                network, task_iterator, active_regularizer, optimizer, 
                scheduler, loss_fn, config['log_every_n_steps'], device, 
                hard_projection=config.get('projection', False),
                grad_clip=config['optimizer'].get('grad_clip', False)
            )
            # 2. Performance Evaluation (The only thing we care about)
            step = (chunk_idx + 1) * config['log_every_n_steps']
            during_eval = training.evaluate_cl_system(network, env, t, config, loss_fn, device)
            
            task_metrics['performance'].append({
                'task': t, 
                'step': step, 
                'train_acc': avg_acc,
                'avg_task_loss': avg_task_loss,
                **during_eval  # Contains 'test' metrics: current, avg_past, etc.
            })

            # Extract metrics for easier printing
            test_metrics = during_eval['test']
            curr_acc = test_metrics['current'] * 100
            past_acc = test_metrics.get('avg_past', 0) * 100
            
            # 3. Print Progress (using tqdm.write to avoid breaking progress bar)
            # This helps you see if the model is learning Task 2 while keeping Task 1
            tqdm.write(
                    f"[Task {t} Step {step:4d}] "
                    f"TrainAcc: {avg_acc*100:5.1f}% | "
                    f"CurrAcc: {curr_acc:5.1f}% | "
                    f"LR: {optimizer.param_groups[0]['lr']:.5f} | "
                    f"PastAcc: {past_acc:5.1f}% | "
                    f"Loss(T/R): {avg_task_loss:.3f}/{avg_reg_loss:.3f}"
                )

        results[t] = task_metrics

        # --- Post-Task Updates (Prepare for Task t+1) ---
        ds_replay, ds_reg = utils_io.get_task_subsets(task_data, config)

        if training_mode == 'regularized':
            active_regularizer.update(
                network, ds_reg, loss_fn, 
                accumulate=config['accumulate']
            )
            
        replay_buffer.append(ds_replay)

        # Print simple summary after each task
        final_acc = task_metrics['performance'][-1]['test']
        print(f"Task {t} Finished | Current: {final_acc['current']*100:.2f}% | Past: {final_acc['avg_past']*100:.2f}%")

    return results

# ==========================================
# 2. The CLI / Main Execution
# ==========================================

def sweep_main():
    parser = argparse.ArgumentParser(description="Run Lean CL Sweep")
    
    # Config File
    parser.add_argument('--config', type=str, default='config_2D_classification.yaml')

    # Overrides (Inherited from your main)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--reg_type', type=str)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--accumulate', action='store_true', default=False, dest='accumulate')
    parser.add_argument('--mode', type=str, default='cl', choices=['sequential', 'regularized', 'replay'])
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--curvature', type=str, default=None, choices=['hessian', 'fisher', 'true_fisher'])
    parser.add_argument('--ignore_gradient', action='store_true')
    parser.add_argument('--spectral_override', action='store_true', default=False)
    
    # --- New Optimization Flags ---
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--scheduler', type=str, choices=['step', 'cosine_anneal', 'plateau', 'one_cycle'])
    parser.add_argument('--step_scheduler_decay', type=float)
    parser.add_argument('--scheduler_step', type=int)
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum for SGD")
    
    parser.add_argument('--run_id', type=str, required=True, help="Directory to save metrics/config (provided by the sweep bash script)")
    args = parser.parse_args()

    # --- Load & Merge Config ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Apply Basic Overrides
    if args.exp_name: config['exp_name'] = args.exp_name
    if args.seed is not None: config['seed'] = args.seed
    if args.reg_type: config['reg_type'] = args.reg_type
    if args.alpha is not None: config['alpha'] = args.alpha
    if args.accumulate is not None: config['accumulate'] = args.accumulate
    if args.mode is not None: config['training_mode'] = args.mode
    if args.curvature is not None: config['curvature_type'] = args.curvature
    
    config['ignore_gradient'] = args.ignore_gradient if args.ignore_gradient else config.get('ignore_gradient', False)
    config['spectral_override'] = args.spectral_override

    # 2. Apply Nested Optimizer/Scheduler Overrides
    if args.optimizer: config['optimizer']['name'] = args.optimizer
    if args.lr is not None: config['optimizer']['lr'] = args.lr
    if args.batch_size is not None: config['batch_size'] = args.batch_size
    if args.momentum is not None: config['optimizer']['momentum'] = args.momentum

    if args.scheduler: 
        config['scheduler']['name'] = args.scheduler
        if args.step_scheduler_decay is not None: config['scheduler']['step_scheduler_decay'] = args.step_scheduler_decay
        if args.scheduler_step is not None: config['scheduler']['scheduler_step'] = args.scheduler_step
    
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Setup ---
    utils.seed_everything(config['seed'])

    # --- Environment ---
    env_args = config['environment_args']
    env_args_obj = utils.Dict2Args(**env_args)
    env, _ = environments.get_environment_from_name(config['environment'], env_args_obj)
    config['environment_args']['num_tasks'] = env.number_tasks 

    # --- Closures (Factories) ---
    def init_net():
        net_args = config['network']
        net_args['num_classes_total'] = env.num_classes
        return networks.get_network_from_name(net_args['name'], **net_args)

    def init_opt(network):
        return utils.setup_optimizer(config['optimizer'], network)
        
    def init_sched(optimizer):
        # We ensure the scheduler resets for every task in run_experiment_loop_sweep
        return utils.setup_scheduler(config['scheduler'], optimizer)

    # --- Minimal Storage Logic for Sweeps ---
    config['storage_folder'] = os.path.join(
        config.get('exp_name', 'default'), 
        args.run_id
    )
    print(f"--- SWEEP RUN: {args.run_id} ---")

    # --- Run Lean Loop ---
    # Using the optimized version without monitors/shadows
    results = run_experiment_loop_sweep(
        config, env, init_net, init_opt, init_sched, 
        training_mode=args.mode
    )

    # --- Save Minimal Output ---
    utils_io.save_experiment_results(results, config, base_dir=args.output_dir)

if __name__ == "__main__":
    sweep_main() # run the sweep