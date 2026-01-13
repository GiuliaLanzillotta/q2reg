""" Training and evaluation functions for continual learning with quadratic regularizers."""
import sys
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

import torch
import numpy as np
import bitbybit.utils as utils
import itertools
from torch.utils.data import ConcatDataset, DataLoader

# In training.py
import utils_landscape
import bitbybit.utils as utils

def train_task(model, train_iterator, regularizer, optimizer, scheduler, 
               loss_fn, num_steps, device, hard_projection=False, grad_clip=False):
    
    model.train()
    # Stats accumulators
    total_acc = 0.0
    total_task_loss = 0.0
    total_reg_loss = 0.0
    


    for step in range(num_steps):
        x, y, _ = next(train_iterator)
        x, y = x.to(device), y.to(device)

        # 1. Snapshot BEFORE step
        previous_params = utils.get_flat_params(model).clone()

        optimizer.zero_grad()
        output = model(x)
        task_loss = loss_fn(output, y)

        # Logic Split
        if hard_projection:
            # A. Hard Mode: Loss is just task loss
            total_loss = task_loss
            reg_loss = torch.tensor(0.0)
        else:
            # B. Soft Mode: Loss includes penalty
            reg_loss = regularizer.compute_total_reg_loss(model)
            total_loss = task_loss + reg_loss
            
        total_loss.backward()
        if grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler[0].step()
        
        # 2. Projection AFTER step (only if hard mode)
        if hard_projection:
            regularizer.project_weights(model, previous_params_flat=previous_params)


        # 5. Track Basic Stats
        total_task_loss += task_loss.item()
        total_reg_loss += reg_loss
        acc = (output.argmax(dim=1) == y).float().mean().item()
        total_acc += acc



    # Return averaged stats
    avg_acc = total_acc / num_steps
    avg_task_loss = total_task_loss / num_steps
    avg_reg_loss = total_reg_loss / num_steps

    # if it's a tensor, convert to float
    if torch.is_tensor(avg_reg_loss):
        avg_reg_loss = avg_reg_loss.item()
    
    return avg_acc,  avg_task_loss, avg_reg_loss


@torch.enable_grad() # Ensure gradients are computed for this function
def evaluate_past_metrics(model, regularizer, loss_fn, device):
    """
    Computes a full suite of metrics for ALL past samples.
    Returns:
        A list of dictionaries, where each dict contains the
        metrics for a single past sample.
    """
    model.eval() # Set to eval mode, but grads are enabled by the decorator
    past_samples = regularizer.get_past_data()
    
    if not past_samples:
        return [] # No past samples, no metrics

    # This will be a list of dicts
    all_sample_metrics = []

    for i, (x_i, y_i) in enumerate(past_samples):
        x_i, y_i = x_i.to(device), y_i.to(device)
        
        # This dict will store metrics for this single sample
        current_sample_metrics = {
            'sample_id': i  # Store the index in the regularizer's memory
        }
        
        model.zero_grad()
        
        # --- 1. True Loss, Accuracy, and Gradient ---
        output = model(x_i)
        true_loss = loss_fn(output, y_i)
        
        # Compute true gradient
        true_grad_tuple = torch.autograd.grad(true_loss, model.parameters())
        true_grad_flat = torch.cat([g.view(-1) for g in true_grad_tuple if g is not None])
        
        # Compute accuracy (forgetting)
        is_correct = (torch.argmax(output, dim=1) == y_i).item()
        current_sample_metrics['accuracy'] = is_correct
        peak_acc = regularizer.past_sample_peak_acc[i]
        current_sample_metrics['forgotten'] = peak_acc - is_correct
        
        # --- 2. Proxy Loss and Gradient ---
        proxy_loss, proxy_grad_flat = regularizer.compute_per_sample_proxy_loss_and_grad(model, i)
        
        # --- 3. Compute Error Metrics ---
        
        # Loss Error (kappa) = \loss_i - \hat\loss_i
        current_sample_metrics['loss_error'] = (true_loss.item() - proxy_loss.item())
        
        # Gradient Error (Norm)
        grad_error_vec = true_grad_flat - proxy_grad_flat
        current_sample_metrics['grad_norm_error'] = torch.norm(grad_error_vec).item()
        
        # Gradient Error (Cosine Similarity)
        cos_sim = torch.nn.functional.cosine_similarity(
            true_grad_flat + 1e-8, proxy_grad_flat + 1e-8, dim=0
        ).item()
        current_sample_metrics['grad_cos_sim'] = cos_sim
        
        # --- 4. Store this sample's metrics ---
        all_sample_metrics.append(current_sample_metrics)

    # Return the full list
    return all_sample_metrics


@torch.no_grad() # This evaluation doesn't need gradients
def evaluate_on_all_past(model, replay_buffer, loss_fn, device):
    """
    Evaluates model's loss and accuracy on ALL samples from past tasks.
    
    Args:
        model (nn.Module): The model to evaluate.
        replay_buffer (list): List of TensorDatasets, one for each past task.
        loss_fn: The loss function.
        device: The device to run on.
    
    Returns:
        A dict containing 'mean_loss' and 'mean_accuracy'.
    """
    model.eval()
    
    if not replay_buffer:
        return {'mean_loss': 0.0, 'mean_accuracy': 0.0}

    # Combine all past datasets into one
    full_past_dataset = ConcatDataset(replay_buffer)
    # Use a loader for efficient batching
    past_loader = DataLoader(full_past_dataset, batch_size=32)

    all_losses = []
    all_corrects = []

    for x, y, _ in past_loader:
        x, y = x.to(device), y.to(device)
        
        output = model(x)
        loss = loss_fn(output, y)
        
        all_losses.append(loss.item() * len(y)) # Store total loss for this batch
        
        preds = torch.argmax(output, dim=1)
        all_corrects.append((preds == y).sum().item())
        
    total_samples = len(full_past_dataset)
    if total_samples == 0:
        return {'mean_loss': 0.0, 'mean_accuracy': 0.0}
        
    mean_loss = np.sum(all_losses) / total_samples
    mean_accuracy = np.sum(all_corrects) / total_samples
    
    return {'mean_loss': mean_loss, 'mean_accuracy': mean_accuracy}

@torch.no_grad()
def evaluate_model(model, data_loader, loss_fn, device):
    """
    Evaluates model's loss and accuracy on a provided DataLoader.
    
    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): The loader containing the evaluation set.
        loss_fn: The loss function.
        device: The device to run on.
    
    Returns:
        A dict containing 'loss' and 'accuracy'.
    """
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y, _ in data_loader: # Unpacking the 3-tuple from your dataset
        x, y = x.to(device), y.to(device)
        
        output = model(x)
        loss = loss_fn(output, y)
        
        # Accumulate metrics
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        
        preds = torch.argmax(output, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += batch_size
        
    if total_samples == 0:
        return {'loss': 0.0, 'accuracy': 0.0}
        
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples
    }

@torch.no_grad()
def evaluate_cl_system(network, env, current_task_idx, config, loss_fn, device):
    """
    Evaluates ALL tasks in the environment to track learning, forgetting, 
    and forward transfer.
    """
    network.eval()
    total_tasks = config['environment_args']['num_tasks']
    
    # Storage for raw task metrics
    raw_results = {'train': {}, 'test': {}}
    
    for t_idx in range(total_tasks):
        for split in ['train', 'test']:
            # Load the full dataset for the task
            data = env.init_single_task(task_number=t_idx, train=(split == 'train'))
            loader = DataLoader(data, batch_size=config.get('batch_size', 32), shuffle=False)
            
            # Use the generic evaluator we built
            metrics = evaluate_model(network, loader, loss_fn, device)
            raw_results[split][t_idx] = metrics

    # --- Compute Specific Averages ---
    summary = {}
    for split in ['train', 'test']:
        accuracies = [raw_results[split][i]['accuracy'] for i in range(total_tasks)]
        
        # 1. Total Average (All tasks)
        avg_total = np.mean(accuracies)
        
        # 2. Current Task Accuracy
        curr_acc = accuracies[current_task_idx]
        
        # 3. Past Average (Tasks 0 to current-1)
        past_accs = accuracies[:current_task_idx]
        avg_past = np.mean(past_accs) if past_accs else 0.0
        
        # 4. Future Average (Tasks current+1 to end)
        future_accs = accuracies[current_task_idx+1:]
        avg_future = np.mean(future_accs) if future_accs else 0.0
        
        summary[split] = {
            'task_raw': raw_results[split],
            'avg_total': avg_total,
            'avg_past': avg_past,
            'current': curr_acc,
            'avg_future': avg_future
        }

    network.train()
    return summary

def track_geometry_dynamics(model, top_eigs, previous_params, task_anchor_params):
    current_params = utils.get_flat_params(model)
    eps = 1e-12
    
    # 1. VELOCITY (Step-wise movement)
    update_vec = current_params - previous_params
    update_norm = update_vec.norm().item()
    
    # 2. DRIFT (Total movement from anchor)
    displacement = current_params - task_anchor_params
    disp_norm = displacement.norm().item()

    # --- Efficiency: One call to compute projections ---
    # We use raw dot products for math, then derive alignment from them
    # proj_k = <vec, v_k>
    vel_dots = torch.mv(top_eigs, update_vec) if update_norm > eps else torch.zeros(len(top_eigs))
    disp_dots = torch.mv(top_eigs, displacement) if disp_norm > eps else torch.zeros(len(top_eigs))

    # --- Calculate Leakage (Pythagorean theorem in the subspace) ---
    # norm_in_subspace = sqrt(sum(proj_k^2))
    norm_vel_on_top = torch.norm(vel_dots).item()
    norm_disp_on_top = torch.norm(disp_dots).item()

    # Leakage = 1 - (proportion of norm captured by top eigenvectors)
    leakage_vel = 1.0 - (norm_vel_on_top / (update_norm + eps))
    leakage_drift = 1.0 - (norm_disp_on_top / (disp_norm + eps))

    # --- Format Outputs for your existing Analysis Code ---
    # align_scores: Dot products normalized by update_norm (Cosine Similarity)
    align_scores = (vel_dots / (update_norm + eps)).tolist()
    # proj_scores: Raw dot products (Distance)
    proj_scores = disp_dots.tolist()

    metrics = {
        'align_scores': align_scores,
        'proj_scores': proj_scores,
        'leakage_vel': max(0.0, leakage_vel),
        'leakage_drift': max(0.0, leakage_drift),
        'total_disp_norm': disp_norm
    }
    
    return metrics, current_params