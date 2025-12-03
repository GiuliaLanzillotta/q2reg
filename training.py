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
               loss_fn, num_steps, device, landscape_interval=1):
    
    model.train()
    avg_acc = 0.0
    
    # Store landscape metrics to average over the chunk
    landscape_metrics = []


    for step in range(num_steps):
        x, y, _ = next(train_iterator)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        
        output = model(x)
        task_loss = loss_fn(output, y)
        reg_loss = regularizer.compute_total_reg_loss(model)
        
        total_loss = task_loss + reg_loss
        total_loss.backward()
        
        # ---  Landscape Monitoring (Pre-Step) ---
        # We do this before optimizer.step() to capture the gradient state
        if step % landscape_interval == 0:
            
            # 1. Get Flat Gradient
            g_flat = utils.get_flat_grad(model)
            
            # 2. Alignment with Past (H_accum)
            if regularizer.past_samples: # Only if we have a regularizer active
                align_res = utils_landscape.compute_grad_hessian_alignment(
                    model, g_flat, regularizer
                )
            else:
                align_res = {}
            
            # 3. Sharpness of Current Task
            # Note: we use the current batch 'x' as a proxy for the task landscape
            sharp_res = utils_landscape.compute_current_task_sharpness(
                model, x, y, loss_fn
            )
            
            # Merge and store
            combined = {**align_res, **sharp_res}
            landscape_metrics.append(combined)
        # --------------------------------------------

        optimizer.step()
        scheduler[0].step()
        
        acc = (output.argmax(1) == y).float().mean().item()
        avg_acc += acc

    avg_acc /= num_steps
    
    # Aggregate landscape metrics (mean) for this chunk
    avg_landscape = {}
    if landscape_metrics:
        keys = landscape_metrics[0].keys()
        for k in keys:
            avg_landscape[k] = sum(d.get(k, 0) for d in landscape_metrics) / len(landscape_metrics)

    return avg_acc, avg_landscape

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
