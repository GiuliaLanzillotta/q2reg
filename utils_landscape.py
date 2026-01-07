# utils_landscape.py
import random
import torch
import bitbybit.utils as utils
import utils_math
from torch.func import functional_call, vmap, hessian
from torch.utils.data import DataLoader, Subset
import random
import numpy as np


def compute_projection(displacement_vector, eigen_vectors):
    """
    Computes the scalar projection (distance) of the displacement onto the eigenvectors.
    Result = (displacement . eigenvector)
    """
    # Ensure device match
    if eigen_vectors.device != displacement_vector.device:
        eigen_vectors = eigen_vectors.to(displacement_vector.device)

    # eigen_vectors: [K, D] (Rows are vectors)
    # displacement: [D]
    
    # Matrix-Vector multiplication: [K, D] @ [D] -> [K]
    # We want the absolute distance along that axis
    projections = torch.mv(eigen_vectors, displacement_vector)
    
    return projections.tolist()

def compute_current_task_sharpness(model, x, y, loss_fn):
    """
    Computes the sharpness (max eigenvalue) using the shared utility.
    """
    # Calculate Full Hessian using the robust utility
    # We pass None for params_flat to use current weights
    _, _, H_full = utils_math.compute_loss_grad_hessian(
        model, loss_fn, x, y, params_flat=None
    )

    
    
    # Compute Eigenvalues (Symmetric)
    _, eigvals, _ = torch.linalg.svd(H_full)

    # 3. Compute Numerical Rank
    # Floating point math is noisy, so we don't check > 0.0.
    # We check if values are > threshold (e.g. 0.0001% of the max eigenvalue).
    max_abs_eig = torch.max(torch.abs(eigvals))
    tolerance = max_abs_eig * 1e-6 if max_abs_eig > 0 else 1e-6
    
    # Count how many eigenvalues represent "real" curvature info
    rank = (eigvals.abs() > tolerance).sum().item()
    
    return {
        'sharpness': eigvals[-1].item(),
        'trace': eigvals.sum().item(),
        'rank': int(rank),
        'rank_fraction': rank / len(eigvals) # Percentage of dimensions that are active
    }

def compute_past_sharpness(model, regularizer, loss_fn, sample_limit=200):
    """
    Computes sharpness on a subset of samples stored in the regularizer.
    Reuses compute_current_task_sharpness.
    """
    if not regularizer.past_samples:
        return {'sharpness_past': 0.0, 'trace_past': 0.0, 'rank_past': 0, 'rank_fraction_past': 0.0}
    
    device = next(model.parameters()).device
    
    # 1. Sample random subset to avoid OOM with Full Hessian
    # regularizer.past_samples is a list of (x,y)
    n_available = len(regularizer.past_samples)
    subset = random.sample(regularizer.past_samples, min(n_available, sample_limit))
    
    # 2. Collate into a single batch
    # We ensure they are on the correct device
    x_list = [s[0].to(device) for s in subset]
    y_list = [s[1].to(device) for s in subset]
    
    # Handle potentially different shapes (unlikely in simple CL, but good practice)
    # Unsqueeze if single sample [D] -> [1, D]
    x_batch = torch.cat([x.unsqueeze(0) if x.dim() == 1 else x for x in x_list])
    y_batch = torch.cat([y.unsqueeze(0) if y.dim() == 0 else y for y in y_list])
    
    # 3. Call your EXISTING function
    # It returns {'sharpness': ..., 'trace': ...}
    res = compute_current_task_sharpness(model, x_batch, y_batch, loss_fn)
    
    # Rename keys for clarity in logs
    return {
        'sharpness_past': res['sharpness'], 
        'trace_past': res['trace'],
        'rank_past': res['rank'],
        'rank_fraction_past': res['rank_fraction']
    }

def compute_grad_hessian_alignment(model, current_grad, regularizer):
    """
    Computes alignment between the current update gradient and the 
    accumulated Hessian of the regularizer.
    """
    # 1. Reconstruct the Total Accumulated Hessian/Fisher Matrix
    # The regularizer stores \Lambda_i (diag or block or full). 
    # We need to sum them to get the total curvature constraint.
    
    if not regularizer.per_sample_importances:
        return {'alignment_energy': 0.0, 'alignment_top_eig': 0.0}

    # Stack and sum to get total Lambda (approx Hessian of Past)
    # Shape depends on structure (Diag: [D], Full: [D, D])
    lambdas = torch.stack(regularizer.per_sample_importances)
    H_accum = torch.sum(lambdas, dim=0).to(current_grad.device)
    
    # 2. Metric A: Quadratic Energy (g^T * H * g)
    # How much does this gradient increase the regularization penalty?
    if regularizer.structure == 'diag':
        energy = torch.sum(H_accum * (current_grad ** 2))
    elif regularizer.structure in ['full','block']:
        energy = current_grad @ H_accum @ current_grad
    else:
        # Handle other cases if needed
        energy = torch.tensor(0.0) 

    # 3. Metric B: Cosine Sim with Top Eigenvector of H_accum
    # (Does the gradient point in the "stiffest" direction of the past?)
    if regularizer.structure == 'diag':
        # For diagonal, eigenvectors are one-hot vectors. 
        # The top eigenvector corresponds to the index of max value.
        # We just check if the gradient is large at that index.
        max_idx = torch.argmax(H_accum)
        top_eig_vec = torch.zeros_like(current_grad)
        top_eig_vec[max_idx] = 1.0
    
    elif regularizer.structure in ['block','full']:
        # Full Eigendecompositio
        eigvals, eigvecs = torch.linalg.eigh(H_accum)
        top_eig_vec = eigvecs[:, -1] # Last column is top vector
    
    alignment = torch.nn.functional.cosine_similarity(
        current_grad.unsqueeze(0), 
        top_eig_vec.unsqueeze(0)
    ).item()

    return {
        'alignment_energy': energy.item(),
        'alignment_top_eig': alignment
    }

def get_total_curvature_from_regularizer(regularizer):
    """
    Aggregates per-sample importances into a single global curvature tensor.
    Handles 'diag' (vector sum) and 'full/block' (matrix sum).
    """
    if not regularizer.per_sample_importances:
        return None
        
    # Summing iteratively is memory-safer than stacking
    # Start with the first one
    total_curvature = regularizer.per_sample_importances[0].clone()
    
    # Add the rest
    for i in range(1, len(regularizer.per_sample_importances)):
        total_curvature += regularizer.per_sample_importances[i]
        
    return total_curvature

def get_top_eigenvectors(matrix, k=1, structure='full', tol = 1e-6):
    """
    Returns the top-k eigenvectors of the aggregated curvature.
    """
    
    if structure == 'diag':
        # matrix is 1D tensor (Diagonal). 

        # matrix is 1D tensor [D]
        # A. Compute Rank (Count of non-zero elements)
        rank = (matrix.abs() > tol).sum().item()

        # Eigenvectors are canonical basis vectors at indices of largest elements.
        top_indices = torch.topk(matrix, k).indices
        vectors = []
        d = matrix.shape[0]
        for idx in top_indices:
            v = torch.zeros(d, device=matrix.device)
            v[idx] = 1.0
            vectors.append(v)
        return torch.stack(vectors), rank # (k, d)

    elif structure in ['full', 'block']:
        # matrix is 2D. 
        try:
            # Use eigh for symmetric matrices (Fisher/Hessian are symmetric)
            # Returns eigenvalues (ascending) and eigenvectors
            L, V = torch.linalg.eigh(matrix)
            rank = (L.abs() > tol).sum().item()
            
            # Take last k columns (largest eigenvalues) and transpose to rows
            top_vectors = V[:, -k:].T.flip(0) 
            return top_vectors, rank
        except Exception as e:
            print(f"Eigen-decomp failed: {e}")
            return None, None
            
def compute_alignment(update_vector, eigen_vectors):
    """
    Computes cosine similarity between update and top eigenvectors.
    """
    # --- FIX: Ensure devices match ---
    if eigen_vectors.device != update_vector.device:
        eigen_vectors = eigen_vectors.to(update_vector.device)

    # Normalize update to unit length
    u_norm = update_vector / (update_vector.norm() + 1e-8)
    
    alignments = []
    for v in eigen_vectors:
        # v should be unit length, but be safe
        v_norm = v / (v.norm() + 1e-8)
        
        # Absolute dot product 
        score = torch.dot(u_norm, v_norm).item()
        alignments.append(score)
        
    return alignments


def evaluate_landscape_sharpness(model, task_data, shadow_monitor, loss_fn, device, sample_size=200):
    """
    Evaluates the local geometry (Hessian Sharpness) on:
    1. The Current Task (using a random subset of task_data)
    2. The Past Tasks (using samples stored in the Shadow Monitor)
    """
    model.eval()
    metrics = {}
    
    # --- 1. CURRENT TASK SHARPNESS ---
    # Quick subsample from the dataset
    indices = np.random.choice(len(task_data), min(len(task_data), sample_size), replace=False)
    subset = Subset(task_data, indices)
    
    # Create a loader just to handle collate_fn automatically
    loader = DataLoader(subset, batch_size=sample_size)
    x_curr, y_curr, _ = next(iter(loader))
    x_curr, y_curr = x_curr.to(device), y_curr.to(device)
    
    # Reuse your existing robust Hessian computation
    res_curr = compute_current_task_sharpness(model, x_curr, y_curr, loss_fn)
    
    metrics['sharpness_curr'] = res_curr['sharpness']
    metrics['trace_curr'] = res_curr['trace']
    metrics['rank_curr'] = res_curr['rank']
    metrics['rank_fraction_curr'] = res_curr['rank_fraction']
    
    # --- 2. PAST TASK SHARPNESS ---
    # We use the Accumulate Monitor as the source of truth for "Past Constraints"
    # It acts as a Regularizer object, so it has .past_samples
    if shadow_monitor is not None and hasattr(shadow_monitor, 'reg_accum'):
        past_reg = shadow_monitor.reg_accum
        
        # Reuse your existing past sharpness wrapper
        # (Ensure utils_landscape.compute_past_sharpness is defined as discussed previously)
        res_past = compute_past_sharpness(model, past_reg, loss_fn, sample_limit=sample_size)
        
        metrics['sharpness_past'] = res_past['sharpness_past']
        metrics['trace_past'] = res_past['trace_past']
        metrics['rank_past'] = res_past['rank_past']
        metrics['rank_fraction_past'] = res_past['rank_fraction_past']
    else:
        metrics['sharpness_past'] = 0.0
        metrics['trace_past'] = 0.0
        metrics['rank_past'] = 0.0
        metrics['rank_fraction_past'] = 0.0
        
    return metrics


# utils.py

def compute_gradient_stats(model, regularizer, loss_fn, task_data, device, sample_size=32):
    """
    Computes Gradient Norms and Cosine Similarity between Task and Reg gradients.
    """
    model.eval() # Use eval to avoid BatchNorm stats updates during checking
    model.zero_grad()
    
    # 1. Get Batch
    indices = np.random.choice(len(task_data), min(len(task_data), sample_size), replace=False)
    loader = DataLoader(Subset(task_data, indices), batch_size=sample_size)
    x, y, _ = next(iter(loader))
    x, y = x.to(device), y.to(device)
    
    # 2. Task Gradient
    out = model(x)
    loss = loss_fn(out, y)
    
    params = [p for p in model.parameters() if p.requires_grad]
    g_task = torch.autograd.grad(loss, params)
    g_task_flat = torch.cat([g.view(-1) for g in g_task])
    
    stats = {'grad_norm_task': g_task_flat.norm().item()}
    
    # 3. Reg Gradient
    if regularizer is not None and regularizer.anchor_param_list:
        model.zero_grad()
        reg_loss = regularizer.compute_total_reg_loss(model)
        
        g_reg = torch.autograd.grad(reg_loss, params)
        g_reg_flat = torch.cat([g.view(-1) for g in g_reg])
        
        stats['grad_norm_reg'] = g_reg_flat.norm().item()
        
        # Cosine Sim
        cos_sim = torch.nn.functional.cosine_similarity(
            g_task_flat.unsqueeze(0), g_reg_flat.unsqueeze(0)
        )
        stats['grad_cos_sim'] = cos_sim.item()
    else:
        stats['grad_norm_reg'] = 0.0
        stats['grad_cos_sim'] = 0.0
        
    return stats