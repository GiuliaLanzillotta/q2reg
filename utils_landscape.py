# utils_landscape.py
import torch
import bitbybit.utils as utils
import utils_math
from torch.func import functional_call, vmap, hessian


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
    eigvals = torch.linalg.eigvalsh(H_full)
    
    return {
        'sharpness': eigvals[-1].item(),
        'trace': eigvals.sum().item()
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

def get_top_eigenvectors(matrix, k=1, structure='full'):
    """
    Returns the top-k eigenvectors of the aggregated curvature.
    """
    if structure == 'diag':
        # matrix is 1D tensor (Diagonal). 
        # Eigenvectors are canonical basis vectors at indices of largest elements.
        top_indices = torch.topk(matrix, k).indices
        
        vectors = []
        d = matrix.shape[0]
        for idx in top_indices:
            v = torch.zeros(d, device=matrix.device)
            v[idx] = 1.0
            vectors.append(v)
        return torch.stack(vectors) # (k, d)

    elif structure in ['full', 'block']:
        # matrix is 2D. 
        try:
            # Use eigh for symmetric matrices (Fisher/Hessian are symmetric)
            # Returns eigenvalues (ascending) and eigenvectors
            L, V = torch.linalg.eigh(matrix)
            
            # Take last k columns (largest eigenvalues) and transpose to rows
            top_vectors = V[:, -k:].T.flip(0) 
            return top_vectors
        except Exception as e:
            print(f"Eigen-decomp failed: {e}")
            return None
            
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
        
        # Absolute dot product (we care about alignment with the axis, +/- doesn't matter)
        score = torch.dot(u_norm, v_norm).item()
        alignments.append(score)
        
    return alignments


