# utils_math.py
import torch
from torch.func import functional_call, hessian, grad

def get_params_dict_from_flat(model, flat_params):
    """
    Reconstructs a parameter dictionary (state_dict style) from a flat vector
    based on the model's structure.
    """
    keys = [k for k, _ in model.named_parameters()]
    shapes = [p.shape for p in model.parameters()]
    
    params_dict = {}
    pointer = 0
    for k, shape in zip(keys, shapes):
        numel = torch.tensor(shape).prod().item()
        # We view the slice to reshape it
        params_dict[k] = flat_params[pointer : pointer+numel].view(shape)
        pointer += numel
        
    return params_dict

def compute_loss_grad_hessian(model, loss_fn, x, y, params_flat=None):
    """
    Computes Loss, Gradient, and Hessian.
    Ensures inputs match the device of the parameters.
    """
    
    # 1. Prepare parameters & Determine Device
    if params_flat is None:
        params_dict = dict(model.named_parameters())
        # Get device from the first parameter of the model
        device = next(model.parameters()).device
    else:
        params_dict = get_params_dict_from_flat(model, params_flat)
        # Get device from the flat vector
        device = params_flat.device

    # 2. Ensure Inputs are on the Correct Device
    x = x.to(device)
    y = y.to(device)

    # 3. Define the functional loss 
    def func_loss(p_dict):
        # Unsqueeze if necessary (for single sample processing)
        x_in = x.unsqueeze(0) if x.dim() == 1 else x
        y_in = y.unsqueeze(0) if y.dim() == 0 else y
        
        out = functional_call(model, p_dict, (x_in,))
        return loss_fn(out, y_in)

    # 4. Compute Functional Hessian & Grad
    H_dict = hessian(func_loss)(params_dict)
    g_dict = grad(func_loss)(params_dict)
    loss_val = func_loss(params_dict)
    
    # 5. Flatten Gradient
    keys = list(params_dict.keys())
    g_flat = torch.cat([g_dict[k].view(-1) for k in keys])
    
    # 6. Flatten Hessian
    h_rows = []
    for k1 in keys:
        row_blocks = []
        for k2 in keys:
            block = H_dict[k1][k2]
            n_k1 = params_dict[k1].numel()
            n_k2 = params_dict[k2].numel()
            block = block.view(n_k1, n_k2)
            row_blocks.append(block)
        h_rows.append(torch.cat(row_blocks, dim=1))
        
    H_full = torch.cat(h_rows, dim=0)
    
    return loss_val, g_flat, H_full

def make_hessian_psd(H, mode='abs'):
    """
    Modifies a Hessian matrix to ensure it is Positive Semi-Definite.
    
    Args:
        H: The Hessian Matrix (2D tensor)
        mode: 'abs' (absolute value of eigenvalues) or 'clip' (floor at 0)
    """
    # 1. If Matrix is Diagonal (heuristic check), just abs the diagonal
    # This is a fast path for 'diag' structure if passed as a matrix
    if H.shape[0] == H.shape[1] and torch.count_nonzero(H - torch.diag(torch.diagonal(H))) == 0:
        return torch.abs(H)

    # 2. Full Eigendecomposition for Dense Matrices
    # eigh is for symmetric matrices (Hessians are symmetric)
    L, Q = torch.linalg.eigh(H)
    
    # 3. Correct Eigenvalues
    if mode == 'abs':
        L_new = torch.abs(L)
    elif mode == 'clip':
        L_new = torch.clamp(L, min=1e-6) # Clip negative to small epsilon
    else:
        raise ValueError(f"Unknown PSD mode: {mode}")
        
    # 4. Reconstruct: Q * L_new * Q^T
    H_psd = Q @ (torch.diag(L_new) @ Q.T)
    
    return H_psd
import torch
from torch.func import functional_call, hessian, grad, vmap

def compute_loss_grad_curvature(model, loss_fn, x, y, params_flat=None, curvature_type='hessian'):
    """
    Computes Loss, Gradient, and Curvature Matrix (H).
    curvature_type: 'hessian', 'fisher' (Empirical), or 'true_fisher' (Analytical)
    """
    # 1. Setup Parameters & Device
    if params_flat is None:
        params_dict = dict(model.named_parameters())
        device = next(model.parameters()).device
    else:
        params_dict = get_params_dict_from_flat(model, params_flat)
        device = params_flat.device

    x = x.to(device)
    y = y.to(device)
    
    # Ensure x is a batch for vmap
    if x.dim() == 1: x = x.unsqueeze(0)
    if y.dim() == 0: y = y.unsqueeze(0)

    # 2. Define Functional Loss (Batch)
    def func_loss(p_dict):
        out = functional_call(model, p_dict, (x,))
        return loss_fn(out, y)

    def func_loss_aux(p_dict):
        out = functional_call(model, p_dict, (x,))
        loss = loss_fn(out, y)
        return loss, loss

    # 3. Compute Gradient (of the actual Loss)
    g_dict, loss_val = grad(func_loss_aux, has_aux=True)(params_dict)
    
    keys = list(params_dict.keys())
    g_flat = torch.cat([g_dict[k].view(-1) for k in keys])
    
    # 4. Compute Curvature Matrix
    if curvature_type == 'hessian':
        # --- A. True Hessian ---
        H_dict = hessian(func_loss)(params_dict)
        h_rows = []
        for k1 in keys:
            row_blocks = []
            for k2 in keys:
                block = H_dict[k1][k2]
                n_k1 = params_dict[k1].numel()
                n_k2 = params_dict[k2].numel()
                block = block.view(n_k1, n_k2)
                row_blocks.append(block)
            h_rows.append(torch.cat(row_blocks, dim=1))
        H_full = torch.cat(h_rows, dim=0)

    elif curvature_type == 'fisher':
        # --- B. Empirical Fisher (Rank 1 Estimate) ---
        # Note: This computes the outer product of the MEAN gradient.
        # This is strictly Rank 1. Use with caution for Spectral Reg.
        H_full = torch.outer(g_flat, g_flat)

    elif curvature_type == 'true_fisher':
        # --- C. True Fisher (Expectation over Model Probabilities) ---
        # F = E_y [ \nabla log p(y|x) \nabla log p(y|x)^T ]
        
        # 1. Get Model Probabilities p(c|x)
        with torch.no_grad():
            logits = functional_call(model, params_dict, (x,))
            probs = torch.softmax(logits, dim=1) # [N, C]
            num_classes = probs.shape[1]
            batch_size = x.shape[0]

        H_accum = 0
        
        # Define per-sample loss for a specific target class 'c'
        def func_loss_for_class_c(p_d, x_i, c_idx):
            out = functional_call(model, p_d, (x_i.unsqueeze(0),))
            # Create a dummy target just for gradient calculation
            t = torch.tensor([c_idx], device=x_i.device)
            return loss_fn(out, t)

        # 2. Sum over all classes
        for c in range(num_classes):
            # A. Compute gradients for class 'c' for ALL samples in batch (vmap)
            # grad return dict -> flatten it
            compute_grad = grad(func_loss_for_class_c)
            g_dict_c = vmap(compute_grad, in_dims=(None, 0, None))(params_dict, x, c)
            g_flat_c = torch.cat([g_dict_c[k].flatten(start_dim=1) for k in keys], dim=1) # [N, D]
            
            # B. Weight by sqrt(probability)
            # p_c is [N] -> [N, 1]
            p_c = probs[:, c].view(-1, 1)
            
            # C. Accumulate weighted outer product
            # sum_i ( p(c|x_i) * g_i * g_i^T )
            # Efficiently: (sqrt(p)*g)^T @ (sqrt(p)*g)
            g_weighted = g_flat_c * torch.sqrt(p_c)
            H_accum += (g_weighted.T @ g_weighted)
            
        # 3. Average over batch
        H_full = H_accum / batch_size

    else:
        raise ValueError(f"Unknown curvature type: {curvature_type}")

    return loss_val, g_flat, H_full