# In utils_monitor.py (create this file)
import torch
import numpy as np
import regularizers  

class ShadowMonitor:
    """
    Maintains two parallel approximation states:
    1. Reset: Re-approximates all past data at every task boundary.
    2. Accumulate: Keeps old approximations fixed, adds new ones.
    """
    def __init__(self, config, loss_fn):
        self.loss_fn = loss_fn
        # We instantiate two regularizers purely for monitoring state
        # The 'alpha' doesn't matter here as we won't backward() them
        self.reg_reset = regularizers.get_regularizer(config)
        self.reg_accum = regularizers.get_regularizer(config)

    def update(self, model, task_dataset):
        """
        Updates both shadow regularizers at the end of a task.
        """
        print("  [Monitor] Updating 'Accumulate' State...")
        # Accumulate only looks at the new task data to add to history
        self.reg_accum.update(model, task_dataset, self.loss_fn, accumulate=True)
        
        print("  [Monitor] Updating 'Reset' State...")
            
        return self.reg_reset.update(model, task_dataset, self.loss_fn, accumulate=False)

    def get_past_data(self):
        # Both regularizers store the same samples (conceptually), 
        # so we can just return one set of (x,y) to iterate over.
        return self.reg_accum.get_past_data()
    
    

@torch.enable_grad()
def evaluate_shadow_monitors(model, monitor, device):
    """
    Computes detailed Taylor approximation metrics for Reset and Accumulate strategies.
    Incorporates normalization and gradient conflict metrics.
    """
    model.eval()
    past_samples = monitor.get_past_data()
    
    if not past_samples:
        return []

    results = []
    eps = 1e-8 

    for i, (x_i, y_i) in enumerate(past_samples):
        x_i, y_i = x_i.to(device), y_i.to(device)
        
        # --- 1. True Landscape (Current State) ---
        model.zero_grad()
        output = model(x_i)
        L_theta = monitor.loss_fn(output, y_i)
        
        # Compute True Gradient (Current task/sample gradient)
        true_grad = torch.autograd.grad(L_theta, model.parameters())
        true_g_flat = torch.cat([g.view(-1) for g in true_grad])
        L_theta_val = L_theta.item()
        
        # --- 2. Anchor References ---
        # Get L(theta*) for both strategies
        L_anchor_acc = monitor.reg_accum.per_sample_losses[i]
        L_anchor_res = monitor.reg_reset.per_sample_losses[i]
        
        # --- 3. Query Proxies ---
        # pl: Total proxy loss (Taylor expanded)
        # pg: Total proxy gradient
        # p2: Second-order term only (1/2 * delta^T * H * delta)
        pl_acc, pg_acc, p2_acc = monitor.reg_accum.compute_per_sample_proxy_loss_and_grad(model, i)
        pl_res, pg_res, p2_res = monitor.reg_reset.compute_per_sample_proxy_loss_and_grad(model, i)
        
        # --- 4. Calculate Changes ---
        # Actual change: ΔL = L(theta) - L(theta*)
        actual_delta_acc = L_theta_val - L_anchor_acc
        actual_delta_res = L_theta_val - L_anchor_res
        
        # Predicted change: Δg = pl - L(theta*)
        pred_delta_acc = pl_acc.item() - L_anchor_acc
        pred_delta_res = pl_res.item() - L_anchor_res

        # --- 5. Compute Normalized Metrics ---
        res_dict = {
            'sample_id': i,
            'true_loss': L_theta_val,
            'accuracy': (output.argmax(1) == y_i).item(),
            
            # --- Taylor Ratios (Rho) ---
            # Perfect = 1.0. High values = expansion is over-optimistic/broken
            'rho_acc': actual_delta_acc / (pred_delta_acc + eps),
            'rho_res': actual_delta_res / (pred_delta_res + eps),
            
            # --- Normalized Kappa (Relative to curvature energy) ---
            'kappa_norm_acc': abs(actual_delta_acc - pred_delta_acc) / (abs(p2_acc) + eps),
            'kappa_norm_res': abs(actual_delta_res - pred_delta_res) / (abs(p2_res) + eps),
            
            # --- Gradient Alignment (Conflict) ---
            # Raw cosine similarity: -1 is direct conflict, +1 is alignment
            'cos_sim_acc': torch.nn.functional.cosine_similarity(true_g_flat.unsqueeze(0), pg_acc.unsqueeze(0)).item(),
            'cos_sim_res': torch.nn.functional.cosine_similarity(true_g_flat.unsqueeze(0), pg_res.unsqueeze(0)).item(),
            
            # --- Gradient Magnitude Ratio (Overshooting detection) ---
            # > 1.0 means the regularizer pull is stronger than the current task gradient
            'grad_mag_ratio_acc': torch.norm(pg_acc).item() / (torch.norm(true_g_flat).item() + eps),
            'grad_mag_ratio_res': torch.norm(pg_res).item() / (torch.norm(true_g_flat).item() + eps),
            
            # Store raw errors for backward compatibility
            'kappa_loss_acc': actual_delta_acc - pred_delta_acc,
            'kappa_grad_acc': torch.norm(true_g_flat - pg_acc).item()
        }
        results.append(res_dict)
        
    return results