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
            
        self.reg_reset.update(model, task_dataset, self.loss_fn, accumulate=False)

    def get_past_data(self):
        # Both regularizers store the same samples (conceptually), 
        # so we can just return one set of (x,y) to iterate over.
        return self.reg_accum.get_past_data()
    
    

@torch.enable_grad()
def evaluate_shadow_monitors(model, monitor, device):
    """
    Computes approximation error (kappa) for both Reset and Accumulate strategies.
    Returns a dict organized by sample ID.
    """
    model.eval()
    past_samples = monitor.get_past_data()
    
    if not past_samples:
        return []

    results = []
    
    # We need to know dimension d for normalization
    d = sum(p.numel() for p in model.parameters())

    for i, (x_i, y_i) in enumerate(past_samples):
        x_i, y_i = x_i.to(device), y_i.to(device)
        
        # 1. True Landscape
        model.zero_grad()
        output = model(x_i)
        true_loss = monitor.loss_fn(output, y_i)
        # Compute True Gradient
        true_grad = torch.autograd.grad(true_loss, model.parameters())
        true_g_flat = torch.cat([g.view(-1) for g in true_grad])
        true_loss = true_loss.item()
        
        # 2. Query Proxies (Accum & Reset)
        # We need compute_per_sample... to return proxy_grad too!
        pl_acc, pg_acc = monitor.reg_accum.compute_per_sample_proxy_loss_and_grad(model, i)
        pl_res, pg_res = monitor.reg_reset.compute_per_sample_proxy_loss_and_grad(model, i)
        
        # 3. Compute Metrics
        res_dict = {
            'sample_id': i,
            'true_loss': true_loss,
            'accuracy': (output.argmax(1) == y_i).item(),
            
            # --- Loss Errors ---
            'kappa_loss_acc': true_loss - pl_acc.item(),
            'kappa_loss_res': true_loss - pl_res.item(),
            
            # --- Gradient Errors (L2 Norm) ---
            'kappa_grad_acc': torch.norm(true_g_flat - pg_acc).item(),
            'kappa_grad_res': torch.norm(true_g_flat - pg_res).item(),
            
            # --- Gradient Cosine Sim (Normalized by sqrt(d)) ---
            'cos_sim_acc': torch.nn.functional.cosine_similarity(true_g_flat.unsqueeze(0), pg_acc.unsqueeze(0)).item() * np.sqrt(d),
            'cos_sim_res': torch.nn.functional.cosine_similarity(true_g_flat.unsqueeze(0), pg_res.unsqueeze(0)).item() * np.sqrt(d)
        }
        results.append(res_dict)
        
    return results