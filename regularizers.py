""" Implementations of quadratic regularizers for continual learning."""
import sys
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))


import torch
from abc import ABC, abstractmethod
from torch.autograd.functional import hessian, jacobian
import bitbybit.utils as utils 
import utils_math


# In utils.py

def get_regularizer(config):
    reg_name = config['reg_type']
    alpha = config['alpha']
    curv_type = config.get('curvature', 'hessian')
    
    # NEW: Read flag, default to False
    ignore_grad = config.get('ignore_gradient', False) 
    
    if reg_name == 'ewc':
        # EWC classically ignores gradient, but our impl allowed it.
        # Let's support the flag here too.
        return EWCRegularizer(alpha=alpha, ignore_gradient=ignore_grad)
        
    if 'taylor' in reg_name:
        if 'diag' in reg_name: struct = 'diag'
        elif 'block' in reg_name: struct = 'block'
        else: struct = 'full'
            
        return TaylorRegularizer(
            alpha=alpha, 
            structure=struct, 
            curvature_type=curv_type, 
            ignore_gradient=ignore_grad 
        )
    
    raise ValueError(f"Unknown reg_type: {reg_name}")

class BaseRegularizer(ABC):
    """
    Abstract Base Class for a quadratic regularizer.
    Stores per-sample contributions \Lambda_i and the anchor \vparam_{t-1}.
    """
    def __init__(self, alpha, structure='diag'):
        self.alpha = alpha
        self.structure = structure # 'diag', 'full', 'block'
        self.anchor_param_list = [] 
        
        # --- Lists for per-sample data ---
        # Stores \Lambda_i for each sample
        self.per_sample_importances = [] 
        # Stores g_i for each sample
        self.per_sample_grads = [] 
        # Stores \loss_i for each sample
        self.per_sample_losses = []
        # Stores (x_i, y_i) for each sample
        self.past_samples = []
        # Stores peak_acc_i for each sample
        self.past_sample_peak_acc = []
        # Stores the *index* k from anchor_param_list for each sample i
        self.past_sample_anchor_idx = []

    def _compute_quadratic_form(self, delta, lambda_i, g_i):
        """Computes the quadratic form \delta^T \Lambda_i \delta"""
        if self.structure == 'diag':
            # lambda_i is a vector
            return g_i @ delta + torch.sum(lambda_i * (delta ** 2))
        elif self.structure in ['block','full']:
            # lambda_i is a matrix
            return g_i @ delta + delta @ lambda_i @ delta
        else:
            raise ValueError(f"Unknown structure: {self.structure}")

    def compute_total_reg_loss(self, model):
        """Computes the total regularization loss \alpha * \sum_i \hat\loss_i(\vparam)"""
        if self.anchor_param_list is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        current_params = utils.get_flat_params(model)
        total_loss = 0.0
        
        # Iterate sample by sample, using the correct anchor for each
        for i in range(len(self.past_samples)):
            lambda_i = self.per_sample_importances[i].to(current_params.device)
            g_i = self.per_sample_grads[i].to(current_params.device)

            # --- Get the sample-specific anchor ---
            anchor_k_idx = self.past_sample_anchor_idx[i]
            anchor_k = self.anchor_param_list[anchor_k_idx].to(current_params.device)
            
            delta = current_params - anchor_k

            total_loss += self._compute_quadratic_form(delta, lambda_i, g_i)
            
        return self.alpha * total_loss

    def compute_per_sample_proxy_loss(self, model, sample_index):
        """Computes the proxy loss \hat\loss_{i|t-1}(\vparam) for a single sample i."""
        if self.anchor_param_list is None:
            raise RuntimeError("Regularizer not initialized.")
            
        current_params = utils.get_flat_params(model)
        
        lambda_i = self.per_sample_importances[sample_index].to(current_params.device)
        g_i = self.per_sample_grads[sample_index].to(current_params.device)

        # --- Get the sample-specific anchor ---
        anchor_k_idx = self.past_sample_anchor_idx[sample_index]
        anchor_k = self.anchor_param_list[anchor_k_idx].to(current_params.device)
        delta = current_params - anchor_k
        
        proxy_loss = self._compute_quadratic_form(delta, lambda_i, g_i) + self.per_sample_losses[sample_index]
        return proxy_loss
    
    def compute_per_sample_proxy_loss_and_grad(self, model, sample_index):
        """
        Computes both the proxy loss \hat\loss_i(\vparam) and its gradient
        \nabla \hat\loss_i(\vparam) for a single sample i.
        """
        if self.anchor_param_list is None:
            raise RuntimeError("Regularizer not initialized.")
            
        current_params = utils.get_flat_params(model)
        # --- Get the sample-specific anchor ---
        anchor_k_idx = self.past_sample_anchor_idx[sample_index]
        anchor_k = self.anchor_param_list[anchor_k_idx].to(current_params.device)
        delta = current_params - anchor_k

        lambda_i = self.per_sample_importances[sample_index].to(current_params.device)
        g_i = self.per_sample_grads[sample_index].to(current_params.device)
        
        # 1. Compute Proxy Loss: \delta^T \Lambda_i \delta
        proxy_loss = self._compute_quadratic_form(delta, lambda_i, g_i) + self.per_sample_losses[sample_index]
        
        # 2. Compute Proxy Gradient: 2 * \Lambda_i * \delta
        if self.structure == 'diag':
            # lambda_i is a vector, \Lambda_i is its diagonal matrix
            proxy_grad = g_i + 2 * lambda_i * delta
        elif self.structure in ['block', 'full']:
            # lambda_i is the full matrix \Lambda_i
            proxy_grad = g_i + 2 * (lambda_i @ delta)
        else:
            raise ValueError(f"Unknown structure: {self.structure}")
            
        return proxy_loss, proxy_grad

    @abstractmethod
    def _compute_importance_and_grad(self, model, x, y, anchor_params, loss_fn):
        """
        (Implemented by subclass)
        Computes the importance \Lambda_i for a single sample (x, y)
        expanded around anchor_params.
        """
        pass

    def update(self, model, task_dataset, loss_fn, accumulate=True, save_path=None):
        """
        Updates the regularizer's state after training on a new task.
        Handles both task-specific (accumulate=True) and
        global-resetting (accumulate=False) anchor strategies.
        """
        model.eval()
        new_anchor_params = utils.get_flat_params(model).clone().detach()

        if accumulate:
            # --- TASK-SPECIFIC ANCHORS ---
            # Add this as a new, permanent anchor
            self.anchor_param_list.append(new_anchor_params)
            new_anchor_idx = len(self.anchor_param_list) - 1
            
            for x, y, _ in task_dataset:
                x, y = x.unsqueeze(0), y.unsqueeze(0)
                
                # Compute \Lambda_i and g_i around this new task-specific anchor
                lambda_i, g_i, l_i = self._compute_importance_and_grad(model, x, y, new_anchor_params, loss_fn)
                # Compute g_i around this new task-specific anchor
                
                # Compute peak acc at this anchor
                with torch.no_grad():
                    output = model(x.to(new_anchor_params.device))
                    peak_acc = (torch.argmax(output, dim=1) == y.to(new_anchor_params.device)).item()
                
                # Store all new data
                self.per_sample_importances.append(lambda_i.cpu())
                self.per_sample_grads.append(g_i.cpu())  
                self.per_sample_losses.append(l_i.cpu())
                self.past_samples.append((x.clone().detach().cpu(), y.clone().detach().cpu()))
                self.past_sample_peak_acc.append(peak_acc)
                self.past_sample_anchor_idx.append(new_anchor_idx)

        else:
            # --- GLOBAL, RESETTING ANCHOR ---
            # Set this as the *only* anchor
            self.anchor_param_list = [new_anchor_params]
            new_anchor_idx = 0
            
            # Rebuild importances and anchor indices
            new_importances = []
            new_grads = []
            new_losses = []
            new_anchor_indices = []
            
            # 1. Re-compute for *existing* samples
            for i, (x, y) in enumerate(self.past_samples):
                # Re-compute \Lambda_i, g_i around the *new* global anchor
                lambda_i, g_i, l_i = self._compute_importance_and_grad(model, x, y, new_anchor_params, loss_fn)
                new_importances.append(lambda_i.cpu())
                new_grads.append(g_i.cpu())
                new_losses.append(l_i.cpu())
                new_anchor_indices.append(new_anchor_idx)
                # self.past_samples[i] and self.past_sample_peak_acc[i] are preserved
            
            # 2. Compute for *new* samples
            for x, y, _ in task_dataset:
                x, y = x.unsqueeze(0), y.unsqueeze(0)
                
                # Compute \Lambda_i, g_i around the *new* global anchor
                lambda_i, g_i, l_i = self._compute_importance_and_grad(model, x, y, new_anchor_params, loss_fn)
                
                # Compute peak acc at this anchor
                with torch.no_grad():
                    output = model(x.to(new_anchor_params.device))
                    peak_acc = (torch.argmax(output, dim=1) == y.to(new_anchor_params.device)).item()

                # *Append* new data
                new_importances.append(lambda_i.cpu())
                new_grads.append(g_i.cpu())
                new_losses.append(l_i.cpu())
                new_anchor_indices.append(new_anchor_idx)
                self.past_samples.append((x.clone().detach().cpu(), y.clone().detach().cpu()))
                self.past_sample_peak_acc.append(peak_acc)

            # Replace the old lists
            self.per_sample_importances = new_importances
            self.per_sample_grads = new_grads
            self.per_sample_losses = new_losses
            self.past_sample_anchor_idx = new_anchor_indices

        if save_path is not None:
            self.save_artifacts(save_path)

    def reset(self):
        """Resets regularizer state completely."""
        self.anchor_param_list = []
        self.per_sample_importances = []
        self.per_sample_grads = []
        self.per_sample_losses = []
        self.past_samples = []
        self.past_sample_peak_acc = []
        self.past_sample_anchor_idx = []

    def get_past_data(self):
        """Returns all stored (x_i, y_i) pairs."""
        return self.past_samples

    def save_artifacts(self, save_dir):
        """
        Internal method to dump the current state of the regularizer to disk.
        """
        os.makedirs(save_dir, exist_ok=True)
        print(f"  [Regularizer] Saving artifacts to {save_dir}...")

        # 1. Save Anchors
        # This is a list of flat tensors. 
        # For Accumulate: multiple anchors. For Reset: one anchor.
        torch.save(self.anchor_param_list, os.path.join(save_dir, 'anchors.pt'))

        # 2. Save Samples
        # self.past_samples is a list of (x, y) tuples. 
        # We verify they are on CPU to save space/compatibility.
        samples_cpu = [(x.cpu(), y.cpu()) for x, y in self.past_samples]
        torch.save(samples_cpu, os.path.join(save_dir, 'samples.pt'))

        # 3. Save Fisher/Hessian (Conditional)
        # If Diag/Block, it's small, so we save it for convenience.
        # If Full, it's massive, so we SKIP it (recompute from samples+anchor later).
        if self.structure in ['diag', 'block']:
            torch.save(self.per_sample_importances, os.path.join(save_dir, 'importances.pt'))
        else:
            print(f"  [Regularizer] Skipping Full Matrix save (structure={self.structure}). "
                  "Recompute from 'anchors.pt' and 'samples.pt'.")
# --- Implementation 1: EWC ---

class EWCRegularizer(BaseRegularizer):
    """Elastic Weight Consolidation (EWC) regularizer."""
    
    def __init__(self, alpha):
        # EWC is by definition diagonal
        super().__init__(alpha, structure='diag')

    def _compute_importance_and_grad(self, model, x, y, anchor_params_flat, loss_fn):
        
        # 1. Compute Loss and Gradient efficiently using utils_math
        # This ensures graph safety via torch.func
        l_i, g_flat = utils_math.compute_loss_and_grad(
            model, loss_fn, x, y, params_flat=anchor_params_flat
        )
        
        # 2. Compute Fisher Information (Diagonal)
        # F_i = g_i^2
        lambda_i = g_flat ** 2
        
        # Note: Standard EWC often ignores 'g_flat' in the regularization term (assuming g=0).
        # However, for your analysis of "Approximation Error", passing the 
        # true gradient 'g_flat' is necessary to have a valid Taylor expansion
        # when the anchor is not at a minimum (e.g., during Reset).
        
        return lambda_i.detach(), g_flat.detach(), l_i.detach()
    

# --- Implementation 2: 2nd-Order Taylor (Hessian) ---
# In regularizers.py

class TaylorRegularizer(BaseRegularizer):
    def __init__(self, alpha, structure='diag', curvature_type='hessian', ignore_gradient=False):
        super().__init__(alpha, structure)
        self.curvature_type = curvature_type
        self.ignore_gradient = ignore_gradient # <--- Store flag

    def _compute_importance_and_grad(self, model, x, y, anchor_params_flat, loss_fn):
        
        # 1. Compute Math (Loss, Grad, Curvature)
        # We still compute the gradient because 'fisher' curvature needs it!
        l_i, g_flat, H_full = utils_math.compute_loss_grad_curvature(
            model, loss_fn, x, y, 
            params_flat=anchor_params_flat,
            curvature_type=self.curvature_type
        )

        # 2. OPTIONAL: Zero out the linear term
        if self.ignore_gradient:
            # We overwrite g_flat with zeros.
            # The regularizer will now define the proxy as:
            # L_proxy = L_true + 0 + 0.5 * delta^T * H * delta
            g_flat = torch.zeros_like(g_flat)

        # 3. Apply Structure
        lambda_i = 0.5 * H_full
        
        if self.structure == 'diag':
            lambda_i = torch.diag(lambda_i)
        elif self.structure == 'block':
            mask = torch.zeros_like(H_full)
            pointer = 0
            for param in model.parameters():
                num_params = param.numel()
                mask[pointer : pointer+num_params, pointer : pointer+num_params] = 1.0
                pointer += num_params
            lambda_i = lambda_i * mask

        return lambda_i.detach(), g_flat.detach(), l_i.detach()
   