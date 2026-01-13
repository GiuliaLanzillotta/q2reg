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
    curv_type = config.get('curvature_type', 'hessian')
    # NEW: Read flag
    spectral_override = config.get('spectral_override', False)
    spectral_threshold = config.get('spectral_threshold', 0.999)
    spectral_rank = config.get('spectral_rank', 50)
    hard_spectral_cut = config.get('hard_spectral_cut', False)
    
    # NEW: Read flag, default to False
    ignore_grad = config.get('ignore_gradient', False) 
    
    if reg_name == 'ewc':
        # EWC classically ignores gradient, but our impl allowed it.
        # Let's support the flag here too.
        return EWCRegularizer(alpha=alpha, ignore_gradient=ignore_grad, spectral_override=spectral_override, spectral_threshold=spectral_threshold, spectral_rank=spectral_rank, hard_spectral_cut=hard_spectral_cut)
        
    if 'taylor' in reg_name:
        if 'diag' in reg_name: struct = 'diag'
        elif 'block' in reg_name: struct = 'block'
        else: struct = 'full'
            
        return TaylorRegularizer(
            alpha=alpha, 
            structure=struct, 
            curvature_type=curv_type, 
            ignore_gradient=ignore_grad, 
            spectral_override=spectral_override, 
            spectral_threshold=spectral_threshold, 
            spectral_rank=spectral_rank,
            hard_spectral_cut=hard_spectral_cut
        )
    
    
    raise ValueError(f"Unknown reg_type: {reg_name}")

class BaseRegularizer(ABC):
    """
    Abstract Base Class for a quadratic regularizer.
    Stores per-sample contributions \Lambda_i and the anchor \vparam_{t-1}.
    """
    def __init__(self, alpha, structure='diag', **kwargs):
        self.alpha = alpha
        self.structure = structure # 'diag', 'full', 'block'
        self.anchor_param_list = [] 
        self.global_basis = None
        
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
        # Pre-computed map of {anchor_idx: [sample_indices...]}
        self.anchor_groups = {}
        # Structure: { anchor_idx: V_k_matrix }
        self.spectral_cache = {}

        # --- Spectral Regularization ---
        self.spectral_override = kwargs.get('spectral_override', False)
        self.spectral_threshold = kwargs.get('spectral_threshold', 0.999)
        self.spectral_rank = kwargs.get('spectral_rank', 50)
        self.hard_spectral_cut = kwargs.get('hard_spectral_cut', False)


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
        """
        Computes regularization loss. 
        If spectral_override=True, uses the cached subspace V_k from update().
        """
        if not self.anchor_param_list:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        current_params = utils.get_flat_params(model)
        device = current_params.device
        
        # --- PATH A: FAST SPECTRAL REGULARIZATION ---
        if self.spectral_override:
            total_spectral_loss = 0.0
            
            # Iterate over the CACHE, which is grouped by anchor
            for anchor_idx, V_k_cpu in self.spectral_cache.items():
                
                # 1. Get Anchor & Delta
                # Move V_k to GPU just-in-time
                V_k = V_k_cpu.to(device)
                anchor_k = self.anchor_param_list[anchor_idx].to(device)
                
                delta = current_params - anchor_k
                
                # 2. Project Delta: || V_k^T delta ||^2
                # O(k * D) operation (Fast Matrix-Vector product)
                projected_delta = V_k.T @ delta
                
                loss_component = torch.sum(projected_delta ** 2)
                total_spectral_loss += loss_component

            return self.alpha * total_spectral_loss

        # --- PATH B: STANDARD REGULARIZATION ---
        # (Your existing code for sum-of-quadratics)
        total_loss = 0.0
        for anchor_idx, sample_indices in self.anchor_groups.items():
            anchor_k = self.anchor_param_list[anchor_idx].to(device)
            delta = current_params - anchor_k
            
            for i in sample_indices:
                lambda_i = self.per_sample_importances[i].to(device)
                g_i = self.per_sample_grads[i].to(device)
                total_loss += self._compute_quadratic_form(delta, lambda_i, g_i)
            
        return self.alpha * total_loss
    
    @torch.no_grad()
    def project_weights(self, model, previous_params_flat):
        """
        Projects the parameter update onto the null space of the Global Basis.
        """
        if self.global_basis is None:
            return

        current_params = utils.get_flat_params(model)
        
        # 1. Get Basis (Move to GPU for operation)
        U = self.global_basis.to(current_params.device)
        
        # 2. Calculate Update (Drift)
        # delta = w_new - w_old
        delta = current_params - previous_params_flat
        print(f"Update magnitude before projection: {torch.norm(delta).item():.6f}")
        
        # 3. Project Delta onto Forbidden Subspace
        # forbidden = U @ (U.T @ delta)
        inner = U.T @ delta
        forbidden = U @ inner
        
        # 4. Remove Forbidden Component
        # w_corrected = w_new - forbidden
        current_params.sub_(forbidden)
        new_delta = current_params - previous_params_flat
        print(f"Update magnitude after projection: {torch.norm(new_delta).item():.6f}")
        utils.set_flat_params(model, current_params)


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
        Refined proxy computation to decompose first and second order terms.
        Returns: proxy_loss, total_proxy_grad, and the predicted_delta_loss (g)
        """
        if self.anchor_param_list is None:
            raise RuntimeError("Regularizer not initialized.")
            
        current_params = utils.get_flat_params(model)
        anchor_k_idx = self.past_sample_anchor_idx[sample_index]
        anchor_k = self.anchor_param_list[anchor_k_idx].to(current_params.device)
        
        # Delta (the step from the anchor)
        delta = current_params - anchor_k

        Lambda_i = self.per_sample_importances[sample_index].to(current_params.device)
        g_anchor = self.per_sample_grads[sample_index].to(current_params.device)
        L_anchor = self.per_sample_losses[sample_index]
        
        # --- 1. First Order Term: g^T * delta ---
        first_order_change = torch.dot(g_anchor, delta)
        
        # --- 2. Second Order Term (g): 1/2 * delta^T * Lambda * delta ---
        # Note: Added the 1/2 coefficient to match standard Taylor expansion
        if self.structure == 'diag':
            second_order_change = 0.5 * torch.sum(Lambda_i * (delta ** 2))
            second_order_grad = Lambda_i * delta # Gradient of the second order term
        elif self.structure in ['block', 'full']:
            second_order_change = 0.5 * delta.t() @ (Lambda_i @ delta)
            second_order_grad = Lambda_i @ delta
        
        # Total Proxy Loss: L(θ*) + g^TΔ + 1/2 Δ^T H Δ
        proxy_loss = L_anchor + first_order_change + second_order_change
        
        # Total Proxy Gradient: g_anchor + H*Δ
        # This is the "pull" the regularizer exerts at the current theta
        total_proxy_grad = g_anchor + second_order_grad
                
        return proxy_loss, total_proxy_grad, second_order_change.item()


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
        drift_stats = {}

        if accumulate:
            # --- TASK-SPECIFIC ANCHORS ---
            # Add this as a new, permanent anchor
            self.anchor_param_list.append(new_anchor_params)
            new_anchor_idx = len(self.anchor_param_list) - 1
            self.anchor_groups[new_anchor_idx] = []
            
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
                self.anchor_groups[new_anchor_idx].append(len(self.per_sample_importances)-1)

        else:
            print("Resetting the regularizer")
            # --- GLOBAL, RESETTING ANCHOR ---
            # Set this as the *only* anchor
            self.anchor_param_list = [new_anchor_params]
            self.anchor_groups = {}
            new_anchor_idx = 0
            self.anchor_groups[new_anchor_idx] = []
            # Rebuild importances and anchor indices
            new_importances = []
            new_grads = []
            new_losses = []
            new_anchor_indices = []


            if len(self.per_sample_importances) > 0:
                old_total_curvature = self.get_total_curvature() # Q at theta_{t-1}
            else:
                old_total_curvature = None

            # 1. Re-compute for *existing* samples
            for i, (x, y) in enumerate(self.past_samples):
                # Re-compute \Lambda_i, g_i around the *new* global anchor
                lambda_i, g_i, l_i = self._compute_importance_and_grad(model, x, y, new_anchor_params, loss_fn)
                new_importances.append(lambda_i.cpu())
                new_grads.append(g_i.cpu())
                new_losses.append(l_i.cpu())
                new_anchor_indices.append(new_anchor_idx)
                # self.past_samples[i] and self.past_sample_peak_acc[i] are preserved
                self.anchor_groups[new_anchor_idx].append(len(new_importances)-1)
            
            # 2. Compute Drift Metrics
            if old_total_curvature is not None:
                # We compute the NEW curvature sum ONLY for the OLD samples
                # (to ensure a fair apples-to-apples comparison)
                new_curvature_old_samples = self._sum_importances(new_importances[:len(self.past_samples)])
                
                drift_stats = self._compute_drift_metrics(
                    old_total_curvature.to(new_anchor_params.device), 
                    new_curvature_old_samples.to(new_anchor_params.device)
                )

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
                self.anchor_groups[new_anchor_idx].append(len(new_importances)-1)

            # Replace the old lists
            self.per_sample_importances = new_importances
            self.per_sample_grads = new_grads
            self.per_sample_losses = new_losses
            self.past_sample_anchor_idx = new_anchor_indices


        if self.spectral_override and self.structure in ['full', 'block']:
            self.precompute_spectral_subspaces()

        if save_path is not None:
            self.save_artifacts(save_path)
        
        return drift_stats

    def _compute_drift_metrics(self, Q_old, Q_new, k=50):
        """
        Quantifies Fisher Drift, specifically focusing on the rotation 
        of the top-K 'protected' subspace.
        """
        with torch.no_grad():
            # 1. Magnitude Drift (Trace Ratio)
            tr_old = torch.trace(Q_old) if Q_old.dim() > 1 else Q_old.sum()
            tr_new = torch.trace(Q_new) if Q_new.dim() > 1 else Q_new.sum()
            mag_drift = (tr_new / (tr_old + 1e-8)).item()

            # 2. Subspace Overlap (Top-K Eigenvectors)
            if Q_old.dim() > 1: # For Full/Block Curvature
                # Get eigenvectors (eigh returns sorted ascending)
                _, evecs_o = torch.linalg.eigh(Q_old)
                _, evecs_n = torch.linalg.eigh(Q_new)

                # Extract top K (the largest eigenvalues are at the end)
                V_old = evecs_o[:, -k:] # [P, K]
                V_new = evecs_n[:, -k:] # [P, K]

                # Subspace Similarity Metric: Trace(V_old^T @ V_new @ V_new^T @ V_old) / K
                # Range [0, 1]. 1.0 means the subspaces are identical.
                # This is essentially the mean squared cosine of the principal angles.
                projection_overlap = torch.matmul(V_old.t(), V_new) # [K, K]
                overlap_score = (torch.norm(projection_overlap)**2 / k).item()
            else:
                # For Diagonal: Use Weighted Cosine Similarity
                # Since we don't have eigenvectors, we check if the importance 
                # has shifted to different parameters.
                overlap_score = torch.nn.functional.cosine_similarity(
                    Q_old.unsqueeze(0), Q_new.unsqueeze(0)
                ).item()

            return {
                'fisher_mag_drift': mag_drift,
                'fisher_subspace_overlap': overlap_score
            }

    def precompute_spectral_subspaces(self):
        """
        Groups samples by anchor, aggregates their Fisher matrices, 
        and caches the top-k subspace (V_k) for fast projection.
        """
        if not self.per_sample_importances:
            return
        if self.hard_spectral_cut:
            print(f"  [Regularizer] Pre-computing spectral subspaces (Rank={self.spectral_rank})...")
        else:
            print(f"  [Regularizer] Pre-computing spectral subspaces (Threshold={self.spectral_threshold})...")

        # 2. Compute and Cache
        self.spectral_cache = {}
        
        # We determine device from the first stored tensor
        # (Assuming tensors are on CPU to save memory, we move to GPU for SVD)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for anchor_idx, sample_indices in self.anchor_groups.items():
            # A. Aggregate Curvature
            # Stack only what we need to minimize RAM usage
            lambdas_stack = torch.stack([
                self.per_sample_importances[i] for i in sample_indices
            ]).to(device)
            
            F_total = torch.sum(lambdas_stack, dim=0)
            
            # B. Decompose
            # eigh is faster/stable for symmetric matrices
            L, V = torch.linalg.eigh(F_total)
            L = L.flip(0) # Descending
            V = V.flip(1)
            
            # --- Capture Sharpness ---
            # We clip at 0 to avoid numerical errors with negative eigenvalues
            lambda_max = torch.clamp(L[0], min=1e-8)
            
            if self.hard_spectral_cut: 
                k = self.spectral_rank
            else: 
                # C. Threshold
                total_energy = L.sum()
                target_energy = self.spectral_threshold * total_energy
                cumulative_energy = torch.cumsum(L, dim=0)
                
                k = torch.searchsorted(cumulative_energy, target_energy).item() + 1
                k = min(k, len(L))
            

            # D. Cache V_k
            V_k = V[:, :k]

            # SCALE THE BASIS:
            # We multiply by sqrt(lambda_max) so that when we compute norm(V.T @ x)^2,
            # the effective penalty magnitude matches the actual sharpness.
            scale_factor = torch.sqrt(lambda_max)
            V_k_scaled = V_k * scale_factor

            # Store on CPU to avoid holding GPU memory during training
            self.spectral_cache[anchor_idx] = V_k_scaled.cpu()


            
            print(f"    Anchor {anchor_idx}: Cached Subspace Rank {k} with scale factor {scale_factor:.2f}")
                #   f"({cumulative_energy[k-1]/total_energy:.2%} energy)")
        
        # 3. BUILD GLOBAL BASIS (For Hard Projection)
        # We compute this once here so the training loop is fast.
        self._update_global_basis(device)

        # Cleanup GPU memory used for SVD
        if device == 'cuda':
            del lambdas_stack, F_total, L, V
            torch.cuda.empty_cache()

    def _update_global_basis(self, device):
        """
        Combines all cached subspaces into a single orthonormal basis (Q).
        Required for correct Hard Projection (GPM style).
        """
        if not self.spectral_cache:
            self.global_basis = None
            return

        print("  [Regularizer] Building Global Orthogonal Basis (QR)...")
        
        # 1. Collect all vectors
        all_vectors = []
        for V_k in self.spectral_cache.values():
            if V_k.shape[1] > 0:
                V_k = V_k.to(device)
                
                # CRITICAL: Re-Normalize!
                # We stored them as V * sqrt(lambda). 
                # For geometric projection, we need unit vectors.
                # Norm columns to 1.0
                V_norm = V_k / (torch.norm(V_k, dim=0, keepdim=True) + 1e-8)
                all_vectors.append(V_norm)
        
        if not all_vectors:
            self.global_basis = None
            return

        # 2. Concatenate [D, Total_K]
        V_matrix = torch.cat(all_vectors, dim=1)
        
        # 3. QR Decomposition
        # Q is [D, Rank_Union], orthonormal
        Q, _ = torch.linalg.qr(V_matrix, mode='reduced')
        
        self.global_basis = Q.cpu() # Store on CPU
        print(f"  [Regularizer] Global Basis Ready. Total Rank: {Q.shape[1]}")

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
    
    def __init__(self, alpha, **kwargs):
        # EWC is by definition diagonal
        super().__init__(alpha, structure='diag', **kwargs)

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
    def __init__(self, alpha, structure='diag', curvature_type='hessian', ignore_gradient=False, **kwargs):
        super().__init__(alpha, structure, **kwargs)
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
   
# regularizers.py

# regularizers.py
