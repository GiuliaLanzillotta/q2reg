import os
import glob
import yaml
import shutil

ROOT_DIR = "results/study_q2approx_v1"

def make_slug_from_path(path_parts):
    # Heuristic to turn your OLD structure into NEW structure
    # Old: .../mode/reg/curv/alpha/seed
    # This depends on your specific old depth, adjust indices as needed
    # assuming path ends in: .../regularized/taylor-block/hessian/alpha_1.0/seed_11
    
    try:
        reg = path_parts[-3]   # taylor-block
        mode = path_parts[-4]  # regularized
        
        return f"mode_{mode}-reg_{reg}-curv_hessia-alpha_1.0"
    except:
        return None

# Find all seed folders
seed_folders = glob.glob(os.path.join(ROOT_DIR, "**", "seed_*"), recursive=True)

for folder in seed_folders:
    # 1. Identify where we are
    parts = folder.split(os.sep)
    seed_part = parts[-1]
    
    # 2. Generate new slug name
    # Option A: Parse from path (fast, brittle)
    # new_slug = make_slug_from_path(parts)
    
    # Option B: Read the config (slow, robust)
    config_path = os.path.join(folder, "config.yaml")
    if not os.path.exists(config_path): continue
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    # Re-use the python function logic
    keys = ['training_mode', 'reg_type', 'curvature_type', 'accumulate', 'alpha']
    slug_parts = [f"{k}_{cfg.get(k)}" for k in keys]
    new_slug = "-".join(slug_parts)

    # 3. Move
    # New path: results/study_v1 / NEW_SLUG / seed_X
    new_parent = os.path.join(ROOT_DIR, new_slug)
    new_full_path = os.path.join(new_parent, seed_part)
    
    if folder == new_full_path: continue
    
    print(f"Moving:\n  {folder}\n  -> {new_full_path}")
    os.makedirs(new_parent, exist_ok=True)
    shutil.move(folder, new_full_path)

# Optional: Clean up empty old directories
#find results/study_q2approx_v1 -type d -empty -delete