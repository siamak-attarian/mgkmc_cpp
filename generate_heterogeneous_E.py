import numpy as np
import yaml
import os
from scipy.ndimage import gaussian_filter

def generate_correlated_field(shape, mean=1.0, std=0.1, corr=10, clip_min=None, clip_max=None):
    field = np.random.normal(size=shape)
    field = gaussian_filter(field, sigma=corr, mode='wrap')
    field = (field - field.mean()) / field.std()
    field = mean + std * field
    if clip_min is not None or clip_max is not None:
        field = np.clip(field, clip_min, clip_max)
    return field

def main():
    yaml_path = r"d:\GoogleDrive\2-MGKMC\mgkmc\examples\3_uniaxial_tension_heterogenous\Damask_comparison\plane_stress_3d_32_DAMASK_comparison.yaml"
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    seed = config.get('seed', 1)
    np.random.seed(seed)
    
    nx = config['system']['nx']
    ny = config['system']['ny']
    nz = config['system'].get('nz', 1)
    dim = config['system'].get('dimensionality', '3d')
    
    shape = (nx, ny, nz) if dim == '3d' else (nx, ny)
    
    e_conf = config['material']['E']
    params = e_conf.get('parameters', {})
    
    mean = params.get('mean', 70.0)
    std = params.get('std', 7.0)
    corr = params.get('corr', 2)
    clip_min = params.get('clip_min', None)
    clip_max = params.get('clip_max', None)
    
    E_field = generate_correlated_field(
        shape, mean=mean, std=std, corr=corr,
        clip_min=clip_min, clip_max=clip_max
    )
    
    output_path = r"d:\GoogleDrive\2-MGKMC\mgkmc_cpp\E_heterogeneous_32x32x1.npy"
    np.save(output_path, E_field)
    
    print(f"Generated E field with shape: {E_field.shape}")
    print(f"Mean: {E_field.mean():.6f}")
    print(f"Std: {E_field.std():.6f}")
    print(f"Min: {E_field.min():.6f}")
    print(f"Max: {E_field.max():.6f}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
