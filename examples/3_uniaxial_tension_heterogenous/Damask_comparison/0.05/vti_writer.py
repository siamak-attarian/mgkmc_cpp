import pyvista as pv
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
# Generate element-centered correlated field
# -----------------------------------------------------------------------------
def generate_correlated_field(
    shape, mean=1.0, std=0.1, corr=10, clip_min=None, clip_max=None,
    seed=None, visualize=False, title="Correlated field"
):
    if seed is not None:
        np.random.seed(seed)

    field = np.random.normal(size=shape)
    field = gaussian_filter(field, sigma=corr, mode='wrap')

    field = (field - field.mean()) / field.std()
    field = mean + std*field

    if clip_min is not None or clip_max is not None:
        field = np.clip(field, clip_min, clip_max)

    if visualize:
        plt.imshow(field[:,:,-1].T, origin='lower', cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.show()
        
    return field

def write_vti_damask(filename, E, nu, C11, C12, C44,
                     spacing=(1.0,1.0,1.0),
                     origin=(0.0,0.0,0.0),
                     match_matplotlib_orientation=True,
                     material_ids=None):
    """
    Write a DAMASK-compatible VTI file with cell-centered fields.
    
    Parameters
    ----------
    filename : str
        Output .vti file path.
    E, nu, C11, C12, C44 : ndarray (nx, ny, nz)
        Cell-centered material property fields.
    spacing : tuple of 3 floats
        Physical cell size (dx, dy, dz).
    origin : tuple of 3 floats
        Spatial origin.
    match_matplotlib_orientation : bool
        If True, ensure ParaView orientation matches matplotlib imshow().
    material_ids : ndarray (nx, ny, nz), optional
        Integer material ID map.
    """

    nx, ny, nz = E.shape

    # -----------------------
    # Step 2 — VTK ImageData
    # -----------------------
    grid = pv.ImageData()
    # For nx×ny×nz CELLS, VTK needs (nx+1)×(ny+1)×(nz+1) POINTS
    grid.dimensions = (nx+1, ny+1, nz+1)
    grid.spacing = spacing
    grid.origin  = origin

    # -----------------------
    # Step 3 — Add cell data
    # -----------------------
    grid.cell_data["youngsModulus"]  = E.ravel(order="F")
    grid.cell_data["poissonsRatio"]  = nu.ravel(order="F")
    grid.cell_data["C_11"]           = C11.ravel(order="F")
    grid.cell_data["C_12"]           = C12.ravel(order="F")
    grid.cell_data["C_44"]           = C44.ravel(order="F")

    if material_ids is not None:
        grid.cell_data["material"] = material_ids.ravel(order="F")

    # -----------------------
    # Step 4 — Save file
    # -----------------------
    grid.save(filename)
    print(f"✔ Saved {filename} (DAMASK-compatible, matplotlib-aligned)")


nx, ny, nz = 32,32,1
dx = dy = dz = 0.5
spacing = (dx, dy, dz)

seed = 1
E = generate_correlated_field(
    shape=(nx, ny, nz),
    mean=70e9,
    std=70e9*0.1,
    corr=2,
    seed=seed,
    visualize=True
)
nu = 0.3 * np.ones_like(E)

# Elastic stiffness for isotropic
C11 = (E*(1-nu)) / ((1+nu)*(1-2*nu))
C12 = (E*nu)     / ((1+nu)*(1-2*nu))
C44 = E / (2*(1+nu))

material_ids = np.arange(E.size).reshape(E.shape, order='F')

write_vti_damask(
    "heteroMap.vti",
    E, nu, C11, C12, C44,
    spacing=spacing,
    origin=(0,0,0),
    match_matplotlib_orientation=True,
    material_ids=material_ids,
)

# import matplotlib.pyplot as plt
# import numpy as np
# mid=-1
# fig, ax = plt.subplots(figsize=(6,6))
# plot_E = E[:,:,mid].T
# # Plot the field (no scientific notation)
# im = ax.imshow(plot_E, origin='lower', cmap='viridis')

# # Loop over all elements and print text
# for i in range(plot_E.shape[0]):
#     for j in range(plot_E.shape[1]):
#         value = plot_E[i,j].T / 1e9  # convert to GPa if needed (optional)
#         ax.text(j, i, f"{value:.2f}", ha='center', va='center', color='white', fontsize=8)

# plt.title("Correlated field with values")
# plt.colorbar(im)
# plt.show()

import yaml
import numpy as np

# ==============================================
# Generate material.yaml for OPTION A:
# One material + one phase PER CELL
# ==============================================

# Flatten cell fields (Fortran order = VTI order)
C11_flat = C11.ravel(order="F")
C12_flat = C12.ravel(order="F")
C44_flat = C44.ravel(order="F")

n_cells = C11_flat.size

# ----------------------------------------------
# PHASE BLOCKS
# ----------------------------------------------
phase_section = {}

for i in range(n_cells):
    phase_name = f"phase_{i}"
    phase_section[phase_name] = {
        "lattice": "cI",
        "mechanical": {
            "output": ["F", "P", "F_p", "v_0"],
            "elastic": {
                "type": "Hooke",
                "C_11": float(C11_flat[i]),
                "C_12": float(C12_flat[i]),
                "C_44": float(C44_flat[i])
            }
        }
    }

# ----------------------------------------------
# MATERIAL BLOCKS
# ----------------------------------------------
material_section = []

for i in range(n_cells):
    material_section.append({
        "homogenization": "volavg",
        "constituents": [
            {
                "phase": f"phase_{i}",
                "v": 1.0,
                "O": [1.0, 0.0, 0.0, 0.0]
            }
        ]
    })

# ----------------------------------------------
# FULL YAML STRUCTURE
# ----------------------------------------------
yaml_dict = {
    "homogenization": {
        "volavg": {
            "N_constituents": 1,
            "mechanical": {"type": "pass"}
        }
    },
    "phase": phase_section,
    "material": material_section
}

# ----------------------------------------------
# WRITE YAML
# ----------------------------------------------
with open("material.yaml", "w") as f:
    yaml.dump(yaml_dict, f, sort_keys=False)

print(f"Saved material.yaml with {n_cells} materials/phases.")