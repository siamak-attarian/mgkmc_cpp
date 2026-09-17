# 2 — Uniaxial tension, 3D solver on a single layer

The same problem as example 1, run through the **3D** code path
(`dimensionality: 3d`) on a grid one voxel deep (`nz: 1`).

This is a consistency check between the 2D and 3D solvers: with matching lateral
stress targets, a 3D grid of thickness one should reproduce the 2D result. Run
these alongside example 1 and compare the final stresses.

Common to all three: 128×128×1 voxels at `pixel: 1.0` nm, `E = 70` GPa,
`nu = 0.3`, `xx` driven to `eps_target: 0.08` in steps of `1e-4`.

| Config | Lateral boundary condition | Compare against |
|---|---|---|
| `plane_strain.yaml` | none — lateral strains held at zero | `1/plane_strain.yaml` |
| `plane_strain_in_zz.yaml` | `yy` relaxed to zero stress | `1/plane_strain_in_zz.yaml` |
| `plane_stress.yaml` | `yy` and `zz` relaxed to zero stress | `1/plane_stress.yaml` |

## Run

```bash
cd examples/2_uniaxial_tension_2d_3d_homogeneous
../../build/mgkmc plane_stress.yaml
```

Slower than example 1 — the 3D path does more work per step even at `nz: 1`.
