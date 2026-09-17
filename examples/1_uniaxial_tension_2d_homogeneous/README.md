# 1 — Uniaxial tension, 2D, homogeneous

Elastic-only baseline, and the fastest way to check that a fresh build works.
A homogeneous 128×128 sheet is pulled along **x** while the spectral solver
enforces equilibrium; no STZ events occur.

Common to all three: `simulation_type: linear_elastic`, 128×128 voxels at
`pixel: 1.0` nm, `E = 70` GPa, `nu = 0.3`, `xx` driven to `eps_target: 0.08` in
steps of `1e-4` (800 steps).

| Config | Plane mode | Lateral boundary condition | Expected σ_xx/ε_xx |
|---|---|---|---|
| `plane_strain.yaml` | `plane_strain` | none — lateral strains held at zero | λ+2μ = 94.231 GPa |
| `plane_strain_in_zz.yaml` | `plane_strain` | `yy` relaxed to zero stress | E/(1−ν²) = 76.923 GPa |
| `plane_stress.yaml` | `plane_stress` | `yy` relaxed to zero stress | E = 70.000 GPa |

The expected values are closed-form isotropic elasticity, so this doubles as a
correctness check: the last line of `summary_log.txt` should give a final stress
of 7.538, 6.154 and 5.600 GPa respectively at ε_xx = 0.08.

## Run

```bash
cd examples/1_uniaxial_tension_2d_homogeneous
../../build/mgkmc plane_stress.yaml
```

Takes a few seconds. Results land in `output_plane_stress/`.
