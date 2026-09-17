# 3 — Uniaxial tension with heterogeneous elasticity

Still elastic-only, but Young's modulus now varies from voxel to voxel as a
spatially correlated random field, so the stress field under a uniform applied
strain becomes non-uniform. This is the precursor to strain localisation once
STZ events are switched on in example 4.

| Config | Grid | Code path |
|---|---|---|
| `plane_stress_2d.yaml` | 128×128, 2D plane stress | 2D |
| `plane_stress_3d.yaml` | 128×128×1 | 3D |

Both generate `E` at run time from `seed: 1` — mean 70 GPa, std 5, correlation
length 10 voxels, clipped below at 50 — with `nu` constant at 0.3. Nothing is
read from disk, so these are self-contained. `xx` is driven to
`eps_target: 0.08` in steps of `1e-4`, with `yy` (and `zz` in 3D) relaxed to
zero stress.

The volume-averaged stiffness comes out slightly below the 70 GPa of the
homogeneous case in example 1, which is the expected direction for a
heterogeneous elastic field.

`.vtu` files are written every 100 steps; open them in ParaView to see the
stress field following the modulus field.

## Run

```bash
cd examples/3_uniaxial_tension_heterogenous
../../build/mgkmc plane_stress_2d.yaml
```
