# 4 — Uniaxial tension with KMC plasticity

The first example that runs the Shear Transformation Zone model rather than pure
elasticity. Each voxel carries `M: 20` candidate shear orientations with
Gaussian activation barriers (mean 1.8 eV); local stress biases those barriers, a
kinetic Monte Carlo step selects an event, the chosen voxel is sheared by
`gamma0: 0.14`, the FFT solver re-equilibrates, and softening lowers the barriers
around the site.

Shared physics: `temperature: 10.0` K, applied strain rate `1e8` s⁻¹, isotropic
softening with `jp: 21.68`, `jt: 3.52`, softening cap 1.082 eV,
`q_act_temp: 0.177` eV, driven to `eps_target: 0.12`.

| Config | What it runs |
|---|---|
| `kmc_config.yaml` | 2D plane stress, 128×128, exact full-FFT stress update after every event |
| `kmc_config_fast_patching.yaml` | identical physics, but `fast_patching.enabled: true` — the stress update uses a precomputed local kernel (radius 3) and resyncs with a full FFT every 100 steps |
| `kmc_3d.yaml` | the 3D code path on a 128×128×1 grid, fast patching enabled |

The `kmc_config` / `kmc_config_fast_patching` pair is the point of this example:
run both and compare `summary_log.txt` to see what the fast-patching
approximation costs in accuracy and saves in time.

## Run

```bash
cd examples/4
../../build/mgkmc kmc_config.yaml
```

These are 128×128 KMC runs to 12% strain and take substantially longer than
examples 1–3.

## Note on config keys

These configs use `dynamics.instability_mode` and `dynamics.cascade_timing`,
which the Python package has since replaced with the boolean
`dynamics.cascade_mode`. The C++ reader still accepts the older spelling, so they
run as written.
