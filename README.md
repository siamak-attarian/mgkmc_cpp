# mgkmc_cpp

A C++17 port of the small-strain solver from [mgkmc](https://github.com/siamak-attarian/mgkmc), for running the same metallic-glass plasticity simulations faster.

## What this is

[mgkmc](https://github.com/siamak-attarian/mgkmc) simulates plastic deformation in
metallic glasses at the mesoscale. The material is a grid of voxels; each voxel
carries a set of candidate shear transformation zone (STZ) orientations with
activation barriers drawn from a random distribution. The local stress lowers the
barrier for favourably-oriented shears, a kinetic Monte Carlo step picks which
zone transforms and when, the transformed voxel is given a shear eigenstrain, and
an FFT-based spectral solver restores mechanical equilibrium across the grid.
Softening then lowers the barriers around the site, which is what lets shear
bands nucleate and grow rather than deforming the sample uniformly.

That loop is inherently serial — one event at a time, with a global solve after
each — so the Python implementation spends most of its time in the inner loop.
This repository re-implements the small-strain part of it in C++ to shorten that
loop. **It exists for speed; the Python package remains the reference
implementation.**

## Building

Header-only dependencies are vendored in `third_party/`, so no package manager or
network access is needed:

- [Eigen 3.4.0](https://eigen.tuxfamily.org) — dense linear algebra
- [pocketfft](https://github.com/mreineck/pocketfft) — FFTs for the spectral solver

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces `build/mgkmc` (`build/mgkmc.exe` on Windows). Requires CMake ≥ 3.16
and a C++17 compiler.

On Windows with MinGW, pass the generator explicitly:

```bash
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The MinGW build links libstdc++ and libwinpthread statically, so the executable
runs without those DLLs on `PATH`.

### Build options

| Option | Default | Effect |
|---|---|---|
| `MGKMC_USE_SYSTEM_EIGEN` | `OFF` | Link an installed Eigen (`find_package(Eigen3 3.4)`) instead of the vendored copy |
| `MGKMC_BUILD_TESTS` | `ON` | Also build `test_landau_stress` (see below) |

## Running

The executable takes one argument, the path to a YAML config. It defaults to
`cpp_config.yaml` if none is given. Output goes to the directory named by
`output.directory` in the config, resolved relative to the **current working
directory** — so run from the directory where you want the results:

```bash
cd examples/1_uniaxial_tension_2d_homogeneous
../../build/mgkmc plane_stress.yaml
```

This writes `output_plane_stress/` containing `summary_log.txt` (per-step strain
and stress), `global_log.txt`, and ParaView-readable `.vtu` files.

See [`examples/`](examples/) — each directory has a README explaining what its
configs do.

## Configuration

Configs use the same schema and key names as the Python package, so a config can
usually be moved between the two. The YAML is read by a small hand-written parser
(`yaml_parser.hpp`) that supports **nested maps of scalar values only** — no
lists, anchors, multi-line strings, or quoted `#`. That is enough for every
config in `examples/`, but it is not a general YAML implementation, and it will
silently ignore a key it cannot parse rather than reporting an error.

Keys are addressed by dotted path (`system.nx`, `physics.jp`, ...). For the
meaning of individual keys, see the
[mgkmc configuration guide](https://mgkmc.readthedocs.io/) and the commented
`config.yaml` in the Python repository.

## Relationship to the Python package

This is a partial port. It implements:

| | Implemented |
|---|---|
| Kinematics | small strain only |
| Dimensionality | 2D (plane strain / plane stress) and 3D |
| Simulation types | `kmc`, and elastic-only (`linear_elastic`) |
| Elasticity | linear (Hooke), and the Landau nonlinear model with strain capping (**2D only**) |
| Material fields | `constant`, `generated` (correlated random field), `file` (`.npy`) |
| Barrier distributions | `gaussian`, `rayleigh`, `modified_rayleigh`, `modified_rayleigh_with_exponential` |
| Softening | isotropic and directional |
| Fast patching | yes (local stress kernel with periodic full-FFT resync) |
| Output | summary/global logs, `.vtu` export |

Not ported, and **refused explicitly at startup rather than silently approximated**:

- `strain_assumption: finite_strain` — the whole finite-strain path
- `hyperelastic_model: landau` in 3D
- HDF5 checkpointing and restart
- the thermal diffusion solver, plotting, and analysis utilities

If you need any of those, use the Python package.

### Measured speed

On one Windows machine (GCC 13.2, `-O2`), running the identical example config
through both implementations, 800 loading steps of the elastic path:

| Example | Python | C++ |
|---|---|---|
| `1/plane_stress.yaml` — 128×128, 2D | 14.7 s | 2.2 s |
| `2/plane_stress.yaml` — 128×128×1, 3D | 49.4 s | 19.2 s |

Both produce σ_xx = 5.600 GPa at ε_xx = 0.08. These are two configs on one
machine and should be treated as an indication, not a benchmark — reproduce them
with the commands above before relying on the ratio.

## Reference check

`test_landau_stress` is a standalone program, not a pass/fail test suite. It
prints the Landau stress response for four hand-picked strain states so they can
be compared by eye against the Python reference values recorded in
`test_landau_stress.cpp`:

```bash
./build/test_landau_stress
```

## Status

Research code supporting a paper in preparation. The API and config schema are
not stable, and this port tracks the Python package rather than leading it.
Issues and questions are welcome via the issue tracker.

## License

MIT — see [LICENSE](LICENSE).

The vendored dependencies keep their own licences: Eigen 3.4.0 is primarily
MPL2, with some files under BSD or LGPL (see
`third_party/eigen-3.4.0/COPYING.README`), and pocketfft is BSD-3-Clause.

## Citation and contact

A paper describing the model is in preparation. Until it appears, please cite
this repository directly and get in touch if you are using it in published work.

Siamak Attarian — <siamak.attarian@gmail.com>
