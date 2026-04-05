# QCPINN: Quantum-Classical Physics-Informed Neural Networks

Source code of QCPINN described in the paper: [QCPINN: Quantum-Classical Physics-Informed Neural Networks for Solving PDEs](https://iopscience.iop.org/article/10.1088/2632-2153/ae1c91).

---

## Project Structure

```
QCPINN/
├── data/               # Cavity datasets from simulation
├── models/             # Saved models from training
├── qcpinn.yaml         # Conda environment file
└── src/
    ├── contour_plots/  # Plotting functions
    ├── data/           # Data generator
    ├── nn/             # Neural network modules
    ├── notebooks/      # Jupyter notebooks (training, testing, visualization)
    ├── trainer/        # Training scripts
    └── utils/          # Utility functions and helpers
```

> See the `src/notebooks/` folder for hands-on examples and further documentation.

## Getting Started

### Prerequisites

[Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or any other Python environment.

### Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/afrah/QCPINN.git
cd QCPINN
conda env create -f qcpinn.yaml
conda activate qcpinn
```

## Training Models

Train models for different PDEs using the following commands:

```bash
# Helmholtz
python -m src.trainer.helmholtz_hybrid_trainer

# Cavity
python -m src.trainer.cavity_hybrid_trainer

# Klein-Gordon
python -m src.trainer.klein_gordon_hybrid_trainer

# Wave
python -m src.trainer.wave_hybrid_trainer

# Diffusion
python -m src.trainer.diffusion_hybrid_trainer
```

Jupyter notebooks for training, testing, and visualization are in `src/notebooks/`.

> **Note:** I used VS Code with the Jupyter extension for working on the notebooks.

## Inference

After training, generate plots and evaluate results:

```bash
# Helmholtz
python -m src.contour_plots.helmholtz_hybrid_plotting

# Cavity
python -m src.contour_plots.cavity_hybrid_plotting

# Klein-Gordon
python -m src.contour_plots.klein_gordon_hybrid_plotting

# Wave
python -m src.contour_plots.wave_hybrid_plotting

# Diffusion
python -m src.contour_plots.diffusion_hybrid_plotting
```

## Testing 

**Amplitude vs. Angle Encodings**

```bash
# Cavity
python -m src.testing.cavity_test

# Helmholtz
python -m src.testing.helmholtz_test
```

Output plots and data are saved in the results directory.

## Results

**Helmholtz Equation**

- Embedding: Angle
- Topology: Cascade
- Configuration [link](https://github.com/afrah/QCPINN/blob/main/src/nn/DVPDESolver.py#L60) 
- Results [folder](doc/results/helmholtz)

**Cavity flow**
- Embedding: Angle
- Topology: Cascade
- Configuration [link](https://github.com/afrah/QCPINN/blob/main/src/nn/DVPDESolver.py#L60) 
- Results [folder](doc/results/cavity)

**Wave Equation**
- Embedding: Angle
- Topology: Cross-mesh
- Configuration [link](https://github.com/afrah/QCPINN/blob/main/src/nn/DVPDESolver.py#L60) 
- Results [folder](doc/results/Wave)

**Klein_Gordon Equation**
- Embedding: Angle
- Topology: Cascade
- Configuration [link](https://github.com/afrah/QCPINN/blob/main/src/nn/DVPDESolver.py#L60) 
- Results [folder](doc/results/klein-Gordon)

**Convection Diffusion**
- Embedding: Angle
- Topology: Cascade
- Configuration [link](https://github.com/afrah/QCPINN/blob/main/src/nn/DVPDESolver.py#L60) 
- Results [folder](doc/results/cavity)


**Comparisio of Different Embeddings**
- Loss convergence Helomholtz [plot](doc/results/helmholtz/2025-10-09_10-46-51-485328/loss_history_helmholtz.png)
- Loss convergence Cavity flow [plot](doc/results/cavity/2025-10-06_19-42-17-416929/loss_history_cavity.png)

**CV-QCPINN model Results**
- Configuration [link](src/nn/CVNeuralNetwork1.py) 
- Loss convergence Helomholtz [plot](doc/results/CV-QCPINN/loss_plots_helmholtz.pdf)
- Loss convergence Cavity flow [plot](doc/results/CV-QCPINN/loss_plots_cavity.pdf)

## Support

If you encounter issues or have questions, please [open an issue](https://github.com/afrah/QCPINN/issues).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT [LICENSE](LICENSE)

## References

If you find this work useful, please consider citing:

```bibtex
@article{Farea:2025:MLST,
	author={Farea, Afrah and Khan, Saiful and ÇELEBİ, Mustafa Serdar},
	title={QCPINN: Quantum-Classical Physics-Informed Neural Networks for Solving PDEs},
	journal={Machine Learning: Science and Technology},
	url={http://iopscience.iop.org/article/10.1088/2632-2153/ae1c91},
	year={2025},
}
```

## Lorenz63 Dataset Utilities

This repository now also includes a configurable Lorenz63 dataset helper and a small verification script:

- `src/data/lorenz63_dataset.py`
- `src/testing/lorenz63_dataset_test.py`

The dataset module is designed in the same spirit as the other files under `src/data/`, but adapted for the Lorenz63 dynamical system rather than a PDE with spatial boundary conditions.

### What `src/data/lorenz63_dataset.py` Provides

- `lorenz_rhs(...)`: Lorenz63 right-hand side
- `rk4_step(...)`: one Runge-Kutta 4 step
- `build_reference_trajectory(...)`: builds a reference trajectory in time
- `u(...)`: returns reference states interpolated at requested time points
- `r(...)`: returns zero residual targets with shape `(N, 3)`
- `generate_training_dataset(...)`: returns the Lorenz63 training samplers

The dataset factory returns:

```python
[ics_sampler, [traj_sampler], res_sampler]
```

where:

- `ics_sampler` gives the initial condition at `t0`
- `traj_sampler` gives trajectory supervision values over the selected time interval
- `res_sampler` gives zero residual targets over the selected time interval

### Configurable Parameters

External callers can choose:

- `initial_state`
- `sigma`
- `rho`
- `beta`
- `t0`
- `t1`
- `dt`

Example:

```python
from src.data.lorenz63_dataset import generate_training_dataset

ics_sampler, traj_samplers, res_sampler = generate_training_dataset(
    device="cpu",
    initial_state=(2.0, 3.0, 4.0),
    sigma=14.0,
    rho=35.0,
    beta=3.0,
    t0=0.0,
    t1=1.0,
    dt=0.002,
)
```

### Verifying The Dataset

The verification script checks that:

- the initial-condition sampler is constant at the chosen initial state
- the trajectory sampler matches the reference trajectory interpolation
- the residual sampler returns zeros with the expected shape

It also saves:

- a time-series plot of `x(t)`, `y(t)`, and `z(t)`
- a 3D phase portrait plot
- a log file with the sampled shapes and verification metrics

Run it from the repository root with:

```bash
venv/bin/python src/testing/lorenz63_dataset_test.py
```

### Running With Custom Lorenz63 Parameters

You can choose the initial point and the Lorenz63 parameters from the command line:

```bash
venv/bin/python src/testing/lorenz63_dataset_test.py \
  --x0 2.0 --y0 3.0 --z0 4.0 \
  --sigma 14.0 --rho 35.0 --beta 3.0 \
  --t0 0.0 --t1 1.0 --dt 0.002
```

The script writes outputs under:

```text
src/testing_checkpoints/lorenz63_dataset/
```

Each run creates a timestamped directory containing:

- `output.log`
- `lorenz63_time_series.png`
- `lorenz63_phase_plot.png`

### Notes

- The script is written so it can be launched directly without manually setting `PYTHONPATH`.
- It also configures a safe non-interactive Matplotlib backend and cache directories automatically.
- Make sure the repository virtual environment contains `torch` and `matplotlib` before running the test script.

### Next Steps

These are just working notes for future Lorenz63 integration work:

- Add a `lorenz63_operator(...)` function to `src/nn/pde.py`.
- Add a dedicated `src/trainer/lorenz63_train.py`.
- Decide whether Lorenz63 training should use only physics loss plus the initial condition, or also include trajectory supervision points.
- Add a Lorenz63 training entry point similar to the other `src/trainer/*_hybrid_trainer.py` files.
- Add a Lorenz63 evaluation or plotting script for trained models, similar to the existing scripts under `src/contour_plots/` or `src/testing/`.
- Decide how Lorenz63 configuration should be passed around consistently, for example through one shared `args` block.
