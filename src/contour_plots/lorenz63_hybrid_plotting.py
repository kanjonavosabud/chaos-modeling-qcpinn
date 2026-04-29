"""
Compare one or more trained Lorenz63 solvers against the RK4 reference.

Usage:

    python -m src.contour_plots.lorenz63_hybrid_plotting \
        --classical models/lorenz63/classical/<run-id> \
        --dv        models/lorenz63/dv/<run-id> \
        --cv        models/lorenz63/cv/<run-id>

Any subset of the three flags may be omitted. Outputs are saved under
`testing_checkpoints/lorenz63/<timestamp>/`.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.lorenz63_dataset import build_reference_trajectory
from src.nn.ClassicalSolver import ClassicalSolver
from src.nn.CVPDESolver import CVPDESolver
from src.nn.DVPDESolver import DVPDESolver
from src.nn.pde import lorenz63_operator
from src.utils.logger import Logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Lorenz63 model comparisons.")
    parser.add_argument("--classical", default=None, help="Run dir for the classical model")
    parser.add_argument("--dv", default=None, help="Run dir for the DV quantum model")
    parser.add_argument("--cv", default=None, help="Run dir for the CV quantum model")
    parser.add_argument(
        "--log-path",
        default=os.path.join("testing_checkpoints", "lorenz63"),
        help="Parent directory for the comparison output run.",
    )
    return parser.parse_args()


def restore_model(solver, run_dir, logger):
    model_path = os.path.join(run_dir, "model.pth")
    if solver == "Classical":
        state = ClassicalSolver.load_state(model_path)
        model = ClassicalSolver(state["args"], logger, device=DEVICE)
        model.preprocessor.load_state_dict(state["preprocessor"])
        model.postprocessor.load_state_dict(state["postprocessor"])
    elif solver == "DV":
        state = DVPDESolver.load_state(model_path)
        model = DVPDESolver(state["args"], logger, device=DEVICE)
        model.preprocessor.load_state_dict(state["preprocessor"])
        model.postprocessor.load_state_dict(state["postprocessor"])
        model.quantum_layer.load_state_dict(state["quantum_layer"])
    elif solver == "CV":
        state = CVPDESolver.load_state(model_path)
        model = CVPDESolver(state["args"], logger, device=DEVICE)
        model.preprocessor.load_state_dict(state["preprocessor"])
        model.postprocessor.load_state_dict(state["postprocessor"])
        model.quantum_layer.load_state_dict(state["quantum_layer"])
    else:
        raise ValueError(f"Unknown solver {solver}")
    model.eval()
    return state, model


def predict(model, state, t_ref):
    cfg = state["args"]["lorenz63"]
    sigma = torch.tensor(cfg["sigma"], device=DEVICE)
    rho = torch.tensor(cfg["rho"], device=DEVICE)
    beta = torch.tensor(cfg["beta"], device=DEVICE)
    u_pred, residual = lorenz63_operator(
        model, t_ref.clone(), sigma=sigma, rho=rho, beta=beta
    )
    return u_pred.detach().cpu().numpy(), residual.detach().cpu().numpy()


def main():
    cli = parse_args()
    requested = [
        ("Classical", cli.classical),
        ("DV", cli.dv),
        ("CV", cli.cv),
    ]
    requested = [(s, p) for s, p in requested if p is not None]
    if not requested:
        raise SystemExit("Provide at least one of --classical/--dv/--cv with a run dir.")

    logger = Logging(cli.log_path)
    output_dir = logger.get_output_dir()

    # Use the first model's stored config to build the reference trajectory; all
    # supplied models should share the same Lorenz63 parameters for a fair plot.
    first_state, first_model = restore_model(*requested[0], logger=logger)
    cfg = first_state["args"]["lorenz63"]
    logger.print(
        "Reference trajectory params: "
        f"initial={cfg['initial_state']}, sigma={cfg['sigma']}, rho={cfg['rho']}, "
        f"beta={cfg['beta']:.4f}, t in [{cfg['t0']}, {cfg['t1']}], dt={cfg['dt']}"
    )

    t_ref, u_ref = build_reference_trajectory(
        device=DEVICE,
        initial_state=cfg["initial_state"],
        sigma=cfg["sigma"],
        rho=cfg["rho"],
        beta=cfg["beta"],
        t0=cfg["t0"],
        t1=cfg["t1"],
        dt=cfg["dt"],
    )
    t_np = t_ref.detach().cpu().numpy().squeeze()
    u_ref_np = u_ref.detach().cpu().numpy()

    results = {}
    losses = {}
    u_pred, res = predict(first_model, first_state, t_ref)
    results[requested[0][0]] = (u_pred, res)
    losses[requested[0][0]] = first_state.get("loss_history", [])

    for solver, run_dir in requested[1:]:
        state, model = restore_model(solver, run_dir, logger)
        u_pred, res = predict(model, state, t_ref)
        results[solver] = (u_pred, res)
        losses[solver] = state.get("loss_history", [])

    # ---- metrics ----
    logger.print("Per-model errors vs reference:")
    for name, (u_pred, _) in results.items():
        err = u_pred - u_ref_np
        rel = np.linalg.norm(err) / np.linalg.norm(u_ref_np) * 100.0
        per_comp = [
            np.linalg.norm(err[:, i]) / np.linalg.norm(u_ref_np[:, i]) * 100.0
            for i in range(3)
        ]
        logger.print(
            f"  {name}: total rel L2 = {rel:.3e} %, "
            f"x={per_comp[0]:.3e} %%, y={per_comp[1]:.3e} %%, z={per_comp[2]:.3e} %%"
        )

    # ---- plots ----
    plot_time_series(t_np, u_ref_np, results, output_dir)
    plot_phase_portraits(u_ref_np, results, output_dir)
    plot_pointwise_error(t_np, u_ref_np, results, output_dir)
    plot_loss_overlay(losses, output_dir)
    logger.print(f"Comparison artifacts saved to: {output_dir}")


def plot_time_series(t, u_ref, results, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["x(t)", "y(t)", "z(t)"]
    for idx, ax in enumerate(axes):
        ax.plot(t, u_ref[:, idx], label="reference (RK4)", color="black", linewidth=1.5)
        for name, (u_pred, _) in results.items():
            ax.plot(t, u_pred[:, idx], "--", label=name, linewidth=1.2, alpha=0.85)
        ax.set_ylabel(labels[idx])
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("t")
    fig.suptitle("Lorenz63: Reference vs Trained Models")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "time_series_compare.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_phase_portraits(u_ref, results, output_dir):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(u_ref[:, 0], u_ref[:, 1], u_ref[:, 2], color="black", label="reference", linewidth=1.0)
    for name, (u_pred, _) in results.items():
        ax.plot(
            u_pred[:, 0], u_pred[:, 1], u_pred[:, 2],
            "--", label=name, linewidth=1.0, alpha=0.85,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Lorenz63: Phase Portrait Comparison")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "phase_compare.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pointwise_error(t, u_ref, results, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    components = ["x", "y", "z"]
    for idx, ax in enumerate(axes):
        for name, (u_pred, _) in results.items():
            err = u_pred[:, idx] - u_ref[:, idx]
            ax.plot(t, err, label=name, linewidth=1.0)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_ylabel(f"err {components[idx]}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("t")
    fig.suptitle("Pointwise Error vs Reference")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pointwise_error_compare.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_overlay(losses, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, history in losses.items():
        if not history:
            continue
        ax.plot(range(len(history)), history, label=name, linewidth=1.0, alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Loss History (log scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_history_compare.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
