import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import src.trainer.lorenz63_train as lorenz63_train
from src.data.lorenz63_dataset import (
    build_reference_trajectory,
    compute_normalization_stats,
    default_beta,
    default_dt,
    default_initial_state,
    default_rho,
    default_sigma,
    default_t0,
    default_t1,
    stats_from_serializable,
    stats_to_serializable,
)
from src.nn.ClassicalSolver import ClassicalSolver
from src.nn.CVPDESolver import CVPDESolver
from src.nn.DVPDESolver import DVPDESolver
from src.nn.pde import lorenz63_operator
from src.utils.logger import Logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a classical / DV / CV solver on the Lorenz63 system."
    )
    parser.add_argument(
        "--solver",
        choices=["Classical", "DV", "CV"],
        default="Classical",
        help="Which model family to train.",
    )
    parser.add_argument(
        "--encoding",
        choices=["angle", "amplitude", "None"],
        default="angle",
        help="Encoding used by the DV solver. Ignored otherwise.",
    )
    parser.add_argument(
        "--q-ansatz",
        choices=["alternate", "layered", "cascade", "cross_mesh", "farhi"],
        default="cascade",
        help="Quantum ansatz topology used by the DV solver.",
    )
    parser.add_argument(
        "--cv-class",
        choices=["CVNeuralNetwork1", "CVNeuralNetwork2", "CVNeuralNetwork3"],
        default="CVNeuralNetwork1",
        help="Variant of the CV quantum network.",
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=50)
    parser.add_argument("--num-quantum-layers", type=int, default=1)
    parser.add_argument("--cutoff-dim", type=int, default=8)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--x0", type=float, default=default_initial_state[0])
    parser.add_argument("--y0", type=float, default=default_initial_state[1])
    parser.add_argument("--z0", type=float, default=default_initial_state[2])
    parser.add_argument("--sigma", type=float, default=default_sigma)
    parser.add_argument("--rho", type=float, default=default_rho)
    parser.add_argument("--beta", type=float, default=default_beta)
    parser.add_argument("--t0", type=float, default=default_t0)
    parser.add_argument("--t1", type=float, default=default_t1)
    parser.add_argument("--dt", type=float, default=default_dt)

    parser.add_argument("--w-ic", type=float, default=100.0)
    parser.add_argument("--w-traj", type=float, default=10.0)
    parser.add_argument("--w-res", type=float, default=1.0)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable input/output normalization (train in physical units).",
    )
    parser.add_argument(
        "--log-path",
        default=os.path.join("models", "lorenz63"),
        help="Parent directory for run checkpoints.",
    )
    return parser.parse_args()


def build_args_dict(cli):
    input_dim = 1
    output_dim = 3
    return {
        "batch_size": cli.batch_size,
        "epochs": cli.epochs,
        "lr": cli.lr,
        "seed": cli.seed,
        "print_every": cli.print_every,
        "log_path": os.path.join(cli.log_path, cli.solver.lower()),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "num_qubits": cli.num_qubits,
        "hidden_dim": cli.hidden_dim,
        "num_quantum_layers": cli.num_quantum_layers,
        "classic_network": [input_dim, cli.hidden_dim, output_dim],
        "q_ansatz": cli.q_ansatz,
        "mode": "hybrid" if cli.solver != "Classical" else "classical",
        "activation": "tanh",
        "shots": None,
        "problem": "lorenz63",
        "solver": cli.solver,
        "device": DEVICE,
        "method": "None",
        "cutoff_dim": cli.cutoff_dim,
        "class": cli.cv_class,
        "encoding": cli.encoding,
        "w_ic": cli.w_ic,
        "w_traj": cli.w_traj,
        "w_res": cli.w_res,
        "lorenz63": {
            "initial_state": (cli.x0, cli.y0, cli.z0),
            "sigma": cli.sigma,
            "rho": cli.rho,
            "beta": cli.beta,
            "t0": cli.t0,
            "t1": cli.t1,
            "dt": cli.dt,
        },
    }


def build_model(args, logger):
    if args["solver"] == "CV":
        model = CVPDESolver(args, logger, device=DEVICE)
        logger.print("Using CV Solver")
    elif args["solver"] == "Classical":
        model = ClassicalSolver(args, logger, device=DEVICE)
        logger.print("Using Classical Solver")
    elif args["solver"] == "DV":
        model = DVPDESolver(args, logger, device=DEVICE)
        logger.print("Using DV Solver")
    else:
        raise ValueError(f"Unknown solver {args['solver']}")
    return model


def evaluate_and_plot(model, args, logger, output_dir, stats=None):
    cfg = args["lorenz63"]

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

    u_pred_state, residual = lorenz63_operator(
        model,
        t_ref.clone(),
        sigma=torch.tensor(cfg["sigma"], device=DEVICE),
        rho=torch.tensor(cfg["rho"], device=DEVICE),
        beta=torch.tensor(cfg["beta"], device=DEVICE),
        stats=stats,
    )

    t_np = t_ref.detach().cpu().numpy().squeeze()
    u_ref_np = u_ref.detach().cpu().numpy()
    u_pred_np = u_pred_state.detach().cpu().numpy()
    res_np = residual.detach().cpu().numpy()

    err = u_pred_np - u_ref_np
    rel_l2 = np.linalg.norm(err) / np.linalg.norm(u_ref_np) * 100.0
    component_rel_l2 = [
        np.linalg.norm(err[:, i]) / np.linalg.norm(u_ref_np[:, i]) * 100.0
        for i in range(3)
    ]
    res_max = np.max(np.abs(res_np))
    logger.print(f"Relative L2 error (state): {rel_l2:.3e} %")
    logger.print(
        "Per-component relative L2 error: x=%.3e %%, y=%.3e %%, z=%.3e %%"
        % tuple(component_rel_l2)
    )
    logger.print(f"Max abs ODE residual on reference grid: {res_max:.3e}")

    save_loss_plot(model, output_dir)
    save_time_series_plot(t_np, u_ref_np, u_pred_np, output_dir, args)
    save_phase_plot(u_ref_np, u_pred_np, output_dir, args)
    save_error_plot(t_np, err, output_dir, args)
    save_residual_plot(t_np, res_np, output_dir, args)

    np.savez(
        os.path.join(output_dir, "evaluation.npz"),
        t=t_np,
        u_ref=u_ref_np,
        u_pred=u_pred_np,
        residual=res_np,
    )


def save_loss_plot(model, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(model.loss_history)), model.loss_history, linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Training Loss History")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_history.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_time_series_plot(t, u_ref, u_pred, output_dir, args):
    cfg = args["lorenz63"]
    title = (
        f"Lorenz63 [{args['solver']}] — sigma={cfg['sigma']}, rho={cfg['rho']}, "
        f"beta={cfg['beta']:.3f}, t in [{cfg['t0']}, {cfg['t1']}]"
    )
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["x(t)", "y(t)", "z(t)"]
    for idx, ax in enumerate(axes):
        ax.plot(t, u_ref[:, idx], label="reference (RK4)", linewidth=1.5)
        ax.plot(t, u_pred[:, idx], "--", label="model", linewidth=1.5, alpha=0.8)
        ax.set_ylabel(labels[idx])
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("t")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "time_series.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_phase_plot(u_ref, u_pred, output_dir, args):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(u_ref[:, 0], u_ref[:, 1], u_ref[:, 2], label="reference", linewidth=1.0)
    ax.plot(
        u_pred[:, 0], u_pred[:, 1], u_pred[:, 2],
        "--", label="model", linewidth=1.0, alpha=0.8,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Phase Portrait — {args['solver']}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "phase_portrait.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_error_plot(t, err, output_dir, args):
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, label in enumerate(["x", "y", "z"]):
        ax.plot(t, err[:, idx], label=f"err {label}", linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("prediction - reference")
    ax.set_title(f"Pointwise Error — {args['solver']}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pointwise_error.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_residual_plot(t, residual, output_dir, args):
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, label in enumerate(["fx", "fy", "fz"]):
        ax.plot(t, residual[:, idx], label=label, linewidth=1.0)
    ax.set_xlabel("t")
    ax.set_ylabel("ODE residual")
    ax.set_title(f"Lorenz63 Residual — {args['solver']}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ode_residual.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    cli = parse_args()
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    args = build_args_dict(cli)
    logger = Logging(args["log_path"])
    output_dir = logger.get_output_dir()

    model = build_model(args, logger)

    logger.print("The settings used:")
    for key, value in args.items():
        logger.print(f"{key} : {value}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.print(f"Total number of parameters: {total_params}")

    cfg = args["lorenz63"]
    normalize = not cli.no_normalize
    stats = None
    if normalize:
        stats = compute_normalization_stats(
            device=DEVICE,
            initial_state=cfg["initial_state"],
            sigma=cfg["sigma"],
            rho=cfg["rho"],
            beta=cfg["beta"],
            t0=cfg["t0"],
            t1=cfg["t1"],
            dt=cfg["dt"],
        )
        # Persist stats into args so the checkpoint round-trips them.
        model.args["lorenz63_stats"] = stats_to_serializable(stats)
        logger.print(
            "Normalization stats: u_mean=%s, u_std=%s, t0=%.4f, t_span=%.4f"
            % (
                stats["u_mean"].cpu().tolist(),
                stats["u_std"].cpu().tolist(),
                float(stats["t0"]),
                float(stats["t_span"]),
            )
        )
    else:
        logger.print("Normalization DISABLED (training in physical units).")

    lorenz63_train.train(
        model,
        initial_state=cfg["initial_state"],
        sigma=cfg["sigma"],
        rho=cfg["rho"],
        beta=cfg["beta"],
        t0=cfg["t0"],
        t1=cfg["t1"],
        dt=cfg["dt"],
        batch_size=cli.batch_size,
        normalize=normalize,
        stats=stats,
    )

    model.save_state()
    logger.print("Training completed successfully.")

    evaluate_and_plot(model, args, logger, output_dir, stats=stats)
    logger.print(f"All artifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()
