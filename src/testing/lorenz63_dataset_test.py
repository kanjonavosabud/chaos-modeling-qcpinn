import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_CHECKPOINT_PATH = os.path.join(REPO_ROOT, "src/testing_checkpoints/lorenz63_dataset")
MATPLOTLIB_CACHE_DIR = os.path.join(REPO_ROOT, ".matplotlib")
XDG_CACHE_DIR = os.path.join(REPO_ROOT, ".cache")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.makedirs(MATPLOTLIB_CACHE_DIR, exist_ok=True)
os.makedirs(XDG_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", MATPLOTLIB_CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", XDG_CACHE_DIR)

from src.data.lorenz63_dataset import (
    build_reference_trajectory,
    default_beta,
    default_initial_state,
    default_rho,
    default_sigma,
    generate_training_dataset,
    u,
)
from src.utils.logger import Logging


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Verify the Lorenz63 dataset module")
    parser.add_argument("--x0", type=float, default=default_initial_state[0])
    parser.add_argument("--y0", type=float, default=default_initial_state[1])
    parser.add_argument("--z0", type=float, default=default_initial_state[2])
    parser.add_argument("--sigma", type=float, default=default_sigma)
    parser.add_argument("--rho", type=float, default=default_rho)
    parser.add_argument("--beta", type=float, default=default_beta)
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--t1", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.001)
    return parser.parse_args()


def setup_logger():
    logger = Logging(TEST_CHECKPOINT_PATH)
    return logger, logger.get_output_dir()


def verify_dataset(logger, initial_state, sigma, rho, beta, t0, t1, dt):
    ics_sampler, traj_samplers, res_sampler = generate_training_dataset(
        DEVICE,
        initial_state=initial_state,
        sigma=sigma,
        rho=rho,
        beta=beta,
        t0=t0,
        t1=t1,
        dt=dt,
    )
    traj_sampler = traj_samplers[0]

    t_ref, u_ref = build_reference_trajectory(
        device=DEVICE,
        initial_state=initial_state,
        sigma=sigma,
        rho=rho,
        beta=beta,
        t0=t0,
        t1=t1,
        dt=dt,
    )

    x_ics, y_ics = ics_sampler.sample(32)
    x_traj, y_traj = traj_sampler.sample(512)
    x_res, y_res = res_sampler.sample(256)

    y_traj_expected = u(x_traj, t_ref=t_ref, u_ref=u_ref)

    ics_time_error = torch.max(torch.abs(x_ics - x_ics[0])).item()
    ics_state_error = torch.max(torch.abs(y_ics - y_ics[0])).item()
    traj_interp_error = torch.max(torch.abs(y_traj - y_traj_expected)).item()
    residual_norm = torch.max(torch.abs(y_res)).item()

    logger.print(f"device: {DEVICE}")
    logger.print(
        f"initial_state={tuple(initial_state)}, sigma={sigma}, rho={rho}, beta={beta}"
    )
    logger.print(f"time interval=[{t0}, {t1}], dt={dt}")
    logger.print(f"reference trajectory shape: {tuple(u_ref.shape)}")
    logger.print(f"ics sample shapes: x={tuple(x_ics.shape)}, y={tuple(y_ics.shape)}")
    logger.print(f"trajectory sample shapes: x={tuple(x_traj.shape)}, y={tuple(y_traj.shape)}")
    logger.print(f"residual sample shapes: x={tuple(x_res.shape)}, y={tuple(y_res.shape)}")
    logger.print(f"ics time spread: {ics_time_error:.3e}")
    logger.print(f"ics state spread: {ics_state_error:.3e}")
    logger.print(f"trajectory interpolation error: {traj_interp_error:.3e}")
    logger.print(f"residual target max abs value: {residual_norm:.3e}")

    return logger.get_output_dir(), t_ref, u_ref, x_traj, y_traj


def save_time_series_plot(output_dir, t_ref, u_ref, x_traj, y_traj, params_label):
    import matplotlib.pyplot as plt

    t_ref_np = t_ref.detach().cpu().numpy().squeeze()
    u_ref_np = u_ref.detach().cpu().numpy()
    x_traj_np = x_traj.detach().cpu().numpy().squeeze()
    y_traj_np = y_traj.detach().cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["x(t)", "y(t)", "z(t)"]

    for idx, ax in enumerate(axes):
        ax.plot(t_ref_np, u_ref_np[:, idx], label="reference", linewidth=1.5)
        ax.scatter(x_traj_np, y_traj_np[:, idx], s=10, alpha=0.35, label="sampled")
        ax.set_ylabel(labels[idx])
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("t")
    fig.suptitle(f"Lorenz63 Dataset Check: Time Series Samples\n{params_label}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "lorenz63_time_series.png"), bbox_inches="tight")
    plt.close(fig)


def save_phase_plot(output_dir, u_ref, y_traj, params_label):
    import matplotlib.pyplot as plt

    u_ref_np = u_ref.detach().cpu().numpy()
    y_traj_np = y_traj.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(u_ref_np[:, 0], u_ref_np[:, 1], u_ref_np[:, 2], linewidth=1.0, label="reference")
    ax.scatter(
        y_traj_np[:, 0],
        y_traj_np[:, 1],
        y_traj_np[:, 2],
        s=8,
        alpha=0.35,
        label="sampled",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Lorenz63 Dataset Check: Phase Portrait\n{params_label}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "lorenz63_phase_plot.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    initial_state = (args.x0, args.y0, args.z0)
    params_label = (
        f"x0={args.x0}, y0={args.y0}, z0={args.z0}, "
        f"sigma={args.sigma}, rho={args.rho}, beta={args.beta}"
    )
    logger, output_dir = setup_logger()
    output_dir, t_ref, u_ref, x_traj, y_traj = verify_dataset(
        logger,
        initial_state=initial_state,
        sigma=args.sigma,
        rho=args.rho,
        beta=args.beta,
        t0=args.t0,
        t1=args.t1,
        dt=args.dt,
    )
    try:
        save_time_series_plot(output_dir, t_ref, u_ref, x_traj, y_traj, params_label)
        save_phase_plot(output_dir, u_ref, y_traj, params_label)
        logger.print(f"Saved Lorenz63 dataset plots to: {output_dir}")
    except ModuleNotFoundError as exc:
        logger.print(f"Plotting skipped because a dependency is missing: {exc}")
        logger.print(f"Dataset verification logs were still saved to: {output_dir}")


if __name__ == "__main__":
    main()
