import time
import torch

from src.data.lorenz63_dataset import compute_normalization_stats, generate_training_dataset
from src.nn.pde import lorenz63_operator


def fetch_minibatch(sampler, N):
    X, Y = sampler.sample(N)
    return X, Y


def train(
    model,
    initial_state=(1.0, 1.0, 1.0),
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
    t0=0.0,
    t1=2.0,
    dt=0.001,
    batch_size=None,
    normalize=True,
    stats=None,
):
    """
    Train a solver on the Lorenz63 system.

    Loss = w_ic * loss_ic + w_traj * loss_traj + w_res * loss_res

    When `normalize=True`, the model trains in normalized coordinates:
      - input  t_norm = (t - t0) / (t1 - t0)         (in [0, 1])
      - target u_norm = (u - u_mean) / u_std         (per-component)
    The IC and trajectory losses are computed in normalized state space, so all
    three components contribute on equal footing. The residual is rescaled by
    the natural derivative scale `u_std / t_span` so the residual loss is
    dimensionless and directly comparable to the other terms.
    """
    if batch_size is None:
        batch_size = model.args.get("batch_size", 128)

    if normalize and stats is None:
        stats = compute_normalization_stats(
            device=model.device,
            initial_state=initial_state,
            sigma=sigma,
            rho=rho,
            beta=beta,
            t0=t0,
            t1=t1,
            dt=dt,
        )
    if not normalize:
        stats = None

    [ics_sampler, traj_samplers, res_sampler] = generate_training_dataset(
        device=model.device,
        initial_state=initial_state,
        sigma=sigma,
        rho=rho,
        beta=beta,
        t0=t0,
        t1=t1,
        dt=dt,
    )
    traj_sampler = traj_samplers[0]

    sigma_t = torch.tensor(sigma, device=model.device, dtype=torch.float32)
    rho_t = torch.tensor(rho, device=model.device, dtype=torch.float32)
    beta_t = torch.tensor(beta, device=model.device, dtype=torch.float32)

    w_ic = model.args.get("w_ic", 100.0)
    w_traj = model.args.get("w_traj", 10.0)
    w_res = model.args.get("w_res", 1.0)

    if stats is not None:
        # Per-component natural scale of du/dt: state std divided by time span.
        # Dividing the physical residual by this scale brings the residual loss
        # onto the same order as the normalized state loss (~unit variance).
        residual_scale = stats["u_std"] / stats["t_span"]
    else:
        residual_scale = None

    def to_model_inputs(t_phys, u_phys):
        if stats is None:
            return t_phys, u_phys
        t_norm = (t_phys - stats["t0"]) / stats["t_span"]
        u_norm = (u_phys - stats["u_mean"]) / stats["u_std"]
        return t_norm, u_norm

    def objective_fn(it):
        start_time = time.time()

        if model.optimizer is not None:
            model.optimizer.zero_grad()

        X_ics_phys, u_ics_phys = fetch_minibatch(ics_sampler, max(batch_size // 4, 1))
        X_traj_phys, u_traj_phys = fetch_minibatch(traj_sampler, batch_size)
        X_res_phys, _ = fetch_minibatch(res_sampler, batch_size)

        X_ics_in, u_ics_target = to_model_inputs(X_ics_phys, u_ics_phys)
        X_traj_in, u_traj_target = to_model_inputs(X_traj_phys, u_traj_phys)

        u_ics_pred = model.forward(X_ics_in)
        u_traj_pred = model.forward(X_traj_in)
        _, residual_phys = lorenz63_operator(
            model, X_res_phys, sigma=sigma_t, rho=rho_t, beta=beta_t, stats=stats
        )

        if residual_scale is not None:
            residual_for_loss = residual_phys / residual_scale
        else:
            residual_for_loss = residual_phys

        loss_ics = model.loss_fn(u_ics_pred, u_ics_target)
        loss_traj = model.loss_fn(u_traj_pred, u_traj_target)
        loss_res = model.loss_fn(residual_for_loss, torch.zeros_like(residual_for_loss))

        loss = w_ic * loss_ics + w_traj * loss_traj + w_res * loss_res

        elapsed = time.time() - start_time

        if it % model.args["print_every"] == 0:
            model.logger.print(
                "It: %d, Loss: %.3e, Loss_ics: %.3e, Loss_traj: %.3e, Loss_res: %.3e, lr: %.3e, Time: %.2e"
                % (
                    it,
                    loss.item(),
                    loss_ics.item(),
                    loss_traj.item(),
                    loss_res.item(),
                    model.optimizer.param_groups[0]["lr"] if model.optimizer else 0.0,
                    elapsed,
                )
            )
            model.save_state()

        return loss

    for it in range(model.epochs + 1):
        loss = objective_fn(it)
        loss.backward()
        if model.args["solver"] == "CV":
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if model.optimizer is not None:
            model.optimizer.step()
        if model.scheduler is not None:
            model.scheduler.step(loss)

        model.loss_history.append(loss.item())

    return stats
