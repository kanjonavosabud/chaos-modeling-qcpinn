import torch

# Parameters of the Lorenz63 system
default_sigma = 10.0
default_rho = 28.0
default_beta = 8.0 / 3.0
default_t0 = 0.0
default_t1 = 2.0
default_dt = 0.001
default_initial_state = (1.0, 1.0, 1.0)


class Sampler:
    def __init__(self, dim, coords, func, name=None, device="cpu"):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name
        self.device = device

    def sample(self, N):
        rand_vals = torch.rand(N, self.dim, device=self.device)
        x = (
            self.coords[0:1, :]
            + (self.coords[1:2, :] - self.coords[0:1, :]) * rand_vals
        )
        y = self.func(x.to(self.device))
        return x, y


def _to_initial_state_tensor(initial_state, device):
    initial_state_tensor = torch.as_tensor(
        initial_state, dtype=torch.float32, device=device
    ).flatten()
    if initial_state_tensor.numel() != 3:
        raise ValueError("initial_state must contain exactly three values")
    return initial_state_tensor


def lorenz_rhs(
    state,
    sigma=default_sigma,
    rho=default_rho,
    beta=default_beta,
):
    x = state[:, 0:1]
    y = state[:, 1:2]
    z = state[:, 2:3]

    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    return torch.cat((dx_dt, dy_dt, dz_dt), dim=1)


def rk4_step(
    state,
    dt,
    sigma=default_sigma,
    rho=default_rho,
    beta=default_beta,
):
    k1 = lorenz_rhs(state, sigma=sigma, rho=rho, beta=beta)
    k2 = lorenz_rhs(state + 0.5 * dt * k1, sigma=sigma, rho=rho, beta=beta)
    k3 = lorenz_rhs(state + 0.5 * dt * k2, sigma=sigma, rho=rho, beta=beta)
    k4 = lorenz_rhs(state + dt * k3, sigma=sigma, rho=rho, beta=beta)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def build_reference_trajectory(
    device="cpu",
    t0=default_t0,
    t1=default_t1,
    dt=default_dt,
    initial_state=default_initial_state,
    sigma=default_sigma,
    rho=default_rho,
    beta=default_beta,
):
    if dt <= 0:
        raise ValueError("dt must be positive")
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")

    initial_state_tensor = _to_initial_state_tensor(initial_state, device)
    num_steps = int(round((t1 - t0) / dt)) + 1
    t_ref = torch.linspace(t0, t1, num_steps, dtype=torch.float32, device=device).unsqueeze(1)
    u_ref = torch.zeros((num_steps, 3), dtype=torch.float32, device=device)
    u_ref[0] = initial_state_tensor

    for idx in range(num_steps - 1):
        current_state = u_ref[idx : idx + 1]
        u_ref[idx + 1 : idx + 2] = rk4_step(
            current_state, dt, sigma=sigma, rho=rho, beta=beta
        )

    return t_ref, u_ref


def interpolate_reference_solution(t, t_ref, u_ref):
    t_flat = t.squeeze(-1)
    ref_flat = t_ref.squeeze(-1)

    upper_idx = torch.searchsorted(ref_flat, t_flat, right=False)
    upper_idx = torch.clamp(upper_idx, min=1, max=ref_flat.shape[0] - 1)
    lower_idx = upper_idx - 1

    t_lower = ref_flat[lower_idx]
    t_upper = ref_flat[upper_idx]
    u_lower = u_ref[lower_idx]
    u_upper = u_ref[upper_idx]

    denom = torch.clamp(t_upper - t_lower, min=1e-12)
    weights = ((t_flat - t_lower) / denom).unsqueeze(1)

    return u_lower + weights * (u_upper - u_lower)


def u(
    t,
    t_ref=None,
    u_ref=None,
    initial_state=default_initial_state,
    sigma=default_sigma,
    rho=default_rho,
    beta=default_beta,
    t0=default_t0,
    t1=default_t1,
    dt=default_dt,
):
    if t_ref is None or u_ref is None:
        t_ref, u_ref = build_reference_trajectory(
            device=t.device,
            t0=t0,
            t1=t1,
            dt=dt,
            initial_state=initial_state,
            sigma=sigma,
            rho=rho,
            beta=beta,
        )

    return interpolate_reference_solution(t, t_ref, u_ref)


def r(t):
    return torch.zeros((t.shape[0], 3), dtype=torch.float32, device=t.device)


def generate_training_dataset(
    device="cpu",
    initial_state=default_initial_state,
    sigma=default_sigma,
    rho=default_rho,
    beta=default_beta,
    t0=default_t0,
    t1=default_t1,
    dt=default_dt,
):
    initial_state_tensor = _to_initial_state_tensor(initial_state, device)
    t_ref, u_ref = build_reference_trajectory(
        device=device,
        t0=t0,
        t1=t1,
        dt=dt,
        initial_state=initial_state_tensor,
        sigma=sigma,
        rho=rho,
        beta=beta,
    )

    ics_coords = torch.tensor(
        [[t0], [t0]], dtype=torch.float32, device=device
    )
    dom_coords = torch.tensor(
        [[t0], [t1]], dtype=torch.float32, device=device
    )

    initial_state_tensor = initial_state_tensor.unsqueeze(0)

    ics_sampler = Sampler(
        1,
        ics_coords,
        lambda t: initial_state_tensor.repeat(t.shape[0], 1),
        name="Initial Condition",
        device=device,
    )

    # Lorenz63 has no spatial boundary conditions, so the middle list contains
    # trajectory supervision samplers instead of boundary samplers.
    traj_sampler = Sampler(
        1,
        dom_coords,
        lambda t: u(t, t_ref=t_ref, u_ref=u_ref),
        name="Trajectory Supervision",
        device=device,
    )

    res_sampler = Sampler(
        1,
        dom_coords,
        r,
        name="Residual",
        device=device,
    )

    return [ics_sampler, [traj_sampler], res_sampler]
