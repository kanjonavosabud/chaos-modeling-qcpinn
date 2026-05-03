import torch


def navier_stokes_2D_operator(model, t, x, y, min_x=0, max_x=1):
    """
    Operator to compute residuals for the 2D Navier-Stokes equation
    """

    mu = 0.00345
    DENSITY = 1056.0

    t.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True

    uvp = model(torch.concatenate((t, x, y), 1))

    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]

    v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

    continuity = u_x + v_y
    f_u = u_t + (u * u_x + v * u_y) + 1.0 / DENSITY * p_x - mu * (u_xx + u_yy)
    f_v = v_t + (u * v_x + v * v_y) + 1.0 / DENSITY * p_y - mu * (v_xx + v_yy)

    return [continuity, f_u, f_v]


def klein_gordon_operator(fluid_model, t, x, x_min=0.0, x_max=1.0):
    """
    Operator to compute residuals for the 1D Klein-Gordon equation
    """

    alpha = -1.0
    beta = 0.0
    gamma = 1.0
    k = 3

    t.requires_grad = True
    x.requires_grad = True

    u = fluid_model(torch.concatenate((t, x), 1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    residual = u_tt + alpha * u_xx + beta * u + gamma * u**k
    return u, residual


def wave_operator(model, t, x, sigma_t=1.0, sigma_x=1.0):
    """
    Operator to compute residuals for the 1D wave equation
    """
    c = 2
    t.requires_grad = True
    x.requires_grad = True

    u = model(torch.concatenate((t, x), 1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    residual = u_tt - c**2 * u_xx
    return u, residual


def diffusion_operator(
    model, t, x, y, sigma_t=1.0, sigma_x=1.0, sigma_y=1.0, D=0.01, v_x=1.0, v_y=1.0
):
    """
    Operator to compute residuals for the 2D convection-diffusion equation
    """

    t.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True

    # forward pass through the model
    u = model(torch.cat((t, x, y), 1))

    # compute derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0] / sigma_t
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0] / sigma_x
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0] / sigma_y

    u_xx = (
        torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        / sigma_x
    )
    u_yy = (
        torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
        / sigma_y
    )

    # convection-diffusion equation residual
    residual = u_t + v_x * u_x + v_y * u_y - D * (u_xx + u_yy)

    return u, residual


def lorenz63_operator(model, t, sigma=10.0, rho=28.0, beta=8.0 / 3.0, stats=None):
    """
    Operator to compute residuals for the Lorenz63 ODE system.

    `t` is always passed in physical units. Returns `(u_phys, residual_phys)`
    in physical units regardless of whether normalization is enabled.

    If `stats` is None: model takes physical `t` and outputs physical `u`.

    If `stats` is a dict with keys {t0, t_span, u_mean, u_std}: model takes
    `t_norm = (t - t0)/t_span` and outputs `u_norm`. The operator wraps the
    model so it still sees physical `t` going in and physical `u` coming out;
    autograd handles the chain rule across the linear normalization layers.
    """
    t.requires_grad_(True)

    if stats is None:
        u = model(t)
    else:
        t_norm = (t - stats["t0"]) / stats["t_span"]
        u_norm = model(t_norm)
        u = u_norm * stats["u_std"] + stats["u_mean"]

    x = u[:, 0:1]
    y = u[:, 1:2]
    z = u[:, 2:3]

    x_t = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    y_t = torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]
    z_t = torch.autograd.grad(z, t, torch.ones_like(z), create_graph=True)[0]

    f_x = x_t - sigma * (y - x)
    f_y = y_t - (x * (rho - z) - y)
    f_z = z_t - (x * y - beta * z)

    residual = torch.cat((f_x, f_y, f_z), dim=1)
    return u, residual


def helmholtz_operator(
    fluid_model,
    x1,
    x2,
):
    """
    Operator to compute residuals for the 2D helmholtz equation
    """

    LAMBDA = 1.0

    x1.requires_grad_(True)
    x2.requires_grad_(True)
    u = fluid_model(torch.concatenate((x1, x2), 1))

    # compute gradients with respect to x1 and x2
    u_x1 = torch.autograd.grad(
        u, x1, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    u_x2 = torch.autograd.grad(
        u, x2, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    u_xx1 = torch.autograd.grad(
        u_x1, x1, grad_outputs=torch.ones_like(u_x1), create_graph=True
    )[0]

    u_xx2 = torch.autograd.grad(
        u_x2, x2, grad_outputs=torch.ones_like(u_x2), create_graph=True
    )[0]

    residual = u_xx1 + u_xx2 + LAMBDA * u

    return [u, residual]
