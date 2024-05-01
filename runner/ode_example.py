import torch.nn as nn
from joblib import Parallel, delayed
import multiprocessing as mp
import numpy as np
import torch


"""
    for more ODE/SDE solver usage please look up at torchdyn package
        
"""

def construct_dopri5_numpy(dtype):
    c = np.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.], dtype=dtype)
    a = [
        np.array([1 / 5], dtype=dtype),
        np.array([3 / 40, 9 / 40], dtype=dtype),
        np.array([44 / 45, -56 / 15, 32 / 9], dtype=dtype),
        np.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729], dtype=dtype),
        np.array([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], dtype=dtype),
        np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], dtype=dtype),
    ]
    bsol = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=dtype)
    berr = np.array([1951 / 21600, 0, 22642 / 50085, 451 / 720, -12231 / 42400, 649 / 6300, 1 / 60.], dtype=dtype)

    dmid = np.array([-1.1270175653862835, 0., 2.675424484351598, -5.685526961588504, 3.5219323679207912,
                         -1.7672812570757455, 2.382468931778144])
    return (c, a, bsol, bsol - berr)

def construct_dopri5(dtype):
    c = torch.tensor([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.], dtype=dtype)
    a = [
        torch.tensor([1 / 5], dtype=dtype),
        torch.tensor([3 / 40, 9 / 40], dtype=dtype),
        torch.tensor([44 / 45, -56 / 15, 32 / 9], dtype=dtype),
        torch.tensor([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729], dtype=dtype),
        torch.tensor([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], dtype=dtype),
        torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], dtype=dtype),
    ]
    bsol = torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=dtype)
    berr = torch.tensor([1951 / 21600, 0, 22642 / 50085, 451 / 720, -12231 / 42400, 649 / 6300, 1 / 60.], dtype=dtype)

    dmid = torch.tensor([-1.1270175653862835, 0., 2.675424484351598, -5.685526961588504, 3.5219323679207912,
                         -1.7672812570757455, 2.382468931778144])
    return (c, a, bsol, bsol - berr)



    


def euler_solve(x0, my_model, N=100):
    h = 1 / (N - 1)
    x_fake = x0
    t = 0
    # from t=0 to t=1
    for i in range(N - 1):
        x_fake = x_fake + h * my_model(t=t, xt=x_fake)
        t = t + h
    return x_fake

def midpoint_solve(x0, my_model, N=100):
    h = 1 / (N - 1)
    x_fake = x0
    t = 0
    # from t=0 to t=1
    for i in range(N - 1):
        x_mid = x_fake + 0.5 * h * my_model(t=t, xt=x_fake)
        x_fake = x_fake + h * my_model(t=t+h/2, xt=x_mid)
        t = t + h
        
    return x_fake


## to torch
def alf_solve(x0, my_model, N=100, const=1):
    half_state_dim = x0.shape[-1] // 2
    
    x, v = x0[..., :half_state_dim], x0[..., half_state_dim:]
    h = 1 / (N - 1)
    t = 0
    for i in range(N - 1):
        x1 = x + 0.5 * h * v
        xt1 = np.tile(x1, 2)
        vt1 = my_model(t=t + 0.5 * h, xt=xt1)
        v1 = 2 * const * (vt1[:len(v)] - v) + v
        x2 = x1 + 0.5 * h * v1
        x_fake = np.concatenate([x2, v1], axis=-1)
        t = t + h
    return x_fake


def dpori5_solve_numpy(x0, my_model, N=100):
    dt = 1 / (N - 1)
    x_fake = x0
    t = 0
    c, a, bsol, berr = construct_dopri5_numpy(float)
    # from t=0 to t=1
    for i in range(N - 1):
        x = x_fake

        k1 = my_model(t=t, xt=x)
        k2 = my_model(t=t + c[0] * dt, xt=x + dt * a[0] * k1)
        k3 = my_model(t=t + c[1] * dt, xt=x + dt * (a[1][0] * k1 + a[1][1] * k2))
        k4 = my_model(t=t + c[2] * dt, xt=x + dt * a[2][0] * k1 + dt * a[2][1] * k2 + dt * a[2][2] * k3)
        k5 = my_model(t=t + c[3] * dt, xt=x + dt * a[3][0] * k1 + dt * a[3][1] * k2 + dt * a[3][2] * k3 + dt * a[3][3] * k4)
        k6 = my_model(t=t + c[4] * dt, xt=x + dt * a[4][0] * k1 + dt * a[4][1] * k2 + dt * a[4][2] * k3 + dt * a[4][3] * k4 + dt * a[4][4] * k5)
        k7 = my_model(t=t + c[5] * dt, xt=x + dt * a[5][0] * k1 + dt * a[5][1] * k2 + dt * a[5][2] * k3 + dt * a[5][3] * k4 + dt * a[5][4] * k5 + dt * a[5][5] * k6)
        x_fake = x + dt * (bsol[0] * k1 + bsol[1] * k2 + bsol[2] * k3 + bsol[3] * k4 + bsol[4] * k5 + bsol[5] * k6)
        t = t + dt
    return x_fake



def _residual(my_model, x, t, dt, x_sol):
    f_sol = my_model(t=t, xt=x_sol.detach())
    return torch.sum((x_sol.detach() - x - dt*f_sol)**2).detach().numpy()

def ieuler_solve(x0, my_model, N=100):
    h = 1 / (N - 1)
    x_fake = torch.tensor(x0)
    t = 0
    #opt = torch.optim.LBFGS
    max_iters = 200
    # from t=0 to t=1
    opt = torch.optim.LBFGS([x_fake], lr=1, max_iter=max_iters, max_eval=10*max_iters, tolerance_grad=1.e-12, tolerance_change=1.e-12, history_size=100, line_search_fn='strong_wolfe')

    for i in range(N - 1):
        x_sol = x_fake.clone()
        x_sol = nn.Parameter(data=x_sol)

        def closure():
            opt.zero_grad()
            residual = _residual(my_model, x_fake.detach(), t, h, x_sol)
            residual_tensor = torch.tensor(residual, requires_grad=True)
            x_sol.grad, = torch.autograd.grad(residual_tensor, x_sol, only_inputs=True, allow_unused=True)
            return residual_tensor.item()

        opt.step(closure)
        x_fake = x_sol
    return x_fake