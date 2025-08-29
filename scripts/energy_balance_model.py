"""
This module provides methods for calibrating two-box, three-box, and four-box
energy balance models.

Author: Donald P. Cummins - July 2025
Modified: Chris Wells - August 2025
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

class TwoThreeFourBox:
    def __init__(self, C, κ, ϵ, F, k):
        self.C = C
        self.κ = κ
        self.ϵ = ϵ
        self.F = F
        self.k = k

        if len(C) != k or len(κ) != k:
            raise ValueError("C and κ must both be of length k.")

        self.A = A_matrix(C, κ, ϵ, k)
        self.B = B_matrix(C, k)

    def __repr__(self):
        return f'TwoThreeFourBox(C={self.C}, κ={self.κ}, ϵ={self.ϵ}, F={self.F}, k={self.k})'

    def timescales(self):
        eigvals = np.linalg.eigvals(self.A)
        tau = -1 / eigvals
        return tau

    def step(self, n, method='analytical'):
        if method == 'analytical':
            return step_analytical(self.A, self.B, self.F, n)
        elif method == 'numerical':
            return step_numerical(self.A, self.B, self.F, n, self.k)
        else:
            raise ValueError("Method must be 'analytical' or 'numerical'.")

    def observe(self, x):
        κ = self.κ
        ϵ = self.ϵ
        F = self.F
        k = self.k
        tas = x[:, 0]
        if k == 2:
            rtmt = F - κ[0]*tas + (1 - ϵ)*κ[1]*(x[:, 0] - x[:, 1])
        elif k == 3:
            rtmt = F - κ[0]*tas + (1 - ϵ)*κ[2]*(x[:, 1] - x[:, 2])
        elif k == 4:
            rtmt = F - κ[0]*tas + (1 - ϵ)*κ[3]*(x[:, 2] - x[:, 3])
        else:
            raise ValueError("Number of boxes must be 2, 3, or 4.")
        y = np.column_stack((tas, rtmt))
        return y

    def forward(self, n, method='analytical'):
        x = self.step(n, method)
        y = self.observe(x)
        return y

def A_matrix(C, κ, ϵ, k):
    if k == 2:
        A = np.zeros((2, 2))
        A[0, :] = np.array([-(κ[0] + ϵ*κ[1]), ϵ*κ[1]]) / C[0]
        A[1, :] = np.array([κ[1], -κ[1]]) / C[1]
    elif k == 3:
        A = np.zeros((3, 3))
        A[0, :] = np.array([-(κ[0] + κ[1]), κ[1], 0]) / C[0]
        A[1, :] = np.array([κ[1], -(κ[1] + ϵ*κ[2]), ϵ*κ[2]]) / C[1]
        A[2, :] = np.array([0, κ[2], -κ[2]]) / C[2]
    elif k == 4:
        A = np.zeros((4, 4))
        A[0, :] = np.array([-(κ[0] + κ[1]), κ[1], 0, 0]) / C[0]
        A[1, :] = np.array([κ[1], -(κ[1] + κ[2]), κ[2], 0]) / C[1]
        A[2, :] = np.array([0, κ[2], -(κ[2] + ϵ*κ[3]), ϵ*κ[3]]) / C[2]
        A[3, :] = np.array([0, 0, κ[3], -κ[3]]) / C[3]
    else:
        raise ValueError("Number of boxes must be 2, 3, or 4.")
    return A

def B_matrix(C, k):
    B = np.zeros((k, 1))
    B[0] = 1/C[0]
    return B

def step_numerical(A, B, F, n, k):
    def ode_system(t, y):
        gradient = A @ y.reshape(-1, 1) + B * F
        return gradient.flatten()

    y0 = np.zeros(k)
    t_span = (0, n)
    t_eval = np.arange(1, n + 1)
    sol = solve_ivp(ode_system, t_span, y0, 'RK45', t_eval)
    return sol.y.T

def step_analytical(A, B, F, n):
    # Eigendecomposition: A = V D V_inv
    eigvals, V = np.linalg.eig(A)
    # Precompute constant term
    BF_transformed = np.linalg.solve(V, (B * F)) # (k, 1)
    # Time steps (n,)
    t = np.arange(1, n + 1)
    # Compute exp(D * t) for all t: shape (n, k)
    exp_diag = np.exp(np.outer(t, eigvals)) # (n, k)
    # Compute (exp - 1) / lambda
    diag_terms = (exp_diag - 1) / eigvals # (n, k)
    # Scale V columns by diag_terms
    scaled_V = V[None, :, :] * diag_terms[:, None, :] # (n, k, k)
    # Multiply by BF in transformed space
    x = (scaled_V @ BF_transformed).squeeze(-1) # (n, k)
    return x

def pack(C, κ, ϵ, F):
    theta = np.concatenate((C, κ, [ϵ], [F]))
    log_theta = np.log(theta)
    return log_theta

def unpack(log_theta, k):
    theta = np.exp(log_theta)
    if k == 2:
        C = theta[:2]
        κ = theta[2:4]
        ϵ = theta[4]
        F = theta[5]
    elif k == 3:
        C = theta[:3]
        κ = theta[3:6]
        ϵ = theta[6]
        F = theta[7]
    elif k == 4:
        C = theta[:4]
        κ = theta[4:8]
        ϵ = theta[8]
        F = theta[9]
    else:
        raise ValueError("Number of boxes must be 2, 3, or 4.")
    return C, κ, ϵ, F

def penalty(tau, k):
    if k == 2:
        target=np.array([1., 1000.])
    elif k == 3:
        target=np.array([1., 30., 1000.])
    elif k == 4:
        target=np.array([1., 10., 100., 1000.])
    else:
        raise ValueError("Number of boxes must be 2, 3, or 4.")
    log_ratio = np.log(tau) - np.log(target)
    return np.sum(log_ratio**2)

def mse_loss(log_theta, y, k, alpha=0):
    C, κ, ϵ, F = unpack(log_theta, k)
    model = TwoThreeFourBox(C, κ, ϵ, F, k)
    y_pred = model.forward(y.shape[0], method='analytical')
    mse = np.mean((y - y_pred)**2) # data loss
    tau = model.timescales()
    penalty_value = penalty(tau, k) # penalty on timescales
    return mse + alpha*penalty_value

def fit_model(y, k, alpha=0, C_init=None, κ_init=None, ϵ_init=None, F_init=None):
    if C_init is None:
        if k == 2:
            C_init = [5., 100.]
        elif k == 3:
            C_init = [5., 20., 100.]
        elif k == 4:
            C_init = [5., 20., 80., 150.]
        else:
            raise ValueError("Number of boxes must be 2, 3, or 4.")
    if κ_init is None:
        if k == 2:
            κ_init = [1., 0.5]
        elif k == 3:
            κ_init = [1., 2., 1.]
        elif k == 4:
            κ_init = [1., 1.5, 0.75, 0.5]
        else:
            raise ValueError("Number of boxes must be 2, 3, or 4.")
    if ϵ_init is None:
        ϵ_init = 1.
    if F_init is None:
        F_init = 5.9
    
    log_theta_init = pack(C_init, κ_init, ϵ_init, F_init)
    res = minimize(mse_loss, log_theta_init, args=(y, k, alpha), method='L-BFGS-B')
    if not res.success:
        raise RuntimeError("Optimization failed: " + res.message)
    C, κ, ϵ, F = unpack(res.x, k)
    model = TwoThreeFourBox(C, κ, ϵ, F, k)
    return model, res