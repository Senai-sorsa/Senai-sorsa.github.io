"""
EDS Maglev model — velocity + levitation ODEs
Implements the equations derived in "Modeling the Meissner effect with
differential equations" using Euler's method and RK4.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Constants (Appendix, Table 1 & 2)
# ---------------------------------------------------------------
m       = 362874        # kg, train mass (12-car config used in the essay)
# Rm and k are solved together, not chosen independently: fixing v_top =
# 167.5 m/s as the true steady-state equilibrium and requiring the model to
# reach that speed within ~4 minutes (matching the real L0's spin-up time)
# pins down both constants via the quadratic in the "Solving for the Unknown
# Constants" section below. Rm = 2.3 Ω, k = 114.223 satisfy F_motor(v_top) =
# F_drag(v_top) exactly (43092.908 N on both sides).
Lm      = 3.3            # H, motor phase inductance
Rm      = 2.3              # ohm, motor resistance (solved jointly with k)
k       = 114.22254       # N/A == V/(m/s), kt ≈ ke ≈ k (ideal-motor assumption)
C1      = 29              # N·s/m, eddy drag coefficient
rho     = 1.225           # kg/m^3, air density
Cd      = 0.25            # drag coefficient
Af      = 8.9             # m^2, frontal area
Vmax    = 20000           # V, max drive voltage
r_ramp  = 40               # s, voltage ramp-rate constant
g       = 9.807            # m/s^2

A_area  = 76.444           # m^2, effective superconductor working area
d_sc    = 0.01             # m, superconductor depth
mu0     = 4 * np.pi * 1e-7 # H/m, permeability of free space
mu      = mu0              # H/m, material permeability ≈ mu0
sigma   = 1e5               # S/m, conductivity
lam_L   = 1e-7              # m, London penetration depth
v0      = 30                 # m/s, velocity where eddy currents dominate
n_decay = 11                  # skin-effect decay exponent
tau     = 0.5                  # m, Halbach array period length

# Table 3 — Fourier / Halbach harmonics
harmonics = [
    dict(kn=12.566, An=0.000, Bn=1.200),
    dict(kn=25.133, An=0.010, Bn=0.300),
    dict(kn=37.699, An=0.005, Bn=0.100),
    dict(kn=50.265, An=0.002, Bn=0.050),
    dict(kn=62.832, An=0.001, Bn=0.0245),
]


# ---------------------------------------------------------------
# 2.1 / 2.2 — Euler's Method and RK4 (generic ODE solvers)
# ---------------------------------------------------------------
def euler_method(f, y0, t0, t_end, dt):
    """f(t, y) -> dy/dt.  y can be a scalar or numpy array (system of ODEs)."""
    ts = np.arange(t0, t_end + dt, dt)
    ys = [np.atleast_1d(y0).astype(float)]
    for t in ts[:-1]:
        y = ys[-1]
        ys.append(y + dt * np.atleast_1d(f(t, y)))
    return ts, np.array(ys)


def rk4_method(f, y0, t0, t_end, dt):
    ts = np.arange(t0, t_end + dt, dt)
    ys = [np.atleast_1d(y0).astype(float)]
    for t in ts[:-1]:
        y = ys[-1]
        k1 = dt * np.atleast_1d(f(t, y))
        k2 = dt * np.atleast_1d(f(t + dt / 2, y + k1 / 2))
        k3 = dt * np.atleast_1d(f(t + dt / 2, y + k2 / 2))
        k4 = dt * np.atleast_1d(f(t + dt, y + k3))
        ys.append(y + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
    return ts, np.array(ys)


# ---------------------------------------------------------------
# 3.2 — Linear Synchronous Motor: dv/dt
# ---------------------------------------------------------------
def velocity_ode(t, y):
    """y = [v].  Returns dv/dt from Newton's 2nd law: m dv/dt = F_motor - F_drag."""
    v = y[0]
    V = Vmax * (1 - np.exp(-t / r_ramp))                       # 3.2 voltage ramp
    I_term = (1 / Rm) * (V - k * v) * (1 - np.exp(-Rm / Lm * t))  # current, 3.2.1
    F_motor = k * I_term
    F_drag = C1 * v + 0.5 * rho * Cd * Af * v ** 2               # 3.2.2
    return [(F_motor - F_drag) / m]


# ---------------------------------------------------------------
# 3.1 — Electrodynamic suspension: F_repulsion and levitation ODE
# ---------------------------------------------------------------
def penetration_depth(v):
    """3.1.5 — penetration depth transitions from skin effect to London depth
    as speed increases."""
    if v <= 1e-6:
        return np.sqrt(tau / (mu * sigma * np.pi * 1e-6))
    if v < v0:
        return np.sqrt(tau / (mu * sigma * np.pi * v))
    return np.sqrt(lam_L ** 2 + (tau / (mu * sigma * np.pi * v)) * (v0 / v) ** n_decay)


def F_repulsion(z, v):
    """3.1.6 — combined Lorentz/London repulsion force at height z, train speed v."""
    lam = penetration_depth(v)
    decay_term = 1 - np.exp(-d_sc / lam)
    fourier_sum = sum(
        np.exp(-2 * h['kn'] * z) * (h['An'] ** 2 + h['Bn'] ** 2) for h in harmonics
    )
    return (decay_term / (2 * mu0)) * A_area * fourier_sum


def levitation_ode(t, y, v_of_t):
    """y = [z, vz].  m*z'' = F_repulsion(z,v) - m*g, clamped so the train
    can't sink through the track (z >= 0)."""
    z, vz = y
    z = max(z, 0.0)
    v_train = v_of_t(t)
    net_accel = F_repulsion(z, v_train) / m - g
    if z <= 0 and net_accel < 0:
        return [0.0, 0.0]
    return [vz, net_accel]


# ---------------------------------------------------------------
# Run: velocity model (Euler vs RK4)
# ---------------------------------------------------------------
t0, t_end, dt = 0, 300, 0.1
t_e, v_e = euler_method(velocity_ode, [0.0], t0, t_end, dt)
t_r, v_r = rk4_method(velocity_ode, [0.0], t0, t_end, dt)

plt.figure(figsize=(7, 5))
plt.plot(t_e, v_e[:, 0], 'r-', label='Euler')
plt.plot(t_r, v_r[:, 0], 'b--', label='RK4')
plt.title('Velocity of the maglev train over Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/claude/maglev/fig_velocity.png', dpi=130)
plt.close()

print('Final RK4 velocity:', v_r[-1, 0], 'm/s  (target ~167.5 m/s)')

# ---------------------------------------------------------------
# Run: levitation model using RK4 velocity as the drive profile
# ---------------------------------------------------------------
from scipy.interpolate import interp1d
v_interp = interp1d(t_r, v_r[:, 0], fill_value='extrapolate')

t_end_z, dt_z = 120, 0.1
t_ze, z_e = euler_method(lambda t, y: levitation_ode(t, y, v_interp), [0.0, 0.0], 0, t_end_z, dt_z)
t_zr, z_r = rk4_method(lambda t, y: levitation_ode(t, y, v_interp), [0.0, 0.0], 0, t_end_z, dt_z)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
axs[0].plot(t_ze, z_e[:, 0], 'r-')
axs[0].set_title("Euler estimation of height z based on time")
axs[1].plot(t_zr, z_r[:, 0], 'b--')
axs[1].set_title("RK4 estimation of height z based on time")
for ax in axs:
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('z (m)')
    ax.grid(True)
plt.tight_layout()
plt.savefig('/home/claude/maglev/fig_levitation.png', dpi=130)
plt.close()

print('Final RK4 levitation height:', z_r[-1, 0], 'm')
