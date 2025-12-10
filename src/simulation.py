"""
sim_ondas_stokes.py

Simulación y animación 2D (plano z=0) de la solución de Green (Stokes) aproximada
para un impacto puntual impulsivo en un medio elástico isótropo.

Ajustes importantes:
 - rho, c_p, c_s: parámetros físicos (puedes ajustarlos).
 - F0: vector impulso (en el plano).
 - L, Nx: dominio espacial y resolución.
 - Tmax, frames: tiempo total y número de frames.
 - sigma_delta: anchura del pulso que aproxima la delta temporal.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# --------------------------
# Parámetros físicos y de simulación
# --------------------------
rho = 2000.0               # densidad efectiva [kg/m^3] (porosa: 1600-2300 typical)
# lambda_GPa = 6.0           # Lame lambda en GPa (rango sugerido: 3 - 10 GPa)
# mu_GPa     = 4.0           # Lame mu (shear modulus) en GPa (rango sugerido: 2 - 8 GPa)

# # Conversion a Pa
# lambda_pa = lambda_GPa * 1e9
# mu_pa     = mu_GPa     * 1e9


# c_p = np.sqrt((lambda_pa + 2*mu_pa)/rho)   # velocidad P (unidades espacio / tiempo)
# c_s = np.sqrt(mu_pa/rho)              # velocidad S

c_p, c_s = 2645.0, 1414.0  # velocidades P y S [m/s] (ajustar según material)
F0 = np.array([1.0, 1.0]) # impulso puntual: vector en plano (x,y)

# Dominio espacial (plano z=0)
L = 30.0      # dominio en x,y: [-L, L]
Nx = 121      # resolución espacial (Nx x Nx)

# Tiempo
Tmax = 12.0
frames = 1000

# Aproximación de delta temporal (gaussiana)
sigma_delta = 0.08

# --------------------------
# Preparación de la malla y precomputados
# --------------------------
x, y = np.linspace(-L, L, Nx), np.linspace(-L, L, Nx)
X, Y = np.meshgrid(x, y)
pos = np.stack([X, Y], axis=-1)

eps = 1e-6
R = np.sqrt(X**2 + Y**2)
R_safe = np.maximum(R, eps)
# vectores unitarios radiales (evitar división por 0)
Xhat = np.where(R[..., None] > eps, pos / R_safe[..., None], np.zeros_like(pos))
hat_outer = np.einsum('...i,...j->...ij', Xhat, Xhat)  # matriz 2x2 por punto
I2 = np.eye(2)

t_array = np.linspace(0, Tmax, frames)

# --------------------------
# Funciones auxiliares
# --------------------------
def delta_approx(t, center):
    """Aproximación gaussiana de la delta centrada en 'center' (puede vectorizarse)."""
    return np.exp(-0.5 * ((t - center) / sigma_delta) ** 2) / (sigma_delta * np.sqrt(2 * np.pi))

def u_field_at_time(t):
    """
    Evalúa u(x,y,t) según la fórmula aproximada:
    u = (1/(4πρ)) [ P-term * delta(t-r/c_p) + S-term * delta(t-r/c_s) + near-field-term ]
    - devuelve (u_x, u_y) arrays de dimensión (Ny, Nx).
    """
    # Heaviside dif. para campo cercano
    H_ps = (t - R_safe / c_s) >= 0
    H_pp = (t - R_safe / c_p) >= 0
    t_between = t * (H_ps.astype(float) - H_pp.astype(float))

    # Término P (longitudinal)
    tau_p = R_safe / c_p
    delta_p = delta_approx(t, tau_p)
    hat_dot_F = np.einsum('...i,i->...', Xhat, F0)  # escalar hat_x · F0
    termP = (hat_dot_F[..., None] * Xhat) / (c_p**2 * R_safe[..., None]) * delta_p[..., None]

    # Término S (transversal)
    tau_s = R_safe / c_s
    delta_s = delta_approx(t, tau_s)
    hatouter_dot_F = Xhat * hat_dot_F[..., None]
    F0_vec = F0[None, None, :]
    termS = (F0_vec - hatouter_dot_F) / (c_s**2 * R_safe[..., None]) * delta_s[..., None]

    # Campo cercano (aprox.)
    H_tensor_factor = 3.0 * hat_outer - I2
    HdotF = np.einsum('...ij,j->...i', H_tensor_factor, F0)
    termNear = (1.0 / (R_safe**3))[..., None] * (t_between[..., None] * HdotF)

    prefactor = 1.0 / (4.0 * np.pi * rho)
    u = prefactor * (termP + termS + termNear)

    # Regularizar centro para evitar valores excesivos
    # center_mask = (R < 1e-3)
    # if np.any(center_mask):
    #     u[center_mask] = 0.0

    return u[..., 0], u[..., 1]

# --------------------------
# Crear figura y animación
# --------------------------
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((Nx, Nx)), origin='lower', extent=[-L, L, -L, L], vmin=0, vmax=1.0)
ax.set_xlabel('x') ; ax.set_ylabel('y')
ax.set_title('Magnitud del desplazamiento |u(x,t)| (2D, z=0)')

# Quiver: subsample para no saturar la figura
step = max(1, Nx // 20)
Q = ax.quiver(X[::step, ::step], Y[::step, ::step],
              np.zeros_like(X[::step, ::step]), np.zeros_like(Y[::step, ::step]),
              scale=50)

def init():
    im.set_data(np.zeros((Nx, Nx)))
    Q.set_UVC(np.zeros_like(X[::step, ::step]), np.zeros_like(Y[::step, ::step]))
    return (im, Q)

def animate(i):
    t = t_array[i]
    ux, uy = u_field_at_time(t)
    mag = np.sqrt(ux**2 + uy**2)
    # Escala dinámica para la visualización (evita saturación por picos)
    vmax = np.percentile(mag, 99.5)
    if vmax <= 0: vmax = 1e-12
    im.set_data(mag)
    im.set_clim(0, vmax)
    U = ux[::step, ::step]
    V = uy[::step, ::step]
    Q.set_UVC(U, V)
    ax.set_title(f'Magnitud |u| en t = {t:.2f}')
    return (im, Q)

anim = animation.FuncAnimation(fig, animate, frames=frames, init_func=init, blit=True, interval=200, repeat=True)
plt.show()