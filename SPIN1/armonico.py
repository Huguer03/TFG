import tqdm
try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp
from pygpe.shared.utils import handle_array
import matplotlib.pyplot as plt
import pygpe.shared.vortices as vort
import pygpe.spinone as gpe

def global_normalize(psi, dx, dy, target_norm):
    # Calcular densidad total
    density = cp.abs(psi.plus_component)**2 + cp.abs(psi.zero_component)**2 + cp.abs(psi.minus_component)**2

    # Calcular integral (número total de partículas)
    current_norm = cp.sum(density) * dx * dy

    if abs(current_norm - target_norm) > 1e-16:
        factor = cp.sqrt(target_norm / current_norm)
        # Aplicar a todas las componentes
        psi.plus_component *= factor
        psi.zero_component *= factor
        psi.minus_component *= factor
    return psi

# Sortu sarea
points=(128,128)
N = points[0]*points[1]
grid_spacing=(0.5,0.5)
grid = gpe.Grid(points, grid_spacing)

# Trampa harmonikoa
omega = 2
trap_strength = omega**2

print(f"=== SISTEMA SPIN-1 CON TRAMPA ARMÓNICA ===")
print(f"ω = {omega}, trap = {trap_strength}")
print(f"Potencial: V(x,y) = 0.5 * {trap_strength} * (x² + y²)")

# Parametroak
params = {
    # Interacciones (Basadas en longitudes de dispersión del Helio-4 metaestable)
    "c0": 100.0,   # Interacción de densidad (repulsiva, mantiene el condensado unido)
    "c2": -10.0,   # Interacción de spin (NEGATIVA para Helio = Ferromagnético)
    # Campos externos
    "p": 0.0,      # Zeeman lineal (usualmente 0 a menos que haya gradiente)
    "q": 0.001,    # Zeeman cuadrático (pequeño pero positivo para estabilidad numérica)
    # Trampa armonica
    "trap": trap_strength,
    "n0": 1.0,
    "dt": -1j * 1e-2,
    "nt": 10000,
    "t": 0,
}

# Sortu uhin funtzioa
psi = gpe.SpinOneWavefunction(grid)

print("\n=== INICIALIZANDO ESTADO ===")
print("Usando estado 'Ferromagnético': ψ = (1, 0, 0)ᵀ")
psi.set_ground_state("ferromagnetic", params)
psi.add_noise("all", 0.0, 1e-2)
# Bortexak sortu
phase = vort.vortex_phase_profile(grid, 100, 1)
psi.apply_phase(phase)

psi = global_normalize(psi, grid_spacing[0], grid_spacing[1],N)

data = gpe.DataManager("spin_one_data.hdf5", "data", psi, params)

print("\n===Calculo del estado fundamental===")
psi.fft() # Preparar el k espacio
for i in tqdm.tqdm(range(params["nt"]), desc = "Oinarrizko egoera sortzen"):
    # Perform the evolution
    gpe.step_wavefunction(psi, params)
    if i % 10 == 0:  # Save data every 10 time steps
        psi = global_normalize(psi, grid_spacing[0], grid_spacing[1],N) # En el metodo de tiempo imaginario es recomendable renormalizar periodicamente
        data.save_wavefunction(psi)
    params["t"] += params["dt"]

psi = global_normalize(psi, grid_spacing[0], grid_spacing[1],N)
# Plot density and phase of zero component
fig, ax = plt.subplots(2, 2, figsize=(12, 5))

density = cp.abs(psi.plus_component)**2 + cp.abs(psi.zero_component)**2 + cp.abs(psi.minus_component)**2

n = cp.sum(density)/N
n_max = cp.max(density)
n_min = cp.min(density)

print(f"·Partikula dentzitatea n={n}\n·Partikula dentzitate maximoa nmax={n_max}\n·Partikula dentzitate minimoa nmin={n_min}")

# Graficar Densidad
im0 = ax[0,0].pcolormesh(
    handle_array(grid.x_mesh),
    handle_array(grid.y_mesh),
    handle_array(density/n_max),
    shading='auto'
)
ax[0,0].set_title(r"Densidad $|\psi_0|^2 / n$")
ax[0,0].set_ylabel(r"$y$")
fig.colorbar(im0, ax=ax[0,0])

# Graficar Fase spin up
im1 = ax[0,1].pcolormesh(
    handle_array(grid.x_mesh),
    handle_array(grid.y_mesh),
    handle_array(cp.angle(psi.plus_component)),
    cmap="jet",
    shading='auto'
)
ax[0,1].set_title(r"Fase $m_s = 1$")
fig.colorbar(im1, ax=ax[0,1])

# Graficar Fase spin cero
im2 = ax[1,0].pcolormesh(
    handle_array(grid.x_mesh),
    handle_array(grid.y_mesh),
    handle_array(cp.angle(psi.zero_component)),
    cmap="jet",
    shading='auto'
)
ax[1,0].set_title(r"Fase $m_s = 0$")
ax[1,0].set_ylabel(r"$y$")
ax[1,0].set_xlabel(r"$x$")
fig.colorbar(im2, ax=ax[1,0])

# Graficar Fase spin minus
im3 = ax[1,1].pcolormesh(
    handle_array(grid.x_mesh),
    handle_array(grid.y_mesh),
    handle_array(cp.angle(psi.minus_component)),
    cmap="jet",
    shading='auto'
)
ax[1,1].set_title(r"Fase $m_s = -1$")
ax[1,1].set_xlabel("x")
fig.colorbar(im3, ax=ax[1,1])

plt.tight_layout()
plt.show()
