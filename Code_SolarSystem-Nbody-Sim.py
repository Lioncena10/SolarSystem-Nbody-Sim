import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. CONFIGURACIÓN GENERAL Y RUTAS
# =============================================================================

ruta_guardado = r"C:\Users\maced\OneDrive\Documents\Licenciatura en Física\4to semestre\Proyecto Modular I"

plt.style.use('default') # estilo académico para las gráficas

# Listas de referencia para gráficas y prints
nombres_planetas = ["Sol", "Mercurio", "Venus", "Tierra", "Marte", "Júpiter", "Saturno", "Urano", "Neptuno"]
colores_planetas = [
    '#DAA520', # Sol
    '#778899', # Mercurio 
    '#B8860B', # Venus
    '#0000CD', # Tierra 
    '#DC143C', # Marte
    '#FF8C00', # Júpiter 
    '#CD853F', # Saturno 
    '#008B8B', # Urano 
    '#00008B'  # Neptuno
]

# =============================================================================
# 2. PARÁMETROS FÍSICOS Y DE TIEMPO
# =============================================================================

G = 6.67430e-11 # constante de gravitación universal en m^3 kg^-1 s^-2

# Configuración del tiempo
años = 165
t_tot = años*365.25*24*3600  # segundos en 165 años
dt = 60*60*24  # paso de tiempo de 1 día en segundos
n_steps = int(t_tot / dt)  # número de pasos

# Masas en kg
m_1 = 1.9885e30 # sol
m_2 = 3.302e23 # mercurio
m_3 = 4.8673e24 # venus
m_4 = 6.046e24 # tierra
m_5 = 6.417e23 # marte
m_6 = 1.899e27 # júpiter
m_7 = 5.685e26 # saturno
m_8 = 8.682e25 # urano
m_9 = 1.024e26 # neptuno

# =============================================================================
# 3. CONDICIONES INICIALES (JPL HORIZONS 1800)
# =============================================================================

# Posiciones iniciales en metros
x_0 = np.array([5.228565477040853e8, -3.104484324352735e10, -9.143099489342408e10, 
                -3.313444644400436e10, -1.634767007860218e11, -3.890829932084564e9, 
                -8.501069348004460e11, -2.732804562914190e12, -3.037845069049272e12])

y_0 = np.array([-8.694713429460899e8, 3.660337033910143e10, 5.448888030418000e10, 
                1.423137635485536e11, -1.668328988149578e11, 7.669043510263376e11, 
                1.062771517621346e12, 1.459858319621716e11, -3.365950082665156e12])

z_0 = np.array([-1.006344787510397e7, 5.954914973596595e9, 6.038157181927174e9, 
                5.344758478002250e7, 6.267065568712205e8, -2.970622843843400e9, 
                1.468370435448796e10, 3.619456553609335e10, 1.392524612433219e11])

# apilar las posiciones como columnas --> aplanar la matriz para convertirla en un vector largo 1D
r_0_stacked = np.column_stack((x_0, y_0, z_0))
r_0 = r_0_stacked.flatten() # vector de posiciones iniciales

# Velocidades iniciales en metros por segundo
vx_0 = np.array([1.487301455852110e1, -4.707720387366800e4, -1.819264983313854e4, 
                 -2.947270619130661e4, 1.819416830659238e4, -1.321319929332112e4, 
                 -8.076450789155558e3, -4.192778418928231e2, 4.001171350038544e3])

vy_0 = np.array([1.821026438460768e0, -2.939581579050967e4, -3.017476002618740e4, 
                 -6.921770020926504e3, -1.497407556055323e4, 5.356831252234682e2, 
                 -6.055684827604679e3, -7.115799833639487e3, -3.610449958840960e3]) 
    
vz_0 = np.array([-4.033999989399463e-1, 1.952467466337460e3, 6.576798702595550e2, 
                 -4.656563755828458e0, -7.676824325484644e2, 2.947598229872944e2, 
                 4.273491150878284e2, -2.138434150049306e1, -1.779670064898675e1])

v_0_stacked = np.column_stack((vx_0, vy_0, vz_0))
v_0 = v_0_stacked.flatten() # vector de velocidades iniciales

f_0 = np.concatenate((r_0, v_0)) # vector de estado inicial (posiciones y velocidades)

# =============================================================================
# 4. MOTOR FÍSICO (FUNCIONES)
# =============================================================================

# función para calcular la derivada del estado actual
def derivada(t_actual, estado_actual, masas, G):

    N = len(masas)
    
    r_vector = estado_actual[:3*N] # las primeras 3N entradas son posiciones
    v_vector = estado_actual[3*N:] # las últimas 3N entradas son velocidades
    
    r_matrix = r_vector.reshape((N, 3)) # reshape a matriz N x 3
    
    x = r_matrix[:, 0]
    y = r_matrix[:, 1]
    z = r_matrix[:, 2]
    
    Rx, Rx_prime = np.meshgrid(x, x)
    Ry, Ry_prime = np.meshgrid(y, y)
    Rz, Rz_prime = np.meshgrid(z, z)
    
    Delta_Rx = Rx_prime - Rx # matriz de distancias entre planetas en x 
    Delta_Ry = Ry_prime - Ry # matriz de distancias entre planetas en y
    Delta_Rz = Rz_prime - Rz # matriz de distancias entre planetas en z 
    
    R = np.sqrt(Delta_Rx**2 + Delta_Ry**2 + Delta_Rz**2) # matriz de distancias entre planetas
    
    R[R == 0] = 1.0 # evitar división por cero
    
    Ax = G * (masas @ Delta_Rx) / R**3 # matriz de aceleraciones en x 
    Ay = G * (masas @ Delta_Ry) / R**3 # matriz de aceleraciones en y
    Az = G * (masas @ Delta_Rz) / R**3 # matriz de aceleraciones en z
    
    # axis=0 suma hacia abajo (por columna) 
    ax_tot = np.sum(Ax, axis=0) # vector de aceleraciones totales en x
    ay_tot = np.sum(Ay, axis=0) # vector de aceleraciones totales en y
    az_tot = np.sum(Az, axis=0) # vector de aceleraciones totales en z
    
    a_stacked = np.column_stack((ax_tot, ay_tot, az_tot))
    a_vector = a_stacked.flatten()
    
    derivada = np.concatenate((v_vector, a_vector))
    
    return derivada

# función para correr la simulación (Integrador RK4)
def simulación(masas):
    M = np.diag(masas)
    
    # Reiniciar condiciones iniciales
    t = 0.0  # tiempo inicial
    estado_actual = f_0.copy() # estado inicial
    historial = np.zeros((n_steps, len(f_0))) # almacenar el historial de posiciones y velocidades
    historial[0] = f_0 # el paso 0 es la condición inicial
    
    for i in range (1, n_steps):
        k1 = derivada(t, estado_actual, M, G)
        k2 = derivada(t + dt * 0.5, estado_actual + 0.5 * dt * k1, M, G)
        k3 = derivada(t + dt * 0.5, estado_actual + 0.5 * dt * k2, M, G)
        k4 = derivada(t + dt, estado_actual + dt * k3, M, G)
    
        estado_actual = estado_actual + (dt/6) * (k1 + 2 * k2 + 2 * k3 + k4)
        historial[i] = estado_actual
        
        t += dt
    
        if i % (n_steps // 10) == 0:
            print(f"Progreso: {round(i/n_steps * 100)}%")
    
    return historial

# =============================================================================
# 5. EJECUCIÓN DE LAS SIMULACIONES
# =============================================================================

# Simulación con Neptuno
print("Iniciando simulación con Neptuno...")
m_con_neptuno = np.array([m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8, m_9]) # masas con Neptuno
hist_con = simulación(m_con_neptuno) # historial de posiciones y velocidades con Neptuno

# Simulación sin Neptuno
print("Iniciando simulación sin Neptuno...")
m_sin_neptuno = np.array([m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8, 0.0]) # masas sin Neptuno
hist_sin = simulación(m_sin_neptuno) # historial de posiciones y velocidades sin Neptuno

# =============================================================================
# 6. ANÁLISIS DE DATOS
# =============================================================================

# --- 6.1 Cálculo de Desviación de Urano ---

idx_urano = 7 * 3  # índice de la columna x de Urano en el historial

# Posiciones CON Neptuno
x_urano_con = hist_con[:, idx_urano]
y_urano_con = hist_con[:, idx_urano + 1]
z_urano_con = hist_con[:, idx_urano + 2]

# Posiciones SIN Neptuno
x_urano_sin = hist_sin[:, idx_urano]
y_urano_sin = hist_sin[:, idx_urano + 1]
z_urano_sin = hist_sin[:, idx_urano + 2]

# distancia entre las posiciones de Urano con y sin Neptuno en cada paso
desviación = np.sqrt((x_urano_con - x_urano_sin)**2 +
                     (y_urano_con - y_urano_sin)**2 +
                     (z_urano_con - z_urano_sin)**2)

# --- 6.2 Movimiento relativo de Urano respecto a su Órbita sin Neptuno ---

x_relativo = x_urano_con - x_urano_sin
y_relativo = y_urano_con - y_urano_sin

# --- 6.3 Número de revoluciones de los planetas ---

def contar_vueltas(historial, idx_planeta):
    # índices planetas
    idx = idx_planeta * 3
    x = historial[:, idx]
    y = historial[:, idx + 1]

    # índice Sol
    x_sol = historial[:, 0]
    y_sol = historial[:, 1]

    # posición relativa planeta-Sol
    x_rel = x - x_sol
    y_rel = y - y_sol

    # ángulo; arctan2 \in [-pi, pi] determina el cuadrante del ángulo
    theta = np.arctan2(y_rel, x_rel)

    # unwrap detecta saltos bruscos y corrige los valores (suma o resta 2pi)
    theta_unwrapped = np.unwrap(theta)

    # contar vueltas
    delta_theta = theta_unwrapped[-1] - theta_unwrapped[0]
    vueltas = delta_theta / (2 * np.pi)

    return vueltas

print("\nNúmero de revoluciones de los planetas")
print("-" * 40)
for i in range(1,len(nombres_planetas)):
    vueltas = contar_vueltas(hist_con, i)
    print(f"{nombres_planetas[i]}: {vueltas:.6f} vueltas")
    
# --- 6.4 Semieje mayor (a) de los planetas ---    

def semieje_mayor_promedio(historial, idx_planeta):
    # índice del planeta
    idx = idx_planeta * 3

    # posiciones del planeta
    x = historial[:, idx]
    y = historial[:, idx + 1]
    z = historial[:, idx + 2]

    # posiciones del Sol
    x_sol = historial[:, 0]
    y_sol = historial[:, 1]
    z_sol = historial[:, 2]

    # posición relativa planeta–Sol
    dx = x - x_sol
    dy = y - y_sol
    dz = z - z_sol

    # distancia planeta–Sol en cada instante
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    # semieje mayor \approx promedio temporal
    a_sim = np.mean(r)

    return a_sim

print("\nSemiejes mayores (a)")
print("-" * 40)
for i in range(1, len(nombres_planetas)):
    a = semieje_mayor_promedio(hist_con, i)
    print(f"{nombres_planetas[i]:<10}: {a:.6f} m")

# --- 6.5 Cálculo de Error vs. NASA (posiciones finales) ---

r_final_sim = hist_con[-1, :27] # vector de posiciones finales según la simulación

# Posiciones finales según datos de la NASA
x_final_nasa = np.array([-5.126708696893009e8, -5.133402097015115e10, -8.091451644819625e10,
                         -2.824560809242684e10, -1.896386842894861e11, 4.196676223120275e11,
                         1.334047715925336e12, -2.609727205941357e12, -3.012512021253848e12])

y_final_nasa = np.array([-2.862449784465670e8, 1.739467391375619e10, -7.260620346100694e10, 
                         1.441747851001168e11, 1.602552108288316e11, 6.203256791684034e11,
                         -5.866293800249232e11, 8.225882449968312e11, -3.389396039538026e12])

z_final_nasa = np.array([1.486614183729426e7, 6.127724009156877e9, 3.676846629851125e9, 
                         2.605051470699906e7, 8.038952737819076e9, -1.194925393035123e10,
                         -4.273628653023371e10, 3.693138235253632e10, 1.391927738092892e11])

r_final_nasa_stacked = np.column_stack((x_final_nasa, y_final_nasa, z_final_nasa))
r_final_nasa = r_final_nasa_stacked.flatten() # vector de posiciones finales según NASA

distancia_sim = []
distancia_nasa = []
distancia_error = []

# calcular error relativo para cada planeta
for i in range(len(r_final_sim)//3):
    r_sim = np.sqrt(r_final_sim[3*i]**2 + r_final_sim[3*i + 1]**2 + r_final_sim[3*i + 2]**2)
    
    r_nasa = np.sqrt(r_final_nasa[3*i]**2 + r_final_nasa[3*i + 1]**2 + r_final_nasa[3*i + 2]**2)

    error_relativo = (np.abs(r_sim - r_nasa) / r_nasa) * 100
    
    distancia_sim.append(r_sim)
    distancia_nasa.append(r_nasa)
    distancia_error.append(error_relativo)

print("\nPosiciones finales (simulación):")
print("-" * 40)
for i in range(len(distancia_sim)):
    print(f"{nombres_planetas[i]:<10}: {distancia_sim[i]:.4f} m")

print("\nPosiciones finales (nasa):")
print("-" * 40)
for i in range(len(distancia_nasa)):
    print(f"{nombres_planetas[i]:<10}: {distancia_nasa[i]:.4f} m")

print("\nError en las posiciones finales:")
print("-" * 40)
for i in range(len(distancia_error)):
    print(f"{nombres_planetas[i]:<10}: {distancia_error[i]:.4f} %")

# =============================================================================
# 7. GRÁFICAS Y RESULTADOS VISUALES
# =============================================================================

# ---------------------------------------------------------------
# Gráfica 1: Barras de Error Relativo de las Posiciones Finales
# ---------------------------------------------------------------

plt.figure(figsize=(10, 6))

barras = plt.bar(nombres_planetas[:], distancia_error[:], color='rebeccapurple', edgecolor='black')

# poner los números encima de cada barra con 2 decimales
for barra in barras:
    altura = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2., altura, f'{altura:.4f}%', ha='center', 
             va='bottom', fontsize=10, fontweight='bold')

plt.axhline(y=1.0, color='magenta', linestyle='--', linewidth=2, label='Umbral: 1%') # umbral del 1%

plt.title('Error Relativo en las Posiciones Finales de los Planetas y el Sol (1965)', fontsize=16, fontweight='bold')
plt.ylabel('Error relativo (%)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tick_params(labelsize=10) # tamaño de los numeritos (ticks)
plt.legend(loc='upper right', fontsize='small', frameon=True, facecolor='white', edgecolor='lightgray')
plt.tight_layout()

plt.savefig(os.path.join(ruta_guardado, "error_simulacion_posición_final.png"), dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------
# Gráfica 2: Desviación de la Posición de Urano por Neptuno
# -------------------------------------------------------------

tiempo_desviación = np.linspace(1800, 1800 + años, n_steps)

plt.figure(figsize=(10, 6))

plt.plot(tiempo_desviación, desviación, color='crimson', linewidth=2)

# Valor máximo
max_desv = np.max(desviación)
id_max = np.argmax(desviación) # índice del valor máximo
año_max = tiempo_desviación[id_max]

max_label = f'Desviación máx.: {max_desv:.2e} m\nAño: {año_max:.0f}'
plt.plot(año_max, max_desv, 'ro', markersize=8, markeredgecolor='black', label=max_label)

plt.title('Desviación de la Posición de Urano inducida por Neptuno', fontsize=16, fontweight='bold')
plt.xlabel('Año', fontsize=12)
plt.ylabel('Distancia (m)', fontsize=12)
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.tick_params(labelsize=10)
plt.legend(loc='best', fontsize='small', frameon=True, facecolor='white', edgecolor='lightgray')
plt.tight_layout()

plt.savefig(os.path.join(ruta_guardado, "desviacion_urano.png"), dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------
# Gráfica 3: Sistema Solar Completo (Órbitas)
# -------------------------------------------------------------

plt.figure(figsize=(12, 12))

for i in range(len(colores_planetas)):
    idx = i * 3
    
    x = hist_con[:, idx]
    y = hist_con[:, idx + 1]
    
    grosor = 1 if i >= 7 else 0.4  # grosor de línea mayor para Urano y Neptuno
    size = 8 if i >= 5 else 1  # tamaño de marcador mayor para Urano y Neptuno
    
    # gráfica de las órbitas de los planetas
    plt.plot(x, y, color=colores_planetas[i], linewidth=grosor, label=nombres_planetas[i])
    
    # posición final (el planeta en 1965)
    plt.plot(x[-1], y[-1], 'o', color=colores_planetas[i], markersize=size)

plt.title(f'Trayectorias Orbitales del Sistema Solar (1800 - 1965)', fontsize=16, fontweight='bold')
plt.xlabel('Posición X (m)', fontsize=12)
plt.ylabel('Posición Y (m)', fontsize=12)
plt.gca().set_aspect('equal') # mantiene proporción 1:1 en los ejes (evitar distorsión); gca(): get current axis
plt.tick_params(labelsize=10)
plt.legend(loc='upper right', fontsize='small', frameon=True, facecolor='white', edgecolor='lightgray')
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig(os.path.join(ruta_guardado, "sistema_solar_completo.png"), dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------
# Gráfica 4: Órbitas de Planetas Interiores y el Sol
# -------------------------------------------------------------

plt.figure(figsize=(10, 10))

for i in range(5): 
    idx = i * 3
    
    x = hist_con[:, idx]
    y = hist_con[:, idx + 1]
    
    marker = 16 if i == 0 else 8  # tamaño de marcador mayor para el Sol
    
    # gráfica de las órbitas de los planetas interiores
    plt.plot(x, y, color=colores_planetas[i], linewidth=0.5, label=nombres_planetas[i])
    
    # posición final (el planeta en 1965)
    plt.plot(x[-1], y[-1], 'o', color=colores_planetas[i], markersize=marker)

plt.title('Órbitas de los Planetas Interiores y el Sol (1800 - 1965)', fontsize=16, fontweight='bold')
plt.xlabel('Posición X (m)', fontsize=12)
plt.ylabel('Posición Y (m)', fontsize=12)
plt.gca().set_aspect('equal')
plt.tick_params(labelsize=10)

limite = 2.8e11
plt.xlim(-limite, limite)
plt.ylim(-limite, limite)

plt.legend(loc='upper right', fontsize='small', frameon=True, facecolor='white', edgecolor='lightgray')
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig(os.path.join(ruta_guardado, "sistema_solar_interiores.png"), dpi=300, bbox_inches='tight')
plt.show()

# --------------------------------------------------------------------
# Gráfica 5: Movimiento de Urano respecto a su órbita sin Neptuno
# --------------------------------------------------------------------

plt.figure(figsize=(10, 8))

# mapa de colores (cmap) para representar el paso del tiempo
plt.scatter(x_relativo, y_relativo, c=tiempo_desviación, cmap='plasma', s=10, alpha=1.0, label='Trayectoria')
cbar = plt.colorbar() # crear la barra de color lateral 
cbar.set_label('Año', fontsize=12)

# inicio (1800)
plt.plot(x_relativo[0], y_relativo[0], 'ro', markersize=10, markeredgecolor='black', label='Inicio (1800)')
plt.text(x_relativo[0], y_relativo[0], '   1800', fontsize=10, fontweight='bold', ha='left', va='center')

# año del descubrimiento de Neptuno (1846)
idx_1846 = np.abs(tiempo_desviación - 1846).argmin() # buscamos el índice más cercano a 1846
plt.plot(x_relativo[idx_1846], y_relativo[idx_1846], 'r*', markersize=16, markeredgecolor='black',
         label='Descubrimiento (1846)')
plt.text(x_relativo[idx_1846], y_relativo[idx_1846], '1846   ', fontsize=10, fontweight='bold',
         ha='right', va='top')

# final (1965)
plt.plot(x_relativo[-1], y_relativo[-1], 'rX', markersize=10, markeredgecolor='black', label='Final (1965)')
plt.text(x_relativo[-1], y_relativo[-1], '1965   ', fontsize=10, fontweight='bold', ha='right', va='top')

plt.title('Movimiento Relativo de Urano respecto a su Órbita sin Neptuno', fontsize=16, fontweight='bold')
plt.xlabel('Desplazamiento en X (m)', fontsize=12)
plt.ylabel('Desplazamiento en Y (m)', fontsize=12)
plt.axhline(0, color='gray', linestyle=':', alpha=0.8)
plt.axvline(0, color='gray', linestyle=':', alpha=0.8)
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.legend(loc='upper left', fontsize='small', frameon=True, facecolor='white', edgecolor='lightgray')
plt.axis('equal')
plt.tight_layout()

plt.savefig(os.path.join(ruta_guardado, "mov_relativo_urano.png"), dpi=300, bbox_inches='tight')
plt.show()
