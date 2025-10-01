
#################################BORRAR O COMENTAR SI NO SOS PAUL##################################
import sys
#como cree un repo local de numpy tengo que pedirle a asis que ignore el repo local en /chomik/
#y que use el repo en usr/..../site packages, que son los paquetes donde esta todo en ysyry
sys.path.insert(0, '/usr/lib64/python3.11/site-packages')
sys.path = [p for p in sys.path if '/home/chomik/.local/lib/python3.11/site-packages' not in p]

#################################FIN de BORRAR O COMENTAR SI NO SOS PAUL##################################
import scipy as sci
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. PARÁMETROS CLAVE DEL EXPERIMENTO ---
D_VARS = 8 # Número de variables (N=8)
DT_INTEGRACION = 0.005 # Paso de integración (dt)
DT_OBS = 0.05 # Intervalo de asimilación (basado en tu bucle original)
N_ENS_REF = 100 # Tamaño del ensamble de referencia (Gold Standard)

# Tiempos de la Simulación
t0_sim = 0.0
tf_sim = 500.0 # Ajustado a 500.0 para 100,000 pasos (500 / 0.005 = 100,000)
T_SPINUP = 10.0 # Período de calentamiento
T_SAMPLE = tf_sim - T_SPINUP

# Parámetros de la Observación
P_OBS = 0.8 # Porcentaje de variables observadas (80%)
n = D_VARS
m = round(n * P_OBS) # Número de observaciones (m = 6)
R = np.diag(0.01**2 * np.ones(m)) # Matriz de covarianza de observación (R)
H_INDICES = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32) # Observamos las primeras 6 variables
H = np.eye(n,n)[H_INDICES, :] # Matriz H fija

# Número total de ciclos de asimilación después del spin-up
M_CYCLES = int(T_SAMPLE / DT_OBS)

# --- 2. DEFINICIÓN DEL MODELO ---
def lorenz96(t, y):
    D = len(y) 
    F = 8.0 # Forzamiento estándar
    dydt = np.zeros(D)
    for i in range(D):
        dydt[i] = (y[(i + 1) % D] - y[(i - 2) % D]) * y[(i - 1) % D] - y[i] + F
    return dydt

# Función para realizar el pronóstico (un paso de asimilación)
def perform_forecast(x, dt):
    sol = solve_ivp(lorenz96, [0, dt], x, t_eval=[dt]);
    return sol.y[:,-1];

# --- 3. GENERACIÓN DE LA TRAYECTORIA VERDADERA (X_TRUE) ---

# Condición inicial: estado cerca del forzamiento
y0_true = 8 * np.ones(D_VARS)
y0_true[0] += 0.001 # Pequeña perturbación
t_simulacion = np.arange(t0_sim, tf_sim + DT_INTEGRACION, DT_INTEGRACION)

# Integración del estado verdadero (Nature Run)
sol_true = solve_ivp(lorenz96, [t0_sim, tf_sim], y0_true, t_eval=t_simulacion);
x_true_all = sol_true.y.T # (Pasos de tiempo, D_VARS)

# Extraemos los estados verdaderos solo en los tiempos de asimilación (el target para el error)
t_asimilacion = np.arange(T_SPINUP, tf_sim + DT_OBS, DT_OBS)
idx_asimilacion = [np.argmin(np.abs(sol_true.t - t_obs)) for t_obs in t_asimilacion]
X_TRUE_ASIM = x_true_all[idx_asimilacion]
print(f"Número de ciclos de asimilación: {len(X_TRUE_ASIM) - 1}") # M_CYCLES

# --- 4. PREPARACIÓN DEL ENKF DE REFERENCIA ---

def get_syn_observations(y_true, R, Nens):
    # m es el número de observaciones (6 en este caso)
    m = R.shape[0] 
    
    # 1. Generar un ensamble de vectores de ruido (m, Nens)
    white_noise = np.random.multivariate_normal(np.zeros(m), R, size=Nens).T 
    
    # 2. Reformar el vector de observación (y_true) de (6,) a (6, 1) 
    y_true_reshaped = y_true.reshape(-1, 1) 
    
    # 3. Ys es el ensamble de observaciones (6, 100)
    Ys = y_true_reshaped + white_noise 
    return Ys

def compute_analysis_enkf_obs(XB, PB, H, Ys, R):
    XB_mean = np.mean(XB, axis=1) # media del ensamble (forecast mean)
    Ds = Ys - H @ XB # Innovación de cada miembro

    # Cálculo de la ganancia de Kalman con PB (matriz de covarianza de background)
    IN = R + H @ PB @ H.T
    K = PB @ H.T @ np.linalg.inv(IN)

    # Análisis de cada miembro
    XA = XB + K @ Ds
    return XA

# --- 5. INICIALIZACIÓN DE LA CORRIDA DE REFERENCIA ---

# Estado inicial del ensamble (tomado al final del spin-up)
ic_ref = X_TRUE_ASIM[0, :]
white_noise = np.random.randn(N_ENS_REF, n)
e_0 = ic_ref + 0.05 * white_noise # Ensamble inicial perturbado
XB = e_0.T # (nvars, Nens)

# Inicializamos las listas para guardar los datos de entrenamiento CON LOS 3 PROXIES

training_data = {
    'X_PREV_A': [],        # x_{k-1}^{a} (análisis medio del paso anterior)
    'X_FORECAST_B': [],    # x_{k}^{f} (pronóstico medio actual)
    
    # LOS 3 PROXIES DEL PAPER:
    'TARGET_COV_MMA': [],  # Proxy MMA: (x_forecast_b - x_analysis_mean) @ (x_forecast_b - x_analysis_mean).T
    'TARGET_COV_MRA': [],  # Proxy MRA: (x_forecast_b - x_analysis_random_member) @ (x_forecast_b - x_analysis_random_member).T  
    'TARGET_COV_MNT': [],  # Proxy MNT: (x_forecast_b - x_true) @ (x_forecast_b - x_true).T
    
    'PB_ENS_REF': [],      # Covarianza del ensamble de referencia (para comparación)
    'XA_ENSEMBLE': []      # Guardamos el ensamble completo de análisis para MRA
}

# --- 6. BUCLE PRINCIPAL DE ASIMILACIÓN (GENERACIÓN DE DATOS) ---

# Iteramos sobre los ciclos de asimilación después del spin-up
for k in range(1, M_CYCLES):
    # Estado verdadero en el tiempo k
    x_true_k = X_TRUE_ASIM[k, :]
    x_prev_a = np.mean(XB, axis=1) # El análisis medio previo (x_{k-1}^a)

    # 1. PASO DE PRONÓSTICO (FORECAST STEP)
    for e in range(N_ENS_REF):
        XB[:, e] = perform_forecast(XB[:, e], DT_OBS)
    
    # 2. CÁLCULO DE LA COVARIANZA Y MEDIA DEL FORECAST
    PB = np.cov(XB)
    xb_mean = np.mean(XB, axis=1) # Forecast mean (x_k^f o x_b)
    
    # 3. GENERACIÓN DE OBSERVACIÓN
    y_true_obs = H @ x_true_k # Observación ideal
    y_obs = y_true_obs + np.random.multivariate_normal(np.zeros(m), R) # Observación ruidosa
    Ys = get_syn_observations(y_obs, R, N_ENS_REF) # Ensamble de Observaciones

    # 4. PASO DE ANÁLISIS (ANALYSIS STEP)
    XA = compute_analysis_enkf_obs(XB, PB, H, Ys, R)
    xa_mean = np.mean(XA, axis=1) # Analysis mean (x_k^a)

    # 5. CÁLCULO DE LOS 3 PROXIES (MMA, MRA, MNT)
    
    # Proxy MMA: Mean Forecast - Mean Analysis
    epsilon_MMA = xb_mean - xa_mean
    TARGET_COV_MMA = np.outer(epsilon_MMA, epsilon_MMA)
    
    # Proxy MRA: Mean Forecast - Random Analysis member
    random_member_idx = np.random.randint(0, N_ENS_REF)
    xa_random = XA[:, random_member_idx]
    epsilon_MRA = xb_mean - xa_random
    TARGET_COV_MRA = np.outer(epsilon_MRA, epsilon_MRA)
    
    # Proxy MNT: Mean Forecast - Nature (true state)
    epsilon_MNT = xb_mean - x_true_k
    TARGET_COV_MNT = np.outer(epsilon_MNT, epsilon_MNT)
    
    # 6. GUARDAR DATOS DE ENTRENAMIENTO
    training_data['X_PREV_A'].append(x_prev_a)
    training_data['X_FORECAST_B'].append(xb_mean)
    training_data['TARGET_COV_MMA'].append(TARGET_COV_MMA)
    training_data['TARGET_COV_MRA'].append(TARGET_COV_MRA)
    training_data['TARGET_COV_MNT'].append(TARGET_COV_MNT)
    training_data['PB_ENS_REF'].append(PB)
    training_data['XA_ENSEMBLE'].append(XA.copy())  # Guardar ensamble completo para posibles usos futuros
    
    # 7. ITERACIÓN
    XB = XA # El análisis se convierte en el initial condition del próximo forecast

    # Progress indicator
    if k % 100 == 0:
        print(f"Completado ciclo {k}/{M_CYCLES}")

print("\n--- Generación de Datos Completa ---")
print(f"Datos de entrenamiento guardados: {len(training_data['X_FORECAST_B'])} ciclos.")

# --- 7. ALMACENAMIENTO DE DATOS ---
# Convertir a arrays para guardado eficiente
for key in training_data:
    training_data[key] = np.stack(training_data[key], axis=0)

# Guardar todos los datos con los 3 proxies
np.savez(
    "data_N8_ENS100_3proxies.npz",
    X_PREV_A=training_data['X_PREV_A'],
    X_FORECAST_B=training_data['X_FORECAST_B'],
    TARGET_COV_MMA=training_data['TARGET_COV_MMA'],
    TARGET_COV_MRA=training_data['TARGET_COV_MRA'], 
    TARGET_COV_MNT=training_data['TARGET_COV_MNT'],
    PB_ENS_REF=training_data['PB_ENS_REF'],
    XA_ENSEMBLE=training_data['XA_ENSEMBLE']
)

print("\nArchivo 'data_N8_ENS100_3proxies.npz' creado con los 3 proxies de entrenamiento.")
print("Proxies incluidos: MMA, MRA, MNT")
