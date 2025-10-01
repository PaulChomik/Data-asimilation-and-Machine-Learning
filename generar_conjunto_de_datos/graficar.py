import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy as sci
from sklearn.model_selection import train_test_split
import scipy as sci
#import seaborn as sns
#y = [x, y, z]

#Segun la consigna se deben usar 8 variables
nvars = 8



def lorenz96(t, y):
  return [(y[(i+1) % nvars] - y[(i-2) % nvars]) * y[(i-1) % nvars] - y[i] + 8 for i in range(nvars)] 

#la seed inicial era 10
n = 12
np.random.seed();

t0 = 0;
tf = 100;
t = np.arange(t0, tf, 0.005);
print("Numero de tiempos :",t.shape)




sol = solve_ivp(lorenz96, [t0, tf], np.random.randn(nvars), t_eval=t);
y = sol.y;

import matplotlib.pyplot as plt
import numpy as np

def plot_lorenz96_variable(sol_object, variable_index):
  """
  Grafica una variable específica del modelo de Lorenz 96 a lo largo del tiempo.

  Args:
    sol_object (scipy.integrate.OdeSolution): El objeto de solución
      devuelto por solve_ivp.
    variable_index (int): El índice de la variable a graficar (de 0 a 39).
  """
  # Extraer los tiempos y las variables de la solución
  times = sol_object.t
  y_values = sol_object.y

  # Seleccionar la variable específica para graficar
  if variable_index < 0 or variable_index >= y_values.shape[0]:
    raise IndexError("El índice de la variable debe estar entre 0 y 39.")

  variable_to_plot = y_values[variable_index, :]

  #  figura y los ejes
  plt.figure(figsize=(10, 6))
  plt.plot(times, variable_to_plot)
  plt.title(f'Evolución $y_{{{variable_index}}}$ del Lorenz 96')
  plt.xlabel('Tiempo')
  plt.ylabel(f'Valor de la variable $y_{{{variable_index}}}$')
  plt.grid(True)
  plt.show()
  plt.close()


#Depues tenemos que cambiar este rango para la entrega
for i in range(0,2):
  plot_lorenz96_variable(sol, i)


def plot_ensemble_evolution(y_truth, eb, variable_index):
  """
  Grafica la evolución de una variable específica de la "verdad" y
  de todos los miembros del ensamble.

  Args:
    y_truth (np.array): La matriz de la trayectoria de verdad (n_vars x n_time).
    eb (list): La lista de matrices de trayectorias del ensamble.
    variable_index (int): El índice de la variable a graficar.
  """
  # Verificar que el índice de la variable sea válido
  if variable_index < 0 or variable_index >= y_truth.shape[0]:
    raise IndexError(f"El índice de la variable debe estar entre 0 y {y_truth.shape[0]-1}.")

  # Extraer los tiempos de la primera trayectoria del ensamble
  # (asumimos que todas tienen los mismos tiempos)
  time_steps = np.arange(y_truth.shape[1]) * 0.005 # Ajustar al paso de tiempo
  
  plt.figure(figsize=(12, 7))

  # 1. Graficar la trayectoria de verdad en rojo
  plt.plot(time_steps, y_truth[variable_index, :], label='Verdad', color='red', linewidth=2)

  # 2. Graficar cada miembro del ensamble en gris claro
  for i, ensemble_member_trajectory in enumerate(eb):
    plt.plot(time_steps, ensemble_member_trajectory[variable_index, :], color='lightgray', alpha=0.7)

  # Para que la leyenda del ensamble no se duplique 20 veces,
  # graficamos uno solo con el label
  plt.plot([], [], color='lightgray', label='Miembros del ensamble')

  # Añadir etiquetas, título y leyenda
  plt.title(f'Evolución de la variable ${variable_index}$ - Verdad vs. Ensamble')
  plt.xlabel('Tiempo')
  plt.ylabel(f'Valor de la variable ${variable_index}$')
  plt.legend()
  plt.grid(True)
  plt.show()
  plt.close()
