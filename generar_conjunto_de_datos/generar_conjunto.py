import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy as sci
from sklearn.model_selection import train_test_split
import scipy as sci
import matplotlib.pyplot as plt
import numpy as np


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




def compute_analysis_enkf_obs(XB, PB, H, Ys, R):
  Ds = Ys - H @ XB;
  IN = R + H @ PB @ H.T;
  DXa = PB @ H.T @ np.linalg.solve(IN, Ds);
  XA = XB + DXa;
  return XA;

def get_syn_observations(y, m, Nens, R):
  white_noise = np.random.multivariate_normal(np.zeros(m), R, size=Nens).T;
  Ys = y + white_noise;
  return Ys;


def perform_forecast(x, t0, tf):
  sol = solve_ivp(lorenz96, [t0, tf], x);
  return sol.y[:,-1];

def get_random_H(p, n):
  indexes = np.arange(0,n);
  ind,_ = train_test_split(indexes, random_state=10, test_size=1-p)
  ind.sort();
  ind = ind.astype(np.int32);
  m = ind.size
  H = np.eye(n,n);
  H = H[ind,:];
  return H;



#condiciones iniciales
ic = y[:,-1];
#tamaño del ensamble
Nens = 20;
#ruido aleatorio 
white_noise = np.random.randn(Nens,nvars);
#ruido aleatorio agredado a los miembros del ensamble

e_0 = ic + 0.05 * white_noise; #perturbed ensemble
print(e_0.shape)
eb = [];
t0 = 0;
tf = 10;
t = np.arange(t0, tf, 0.005);
for e in e_0:

  sol = solve_ivp(lorenz96, [t0, tf], e, t_eval=t);
  eb.append(sol.y);
  #len=20 miembros del nesamble, con shape (8,2000) por ser 8 variables analizadas y 2000 tiempos


n = 8#40;
p = 0.8;
m = round(n*p)
Nens = Nens;
R = np.diag(0.01**2 * np.ones(m));
M = 20000;


#reference solution
xref = ic;

#initial ensemble
XB = [e[:,-1] for e in eb]
XB = np.array(XB, dtype=np.float32).T;
#XB = e_0.T
era = np.zeros(M);
erb = np.zeros(M);

matrices_xa   = []
matrices_xb   = []
matrices_pb   = []
matrices_xref   = []
evol_ensamble = []
observaciones = []


for k in range(0, M):
  print(k)
  #forecast for the reference solution (for reference)
  xref = perform_forecast(xref, 0, 0.05);

  #forecast step (background step)
  for e in range(0,Nens):
    XB[:,e] = perform_forecast(XB[:,e], 0, 0.05);

  #Forecast
  PB = np.cov(XB);

  #get the observation
  H = get_random_H(p, n);
  
  y = H @ xref.reshape(-1,1) + np.random.multivariate_normal(np.zeros(m), R).reshape(-1,1);
  Ys = get_syn_observations(y, m, Nens, R);

  #analysis step
  XA = compute_analysis_enkf_obs(XB, PB, H, Ys, R);
  
  xb = np.mean(XB, axis=1);
  xa = np.mean(XA, axis=1);
  
  matrices_xa .append(xa)
  matrices_xb .append(xb)
  matrices_pb .append(PB)
  matrices_xref.append(xref)
  observaciones.append(Ys)


  #L-2 norm of errors
  erb[k] = np.linalg.norm(xb-xref);
  era[k] = np.linalg.norm(xa-xref);

  XB = XA;

 
matrices_xa = np.stack(matrices_xa, axis=0)
matrices_xb = np.stack(matrices_xb, axis=0)
matrices_pb = np.stack(matrices_pb, axis=0)
matrices_xref   = np.stack(matrices_xref, axis=0)
#evol_ensamble = np.stack(evol_ensamble, axis=0)


np.savez("data100_miembros"+str(n),matrices_xa=matrices_xa,matrices_xb=matrices_xb,matrices_pb=matrices_pb,
            matrices_xref=matrices_xref ,#evol_ensamble=evol_ensamble,
            observaciones=observaciones)
