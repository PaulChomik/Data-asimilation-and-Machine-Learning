import sys, os, shutil, importlib
sys.path.insert(0, '../')
#como cree un repo local de numpy tengo que pedirle a asis que ignore el repo local en /chomik/
#y que use el repo en usr/..../site packages, que son los paquetes donde esta todo en ysyry
sys.path.insert(0, '/usr/lib64/python3.11/site-packages')
#sys.path.insert(0, "/usr/local/ml/lib64/python3.11/site-packages")    #ENTORNO DE PYTHON
sys.path = [p for p in sys.path if '/home/chomik/.local/lib/python3.11/site-packages' not in p]
#sys.path = [p for p in sys.path if '/usr/local/ml/lib64/python3.11/site-packages/numpy' not in p]
#sys.path.insert(0, "/usr/lib64/python3.11/site-packages/numpy")

from arquitecturas import NN_BASE_COMPACTA
import numpy  as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from graphics import *

#############################################################################################################
#CARGAR LOS DATOS
device=torch.device("cuda:0")
data = "/home/chomik/scripts/cursos_De_posgrado/asimilacionUBA23025/tp_final/generar_conjunto_de_datos/data.npz"
data = np.load(data)

xref = torch.from_numpy(data["matrices_xref"][:20,:])
xa   = torch.from_numpy(data["matrices_xa"][:20,:])

inputs = torch.cat([xref , xa], dim=1).float().to(device) 
#print(data["matrices_pb"].shape)
#exit()
targets=torch.from_numpy(data["matrices_pb"][:20,:,:]).float().to(device) 


x_train,Y_train = inputs [:1600,...],targets[:1600,...]
x_val,y_val     = inputs [1600:,...],targets[1600:,...]
"""codigo sugerencia de manuel
https://github.com/husnejahan/DeepAR-pytorch/blob/master/model/LSTM.py"""
#############################################################################################################3
cant_neuronas = 10
nombre = "Red de funcion Linear-Lr=e-3 -optim=Adam-"+str(cant_neuronas)+"con_2_Activacion_RELU-1 Capa Ocultas"
print(" ---entrenando red con "+str(cant_neuronas)+" neuronas---")
#va a largar un dato a la vez
model = NN_BASE_COMPACTA().to(device) 
MSE   = nn.L1Loss()
criterion = torch.nn.MSELoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


epocas = 10000

error_entrenamiento = []
error_validacion    = []

for epoca in range(epocas):        
    #antes del entrenamiento hay que dejar en zero los versores
    optimizer.zero_grad()
    #PASO HACIA ADELANTE
    outputs = model(x_train)
    loss    = criterion(outputs,Y_train)
        
        
    #PASO HACIA ATRAS PARA OPTIMIZAR
    loss.backward()
    optimizer.step()
        
    error_entrenamiento.append(loss.item())
        
    with torch.no_grad():
        val_loss = 0
        outputs_val = model(x_val)
        #print (x_train)
        val_loss = criterion(outputs_val,y_val)
            
        error_validacion.append(val_loss.item())
            
        #------------GRAFICOS-------------
        if epoca % 20 ==0: 
            numero = clean_number(epoca,epocas+1)
            """
            plot_nube_funcion_losses_and_log_LOSSes("gif_modelo/"+numero+"__"+nombre,
                                    x_val.cpu().numpy(),np.transpose(outputs_val.cpu().detach().numpy()),
                                    x_val.cpu().numpy(),y_val.cpu().numpy(),
                                    #AGREGAR curva de entrenamiento y validacion
                                    np.arange(epoca+1),np.array(error_validacion),np.array(error_entrenamiento))
                                    """
        #----------FIN GRAFICOS----------
        
    if epoca % 500==0:
        print("ENTRENANDO ,EPOCA nº:",epoca)
            

print("Etrenamiento Finalizado, GUARDANDO MODELO...")
torch.save(model,f'modelos/{nombre}_model.ckpt')
makegif("gif_modelo/",nombre,delete_PNGS=True)
entrenamiento_validacion(f'modelos/{nombre}.png',range(epocas),error_validacion,
                        range(epocas),error_validacion)
 
