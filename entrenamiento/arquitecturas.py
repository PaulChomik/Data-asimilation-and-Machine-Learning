import numpy  as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
#DEFINIR LA RED

class NN_BASE(nn.Module):
    def __init__(self,Cant_Neuronas=10):
        super(NeuralNetwork, self).__init__()
        #VApa de una neurona con 10 datos de entrada
        self.Capa_Entrada = nn.Linear(16  ,Cant_Neuronas)
        self.Capa_oculta  = nn.Linear(Cant_Neuronas ,Cant_Neuronas)
        self.capa_Salida  = nn.Linear(Cant_Neuronas ,64)
        self.relu = nn.ReLU() 
        
    def forward(self,x):
        x    =  self.relu(self.Capa_Entrada(x))
        #esta capa oculta se puede comentar para no ejecutarse
        x    = self.relu(self.Capa_oculta(x))
        #x    = self.relu(self.Capa_oculta(x))
        x    = self.capa_Salida(x)
        return x 
class NN_BASE_COMPACTA(nn.Module):
    def __init__(self, Cant_Neuronas=10):
        super(NN_BASE_COMPACTA, self).__init__()
        self.Capa_Entrada = nn.Linear(16, Cant_Neuronas)
        self.Capa_oculta = nn.Linear(Cant_Neuronas, Cant_Neuronas)
        self.capa_Salida = nn.Linear(Cant_Neuronas, 36)
        self.relu = nn.ReLU() 
        
    def forward(self, x):
        x = self.relu(self.Capa_Entrada(x))
        x = self.relu(self.Capa_oculta(x))
        x = self.capa_Salida(x)
        
        # Crear matriz simétrica directamente
        batch_size = x.shape[0]
        matriz_simetrica = torch.zeros(batch_size, 8, 8, device=x.device)
        
        # Llenar la parte triangular inferior
        indices = torch.tril_indices(8, 8)
        matriz_simetrica[:, indices[0], indices[1]] = x
        
        # Copiar a la parte triangular superior (excluyendo diagonal)
        matriz_simetrica = matriz_simetrica + matriz_simetrica.transpose(1, 2)
        diagonal = torch.diag_embed(torch.diagonal(matriz_simetrica, dim1=1, dim2=2) / 2)
        matriz_simetrica = matriz_simetrica - diagonal
        
        return matriz_simetrica