import numpy as np
import os
import math
import torch
from torch.utils import data
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image

def ploteo_funcion_nube_puntos(image_name_to_print,x,y,x_curva,y_curva):

    # Crear el gráfico
    plt.figure(figsize=(5, 5))  # Tamaño de la figura

    # Graficar la nube de puntos en rojo
    plt.scatter(x, y, c='red', label='Nube de Puntos')
    lab='FUNCION ORIGINAL'
    # Graficar la curva en azul
    plt.plot(x_curva, y_curva, 'b-', label=lab)

    # Etiquetas de los ejes
    plt.xlabel('X')
    plt.ylabel('Y')

    # Leyenda
    plt.legend()

    # Título
    plt.title(image_name_to_print)

    # Mostrar el gráfico
    plt.grid(True)
    plt.savefig(image_name_to_print+'.png')
    plt.close('all')
    
    
def plot_nube_funcion_losses_and_log_LOSSes(image_name_to_print,x,y,x_curva,y_curva,
                                            iterations,loss_train,loss_val):
    #esta funcion plotea :
    #-una imagen de la nube de pntos predicha por la red con la curva objetivo:
    #-las curvas de entrenamiento y validacion en escala natural
    #-curvas de etrenamiento y validacion en escala logaritmica base 10
    
    #creal el objeto figura
    fig = plt.figure(figsize=(7,5 ))
    #crear los objetos de cada panel dentro de la grilla
    ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2,colspan=2,)
    ax2 = plt.subplot2grid((2, 3), (0, 2), colspan=1)
    ax3 = plt.subplot2grid((2, 3), (1, 2), colspan=1)
    
    # Crear el gráfico de curva objetivo y prediccion de la red

    # Graficar la nube de puntos en rojo
    ax1.scatter(x, y, c='red', label='Nube de Puntos')
    # Graficar la curva en azul
    ax1.plot(x_curva, y_curva, 'b-', label='FUNCION ORIGINAL')
    # Etiquetas de los ejes
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    # Leyenda
    ax1.legend(loc="upper right")

    

    # Graficar loss _val y tain val en escala natural
    # 
    ax2.plot(iterations, loss_train, 'b-', label='Entrenamiento')
    ax2.plot(iterations, loss_val  , 'r-', label='Validacion')
    # Etiquetas de los ejes
    ax2.set_xlabel('epocas')
    ax2.set_ylabel('Error' )
    # Leyenda
    ax2.legend(loc="upper right")
    
    
     # Graficar loss _val y tain val en escala logatimica
    # 
    ax3.plot(iterations, loss_train, 'b-', label='Entrenamiento')
    ax3.plot(iterations, loss_val  , 'r-', label='Validacion')
    # Etiquetas de los ejes
    ax3.set_xlabel('epocas escala logaritmica')
    ax3.set_ylabel('Error escala logaritmica' )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    # Leyenda
    ax3.legend(loc="upper right")
    
    # Título
    fig.suptitle(image_name_to_print)

    # Mostrar el gráfico
    fig.savefig(image_name_to_print+'.png')
    plt.close('all')

def entrenamiento_validacion(image_name_to_print,x_val,y_val,x_train,y_train):

    # Crear el gráfico
    plt.figure(figsize=(6, 6))  # Tamaño de la figura
    
    lab='FUNCION ORIGINAL'
    # Graficar la curva en azul
    plt.plot(x_train, y_train, 'b-', label="ENTRENAMIENTO")
    plt.plot(x_val  , y_val  , 'r-', label="VALIDACION")
    # Etiquetas de los ejes
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR')

    # Leyenda
    #plt.legend()

    # Título
    plt.title('CURVAS DE ENTRENAMIENTO Y VALIDACION')

    # Mostrar el gráfico
    plt.grid(True)
    plt.savefig(image_name_to_print+'.png')
    plt.close('all')

def clean_number(number,max_number):
    '''esta funcion se usa para crear numeros ordenados para
    luego generar gif con pil.
    pil solo reconoce orden alfabetico, asi por ejemplo
    si le damos un path para crear un gif, los frames corresponden a su orden
    alfabetico:
    1.png->1er frame;   2.png->2do frame... y asi sucesivamente
    el problema suger con numeros de mas de 2 o 3 digitos, el si hubiera los siguientes archivos:
    1.png ;2.png ;3.png ;10.png; 11.png ;004.png
    el orden seria:
    004.png->1er frame ;1.png->2do frame ;10.png->3er frame ;11.png->4to frame  ;2.png->5to frame ;3.png 6to frame
    para corregir esto deben ponerse ceros si gorresponde:
    001.png ;002.png ;003.png ;004.png ;010.png ;011.png
    '''
    #podemos saber la cantidad de digitos haciendo el log en base  max_number de 10
    max_digits   = math.ceil (math.log(max_number,10))
    if number!=0:
        my_digits    = math.floor(math.log(number,10))
    else:
        my_digits    =1
    extra_digits = max_digits-my_digits
    zeros=""
    for i in range(0,extra_digits):
        zeros=zeros+"0"
    return (zeros+str(number))
    
#TEST
#print(clean_number(200,10000))
#this returns 00200
def makegif(path,name,delete_PNGS=False):
    # Especifica la carpeta que contiene las imágenes .png
    carpeta_imagenes = path

    # Lista de nombres de archivo de imágenes en la carpeta
    nombres_imagenes = [archivo for archivo in os.listdir(carpeta_imagenes) if archivo.endswith('.png')]

    # Ordena los nombres de archivo para asegurarse de que están en el orden correcto
    nombres_imagenes.sort()

    # Crea una lista para almacenar los objetos de imagen
    imagenes = []

    # Carga cada imagen y la agrega a la lista
    for nombre_imagen in nombres_imagenes:
        ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)
        imagen = Image.open(ruta_imagen)
        imagenes.append(imagen)

    # Guarda las imágenes como un archivo GIF
    imagenes[0].save(name+'.gif', save_all=True, append_images=imagenes[1:], duration=60, loop=0,)# background=(204, 255, 153))
    if delete_PNGS:
        nombres_imagenes = [archivo for archivo in os.listdir(carpeta_imagenes) if archivo.endswith('.png')]
        for file in nombres_imagenes:
            try:
                os.remove(path+file)
            except FileNotFoundError:
                print(f"{file} no se encontró.")
            except PermissionError:
                print(f"No se tienen permisos para eliminar {file}.")
            except Exception as e:
                print(f"Ocurrió un error al intentar eliminar {file}: {e}")
#TEST
#makegif("/home/chomik/scripts/data_sima_simulations/","prueba")



