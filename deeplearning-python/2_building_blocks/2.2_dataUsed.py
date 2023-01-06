"""
Tensori sono contenitori di dati. Le matrici possono essere considerati Tensori di 2D ma i tensori possono avere N dimensioni. Dimensione è chiamata anche asse. Il numero di assi determina il Rank
Diversi tipi di tensori:
- Scalari (0D) -> numero singolo
- Vettori (1D)
- Matrici (2D)
- 3D Tensors (>=3D)
"""

import numpy as np

# Scalari
x = np.array(12)
print(x)
x.ndim

#Vettori
x = np.array([12,23,32,1])
print(x)
x.ndim
x.shape
#Matrici
x = np.array([[23,32,1],[23,32,1],[23,32,1],[23,32,1]]) #4 Dimension Vector
print(x)
x.ndim

#Tensor 3D
x = np.array([[[12,23,32,1],
              [12,23,32,1],
              [12,23,32,1],
              [12,23,32,1]],
              [[12,23,32,1],
              [12,23,32,1],
              [12,23,32,1],
              [12,23,32,1]],
              [[12,23,32,1],
              [12,23,32,1],
              [12,23,32,1],
              [12,23,32,1]]]) #4 Dimension Vector
print(x)
x.ndim


assert len(x.shape)
