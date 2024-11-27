import pandas as pd
import numpy as np

def idx(P, mX, no_TH):
    """
    Calcule les indices des variables, catégories, et fréquences.
    P : nombre de variables
    mX : marginales bivariées (liste de DataFrames)
    no_TH : nombre de catégories par variable
    """
    k = 0
    Ind = []
    for i in range(P - 1):
        for j in range(i + 1, P):
            for ci in range(no_TH[i]):
                for cj in range(no_TH[j]):
                    fr = mX[k].values[ci , cj ]
                    # if fr > 0:
                    Ind.append((i, j, ci, cj, k, fr))
            k=k+1
    return np.array(Ind)