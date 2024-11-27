import numpy as np
from more_itertools import flatten

def parl2v(theta, P, no_TH, G):
    """
    Convertit une structure de paramètres (liste) en un vecteur.

    :param theta: Dictionnaire contenant les paramètres (Mu, Sigma, ts, a).
    :param P: Nombre de variables.
    :param no_TH: Nombre de seuils par variable (liste ou tableau).
    :param G: Nombre de groupes.
    :return: Vecteur des paramètres.
    """
    # Extraction des moyennes (sans la première ligne)
    param0 = list(flatten(theta['Mu'][1:]))  # Suppression de la première ligne

    # Ajout des éléments des matrices `a` (Cholesky)
    # Premier groupe (triangle supérieur sans la diagonale)
    A = theta['a'][0]
    param0 = np.concatenate([param0, A[np.triu_indices_from(A, k=1)]])

    # Autres groupes (triangle supérieur avec la diagonale)
    for g in range(1, G):
        A = theta['a'][g]
        param0 = np.concatenate([param0, A[np.triu_indices_from(A, k=0)]])

    # Ajout des seuils
    for p in range(P):
        ts = theta['ts'][p]
        ts_clean = ts[1:-1]  # Suppression des premières et dernières valeurs
        param0 = np.concatenate([param0, [ts_clean[0]], np.log(np.diff(ts_clean))])
        
    return param0
