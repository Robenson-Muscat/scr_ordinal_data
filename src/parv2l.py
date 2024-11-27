import numpy as np

def parv2l(param, P, no_TH, G):
    """
    Convertit un vecteur de paramètres en une liste de paramètres.

    :param param: Vecteur de paramètres.
    :param P: Nombre de variables.
    :param no_TH: Liste contenant le nombre de catégories par variable.
    :param G: Nombre de groupes.
    :return: Un dictionnaire contenant Mu, Sigma, ts, et a.
    """
    id1 = P * (G - 1)
    Mu = np.reshape(param[:id1], (G - 1, P))
    Mu = np.vstack([np.zeros(P), Mu])  # Ajouter une ligne de zéros au début

    id2 = id1 + P * (P - 1) // 2
    Sigma = []
    a = []

    A = np.zeros((P, P))
    A[np.triu_indices(P, 1)] = param[id1:id2]
    np.fill_diagonal(A, 1)  # Remplir la diagonale de A avec des 1
    a.append(A)
    Sigma.append(np.dot(A.T, A))

    id3 = P * (P + 1) // 2
    for g in range(1, G):
        A[np.triu_indices(P, 0)] = param[id2 + (g - 1) * id3:id2 + g * id3]
        Sigma.append(np.dot(A.T, A))
        a.append(A)

    id4 = id2 + (G - 1) * id3
    ts = []

    for p in range(P):
        id5 = id4 + no_TH[p] - 1
        a_p = param[id4:id5]
        ts.append(np.concatenate([[-np.inf], a_p[:1], a_p[1:] + np.cumsum(np.exp(a_p[1:])), [np.inf]]))
        id4 = id5

    return {'Mu': Mu, 'Sigma': Sigma, 'ts': ts, 'a': a}