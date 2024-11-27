import numpy as np
from scipy.stats import multivariate_normal

def E_step(theta, pg, Ind, P, G):
    """
    Effectue l'E-step de l'algorithme EM. Calcule les probabilités d'appartenance au groupe 
    pour chaque sujet. Les dimensions de retour sont: nrow=le nombre total de pi^xi_ci (pas égal à N),
    ncol=le nombre total de groupes G.
    
    :param theta: Liste contenant les paramètres de chaque groupe (par ligne). 
                  Comprend les moyennes, écarts-types, corrélations et les seuils invariants.
    :param pg: Les probabilités a priori de chaque groupe.
    :param Ind: Indices des observations à utiliser.
    :param P: Nombre de variables.
    :param G: Nombre de groupes.
    
    :return: La matrice de probabilités d'appartenance au groupe pour chaque sujet.
    """
    rU = Ind.shape[0]  # Nombre d'observations
    U = np.zeros((rU, G))  # Matrice pour les probabilités d'appartenance
    L = np.zeros((2, 2))  # Matrice pour les seuils

    for g in range(G):
        ds = 1 / np.sqrt(np.diag(theta['Sigma'][g]))  # Ecarts-types
        R = np.outer(ds, ds) * theta['Sigma'][g]  # Corrélation
        for i in range(rU):
            L[0, 0] = theta['ts'][Ind[i, 0]][Ind[i, 2]]
            L[0, 1] = theta['ts'][Ind[i, 1]][Ind[i, 3]]
            L[1, 0] = theta['ts'][Ind[i, 0]][Ind[i, 2] + 1]
            L[1, 1] = theta['ts'][Ind[i, 1]][Ind[i, 3] + 1]

            # Transformation selon les seuils et la moyenne pour les groupes
            Lg = (L - np.tile(theta['Mu'][g][Ind[i, 0:2]], (2, 1))) * np.outer(ds[Ind[i, 0:2]], ds[Ind[i, 0:2]])
            
            # Calcul de la densité multivariée
            up = multivariate_normal.cdf(Lg[1, :], mean=np.zeros(2), cov=R[Ind[i, 0:2], Ind[i, 0:2]])
            lo = multivariate_normal.cdf(Lg[0, :], mean=np.zeros(2), cov=R[Ind[i, 0:2], Ind[i, 0:2]])
            if np.isnan(up):
                up = 0
            if np.isnan(lo):
                lo = 0
            prg = up-lo
            uig = pg[g] * prg
            if uig < np.finfo(float).eps:
                uig = np.finfo(float).eps  # Eviter les valeurs très petites

            U[i, g] = uig
    
    return U/U.sum(axis=1, keepdims=True)
