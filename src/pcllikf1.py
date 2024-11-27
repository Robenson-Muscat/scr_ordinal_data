import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm

def pcllikf1(thetav, U, Ind, P, no_TH, G, parv2l):
    """
    Calcul de la log-vraisemblance complète par paires.

    :param thetav: Vecteur des paramètres.
    :param U: Matrice des probabilités d'appartenance (E-step).
    :param Ind: Tableau des indices d'observations.
    :param P: Nombre de variables.
    :param no_TH: Nombre de catégories par variable.
    :param G: Nombre de groupes.
    :param parv2l: Fonction convertissant le vecteur des paramètres en une structure (liste/dictionnaire).
    :return: Log-vraisemblance complète.
    """
    # Conversion du vecteur des paramètres en structure
    theta = parv2l(thetav, P, no_TH, G)
    rU = Ind.shape[0]
    pcll = 0
    l = np.zeros((rU, G))  # Matrice temporaire pour stocker les probabilités
    L = np.zeros((2, 2))  # Matrice des bornes inférieures et supérieures

    for g in range(G):
        # Calcul des paramètres de la corrélation normalisée
        ds = 1 / np.sqrt(np.diag(theta['Sigma'][g]))
        R = theta['Sigma'][g] * np.outer(ds, ds)

        for i in range(rU):
            l1 = theta['ts'][Ind[i, 0] ][Ind[i, 2] ]
            l2 = theta['ts'][Ind[i, 1] ][Ind[i, 3] ]
            u1 = theta['ts'][Ind[i, 0] ][Ind[i, 2]+1]
            u2 = theta['ts'][Ind[i, 1] ][Ind[i, 3]+1]
            lower = np.array([l1, l2])
            upper = np.array([u1, u2])
            
            mean = theta['Mu'][g][[Ind[i, 0], Ind[i, 1]]]
            cov = theta['Sigma'][g][np.ix_([Ind[i, 0], Ind[i, 1]],
                                           [Ind[i, 0], Ind[i, 1]])]
            
            up = multivariate_normal.cdf(upper, mean=mean, cov=cov)
            lo = multivariate_normal.cdf(lower, mean=mean, cov=cov)
            if np.isnan(up):
                up = 0
            if np.isnan(lo):
                lo = 0
            prig = up-lo
            # Gestion des petites valeurs
            if prig < np.finfo(float).eps:
                prig = np.finfo(float).eps

            # Mise à jour des contributions à la log-vraisemblance
            pcll += U[i, g] * Ind[i, 5] * np.log(prig)
            l[i, g] = prig

    return pcll
