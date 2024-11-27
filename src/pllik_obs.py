import numpy as np
from scipy.stats import multivariate_normal

def pllik_obs(theta, pg, Ind, P, G):
    """
    Calcule la log-vraisemblance observée.
    theta : dictionnaire contenant ts (seuils), Mu (moyennes), et Sigma (covariances)
    pg : vecteur des probabilités des groupes
    Ind : tableau des indices généré par idx()
    P : nombre de variables
    G : nombre de groupes
    """
    pllik = 0
    eps = np.finfo(float).eps  # eps machine

    for i in range(Ind.shape[0]):
        pr = 0
        for g in range(G):
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
            prg = up-lo
            
            pr += pg[g] * prg

        pr = max(pr, eps)  # Évite les log(0)
        pllik += Ind[i, 5] * np.log(pr)

    return pllik