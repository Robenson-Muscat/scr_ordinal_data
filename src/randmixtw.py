import numpy as np
from scipy.stats import multivariate_normal

def randmixtw(N, Mu, Sigma, pg, no_TH):
    """
    Génération aléatoire d'un mélange de distributions gaussiennes multivariées.
    N : taille de l'échantillon
    Mu : liste de vecteurs de moyennes
    Sigma : liste de matrices de covariance
    pg : vecteur de probabilités associé à chaque groupe
    no_TH : nombre de seuils par dimension
    """
    G = len(pg)
    P = Sigma[0].shape[0]
    
    # Étape 1 : Échantillonnage des groupes
    n_samp = np.random.choice(np.arange(1, G + 1), size=N, p=pg)
    n_pg = np.array([np.sum(n_samp == g) for g in range(1, G + 1)])
    cn_pg = np.cumsum(np.hstack(([0], n_pg)))
    
    # Étape 2 : Génération des échantillons
    X = np.zeros((N, P))
    for g in range(G):
        id_start = cn_pg[g]
        id_end = cn_pg[g + 1]
        X[id_start:id_end, :] = multivariate_normal.rvs(mean=Mu[g], cov=Sigma[g], size=n_pg[g])
    
    # Étape 3 : Discrétisation des données
    ts = []
    for p in range(P):
        st = (np.max(X[:, p]) - np.min(X[:, p])) / no_TH[p]
        b = np.linspace(np.min(X[:, p]), np.max(X[:, p]), no_TH[p] + 1)
        b[0], b[-1] = -100000, 100000
        ts.append(b)
        X[:, p] = np.digitize(X[:, p], bins=b)
    
    # Sortie
    tcl = np.concatenate([np.full(n_pg[g], g + 1) for g in range(G)])
    return {"X": X, "tcl": tcl, "ts": ts}