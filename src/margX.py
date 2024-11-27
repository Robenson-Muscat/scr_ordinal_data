import pandas as pd

def margX(X, no_TH):
    """
    Calcule toutes les marginales bivariées.
    X : matrice des données (N, P)
    no_TH : nombre de catégories par variable
    """
    P = X.shape[1]
    mX = []
    for p in range(P - 1):
        x = pd.Categorical(X[:, p], categories=range(1, no_TH[p] + 1))
        for r in range(p + 1, P):
            y = pd.Categorical(X[:, r], categories=range(1, no_TH[r] + 1))
            mX.append(pd.crosstab(x, y))
    return mX