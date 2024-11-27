import numpy as np

from src.randmixtw import randmixtw
from src.idx import idx
from src.margX import margX
from src.pllik_obs import pllik_obs
from src.E_step import E_step
from src.parl2v import parl2v
from src.parv2l import parv2l
from src.pcllikf1 import pcllikf1
from src.M_step import M_step

def EMtw(theta, pg, mX, Ind, G, P, no_TH, tol=1e-6):
    """
    Combine les étapes E et M jusqu'à convergence.
    Le critère de convergence est donné par la différence de vraisemblance.

    :param theta: Paramètres du modèle.
    :param pg: Probabilités a priori des groupes.
    :param mX: Données marginales.
    :param Ind: Indicateurs des cellules.
    :param G: Nombre de groupes.
    :param P: Nombre de variables.
    :param tol: Tolérance pour la convergence.
    :return: Un dictionnaire avec les résultats de l'EM.
    """
    print("|-------------|-------------|-------------|-------------|")
    print("|     iter    |    imp      |     lik     |  llik-llko  |")
    print("|-------------|-------------|-------------|-------------|")
    
    it = 0
    likold = -np.inf
    dif = np.inf
    linf = np.zeros(3)
    
    while dif > tol and it < 10:
        it += 1
        # Étape E
        U = E_step(theta, pg, Ind, P, G)
        entr = -np.sum(np.dot(Ind[:, 5].T, U * np.log(U)))
        
        # Étape M
        M = M_step(theta, U, mX, Ind, P, no_TH, G, parl2v, pcllikf1, parv2l)
        theta = M['theta']
        pg = M['pg']
        lik = M['flik'] + entr + np.dot(np.dot(Ind[:, 5].T, U), np.log(pg))
        if it <= 3:
            dif = lik - likold
            linf[it - 1] = lik
            if it == 3:
                ci = (linf[2] - linf[1]) / (linf[1] - linf[0])
                linf1 = linf[1] + (linf[2] - linf[1]) / (1 - ci)
        else:
            linf = np.concatenate([linf[:2], [lik]])
            ci = (linf[2] - linf[1]) / (linf[1] - linf[0])
            linf0 = linf1
            linf1 = linf[1] + (linf[2] - linf[1]) / (1 - ci)
            dif = abs(linf1 - linf0)
        
        likold = lik
        print(f"{it:11d} | {M['improv']:11g} | {lik:11g} | {dif:11g}")
    
    print("|-------------|-------------|-------------|-------------|")
    
    out1 = {
        'lik': lik,
        'U': U,
        'pg': pg,
        'mu': theta['Mu'],
        'sigma': theta['Sigma'],
        'gamma': theta['ts']
    }
    
    return out1
