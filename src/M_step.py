import numpy as np
from scipy.optimize import minimize

def M_step(theta0, U, mX, Ind, P, no_TH, G, parl2v, pcllikf1, parv2l):
    """
    Effectue l'étape M de l'algorithme EM :
    1. Mise à jour des probabilités a priori des groupes (pg).
    2. Mise à jour des paramètres theta en maximisant la log-vraisemblance.

    :param theta0: Paramètres initiaux.
    :param U: Matrice des probabilités d'appartenance aux groupes.
    :param mX: Non utilisé dans cette version.
    :param Ind: Indices des observations.
    :param P: Nombre de variables.
    :param no_TH: Nombre de seuils par variable.
    :param G: Nombre de groupes.
    :param parl2v: Fonction pour transformer les paramètres en un vecteur.
    :param pcllikf1: Fonction de log-vraisemblance à maximiser.
    :param parv2l: Fonction pour reconvertir un vecteur de paramètres en une liste structurée.

    :return: Un dictionnaire contenant :
        - 'theta': Les nouveaux paramètres optimisés.
        - 'pg': Les nouvelles probabilités a priori des groupes.
        - 'flik': La log-vraisemblance finale.
        - 'improv': L'amélioration de la log-vraisemblance.
    """
    # Mise à jour de pg
    pg = np.dot(Ind[:, 5].T, U) / np.sum(Ind[:, 5])

    # Transformation des paramètres initiaux en vecteur
    thetav0 = parl2v(theta0, P, no_TH, G)
    npar = len(thetav0)

    # Calcul de la log-vraisemblance initiale
    oldvalue = pcllikf1(thetav0, U, Ind, P, no_TH, G, parv2l)

    ctrl = 0
    while ctrl == 0:
        ctrl = 1
        # Optimisation avec L-BFGS-B
        result = minimize(
            fun=lambda params: -pcllikf1(params, U, Ind, P, no_TH, G, parv2l),  # Maximiser => Minimiser l'opposé
            x0=thetav0,
            method="L-BFGS-B",
            options={"maxiter": max(1, round(npar / 4)), "disp": False}
        )
        
        # Vérification de l'amélioration de la log-vraisemblance
        if (result.fun - oldvalue) < 0:
            ctrl = 0
            thetav0 = result.x

    # Mise à jour des paramètres
    theta = parv2l(result.x, P, no_TH, G)

    # Résultats finaux
    out = {
        "theta": theta,
        "pg": pg,
        "flik": -result.fun,  # Restaurer la valeur de la log-vraisemblance originale
        "improv": -result.fun - oldvalue
    }
    return out
