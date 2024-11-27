# SIMULTANEOUS CLUSTERING AND DIMENSIONAL REDUCTION OF ORDINAL DATA 

L'article intitulé A MODEL-BASED APPROACH TO SIMULTANEOUS CLUSTERING AND
DIMENSIONAL REDUCTION OF ORDINAL DATA écrit par _Ranalli et al._(2017) présente une analyse complète de diverses méthodes de clustering et de réduction de dimensionnalité, notamment par l'identification des dimensions de bruit, pour les données ordinales.



Nous allons procédér à des démonstrations des différents résultats ainsi qu'à une implémentation des méthodes présentés.
 # Description

L'implémentation qu'on a essayé sur plusieurs jeu de donnéez se trouvent dans le notebook `SCR_ordinal_data.ipynb` qui fait à la fonction `EMTw`, fonction se trouvant dans le dossier `src` et reprenant les phases E et M de l'algorithme implémenté en `E_step.py` et `M_step.py`. Le dossier `src`regroupe les différentes fonctions utiles à la construction de l'algorithme EM.

# Requirements

```
pip3 install more_itertools
```


