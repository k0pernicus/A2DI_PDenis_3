# A2DI_PDenis_3

# Auteurs

*	Quentin Baert (quentin[dot]baert[at]etudiant[dot]univ-lille1[dot]fr)
*	Antonin Carette (antonin[dot]carette[at]etudiant[dot]univ-lille1[dot]fr)

# Instructions

*	Télécharger les données IRIS.
*	Construire un epsilon-graph des similarités (en utilisant s(x_i, x_j) = exp( -(||x_i - x_j||^2) / theta^2)).
*	Choisir 3 points au hasard (un point par classe), et...
	*	Appliquer l'algorithme itératif harmonique pour prédire les labels inconnus.
	*	Mesurer l'erreur sur les labels inconnus. Justifier la mesure choisie.
	*	Que se passe-t-il lorsque l'on augmente le nombre de labels connus?
*	En quoi le choix du epsilon influence l'algorithme?
*	Boire un coca, et se mater Mr Robot (encore).

# Utilisation

```
python2.7 main.py <k> <epsilon> <sigma>
```

## Exemple

```
> python2.7 main.py 15 0.5 1.0
k : 15
epsilon : 0.5
sigma : 1.0
Class 0 : 7 learned
Class 1 : 3 learned
Class 2 : 5 learned
Nb false : 3
	Class 0 : 0 errors
	Class 1 : 2 errors
	Class 2 : 1 errors
```
