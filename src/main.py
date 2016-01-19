#-*- coding: utf-8 -*-

import numpy as np
import math
import random

from sklearn import datasets

def load_iris_data():
    """
    Fonction permettant de charger en mémoire les données IRIS, contenues dans scikit learn.
    """

    iris = datasets.load_iris()
    X = iris.data[:, :4]
    C = iris.target

    return (X, C)

def similarity(xi, xj, sigma=1):
    """
    Donne la similarité entre deux données.
    """

    return math.exp(-(np.linalg.norm(np.subtract(xi, xj))) / math.pow(sigma, 2))

def construct_degree_matrix(epsilon_graph):
    """
    """

    degree_matrix = np.zeros((len(epsilon_graph), len(epsilon_graph)))

    i = 0
    while i < len(epsilon_graph):
        line = epsilon_graph[i]
        degree_matrix[i,i] = np.sum(line)
        i += 1

    return degree_matrix
    """
    Construit l'epsilon-graph associé aux données de l'IRIS
    """

    return [[similarity(xi, xj) for xi in X] for xj in X]

def choose_3_points_from(X, C):
    """
    Fonction permettant de choisir 3 points aléatoires - chacun étant d'une classe
    inférieure à l'autre
    """

    # Génère des nombres aléatoires, chacun dans des intervalles permettant
    # de choisir 1 point pour chaque classe
    r_0 = random.randrange(0, 50)
    r_1 = random.randrange(50, 100)
    r_2 = random.randrange(100, 150)

    # Prendre les valeurs -> vecteur X
    c_0 = X[r_0]
    c_1 = X[r_1]
    c_2 = X[r_2]

    # print("c_0 {}".format(c_0))
    # print("c_1 {}".format(c_1))
    # print("c_2 {}".format(c_2))

    # Suppression des valeurs
    X = np.delete(X, r_0, 0)
    X = np.delete(X, r_1, 0)
    X = np.delete(X, r_2, 0)

    # Ajout des valeurs dans les 3 premières positions du nouveau vecteur X
    X = np.insert(X, 0, c_0, axis=0)
    X = np.insert(X, 1, c_1, axis=0)
    X = np.insert(X, 2, c_2, axis=0)

    # print("X {}".format(X))

    return X

def compute_y_0():

    # Initialisation du vecteur y_0 (les 3 premières du X modifié dans
    # choose_3_points_from)
    y_0 = [[1,0,0], [0,1,0], [0,0,1]]

    # Ajout des valeurs nulles
    for i in range(0, len(X) - len(y_0)):
        y_0.append([0,0,0])

    return y_0

def iterative_harmonic_algorithm(X, C):
    """
    Fonction permettant de faire tourner l'algorithme itératif harmonique, afin de prédire les labels inconnus
    """

    # Construction de l'epsilon-graph
    epsilon_graph = construct_epsilon_graph(X)
    # Choix de 3 points aléatoires -> new_X
    new_X = choose_3_points_from(X, C)
    # Construction de y_0
    y_0 = compute_y_0()

if __name__ == '__main__':
    X,C = load_iris_data()
    iterative_harmonic_algorithm(X,C)
