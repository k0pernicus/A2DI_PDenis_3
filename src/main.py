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

def construct_epsilon_graph(X):
    """
    Construit l'epsilon-graph associé aux données de l'IRIS
    """

    return [[similarity(xi, xj) for xi in X] for xj in X]

def choose_3_points_from(X, C):
    """
    Fonction permettant de choisir 3 points aléatoires - chacun étant d'une classe
    inférieure à l'autre
    """

    r_0 = random.randrange(0, 50)
    r_1 = random.randrange(50, 100)
    r_2 = random.randrange(100, 150)

    c_0 = (X[r_0], C[r_0])
    c_1 = (X[r_1], C[r_1])
    c_2 = (X[r_2], C[r_2])

    return (c_0, c_1, c_2)
