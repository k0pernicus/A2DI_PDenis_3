#-*- coding: utf-8 -*-

import numpy as np
import math
import random

from sklearn import datasets

def load_iris_data():
    """
    Méthode permettant de charger en mémoire les données IRIS, contenues dans scikit learn.
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
