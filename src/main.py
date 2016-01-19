#-*- coding: utf-8 -*-

import numpy as np
import math
import random
import sys

from sklearn import datasets

global r_0
global r_1
global r_2

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
        count = 0
        for j in range(0, len(line)):
            if line[j] != 0:
                count+=1
        degree_matrix[i,i] = count
        i += 1

    return degree_matrix

def construct_epsilon_graph(X, epsilon = 0.5):
    """
    Construit l'epsilon-graph associé aux données de l'IRIS
    """

    return [[(similarity(xi, xj) if (similarity(xi, xj) > epsilon) else 0) for xi in X] for xj in X]

def choose_3_points_from(X, C):
    """
    Fonction permettant de choisir 3 points aléatoires - chacun étant d'une classe
    inférieure à l'autre
    """

    global r_0
    global r_1
    global r_2

    # Génère des nombres aléatoires, chacun dans des intervalles permettant
    # de choisir 1 point pour chaque classe
    r_0 = random.randrange(0, 50)
    r_1 = random.randrange(50, 100)
    r_2 = random.randrange(100, 150)

def compute_y_0():

    global r_0
    global r_1
    global r_2

    # Initialisation du vecteur y_0 (les 3 premières du X modifié dans
    # choose_3_points_from)
    y_0 = np.zeros((150, 3))

    y_0[r_0] = [1,0,0]
    y_0[r_1] = [0,1,0]
    y_0[r_2] = [0,0,1]

    return y_0

def iterate_over_y(epsilon_graph, degree_matrix, y_i):
    """
    Fonction permettant d'itérer sur le vecteur y (jusqu'à convergence)
    """

    return np.dot(np.dot(np.linalg.inv(degree_matrix), epsilon_graph), y_i)

def iterative_harmonic_algorithm(X, C):
    """
    Fonction permettant de faire tourner l'algorithme itératif harmonique, afin de prédire les labels inconnus
    """

    global r_0
    global r_1
    global r_2

    # Construction de l'epsilon-graph
    epsilon_graph = construct_epsilon_graph(X)
    # Calcul de la matrice des degrés
    # Choix de 3 points aléatoires -> new_X
    choose_3_points_from(X, C)
    degree_matrix = construct_degree_matrix(epsilon_graph)
    # Construction de y_0
    y_0 = compute_y_0()
    y_i_b = np.zeros((len(y_0), 3))

    y_0_0 = [1,0,0]
    y_0_1 = [0,1,0]
    y_0_2 = [0,0,1]

    while not np.array_equal(y_0,y_i_b):
        y_i_b = y_0
        y_0 = iterate_over_y(epsilon_graph, degree_matrix, y_0)
        y_0[r_0] = y_0_0
        y_0[r_1] = y_0_1
        y_0[r_2] = y_0_2

    i = 0
    f_nb = 0
    for line in y_0:
        am = np.argmax(line)
        if am != C[i]:
            f_nb += 1
        print("Line: {} - {} - {}".format(line, am, C[i]))
        i += 1

    print("Nb false : {}".format(f_nb))

if __name__ == '__main__':
    X,C = load_iris_data()
    iterative_harmonic_algorithm(X,C)
