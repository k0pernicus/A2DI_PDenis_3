#-*- coding: utf-8 -*-

import numpy as np
import math
import random
import sys

from sklearn import datasets

global r

def load_iris_data():
    """
    Fonction permettant de charger en mémoire les données IRIS, contenues dans scikit learn.
    """

    iris = datasets.load_iris()
    X = iris.data[:, :4]
    C = iris.target

    return (X, C)

def similarity(xi, xj, sigma):
    """
    Donne la similarité entre deux données.
    """

    return math.exp(-(np.linalg.norm(np.subtract(xi, xj))) / math.pow(sigma, 2))

def construct_degree_matrix(epsilon_graph):
    """
    Construit la matrice des degrés, pour un epsilon graph donné
    """

    len_epsilon = len(epsilon_graph)

    degree_matrix = np.zeros((len_epsilon, len_epsilon))

    for i, line in enumerate(epsilon_graph):
        count = 0
        for j in range(0, len(line)):
            count += 1
        degree_matrix[i,i] = count

    return degree_matrix

def construct_epsilon_graph(X, epsilon, sigma):
    """
    Construit l'epsilon-graph associé aux données de l'IRIS
    """

    len_X = len(X)

    epsilon_graph = np.zeros((len_X, len_X))

    for i, xi in enumerate(X):
        for j, xj in enumerate(X):
            sim = similarity(xi, xj, sigma)
            if sim > epsilon:
                epsilon_graph[i,j] = sim
            else:
                epsilon_graph[i,j] = 0

    return epsilon_graph

def choose_k_points_from(X, C, k):
    """
    Fonction permettant de choisir 3 points aléatoires - chacun étant d'une classe
    inférieure à l'autre
    """

    global r

    r = []

    # Génère des nombres aléatoires, chacun dans des intervalles permettant
    # de choisir 1 point pour chaque classe
    r.append(random.randrange(0, 50))
    r.append(random.randrange(50, 100))
    r.append(random.randrange(100, 150))

    for elt in range(k-3):
        r.append(random.randrange(0, 150))

    nb_learned = [0] * 3

    for n in r:
        nb_learned[C[n]] += 1

    for i, r_0 in enumerate(nb_learned):
        print("Class {} : {} learned".format(i, r_0))

def compute_y_0(C, k):

    global r
    # Initialisation du vecteur y_0 (les 3 premières du X modifié dans
    # choose_3_points_from)
    y_0 = np.zeros((len(C), 3))

    for r_0 in r:
        if C[r_0] == 0:
            y_0[r_0] = [1,0,0]
        elif C[r_0] == 1:
            y_0[r_0] = [0,1,0]
        else:
            y_0[r_0] = [0,0,1]

    return y_0

def iterate_over_y(epsilon_graph, degree_matrix, y_i):
    """
    Fonction permettant d'itérer sur le vecteur y (jusqu'à convergence)
    """

    global r

    y_0 = np.dot(np.dot(np.linalg.inv(degree_matrix), epsilon_graph), y_i)

    for r_0 in r:
        if C[r_0] == 0:
            y_0[r_0] = [1,0,0]
        elif C[r_0] == 1:
            y_0[r_0] = [0,1,0]
        else:
            y_0[r_0] = [0,0,1]

    return y_0

def evaluate(y_0, C):

    f_nb = 0

    errors = [0] * 3

    for i, line in enumerate(y_0):
        am = np.argmax(line)
        if am != C[i]:
            f_nb += 1
            errors[C[i]] += 1

    print("")
    print("RESULTS...")
    print("*"*80)
    for i, error in enumerate(errors):
        print("Class {} : {} errors".format(i, error))
    print("*"*80)

def iterative_harmonic_algorithm(X, C, k, epsilon, sigma):
    """
    Fonction permettant de faire tourner l'algorithme itératif harmonique, afin de prédire les labels inconnus
    """

    global r

    # Construction de l'epsilon-graph
    epsilon_graph = construct_epsilon_graph(X, epsilon, sigma)
    # Calcul de la matrice des degrés
    degree_matrix = construct_degree_matrix(epsilon_graph)
    # Choix de 3 points aléatoires -> new_X
    choose_k_points_from(X, C, k)
    # Construction de y_0
    y_0 = compute_y_0(C, k)
    y_i_b = np.zeros((len(C), 3))

    while not np.array_equal(y_0,y_i_b):
        y_i_b = y_0
        y_0 = iterate_over_y(epsilon_graph, degree_matrix, y_0)

    evaluate(y_0, C)

if __name__ == '__main__':
    k = int(sys.argv[1])
    epsilon = float(sys.argv[2])
    sigma = float(sys.argv[3])
    print("*"*80)
    print("* PARAMETERS...")
    print("* k : {}".format(k))
    print("* epsilon: {}".format(epsilon))
    print("* sigma: {}".format(sigma))
    print("*"*80)
    X,C = load_iris_data()
    iterative_harmonic_algorithm(X,C,k,epsilon,sigma)
