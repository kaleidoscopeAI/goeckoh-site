import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import ConvexHull

from scipy.sparse.linalg import eigsh

import networkx as nx


class MolecularCube:

def __init__(self, num_nodes=50):

