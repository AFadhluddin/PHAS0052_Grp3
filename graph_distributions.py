import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix
from Node_Test import Node


def age_distribution(graph_type):
    """
    Calulate the age of the person from the real-data distribution depending on the nature 
    of the node
    Input: graph_type  nature of the graph

    Return: Age_band    band age of the node (eg. 2 between 21 and 30 years old)
    """
    if graph_type == 'family' or graph_type == 'random':
        # calulate the age by the real data distribution 
        age = np.random.normal(45, 20)
        # calulate the age band
        age_band = int(age/10 - (int(age/10) - age/10))
        if age_band > 9: # for >90 of age are all the same 
            age_band = 9
        return age_band
    
    elif graph_type == 'workers' or graph_type == 'essential_workers' or graph_type == 'unemployed':
        # calulate the age by the real data distribution 
        age = np.random.choice(range(16,75))
        # calulate the age band
        age_band = int(age/10 - (int(age/10) - age/10))
        if age_band > 9: # for >90 of age are all the same 
            age_band = 9
        return age_band
    
    elif graph_type == 'student':
        # calulate the age by the real data distribution 
        age = np.random.choice(range(3,23))
        # calulate the age band
        age_band = int(age/10 - (int(age/10) - age/10))
        if age_band > 9: # for >90 of age are all the same 
            age_band = 9
        return age_band

def family_size_distribution():
    """
    Calulate the size of a family from the real-data distribution
    Return:      family_size     size of the family
    """

    # calulate the family size by the real data distribution 
    family_size = np.random.normal(4, 2)
    if family_size < 1:
        family_size = 1
    return int(family_size)

def workers_size_distribution():
    """
    Calulate the size of a working team from the real-data distribution
    Return:      worker_size     size of the worker
    """

    # calulate the workers size by the real data distribution 
    workers_size = np.random.normal(4, 2)
    if workers_size < 1:
        workers_size = 1
    return int(workers_size)


def student_size_distribution():
    """
    Calulate the size of a student group from the real-data distribution
    Return:      worker_size     size of the worker
    """

    # calulate the student size by the real data distribution 
    student_size = np.random.normal(8, 2)
    if student_size < 1:
        studnet_size = 1
    return int(student_size)


def essential_workers_size_distribution():
    """
    Calulate the size of a essential working team from the real-data distribution
    Return:      worker_size     size of the worker
    """

    # calulate the worker size by the real data distribution 
    essential_workers_size = np.random.normal(10, 2)
    if essential_workers_size < 1:
        essential_workers_size = 1
    return int(essential_workers_size)
