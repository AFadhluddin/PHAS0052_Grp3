import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random
import networkx as nx
from networkx import convert_matrix
import pathlib

from node_class import *

#import sys
#np.set_printoptions(threshold=sys.maxsize)


def node_generation(number_nodes):
    """
    Returns a list containing of the Nodes

    Inputs: 
    number_nodes (int)  The size of the population

    Return:
    nodes_list (list)   Contains list of the nodes created
    """
    nodes_list = []
    for i in range(number_nodes):
        node = Node()
        nodes_list.append(node)

    return nodes_list


def count_list(graph_index):
    """
    Returns a list the occurence of the element in its corrrect position

    Inputs: 
    graph_index (list)  Binary list which is 1 if correct job in nodes

    Return:
    index_occurrence (list)  elements in the correct positions
    """

    index_occurrence = [] 
    count_ones = 1   

    # Loop over index list and create occurrence loop
    for i in range(len(graph_index)):
        if graph_index[i] == 0:
            index_occurrence.append(0)
        elif graph_index[i] == 1:
            index_occurrence.append(count_ones)
            count_ones += 1
    return index_occurrence


def total_nodes_sq_mtrx_from_job_graph(full_size_graph, job_graph, number_occurences):
    """
    Returns a list the occurence of the element in its corrrect position

    Inputs: 
    full_size_graph (2d arrrau)  Zero array of nxn where n is population size
    job_graph (2d arrrau)  Zero array of mxm where m is job size
    number_occurrence (list)  elements in the correct positions

    Return:
    full_size_graph (n array)  filled array mapping job_graph
    """

    # enumerate over the occurences
    translate = []
    for i, n in enumerate(number_occurences):
        if n != 0:
            translate.append(i+1)

    for i, row in enumerate(job_graph):
        for j, n in enumerate(row):
            if n != 0:
                full_size_graph[translate[j]-1][translate[i]-1] = 1


    # Remove the diagonal elements
    np.fill_diagonal(full_size_graph, 0)
    
    return full_size_graph


def return_nodes_list_distributions(nodes_list):
    """
    Return information on the node of the network
    
    Inputs:
    nodes_list (list)                       Lists of the nodes
    
    Returns:
    total_worker_distribution (list)                Workers distribution by age (including essential and nonessential)
    worker_distributon (list)                       Nonessential orkers distribution by age
    essential_worker_distribution (list)            Essential workers distribution by age
    student_distribution (list)                     Students distribution by age
    total_number_worker (int)                       Total number of workers (including essential and nonessential)
    number_worker (int)                             Number of nonessential workers
    number_essential_worker (int)                   Number of essential workers
    number_student (int)                            Number of students
    student_index_occurrence (list)                 elements in the correct positions
    essential_worker_index_occurrence (list)        elements in the correct positions
    worker_index_occurrence (list)                  elements in the correct positions
    """

    num_nodes = len(nodes_list)
    total_worker_distribution, worker_distributon, essential_worker_distribution, student_distribution = np.zeros(
        num_nodes), np.zeros(num_nodes), np.zeros(num_nodes), np.zeros(num_nodes)
    total_worker_index, worker_index, essential_worker_index, student_index = np.zeros(
        num_nodes), np.zeros(num_nodes), np.zeros(num_nodes), np.zeros(num_nodes)

    # Loop over nodes list and populate respective arrays for the job type
    for i, node in enumerate(nodes_list):
        if node.job == 'worker':
            worker_distributon[node.age] += 1
            worker_index[i] = 1

        elif node.job == 'essential_worker':
            essential_worker_distribution[node.age] += 1
            essential_worker_index[i] = 1

        elif node.job == 'student':
            student_distribution[node.age] += 1
            student_index[i] = 1

    # Create the occurence arrays
    worker_index_occurrence = count_list(worker_index)
    essential_worker_index_occurrence = count_list(essential_worker_index)
    student_index_occurrence = count_list(student_index)

    # Variabls holding outputs created in function
    total_worker_distribution = worker_distributon + essential_worker_distribution
    total_number_worker = np.sum(total_worker_distribution)
    number_worker = np.sum(worker_distributon)
    number_essential_worker = np.sum(essential_worker_distribution)
    number_student = np.sum(student_distribution)

    return worker_index, essential_worker_index, student_index, total_number_worker, number_worker, number_essential_worker, number_student, student_index_occurrence, essential_worker_index_occurrence, worker_index_occurrence


def avg_deg(graph): 
    """
    Returns average degree of the graph

    Input:
    graph (2d array)   adjacency matrix

    Output:
    avg (float)    average degree
    
    """

    num_nodes = len(graph[0,:])
    degree_list = []
    
    for i in range(num_nodes):
        n=np.sum(graph[i,:])
        degree_list.append(n)

    avg = np.mean(np.array(degree_list))
    return avg


