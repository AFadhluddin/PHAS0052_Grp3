import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random
import networkx as nx
from networkx import convert_matrix
import pathlib

from utils_network_generation import *


class Network_Generation:
    """
    Network generation class
    """

    def __init__(self, number_nodes):
        """
        Constructor for network generation

        Inputs: 
        number_nodes (int)  size of the population 
        """

        self.number_nodes = number_nodes
        self.node_list = node_generation(number_nodes)
        self.total_init_matrix = np.zeros((number_nodes, number_nodes))

        self.worker_network_iarray, self.essential_worker_network_iarray, self.student_network_iarray, self.total_number_worker, self.number_worker, self.number_essential_worker, self.number_student, self.student_occurence_index, self.essential_worker_occurence_index, self.worker_occurence_index = return_nodes_list_distributions(self.node_list)



    def family_network(self):
        """
        Creates a family network

        Returns:
        family_network (2d array)  matrix of the family network
        """

        family_network = self.total_init_matrix
        nodes_remaining = self.number_nodes
        nodes_done = 0
        while nodes_remaining != 0:  # while it is possible to generate nodes

            # calulate the size of the family
            family_size = set_family_size()
            if family_size > nodes_remaining:  # if the family is larger of the remaining nodes
                family_size = nodes_remaining  # set the family size as the remaining nodes

            for i in range(int(family_size)):

                # create the family subgraph
                for j in range(int(family_size)):
                    family_network[nodes_done+i, nodes_done+j] = 1

            nodes_remaining -= family_size  # update the remaning nodes
            nodes_done += family_size  # update the done nodes
            

        return family_network

    def random_social_network(self,  m=1):
        """
        Creates a random network

        Returns:
        random_network (2d array)  matrix of the random network
        """
        
        # Use NetworK BA model to create network
        G_BA_social = nx.barabasi_albert_graph(self.number_nodes, m)

        social_network = self.total_init_matrix

        # Decompose nx graph to matrix
        for edge in G_BA_social.edges():
            social_network[edge[0], edge[1]] = 1
            social_network[edge[1], edge[0]] = 1

        # Set diagonal elements to 0
        np.fill_diagonal(social_network, 0)

        return social_network

    def worker_network(self, m=3):
        """
        Creates a worker network

        Returns:
        worker_network (2d array)  matrix of the worker network
        """
        # Variables to hold the constructor variables
        worker_network_iarray = self.worker_network_iarray
        number_worker = int(self.number_worker)
        worker_occurence_index = self.worker_occurence_index

        # Initialising the job and population matrices
        worker_network = np.zeros((number_worker, number_worker))
        worker_total_network = self.total_init_matrix

        # Using NetworkX to make a free scale graph
        G_BA_worker = nx.barabasi_albert_graph(number_worker, m)

        # Decompose nx graph to matrix
        for edge in G_BA_worker.edges():
            worker_network[edge[0], edge[1]] = 1
            worker_network[edge[1], edge[0]] = 1

        # Create final filled population-sized matrix for job type
        worker_total_network_filled = total_nodes_sq_mtrx_from_job_graph(
            worker_total_network, worker_network, worker_occurence_index)

        
        return worker_total_network_filled

    def essential_worker_network(self, m=1):
        """
        Creates a essential worker network

        Returns:
        essential_worker_network (2d array)  matrix of the essential worker network
        """

        # Variables to hold the constructor variables
        essential_worker_network_iarray = self.essential_worker_network_iarray
        number_essential_worker = int(self.number_essential_worker)
        essential_worker_occurence_index = self.essential_worker_occurence_index

        # Initialising the job and population matrices
        essential_worker_network = np.zeros(
            (number_essential_worker, number_essential_worker))
        essential_worker_total_network = self.total_init_matrix

        # Using NetworkX to make a free scale graph
        G_BA_essential_worker = nx.barabasi_albert_graph(
            number_essential_worker, m)

        # Decompose nx graph to matrix
        for edge in G_BA_essential_worker.edges():
            essential_worker_network[edge[0], edge[1]] = 1
            essential_worker_network[edge[1], edge[0]] = 1

        # Create final filled population-sized matrix for job type
        essential_worker_total_network_filled = total_nodes_sq_mtrx_from_job_graph(
            essential_worker_total_network, essential_worker_network, essential_worker_occurence_index)

        return essential_worker_total_network_filled

    def student_network(self, m=1):
        """
        Creates a student network

        Returns:
        student_network (2d array)  matrix of the student network
        """

        # Variables to hold the constructor variables
        student_network_iarray = self.student_network_iarray
        number_student = int(self.number_student)
        student_occurence_index = self.student_occurence_index

        # Initialising the job and population matrices
        student_network = np.zeros((number_student, number_student))
        student_total_network = self.total_init_matrix

         # Using NetworkX to make a free scale graph
        G_BA_student = nx.barabasi_albert_graph(number_student, m)

        # Decompose nx graph to matrix
        for edge in G_BA_student.edges():
            student_network[edge[0], edge[1]] = 1
            student_network[edge[1], edge[0]] = 1

        # Create final filled population-sized matrix for job type
        student_total_network_filled = total_nodes_sq_mtrx_from_job_graph(
            student_total_network, student_network, student_occurence_index)

        return student_total_network_filled  
        
        

def main_generation(number_nodes):
    """
    Main loop to be executed
    
    Input: 
    number_nodes (int) size of the population

    Return:
    Creates networkx graphs
    Plots them
    """

    # call the functions to generate the networks
    network_init = Network_Generation(number_nodes)
    network_init_mtrx = network_init.total_init_matrix
    family_network_nx = nx.convert_matrix.from_numpy_matrix(
        network_init.family_network())
    worker_network_nx = nx.convert_matrix.from_numpy_matrix(
        network_init.worker_network())
    essential_network_graph_nx = nx.convert_matrix.from_numpy_matrix(
        network_init.essential_worker_network())
    student_network_nx = nx.convert_matrix.from_numpy_matrix(
        network_init.student_network())
    random_network_nx = nx.convert_matrix.from_numpy_matrix(
        network_init.random_social_network())


   # Plotting the graph
    plt.figure(1)
    family_plot = nx.draw(
        family_network_nx, with_labels=True, node_color='green')

    # Plotting the graph
    plt.figure(2)
    workers_plot = nx.draw(
        worker_network_nx, with_labels=True, node_color='blue')

    # Plotting the graph
    plt.figure(3)
    essential_workers_plot = nx.draw(
        essential_network_graph_nx, with_labels=True, node_color='yellow')

    # Plotting the graph
    plt.figure(4)
    random_plot = nx.draw(
        random_network_nx, with_labels=True, node_color='red')

    # Plotting the graph
    plt.figure(5)
    random_plot = nx.draw(student_network_nx,
                          with_labels=True, node_color='pink')

    # Composition of the graph together
    composition_graph = nx.compose_all(
        [family_network_nx, worker_network_nx, essential_network_graph_nx, random_network_nx, student_network_nx])

    # Ploting the graph
    plt.figure(6)
    joined_plot = nx.draw(composition_graph, with_labels=True)


if __name__ == "__main__":
    number_nodes = 50
    main_generation(number_nodes)
    plt.show()
