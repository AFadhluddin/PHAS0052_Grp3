import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix
from Node_Test import *
from graph_dist_class import *


#def unemployed_distribution():


def random_social_network(number_nodes, m):
    """ Creates a randome free-scale network using the BA model in Networkx"""
    G_BA = nx.barabasi_albert_graph(number_nodes,2)
    return G_BA


def family_network(number_nodes):
    """ Generates a family network
    Input: number_nodes   Total amount of people"""

    # initalise the node_list and family_graph
    nodes_list = []
    family_graph = np.zeros((number_nodes,number_nodes))

    nodes_remaining = number_nodes
    nodes_done = 0
    while nodes_remaining != 0: # while it is possible to generate nodes

        # calulate the size of the family 
        family_size = graph_distributions('family').size_distribution()
        if family_size > nodes_remaining: # if the family is larger of the remaining nodes
            family_size = nodes_remaining # set the family size as the remaining nodes


        for i in range(int(family_size)):
            # create the nodes
            age = graph_distributions('family').age_distribution()
            nodes_list.append(Node(age))
            # create the family subgraph 
            for j in range(int(family_size)):
                family_graph[nodes_done+i,nodes_done+j] = 1

        nodes_remaining -= family_size # update the remaning nodes
        nodes_done += family_size # update the done nodes
        np.fill_diagonal(family_graph,0)

    # Networkx graph
    family_graph_nx = nx.convert_matrix.from_numpy_matrix(family_graph)
    return nodes_list, family_graph_nx


def workers_network(number_nodes, prop_of_pop):
    """ Generates a workers network
    Input:
    number_of_nodes  Total number of people in the simulation
    prop_of_pop  Proportion of the population which are workers"""

    # initalise the node_list and workers_graph
    nodes_list = []
    workers_graph = np.zeros((number_nodes,number_nodes))

    nodes_remaining = int(number_nodes*prop_of_pop)
    nodes_done = 0
    while nodes_remaining != 0: # while it is possible to generate nodes

        # calulate the size of the worker group 
        workers_size = graph_distributions('workers').size_distribution()
        if workers_size > nodes_remaining: # if the worker group is larger of the remaining nodes
            workers_size = nodes_remaining # set the worker group size as the remaining nodes


        for i in range(int(workers_size)):
            # create the nodes
            age = graph_distributions('workers').age_distribution()
            person = Node(age)
            if person.job == 'worker': # only adds nodes that fit the network type
                nodes_list.append(person)
                # create the worker subgraph 
                for j in range(int(workers_size)):
                    workers_graph[nodes_done+i,nodes_done+j] = 1
            else:
                pass

        nodes_remaining -= workers_size # update the remaning nodes
        nodes_done += workers_size # update the done nodes
        np.fill_diagonal(workers_graph,0)

    # Networkx graph
    workers_graph_nx = nx.convert_matrix.from_numpy_matrix(workers_graph)
    workers_graph_nx.remove_nodes_from(list(nx.isolates(workers_graph_nx)))
    return nodes_list, workers_graph_nx


def essential_workers_network(number_nodes, prop_of_pop):
    """ Generates a essential workers network
    Input:
    number_of_nodes  Total number of people in the simulation
    prop_of_pop  Proportion of the population which are essential workers"""

    # initalise the node_list and essential_workers_graph
    nodes_list = []
    essential_workers_graph = np.zeros((number_nodes,number_nodes))

    nodes_remaining = int(number_nodes*prop_of_pop)
    nodes_done = 0
    while nodes_remaining != 0: # while it is possible to generate nodes

        # calulate the size of the worker group 
        essential_workers_size = graph_distributions('essential_workers').size_distribution()
        if essential_workers_size > nodes_remaining: # if the essential workers gruop is larger of the remaining nodes
            essential_workers_size = nodes_remaining # set the essential workers size as the remaining nodes


        for i in range(int(essential_workers_size)):
            # create the nodes
            age = graph_distributions('essential_workers').age_distribution()
            person = Node(age)
            if person.job == 'essential_worker': # only adds nodes that fit the network type
                nodes_list.append(person)
            # create the worker subgraph 
                for j in range(int(essential_workers_size)):
                    essential_workers_graph[nodes_done+i,nodes_done+j] = 1
            else:
                pass

        nodes_remaining -= essential_workers_size # update the remaning nodes
        nodes_done += essential_workers_size # update the done nodes
        np.fill_diagonal(essential_workers_graph,0)

    # Networkx graph
    essential_workers_graph_nx = nx.convert_matrix.from_numpy_matrix(essential_workers_graph)
    essential_workers_graph_nx.remove_nodes_from(list(nx.isolates(essential_workers_graph_nx)))
    return nodes_list, essential_workers_graph_nx

def student_network(number_nodes, prop_of_pop):
    """ Generates a student network
    Input:
    number_of_nodes  Total number of people in the simulation
    prop_of_pop  Proportion of the population which are students"""

    # initalise the node_list and workers_graph
    nodes_list = []
    student_graph = np.zeros((number_nodes,number_nodes))

    nodes_remaining = int(number_nodes*prop_of_pop)
    nodes_done = 0
    while nodes_remaining != 0: # while it is possible to generate nodes

        # calulate the size of the worker group
        
        student_size = graph_distributions('student').size_distribution()
        if student_size > nodes_remaining: # if the student group is larger of the remaining nodes
            student_size = nodes_remaining # set the student size as the remaining nodes


        for i in range(int(student_size)):
            # create the nodes
            age = graph_distributions('student').age_distribution()
            person = Node(age)
            if person.job == 'student':
                nodes_list.append(person)
                # create the worker subgraph 
                for j in range(int(student_size)):
                    student_graph[nodes_done+i,nodes_done+j] = 1
            else:
                pass

        nodes_remaining -= student_size # update the remaning nodes
        nodes_done += student_size # update the done nodes
        np.fill_diagonal(student_graph,0)

    # Networkx graph
    student_graph_nx = nx.convert_matrix.from_numpy_matrix(student_graph)
    student_graph_nx.remove_nodes_from(list(nx.isolates(student_graph_nx)))
    return nodes_list, student_graph_nx




def main(number_nodes):
    """Main loop to be executed

    input: number nodes"""

    # Define the variable for the proportion of populations 
    prop_workers = 0.6
    prop_essential_workers = 0.1
    prop_students = 0.3
    
    # call the functions to generate the networks 
    family_nodes_list, family_graph_nx = family_network(number_nodes)
    workers_nodes_list, workers_graph_nx = workers_network(number_nodes,prop_workers)
    essential_workers_nodes_list, essential_workers_graph_nx = essential_workers_network(number_nodes, prop_essential_workers)
    student_nodes_list, student_graph_nx = student_network(number_nodes, prop_students)
    random_graph_nx = random_social_network(number_nodes, 3)


   # Plotting the graph
    plt.figure(1)
    family_plot = nx.draw(family_graph_nx, with_labels=True, node_color='green')

    # Plotting the graph
    plt.figure(2)
    workers_plot = nx.draw(workers_graph_nx, with_labels=True, node_color='blue')

    # Plotting the graph
    plt.figure(3)
    essential_workers_plot = nx.draw(essential_workers_graph_nx, with_labels=True, node_color='yellow')

    # Plotting the graph
    plt.figure(4)
    random_plot = nx.draw(random_graph_nx, with_labels=True, node_color='red')

    # Plotting the graph
    plt.figure(5)
    random_plot = nx.draw(student_graph_nx, with_labels=True, node_color='pink')

    # Composition of the graph together
    composition_graph = nx.compose_all([family_graph_nx, workers_graph_nx, essential_workers_graph_nx, random_graph_nx])

    #Ploting the graph
    plt.figure(6)
    joined_plot = nx.draw(composition_graph, with_labels=True)

    plt.show()
   



# Runs only the main loop defined in this file
if __name__ == '__main__':
    number_nodes = 300 # sets the number of total nodes
    main(number_nodes) #execute main loop



