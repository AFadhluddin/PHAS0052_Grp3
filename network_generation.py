
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random
import networkx as nx
from networkx import convert_matrix
import pathlib
from data_importing_tool import *
from node_class import *
from real_data_distributions import *

def return_nodes_list_distributions(nodes_list):
	"""
	Return information on the node of the network
	Inputs: 
	nodes_list                        Lists of the nodes
	Ouputs:
	total_worker_distribution         Workers distribution by age (including essential and nonessential)
	worker_distributon                Nonessential orkers distribution by age
	essential_worker_distribution     Essential workers distribution by age
	student_distribution              Students distribution by age
	total_number_worker               Total number of workers (including essential and nonessential)
	number_worker                     Number of nonessential workers
	number_essential_worker           Number of essential workers
	number_student                    Number of students 
	"""
	total_worker_distribution, worker_distributon, essential_worker_distribution, student_distribution = np.zeros(20), np.zeros(20), np.zeros(20), np.zeros(20) 
	for node in nodes_list:
		if node.job == 'worker':
			worker_distributon[node.age] += 1

		elif node.job == 'essential_worker':
			essential_worker_distribution[node.age] += 1

		elif node.job == 'student':
			student_distribution[node.age] += 1
	total_worker_distribution = worker_distributon + essential_worker_distribution
	total_number_worker = np.sum(total_worker_distribution)
	number_worker = np.sum(worker_distributon)
	number_essential_worker = np.sum(essential_worker_distribution)
	number_student = np.sum(student_distribution)

	return total_worker_distribution, worker_distributon, essential_worker_distribution, student_distribution, total_number_worker, number_worker, number_essential_worker, number_student

def generate_nodes(number_nodes):
	"""
	Generates the nodes of the simulation and the f
	Input:
	number_nodes       Number of nodes of the simulation
	Return: 
	nodes_list         List of the nodes
	family_graph       Matrix representing the faliy subgraph 
	"""
	# initalise the node_list and family_graph
	nodes_list = []
	family_graph = np.zeros((number_nodes, number_nodes))

	nodes_remaining = number_nodes
	nodes_done = 0 
	while nodes_remaining != 0: # while it is possible to generate nodes
	
		# calulate the size of the family 
		family_size = set_family_size()
		if family_size > nodes_remaining: # if the family is larger of the remaining nodes
			family_size = nodes_remaining # set the family size as the remaining nodes

		for i in range(int(family_size)):

			# create the nodes
			nodes_list.append(Node())
			# create the family subgraph 
			for j in range(int(family_size)):
				family_graph[nodes_done+i,nodes_done+j] = 1

		nodes_remaining -= family_size # update the remaning nodes
		nodes_done += family_size # update the done nodes
		
	return nodes_list, family_graph

def main_generation(number_nodes):
	"""
	Main function to call to generate the network
	Input:
	number_nodes       Number of nodes of the simulation
	Return: 
	nodes_list         List of the nodes
	total_network      Matrix of the total network 
	"""
	nodes_list, family_graph = generate_nodes(number_nodes)
	total_network = family_graph
	return nodes_list, total_network

number_nodes = 1000
nodes_list, total_network = main_generation(number_nodes)
total_worker_distribution, worker_distributon, essential_worker_distribution, student_distribution, total_number_worker, number_worker, number_essential_worker, number_student = return_nodes_list_distributions(nodes_list)
#print(total_network)
print(total_worker_distribution)
print(worker_distributon)
print(essential_worker_distribution)
