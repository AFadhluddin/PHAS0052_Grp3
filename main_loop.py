import numpy as np
from numpy import random 
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix

from network_generation import *

def main_loop(nodes_list, graph):
	"""
	aaaa
	"""

	#### Infect the new nodes ####

	# Calculate the infectivity vector
	infection_rates = np.zeros(number_nodes)
	for i in range(number_nodes):
		infection_rates[i] = nodes_list[i].return_infectivity()

	# infect 
	infection_probability = np.matmul(graph, infection_rates) # matrix multiplication 
	
	n_infected = 0
	for i in range(number_nodes):
		if infection_probability[i] > np.random.rand() and nodes_list[i].status == 'healthy' and nodes_list[i].immune == False:
			nodes_list[i].infect()
			n_infected += 1

	#### kill, heal, become contageus ####
	n_death, n_recovery = 0,0
	for node in nodes_list:
		
		if node.day_of_death == node.day_from_infection:
			node.kill()
			n_death += 1 
		
		if node.day_of_heal == node.day_from_infection:
			node.heal()
			n_recovery += 1

		if node.day_first_symptoms == node.day_from_infection:
			node.set_contagious()

		node.update_days_from_infection()

	return nodes_list, n_infected, n_death, n_recovery

#### Generates the graphs ####
number_nodes = 20
network_init = Network_Generation(number_nodes)

family_graph = network_init.family_network()
worker_graph = network_init.worker_network()
essential_network_graph = network_init.essential_worker_network()
student_graph = network_init.student_network()
random_graph = network_init.random_social_network()

total_network = family_graph#+worker_graph + essential_network_graph + student_graph + random_graph

# set the initial infected 
n_infected = 15 
network_init.node_list = initial_infect(n_infected, network_init.node_list)

#array_edges = np.sum(total_network, axis= 0)
#print(family_graph)
#distribution_y = np.zeros(int(np.max(array_edges)) + 1)
#for i in range(len(array_edges)):
#	distribution_y[int(array_edges[i])] += 1

#plt.plot(distribution_y)
#plt.show()

n_days = 200 # number of days in the simulation
history_infected, history_death, history_recovery = np.zeros(n_days), np.zeros(n_days), np.zeros(n_days)
for i in range(n_days):# iterate over the days the simualtion 
	network_init.node_list, history_infected[i], history_death[i], history_recovery[i] = main_loop(network_init.node_list, total_network)

plt.plot(history_death, color='r', label='death')
plt.plot(history_infected, color='b', label='infected')
plt.plot(history_recovery, color='g', label='recovery')
plt.legend()
plt.show()







