import numpy as np
from numpy import random 
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix
from node_creation import *
from probability_distributions import *

def main_loop(nodes_list, graph):
	"""
	aaaa
	"""

	#### Infect the new nodes ####

	# Calculate the infectivity vector
	incetion_rates = np.zeros(number_nodes)
	for i in range(number_nodes):
		incetion_rates[i] = nodes_list[i].return_infectivity()

	# infect 
	infection_probability = np.matmul(graph, incetion_rates) # matrix multiplication 
	infection_probability -= np.random.rand(number_nodes) # subtract a random number
	n_infected = 0
	for i in range(number_nodes):
		if infection_probability[i] > 0:
			nodes_list[i].infect()
			n_infected += 1


	#### kill, heal, become contageus ####
	prob_death, prob_heal, prob_contagious = 0,0,0
	n_death, n_recovery = 0,0
	for i in range(number_nodes):
		prob_death = death_probability(nodes_list[i].age, nodes_list[i].status, nodes_list[i].days_from_infection)
		prob_heal = heal_probability(nodes_list[i].age, nodes_list[i].status, nodes_list[i].days_from_infection) 
		prob_contagious = contagious_probability(nodes_list[i].age, nodes_list[i].status, nodes_list[i].days_from_infection) 
		
		if prob_death  - np.random.rand() > 0:
			nodes_list[i].kill()
			n_death += 1 
		if prob_heal  - np.random.rand() > 0:
			nodes_list[i].heal()
			n_recovery += 1
		if prob_contagious  - np.random.rand() > 0:
			nodes_list[i].set_contagious()

		nodes_list[i].update_days_from_infection()

	return nodes_list, n_infected, n_death, n_recovery




if __name__ == '__main__':

	#### Generates the graphs ####

	number_nodes = 2000
	nodes_list, family_graph = generate_nodes(number_nodes)
	graph = family_graph

	# set the initial infected 
	n_infected = 15 
	nodes_list = initial_infect(n_infected, nodes_list)


	n_days = 200 # number of days in the simulation
	history_infected, history_death, history_recovery = np.zeros(n_days), np.zeros(n_days), np.zeros(n_days)
	for i in range(n_days):# iterate over the days the simualtion 
		nodes_list, history_infected[i], history_death[i], history_recovery[i] = main_loop(nodes_list, graph)

	plt.plot(history_death, color='r', label='death')
	plt.plot(history_infected, color='b', label='infected')
	plt.plot(history_recovery, color='g', label='infected')
	plt.legend()
	plt.show()







