import numpy as np
from numpy import random 
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix
import pandas as pd

from network_generation import *

def main_loop(nodes_list, graph):
	"""
	Main loop: to be called each day of the simulation
	Inputs: 
	nodes_list          List of the nodes
	graph               Matrix of the graph
	Return: nodes_list, n_infected, n_death, n_recovery
	"""

	#### Infect the new nodes ####
	number_nodes = len(nodes_list)
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

def main_algorithm(n_simulations, n_days, n_nodes, n_initial_infected, array_network_parameters, array_weights):
	"""
	Creates n simulations by iterating the main loop on each day
	Inputs:
	n_simulations              Number of simulations
	n_days                     Number of days per simulation
	n_nodes                    Number of nodes in the simulations
	n_initial_infected         Number of initial infected nodes
	array_network_parameters   Vector of parameters for the subgraphs
	array_weights              Vector of weights for the subgraphs
	Outputs: matrix_infected, matrix_death, matrix_recovery
	"""
	matrix_infected = np.zeros((n_simulations, n_days))
	matrix_death = np.zeros((n_simulations, n_days))
	matrix_recovery = np.zeros((n_simulations, n_days))

	for i in range(n_simulations):

		network = Network_Generation(n_nodes) # generate the network

		# creates all the subgraphs
		family_graph = network.family_network()
		worker_graph = network.worker_network(array_network_parameters[0])
		essential_network_graph = network.essential_worker_network(array_network_parameters[1])
		student_graph = network.student_network(array_network_parameters[2])
		random_graph = network.random_social_network(array_network_parameters[3])

		# weighted sum of the network 
		### NEED TO CHANGE FOR LOCKDOWN, PASS EVERYTHING IN THE MAIN LOOP
		total_network = (array_weights[0]*family_graph + array_weights[1]*worker_graph +
			array_weights[2]*essential_network_graph + array_weights[3]*student_graph +
			array_weights[4]*random_graph)

		network.node_list = initial_infect(n_initial_infected, network.node_list) #infect the intial nodes
		
		for j in range(n_days):
			network.node_list, matrix_infected[i,j], matrix_death[i,j], matrix_recovery[i,j] = main_loop(network.node_list, total_network)

	return matrix_infected, matrix_death, matrix_recovery

def plot_results(matrix_infected, matrix_death, matrix_recovery):
	"""
	Plot the simulation results and save the plots
	Inputs: 
	matrix_infected      Matrix of infected (n_simulation X n_day)
	matrix_death         Matrix of death (n_simulation X n_day)
	matrix_recovery      Matrix of recovery (n_simulation X n_day)
	"""
	
	average_infected = np.mean(matrix_infected, axis = 0)
	average_death = np.mean(matrix_death, axis = 0)
	average_recovery = np.mean(matrix_recovery , axis = 0)

	fig = plt.figure(figsize = (14, 8))
	
	ax1 = fig.add_subplot(131)
	ax1.set_title('Infections')
	for i in range(n_simulations):
		ax1.plot(matrix_infected[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax1.plot(average_infected, color = 'b')

	ax2 = fig.add_subplot(132)
	ax2.set_title('Deaths')
	for i in range(n_simulations):
		ax2.plot(matrix_death[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax2.plot(average_death, color = 'r')
	
	ax3 = fig.add_subplot(133)
	ax3.set_title('Recoveries')
	for i in range(n_simulations):
		ax3.plot(matrix_recovery[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax3.plot(average_recovery, color = 'g')

	plt.savefig('Results.pdf')
	return

def save_results(matrix_infected, matrix_death, matrix_recovery):
	"""
	Saves the simulation results in csv files 
	Inputs: 
	matrix_infected      Matrix of infected (n_simulation X n_day)
	matrix_death         Matrix of death (n_simulation X n_day)
	matrix_recovery      Matrix of recovery (n_simulation X n_day)
	"""
	# convert to dataframe
	df_matrix_infected = pd.DataFrame(data=matrix_infected)
	df_matrix_death = pd.DataFrame(data=matrix_death)
	df_matrix_recovery = pd.DataFrame(data=matrix_recovery)

	# save 
	df_matrix_infected.to_csv("infected_results.csv")
	df_matrix_death.to_csv("infected_results.csv")
	df_matrix_recovery.to_csv("infected_results.csv")	
	
	return

if __name__ == "__main__":
	# set initial parameters
	n_simulations = 10
	n_days = 10
	n_nodes = 1000
	n_initial_infected = 3
	array_network_parameters = np.array([0.001,0.001,0.001,0.001])
	array_weights = np.array([1,1,1,1,1])

	# call the simulations and save
	matrix_infected, matrix_death, matrix_recovery = main_algorithm(n_simulations, n_days, n_nodes, n_initial_infected, array_network_parameters, array_weights)
	plot_results(matrix_infected, matrix_death, matrix_recovery)
	save_results(matrix_infected, matrix_death, matrix_recovery)










