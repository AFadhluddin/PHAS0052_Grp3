import numpy as np
from numpy import random 
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix
import pandas as pd

from network_generation import *
from interventions import *

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
		if nodes_list[i].status == 'healthy' and nodes_list[i].immune == False: # check that it can be infected
			if infection_probability[i] > np.random.rand():
				nodes_list[i].infect()
				n_infected += 1

	#### kill, heal, become contageus ####
	n_death, n_recovery = 0,0
	for i in range(number_nodes):
		if nodes_list[i].status == 'infected': # check that it is infected 
			if nodes_list[i].day_from_infection != -1:
				if nodes_list[i].day_of_death == nodes_list[i].day_from_infection:
					nodes_list[i].kill()
					n_death += 1 
				if nodes_list[i].day_of_heal == nodes_list[i].day_from_infection:
					nodes_list[i].heal()
					n_recovery += 1
				if nodes_list[i].day_first_symptoms == nodes_list[i].day_from_infection:
					nodes_list[i].set_contagious()
		nodes_list[i].update_days_from_infection()

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
	matrix_vaccination = np.zeros((n_simulations, n_days))

	for i in range(n_simulations):

		network = Network_Generation(n_nodes) # generate the network

		# creates all the subgraphs
		family_graph = network.family_network()
		worker_graph = network.worker_network(array_network_parameters[0])
		essential_worker_graph = network.essential_worker_network(array_network_parameters[1])
		student_graph = network.student_network(array_network_parameters[2])
		random_graph = network.random_social_network(array_network_parameters[3])
		essential_random_graph = network.essential_random_network(array_network_parameters[4])

		# weighted sum of the network 
		total_network = (array_weights[0]*family_graph + array_weights[1]*worker_graph +
			array_weights[2]*essential_worker_graph + array_weights[3]*student_graph +
			array_weights[4]*random_graph + array_weights[5]+essential_random_graph)

		network.node_list = initial_infect(n_initial_infected, network.node_list) #infect the intial nodes
		
		vaccinations_number_array = vaccinations_array(n_days)

		for j in range(n_days):
			array_weights, change = lockdowns(array_weights, j,days_lockdown_start, days_lockdown_end, day_school_close, day_school_open, original_array_weights)
			if change == True:
				total_network = (array_weights[0]*family_graph + array_weights[1]*worker_graph +
					array_weights[2]*essential_worker_graph + array_weights[3]*student_graph +
					array_weights[4]*random_graph)
			network.node_list, matrix_infected[i,j], matrix_death[i,j], matrix_recovery[i,j] = main_loop(network.node_list, total_network)
			network.node_list, matrix_vaccination[i,j] = vaccination(network.node_list, vaccinations_number_array[j])

	return matrix_infected, matrix_death, matrix_recovery, matrix_vaccination

def plot_results(matrix_infected, matrix_death, matrix_recovery, matrix_vaccination):
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
	
	######## Infections subplots ########
	ax1 = fig.add_subplot(141)
	ax1.set_title('Infections')

	# plot each simulation
	for i in range(n_simulations):
		ax1.plot(matrix_infected[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax1.plot(average_infected, color = 'b') # plot the average

	# plot lockdowns
	#for i in range(len(days_lockdown_start)):
	#	ax1.axvline(days_lockdown_start[i], color='r', linewidth=0.5,alpha = 0.5)
	#	ax1.axvline(days_lockdown_end[i], color='r', linewidth=0.5,alpha = 0.5)

	#for i in range(len(day_school_close)):
	#	ax1.axvline(day_school_close[i], color='m', linewidth=0.5,alpha = 0.5)
	#	ax1.axvline(day_school_open[i], color='m', linewidth=0.5,alpha = 0.5)

	######## Death subplots ########
	ax2 = fig.add_subplot(142)
	ax2.set_title('Deaths')

	# plot each simulation
	for i in range(n_simulations):
		ax2.plot(matrix_death[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax2.plot(average_death, color = 'r') # plot the average
	
	# plot lockdowns
	#for i in range(len(days_lockdown_start)):
	#	ax2.axvline(days_lockdown_start[i], color='r', linewidth=0.5,alpha = 0.5)
	#	ax2.axvline(days_lockdown_end[i], color='r', linewidth=0.5,alpha = 0.5)

	#for i in range(len(day_school_close)):
	#	ax2.axvline(day_school_close[i], color='m', linewidth=0.5,alpha = 0.5)
	#	ax2.axvline(day_school_open[i], color='m', linewidth=0.5,alpha = 0.5)

	######## Infections subplots ########
	ax3 = fig.add_subplot(143)
	ax3.set_title('Recoveries')

	# plot each simulation
	for i in range(n_simulations):
		ax3.plot(matrix_recovery[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax3.plot(average_recovery, color = 'g') # plot the average

	# plot lockdowns
	#for i in range(len(days_lockdown_start)):
	#	ax3.axvline(days_lockdown_start[i], color='r', linewidth=0.5,alpha = 0.5)
	#	ax3.axvline(days_lockdown_end[i], color='r', linewidth=0.5,alpha = 0.5)

	#for i in range(len(day_school_close)):
	#	ax3.axvline(day_school_close[i], color='m', linewidth=0.5,alpha = 0.5)
	#	ax3.axvline(day_school_open[i], color='m', linewidth=0.5,alpha = 0.5)

	# plot vaccination 
	ax4 = fig.add_subplot(144)
	for i in range(n_simulations):
		ax4.set_title('Vaccinations')
	ax4.plot(matrix_vaccination[i], color='black', linewidth=0.5,alpha = 0.5)

	plt.savefig('Results.pdf')
	return


def save_results(matrix_infected, matrix_death, matrix_recovery, matrix_vaccination):
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
	df_matrix_vaccination = pd.DataFrame(data=matrix_vaccination)

	# save 
	df_matrix_infected.to_csv("infected_results.csv")
	df_matrix_death.to_csv("infected_results.csv")
	df_matrix_recovery.to_csv("infected_results.csv")	
	#df_matrix_vaccination.to_csv("vaccination_results.csv")
	
	return

if __name__ == "__main__":
	n_simulations = 5
	n_days = 60
	n_nodes = 1000000 
	n_initial_infected = 8
	array_network_parameters = np.array([2,2,2,2,2])
	array_weights = np.array([1,1,1,1,1,1])
	original_array_weights = array_weights

	# days of lockdown data
	days_lockdown_start = [30, 100]
	days_lockdown_end = [60, 130]
	day_school_close = [30]
	day_school_open = [65]

	# call the simulations and save
	matrix_infected, matrix_death, matrix_recovery, matrix_vaccination = main_algorithm(n_simulations, n_days, n_nodes, n_initial_infected, array_network_parameters, array_weights)
	plot_results(matrix_infected, matrix_death, matrix_recovery, matrix_vaccination)
	save_results(matrix_infected, matrix_death, matrix_recovery, matrix_vaccination)










