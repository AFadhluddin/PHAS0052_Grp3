import numpy as np
from numpy import random 
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix
import pandas as pd
from tqdm import tqdm

from network_generation import *
from interventions import *

############################## import data ##############################

# create a dataframe for parameter where the index is age
parameter_age = parameter_importer(pathandfile('parameters_age.csv'))
# this is to form a new column 'percentage_population' that shows the percentage of the population in certain age range, in case you need it
parameter_age['percentage_population'] = parameter_age['population_UK']/parameter_age['population_UK'].sum()
# create a dataframe for parameter where the index is day
parameter_day = parameter_importer(pathandfile('parameters_day.csv'))

# columns 
probability_day_death = column_extractor(parameter_day, 'Probability_of_Death')
probability_day_death = probability_day_death/np.sum(probability_day_death)

probability_day_incubation = column_extractor(parameter_day, 'Incubation_Period')
probability_day_incubation = probability_day_incubation/np.sum(probability_day_incubation)


def simulation(n_days, n_nodes, n_initial_infected, array_network_parameters, array_weights):
	
	network = Network_Generation(n_nodes) # generate the network

	# creates all the subgraphs
	family_graph = network.family_network()
	worker_graph = network.worker_network(array_network_parameters[0])
	essential_worker_graph = network.essential_worker_network(array_network_parameters[1])
	student_graph = network.student_network(array_network_parameters[2])
	random_graph = network.random_social_network(array_network_parameters[3])
	essential_random_graph = network.essential_random_network(array_network_parameters[4])

	# weighted sum of the network 
	original_array_weights = array_weights
	total_network = (array_weights[0]*family_graph + array_weights[1]*worker_graph + 
		array_weights[2]*essential_worker_graph + array_weights[3]*student_graph + 
		array_weights[4]*random_graph + array_weights[5]*essential_random_graph)
	
	df_initial_graph = pd.DataFrame(data=total_network) 
	df_initial_graph.to_csv("initial_network.csv")
	#df_family_graph = pd.DataFrame(data=family_graph) 
	#df_family_graph.to_csv("family_graph.csv")
	#df_work_graph = pd.DataFrame(data=worker_graph) 
	#df_work_graph.to_csv("worker_graph.csv")

	network.node_list = initial_infect(n_initial_infected, network.node_list) #infect the intial nodes
	
	vaccinations_number_array = np.zeros(n_days) # vaccinations_array(n_days) #TO BE CHANGED 

	days_lockdown_start = []
	days_lockdown_end = []
	day_school_close = []
	day_school_open = []

	############################## generation arrays ##############################

	# constant vector by age
	contageus_prob_age_vector = np.zeros(n_nodes)
	death_prob_age_vector = np.zeros(n_nodes)
	for i in range(n_nodes):
		contageus_prob_age_vector[i] = contagious_probability_age()
		death_prob_age_vector[i] = death_probability_age(network.node_list[i].age)

	# constant probabilities by day from data

	probability_day_death = column_extractor(parameter_day, 'Probability_of_Death')
	probability_day_death = probability_day_death/np.sum(probability_day_death)

	probability_day_incubation = column_extractor(parameter_day, 'Incubation_Period')
	probability_day_incubation = probability_day_incubation/np.sum(probability_day_incubation)

	infectivity_day = column_extractor(parameter_day, 'Probability_of_infecting')


	day_array = np.arange(0, len(infectivity_day)) + 1

	# vectors new at each loop
	infectivity_vector = np.zeros(n_nodes)
	infection_prob_vector = np.zeros(n_nodes)

	new_will_symptoms_vector = np.zeros(n_nodes)
	new_will_death_vector = np.zeros(n_nodes)
	new_infections_vector = np.zeros(n_nodes)

	new_recovery_vector = np.zeros(n_nodes)
	new_death_vector = np.zeros(n_nodes)
	new_contageus_vector = np.zeros(n_nodes)

	# days vector 
	day_from_infection_vector = np.zeros(n_nodes)
	day_first_symptoms_vector = np.zeros(n_nodes)
	day_death_vector = np.zeros(n_nodes)
	day_recovery_vector = np.zeros(n_nodes)

	# total vectors
	will_contag_vector = np.zeros(n_nodes)
	will_death_vector = np.zeros(n_nodes)

	infected_vector = np.zeros(n_nodes)

	dead_vector = np.zeros(n_nodes)
	contageus_vector = np.zeros(n_nodes)
	immune_vecor = np.zeros(n_nodes)

	# outputs array 
	infection_array = np.zeros(n_days)
	death_array = np.zeros(n_days)
	recovery_array = np.zeros(n_days)

	# other constants
	immune_fraction_healing = 0.8 # fraction of immune people after healing
	day_to_recovery = 14 # number of day to recovery

	##############################  initial infections ##############################
	actually_infected = 0 
	while actually_infected < n_initial_infected:
		random_index = np.random.randint(0, n_nodes)
		if infected_vector[random_index] == 0:
			infected_vector[random_index] = 1
			contageus_vector[random_index] = 1
			day_first_symptoms_vector[random_index] = 1
			day_recovery_vector[random_index] = 14

			actually_infected +=1

	##############################  iterate over tha days  ##############################
	
	for j in range(n_days):
		############################## Lockdowns ##############################

		array_weights, change = lockdowns(array_weights, j,days_lockdown_start, days_lockdown_end, day_school_close, day_school_open, original_array_weights)
		if change == True:
			total_network = (array_weights[0]*family_graph + array_weights[1]*worker_graph +
				array_weights[2]*essential_worker_graph + array_weights[3]*student_graph +
				array_weights[4]*random_graph + array_weights[5]*essential_random_graph)

		############################## the main loop ##############################
		# calculate probability of infection
		day_from_infection_vector = np.where(day_from_infection_vector < 21, day_from_infection_vector, 21)
		for i in range(n_nodes):
			infectivity_vector[i] = infectivity_day[int(day_from_infection_vector[i])] * contageus_vector[i]

		infection_prob_vector = np.matmul(total_network, infectivity_vector)
		#print(infection_prob_vector)

		# infections 
		new_infections_vector = abs(np.where((infection_prob_vector - np.random.rand(n_nodes)) > 0, 1, 0) * (1 - immune_vecor) * (1 - dead_vector) * (1 - infected_vector))
		new_will_symptoms_vector = np.where((new_infections_vector * contageus_prob_age_vector - np.random.rand(n_nodes)) > 0, 1, 0)
		new_will_death_vector = np.where((new_will_symptoms_vector * death_prob_age_vector/contageus_prob_age_vector - np.random.rand(n_nodes)) > 0, 1, 0)

		# update arrays 
		will_contag_vector += new_will_symptoms_vector
		will_death_vector += new_will_death_vector
		infected_vector += new_infections_vector

		# set days 
		day_from_infection_vector += infected_vector

		new_day_first_sympoms_vector = np.random.choice(day_array, n_nodes, p=probability_day_incubation) 
		day_first_symptoms_vector = day_first_symptoms_vector + new_day_first_sympoms_vector * new_will_symptoms_vector
		day_recovery_vector += (new_day_first_sympoms_vector + day_to_recovery) * new_infections_vector 
		day_death_vector += np.random.choice(day_array, n_nodes, p=probability_day_death)*new_will_death_vector

		# check days (is this day a day to recovery, show symptoms or death?)
		new_contageus_vector = np.where((day_first_symptoms_vector - day_from_infection_vector) == 0, 1, 0)*will_contag_vector
		new_death_vector = np.where((day_death_vector - day_from_infection_vector) == 0, 1, 0)*will_death_vector
		new_recovery_vector = np.where((day_recovery_vector - day_from_infection_vector) == 0, 1, 0)*infected_vector

		dead_vector += new_death_vector
		contageus_vector += new_contageus_vector
		immune_vecor += np.where(np.random.rand(n_nodes) < immune_fraction_healing, 1,0)*new_recovery_vector

		# update arrays

		infected_vector -= new_death_vector + new_recovery_vector
		will_contag_vector *= (1 - new_recovery_vector) * (1 - new_death_vector)
		day_from_infection_vector *= (1 - new_recovery_vector) * (1 - new_death_vector)
		day_recovery_vector *= (1 - new_recovery_vector) * (1 - new_death_vector)
		# putputs 
		infection_array[j] = np.sum(new_infections_vector)
		death_array[j] = np.sum(new_death_vector)
		recovery_array[j] = np.sum(new_recovery_vector)

		

		############################## vaccinations ##############################

		#network.node_list, matrix_vaccination[i,j] = vaccination(network.node_list, vaccinations_number_array[j])
	
	return infection_array, death_array, recovery_array

def main_algorithm_fast(n_simulations, n_days, n_nodes, n_initial_infected, array_network_parameters, array_weights):
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

	for i in tqdm(range(n_simulations)):
		matrix_infected[i], matrix_death[i], matrix_recovery[i] = simulation(n_days, n_nodes, n_initial_infected, array_network_parameters, array_weights)

	return matrix_infected, matrix_death, matrix_recovery, matrix_vaccination

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
	original_array_weights = array_weights

	matrix_infected = np.zeros((n_simulations, n_days))
	matrix_death = np.zeros((n_simulations, n_days))
	matrix_recovery = np.zeros((n_simulations, n_days))
	matrix_vaccination = np.zeros((n_simulations, n_days))

	for i in tqdm(range(n_simulations)):

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
			array_weights[4]*random_graph + array_weights[5]*essential_random_graph)

		network.node_list = initial_infect(n_initial_infected, network.node_list) #infect the intial nodes
		
		vaccinations_number_array = np.zeros(n_days) # vaccinations_array(n_days) #TO BE CHANGED 

		days_lockdown_start = []
		days_lockdown_end = []
		day_school_close = []
		day_school_open = []

		for j in range(n_days):
			array_weights, change = lockdowns(array_weights, j,days_lockdown_start, days_lockdown_end, day_school_close, day_school_open, original_array_weights)
			if change == True:
				total_network = (array_weights[0]*family_graph + array_weights[1]*worker_graph +
					array_weights[2]*essential_worker_graph + array_weights[3]*student_graph +
					array_weights[4]*random_graph + array_weights[5]*essential_random_graph)
			network.node_list, matrix_infected[i,j], matrix_death[i,j], matrix_recovery[i,j] = main_loop(network.node_list, total_network)
			network.node_list, matrix_vaccination[i,j] = vaccination(network.node_list, vaccinations_number_array[j])

	return matrix_infected, matrix_death, matrix_recovery, matrix_vaccination

def plot_results(name_file, matrix_infected, matrix_death, matrix_recovery, matrix_vaccination, plot_lockdowns = False):
	"""
	Plot the simulation results and save the plots
	Inputs: 
	matrix_infected      Matrix of infected (n_simulation X n_day)
	matrix_death         Matrix of death (n_simulation X n_day)
	matrix_recovery      Matrix of recovery (n_simulation X n_day)
	"""

	fig = plt.figure(figsize = (14, 8))

	################## Plot day data ##################
	
	#### Infections subplots ####
	ax1 = fig.add_subplot(241)
	ax1.set_title('Infections')

	# plot each simulation
	for i in range(n_simulations):
		ax1.plot(matrix_infected[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax1.plot(np.mean(matrix_infected, axis = 0), color = 'b') # plot the average

	#### Death subplots ####
	ax2 = fig.add_subplot(242)
	ax2.set_title('Deaths')

	# plot each simulation
	for i in range(n_simulations):
		ax2.plot(matrix_death[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax2.plot(np.mean(matrix_death, axis = 0), color = 'r') # plot the average

	#### Infections subplots ####
	ax3 = fig.add_subplot(243)
	ax3.set_title('Recoveries')

	# plot each simulation
	for i in range(n_simulations):
		ax3.plot(matrix_recovery[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax3.plot(np.mean(matrix_recovery , axis = 0), color = 'g') # plot the average

	#### plot vaccination ####
	ax4 = fig.add_subplot(244)
	ax4.set_title('Vaccinations')
	for i in range(n_simulations):
		ax4.plot(matrix_vaccination[i], color='black', linewidth=0.5,alpha = 0.5)
	ax3.plot(np.mean(matrix_vaccination , axis = 0), color = 'g') # plot the average

	################## Plot cumulative data ##################
	
	cumulative_infections = cumulative(matrix_infected)
	cumulative_deaths = cumulative(matrix_death)
	comulative_recoveries = cumulative(matrix_recovery)
	comulative_vaccinations = cumulative(matrix_vaccination)

	current_infected = cumulative_infections -cumulative_deaths - comulative_recoveries

	#### Infections subplots ####
	ax5 = fig.add_subplot(245)
	# plot each simulation
	for i in range(n_simulations):
		ax5.plot(cumulative_infections[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax5.plot(np.mean(cumulative_infections , axis = 0), color = 'b') # plot the average
	ax5.set_title('Comulative Infections')

	#### Death subplots ####
	ax6 = fig.add_subplot(246)

	# plot each simulation
	for i in range(n_simulations):
		ax6.plot(cumulative_deaths[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax6.plot(np.mean(cumulative_deaths , axis = 0), color = 'r') # plot the average
	ax6.set_title('Comulative Deaths')

	#### Infections subplots ####
	ax7 = fig.add_subplot(247)

	# plot each simulation
	for i in range(n_simulations):
		ax7.plot(comulative_recoveries[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax7.plot(np.mean(comulative_recoveries , axis = 0), color = 'g') # plot the average
	ax7.set_title('Comulative Recoveries')

	#### plot vaccination ####
	ax8 = fig.add_subplot(248)
	for i in range(n_simulations):
		ax8.plot(current_infected[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax8.plot(np.mean(current_infected, axis = 0), color='black')
	ax8.set_title('Current Infected')

	################## Plot lockdowns ##################
	if plot_lockdowns == True:
		list_subplots = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
		for j in range(8):
			for i in range(len(days_lockdown_start)):
				list_subplots[j].axvline(days_lockdown_start[i], color='r', linewidth=0.5,alpha = 0.5)
				list_subplots[j].axvline(days_lockdown_end[i], color='r', linewidth=0.5,alpha = 0.5)

			for i in range(len(day_school_close)):
				list_subplots[j].axvline(day_school_close[i], color='m', linewidth=0.5,alpha = 0.5)
				list_subplots[j].axvline(day_school_open[i], color='m', linewidth=0.5,alpha = 0.5)

	plt.savefig(name_file)
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

def cumulative(matrix):
	
	cumulative_matrix = np.zeros((matrix.shape[0], matrix.shape[1] + 1))
	for i in range(n_simulations):
		for j in range(n_days):
			cumulative_matrix[i,j + 1] = matrix[i,j] + cumulative_matrix[i,j]
	return cumulative_matrix

def filter_non_spread(matrix_infected, matrix_death, matrix_recovery, matrix_vaccination):
	n_non_spread = 0
	threshold = 0.8*n_days

	for i in range(n_simulations):
		n_zeros = np.sum(np.where(matrix_infected[i - n_non_spread] == 0, 1, 0))
		if n_zeros > threshold:
			matrix_infected = np.delete(matrix_infected, i - n_non_spread, 0)
			matrix_death = np.delete(matrix_death, i - n_non_spread, 0)
			matrix_recovery = np.delete(matrix_recovery, i - n_non_spread, 0)
			matrix_vaccination = np.delete(matrix_vaccination, i - n_non_spread, 0)
			n_non_spread += 1 

	return matrix_infected, matrix_death, matrix_recovery, matrix_vaccination, n_non_spread

if __name__ == "__main__":
	n_simulations = 10
	n_days = 90
	n_nodes = 5000 
	n_initial_infected = 8
	array_network_parameters = np.array([2,2,2,2,2])
	array_weights = np.array([0.7,0.1,0.1,0.1,0.2,0.2])


	# days of lockdown data
	days_lockdown_start = []
	days_lockdown_end = []
	day_school_close = []
	day_school_open = []

	# call the simulations and save
	matrix_infected, matrix_death, matrix_recovery, matrix_vaccination = main_algorithm_fast(n_simulations, n_days, n_nodes, n_initial_infected, array_network_parameters, array_weights)
	for i in range(n_simulations):
		matrix_infected[i,0] += n_initial_infected
	matrix_infected, matrix_death, matrix_recovery, matrix_vaccination, n_non_spread = filter_non_spread(matrix_infected, matrix_death, matrix_recovery, matrix_vaccination)
	print(n_non_spread)
	n_simulations -= n_non_spread
	plot_results("reults_witout_non_spread.pdf",matrix_infected, matrix_death, matrix_recovery, matrix_vaccination)
	save_results(matrix_infected, matrix_death, matrix_recovery, matrix_vaccination)










