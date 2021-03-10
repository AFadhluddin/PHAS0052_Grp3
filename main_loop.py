import numpy as np
from numpy import random 
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix
import pandas as pd
from tqdm import tqdm
from utils_network_generation import *
from network_generation import *
from interventions import *

def simulation(n_days, n_nodes, n_initial_infected, array_network_parameters, array_weights):

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

	
	network = Network_Generation(n_nodes) # generate the network

	# creates all the subgraphs
	family_graph = network.family_network()
	deg_family = degree_distribution(nx.convert_matrix.from_numpy_matrix(
        family_graph))
		
	worker_graph = network.worker_network(array_network_parameters[0])
	deg_worker = degree_distribution(nx.convert_matrix.from_numpy_matrix(
        worker_graph))

	essential_worker_graph = network.essential_worker_network(array_network_parameters[1])
	deg_essential_worker= degree_distribution(nx.convert_matrix.from_numpy_matrix(
        essential_worker_graph))

	student_graph = network.student_network(array_network_parameters[2])
	deg_sudent = degree_distribution(nx.convert_matrix.from_numpy_matrix(
        student_graph))

	random_graph = network.random_social_network(array_network_parameters[3])
	deg_random = degree_distribution(nx.convert_matrix.from_numpy_matrix(
        random_graph))

	essential_random_graph = network.essential_random_network(array_network_parameters[4])
	deg_essential_random = degree_distribution(nx.convert_matrix.from_numpy_matrix(
        essential_random_graph))

	# weighted sum of the network 
	original_array_weights = array_weights
	total_network = (array_weights[0]*family_graph + array_weights[1]*worker_graph + 
		array_weights[2]*essential_worker_graph + array_weights[3]*student_graph + 
		array_weights[4]*random_graph + array_weights[5]*essential_random_graph)

	deg_total_network = degree_distribution(nx.convert_matrix.from_numpy_matrix(
        total_network))
	
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
		matrix_infected[i,0] += n_initial_infected	# add initial infections

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

	# Lists to hold degree distributions per simulation
	family_deg_list = []
	worker_deg_list = []
	essential_worker_deg_list = []
	student_deg_list = []
	random_deg_list = []
	essential_random_deg_list = []
	total_network_deg_list = []



	for i in tqdm(range(n_simulations)):

		network = Network_Generation(n_nodes) # generate the network

		# creates all the subgraphs
		family_graph = network.family_network()
		deg_family = degree_distribution(nx.convert_matrix.from_numpy_matrix(
			family_graph))
			
		worker_graph = network.worker_network(array_network_parameters[0])
		deg_worker = degree_distribution(nx.convert_matrix.from_numpy_matrix(
			worker_graph))

		essential_worker_graph = network.essential_worker_network(array_network_parameters[1])
		deg_essential_worker= degree_distribution(nx.convert_matrix.from_numpy_matrix(
			essential_worker_graph))

		student_graph = network.student_network(array_network_parameters[2])
		deg_sudent = degree_distribution(nx.convert_matrix.from_numpy_matrix(
			student_graph))

		random_graph = network.random_social_network(array_network_parameters[3])
		deg_random = degree_distribution(nx.convert_matrix.from_numpy_matrix(
			random_graph))

		essential_random_graph = network.essential_random_network(array_network_parameters[4])
		deg_essential_random = degree_distribution(nx.convert_matrix.from_numpy_matrix(
			essential_random_graph))

		# weighted sum of the network 
		original_array_weights = array_weights
		total_network = (array_weights[0]*family_graph + array_weights[1]*worker_graph + 
			array_weights[2]*essential_worker_graph + array_weights[3]*student_graph + 
			array_weights[4]*random_graph + array_weights[5]*essential_random_graph)

		deg_total_network = degree_distribution(nx.convert_matrix.from_numpy_matrix(
			total_network))

		# Apending degree distributions lists 
		family_deg_list.append(deg_family)
		worker_deg_list.append(deg_worker)
		essential_worker_deg_list.append(deg_essential_worker)
		student_deg_list.append(deg_sudent)
		random_deg_list.append(deg_random)
		essential_random_deg_list.append(deg_essential_random)
		total_network_deg_list.append(deg_total_network)

		# Saving them to csv files 
		np.savetxt("family_deg_dist.csv",  family_deg_list, delimiter =", ",  fmt ='% s')
		np.savetxt("worker_deg_dist.csv",  worker_deg_list, delimiter =", ",  fmt ='% s') 
		np.savetxt("essential_worker_deg_dist.csv",  essential_worker_deg_list, delimiter =", ",  fmt ='% s') 
		np.savetxt("student_deg_dist.csv",  student_deg_list, delimiter =", ",  fmt ='% s') 
		np.savetxt("random_deg_dist.csv",  random_deg_list, delimiter =", ",  fmt ='% s') 
		np.savetxt("essential_random_deg_dist.csv",  essential_random_deg_list, delimiter =", ",  fmt ='% s')
		np.savetxt("total_deg_dist.csv",  total_network_deg_list, delimiter =", ",  fmt ='% s')  

		network.node_list = initial_infect(n_initial_infected, network.node_list) #infect the intial nodes
		
		vaccinations_number_array = np.zeros(n_days) # vaccinations_array(n_days) #TO BE CHANGED 

		for j in range(n_days):
			array_weights, change = lockdowns(array_weights, j)
			if change == True:
				total_network = (array_weights[0]*family_graph + array_weights[1]*worker_graph +
					array_weights[2]*essential_worker_graph + array_weights[3]*student_graph +
					array_weights[4]*random_graph + array_weights[5]*essential_random_graph)
			network.node_list, matrix_infected[i,j], matrix_death[i,j], matrix_recovery[i,j] = main_loop(network.node_list, total_network)
			network.node_list, matrix_vaccination[i,j] = vaccination(network.node_list, vaccinations_number_array[j])
		matrix_infected[i,0] += n_initial_infected	
	
	return matrix_infected, matrix_death, matrix_recovery, matrix_vaccination





