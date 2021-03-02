import numpy as np
from numpy import random 
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix
import pandas as pd

from main_loop import *



#############################


def initialise_matrix_parameters(n_parameters_combinations):
	matrix_parameters = np.array([
		[2,2,2,2,2],
		[2,2,2,1,1],
		[2,2,1,1,1],
		[1,1,1,1,1]])
	return matrix_parameters

def main_parameters_iteration(matrix_parameters, n_parameters_combinations, n_simulations, n_days, n_nodes, n_initial_infected, array_weights):
	matrix_averages_infected = np.zeros((n_parameters_combinations, n_days))
	matrix_averages_death = np.zeros((n_parameters_combinations, n_days))
	matrix_averages_recovery = np.zeros((n_parameters_combinations, n_days))

	for i in range(n_parameters_combinations):
		array_network_parameters = matrix_parameters[i]
		days_lockdown_start = []
		days_lockdown_end = []
		day_school_close = []
		day_school_open = []
		matrix_infected, matrix_death, matrix_recovery, matrix_vaccination = main_algorithm(n_simulations, n_days, n_nodes, n_initial_infected, array_network_parameters, array_weights)

		matrix_averages_infected[i] = np.mean(matrix_infected, axis = 0)
		matrix_averages_death[i] = np.mean(matrix_death, axis = 0)
		matrix_averages_recovery[i] = np.mean(matrix_recovery, axis = 0)

	return matrix_averages_infected, matrix_averages_death, matrix_averages_recovery

def plot_results_parameters(matrix_averages_infected, matrix_averages_death, matrix_averages_recovery, n_nodes):
	"""
	Plot the simulation results and save the plots
	Inputs: 
	matrix_infected      Matrix of infected (n_simulation X n_day)
	matrix_death         Matrix of death (n_simulation X n_day)
	matrix_recovery      Matrix of recovery (n_simulation X n_day)
	"""

	fig = plt.figure(figsize = (14, 8))
	
	######## Infections subplots ########
	ax1 = fig.add_subplot(131)
	ax1.set_title('Infections')

	# plot each simulation
	for i in range(n_parameters_combinations):
		ax1.plot(matrix_averages_infected[i]/n_nodes, color='b', linewidth=1,alpha = 0.8)
	ax1.plot(data_infections, color='m', label='data')
	ax1.legend()
	# plot lockdowns
	#for i in range(len(days_lockdown_start)):
	#	ax1.axvline(days_lockdown_start[i], color='r', linewidth=0.5,alpha = 0.5)
	#	ax1.axvline(days_lockdown_end[i], color='r', linewidth=0.5,alpha = 0.5)

	#for i in range(len(day_school_close)):
	#	ax1.axvline(day_school_close[i], color='m', linewidth=0.5,alpha = 0.5)
	#	ax1.axvline(day_school_open[i], color='m', linewidth=0.5,alpha = 0.5)

	######## Death subplots ########
	ax2 = fig.add_subplot(132)
	ax2.set_title('Deaths')

	# plot each simulation
	for i in range(n_parameters_combinations):
		ax2.plot(matrix_averages_death[i]/n_nodes, color='r', linewidth=1,alpha = 0.8)
	ax2.plot(n_nodes*data_death, color='m', label='data')
	ax2.legend()
	
	# plot lockdowns
	#for i in range(len(days_lockdown_start)):
	#	ax2.axvline(days_lockdown_start[i], color='r', linewidth=0.5,alpha = 0.5)
	#	ax2.axvline(days_lockdown_end[i], color='r', linewidth=0.5,alpha = 0.5)

	#for i in range(len(day_school_close)):
	#	ax2.axvline(day_school_close[i], color='m', linewidth=0.5,alpha = 0.5)
	#	ax2.axvline(day_school_open[i], color='m', linewidth=0.5,alpha = 0.5)

	######## Infections subplots ########
	ax3 = fig.add_subplot(133)
	ax3.set_title('Recoveries')

	# plot each simulation
	for i in range(n_parameters_combinations):
		ax3.plot(matrix_averages_recovery[i]/n_nodes, color='g', linewidth=1,alpha = 0.8)

	# plot lockdowns
	#for i in range(len(days_lockdown_start)):
	#	ax3.axvline(days_lockdown_start[i], color='r', linewidth=0.5,alpha = 0.5)
	#	ax3.axvline(days_lockdown_end[i], color='r', linewidth=0.5,alpha = 0.5)

	#for i in range(len(day_school_close)):
	#	ax3.axvline(day_school_close[i], color='m', linewidth=0.5,alpha = 0.5)
	#	ax3.axvline(day_school_open[i], color='m', linewidth=0.5,alpha = 0.5)

	plt.savefig('Results_parameters.pdf')
	return

if __name__ == "__main__":

	n_parameters_combinations = 4

	n_simulations = 10
	n_days = 90
	n_nodes = 1000 
	n_initial_infected = 5
	array_weights = np.array([0.7,0.1,0.1,0.1,0.2,0.2]) #from data
	original_array_weights = array_weights

	######## Import data ########

	gov_data = parameter_importer(pathandfile('gov_data_fin.csv'))

	data_infections = column_extractor(gov_data, 'percentage_new_case')[65:65+n_days]
	data_death = column_extractor(gov_data, 'percentage_new_death')[65:65+n_days]

	# days of lockdown data

	matrix_parameters = initialise_matrix_parameters(n_parameters_combinations)
	matrix_averages_infected, matrix_averages_death, matrix_averages_recovery = main_parameters_iteration(matrix_parameters, n_parameters_combinations, n_simulations, n_days, n_nodes, n_initial_infected, array_weights)
	plot_results_parameters(matrix_averages_infected, matrix_averages_death, matrix_averages_recovery, n_nodes)


