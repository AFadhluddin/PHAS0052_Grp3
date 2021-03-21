import numpy as np
from numpy import random 
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix
import pandas as pd

from main_loop import *
from data_analysis import *

#############################

def initialise_matrix_parameters(n_parameters_combinations):
	matrix_parameters = np.array([
		[2,2,2,2,2],
		[2,2,2,1,1],
		[2,2,1,2,2],
		[2,2,1,1,1],
		[1,1,2,2,2],
		[1,1,2,1,1],
		[1,1,1,1,1]])
	return matrix_parameters

def main_parameters_iteration(matrix_parameters, n_parameters_combinations, n_simulations, n_days, n_nodes, n_initial_infected, array_weights):
	matrix_averages_infected = np.zeros((n_parameters_combinations, n_days))
	matrix_averages_death = np.zeros((n_parameters_combinations, n_days))
	matrix_averages_recovery = np.zeros((n_parameters_combinations, n_days))
	non_spread_array = np.zeros(n_parameters_combinations)

	for i in range(n_parameters_combinations):
		array_network_parameters = matrix_parameters[i]
		days_lockdown_start = []
		days_lockdown_end = []
		day_school_close = []
		day_school_open = []
		matrix_infected, matrix_death, matrix_recovery, matrix_vaccination = main_algorithm(n_simulations, n_days, n_nodes, n_initial_infected, array_network_parameters, array_weights)
		matrix_infected, matrix_death, matrix_recovery, matrix_vaccination, non_spread_array[i] = filter_non_spread(matrix_infected, matrix_death, matrix_recovery, matrix_vaccination)

		matrix_averages_infected[i] = np.mean(matrix_infected, axis = 0)
		matrix_averages_death[i] = np.mean(matrix_death, axis = 0)
		matrix_averages_recovery[i] = np.mean(matrix_recovery, axis = 0)

	return matrix_averages_infected, matrix_averages_death, matrix_averages_recovery, non_spread_array

if __name__ == "__main__":

	n_parameters_combinations = 7

	n_simulations = 40
	n_days = 100
	n_nodes = 10000 
	n_initial_infected = 10
	array_weights = np.array([0.7,0.1,0.1,0.1,0.2,0.2]) #from data
	original_array_weights = array_weights

	# days of lockdown data

	matrix_parameters = initialise_matrix_parameters(n_parameters_combinations)
	matrix_averages_infected, matrix_averages_death, matrix_averages_recovery, non_spread_array = main_parameters_iteration(matrix_parameters, n_parameters_combinations, n_simulations, n_days, n_nodes, n_initial_infected, array_weights)
	save_results_parameters(matrix_parameters, matrix_averages_infected, matrix_averages_death, matrix_averages_recovery, non_spread_array)




