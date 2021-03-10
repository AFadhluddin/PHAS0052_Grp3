
import numpy as np
from numpy import random 
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix
import pandas as pd
from tqdm import tqdm

from plots_functions import *
from save_files import * 

def filter_non_spread(matrix_infected, matrix_death, matrix_recovery, matrix_vaccination):
	n_simulations, n_days = np.shape(matrix_infected) 
	n_non_spread = 0
	threshold = 0.6*n_days

	for i in range(n_simulations):
		n_zeros = np.sum(np.where(matrix_infected[i - n_non_spread] == 0, 1, 0))
		if n_zeros > threshold:
			matrix_infected = np.delete(matrix_infected, i - n_non_spread, 0)
			matrix_death = np.delete(matrix_death, i - n_non_spread, 0)
			matrix_recovery = np.delete(matrix_recovery, i - n_non_spread, 0)
			matrix_vaccination = np.delete(matrix_vaccination, i - n_non_spread, 0)
			n_non_spread += 1 

	return matrix_infected, matrix_death, matrix_recovery, matrix_vaccination, n_non_spread

def import_results(name_infected_file, name_death_file, name_recovery_file, name_vaccination_file = False):
	
	matrix_infected = np.genfromtxt(name_infected_file, delimiter=',')[1:,1:]
	matrix_death = np.genfromtxt(name_death_file, delimiter=',')[1:,1:]
	matrix_recovery = np.genfromtxt(name_recovery_file, delimiter=',')[1:,1:]
	
	if name_vaccination_file == False:
		matrix_vaccination = np.zeros(np.shape(matrix_infected))
		pass
	else:
		matrix_vaccination = np.genfromtxt(name_vaccination_file, delimiter=',')[1:,1:]

	return matrix_infected, matrix_death, matrix_recovery, matrix_vaccination

if __name__ == "__main__":

	# import results
	matrix_infected, matrix_death, matrix_recovery, matrix_vaccination = import_results('infected_results.csv', 'death_results.csv', 'recovery_results.csv')
	n_simulations, n_days = np.shape(matrix_infected) 
	matrix_infected, matrix_death, matrix_recovery, matrix_vaccination, n_non_spread = filter_non_spread(matrix_infected, matrix_death, matrix_recovery, matrix_vaccination)
	print(n_non_spread)
	n_simulations -= n_non_spread
	plot_results("results.pdf", matrix_infected, matrix_death, matrix_recovery, matrix_vaccination, True)

