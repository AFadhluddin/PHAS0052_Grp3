
import numpy as np
from numpy import random 
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix
import pandas as pd
from tqdm import tqdm

from main_loop import *
from save_files import *
from plots_functions import *
from data_analysis import *

if __name__ == "__main__":
	
	n_simulations = 10
	n_days = 90
	n_nodes = 1000
	n_initial_infected = 8
	array_network_parameters = np.array([2,2,2,2,2])
	array_weights = np.array([0.7,0.1,0.1,0.1,0.2,0.2])


	# days of lockdown data
	days_lockdown_start = []
	days_lockdown_end = []
	day_school_close = []
	day_school_open = []

	# call the simulations and save
	matrix_infected, matrix_death, matrix_recovery, matrix_vaccination = main_algorithm(n_simulations, n_days, n_nodes, n_initial_infected, array_network_parameters, array_weights)

	# save the results 
	save_results(matrix_infected, matrix_death, matrix_recovery, matrix_vaccination)


