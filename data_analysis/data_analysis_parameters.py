import numpy as np
from numpy import random 
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix
import pandas as pd
from tqdm import tqdm

from data_analysis import * 
from plots_functions import *
from save_files import * 
from data_importing_tool import *



if __name__ == "__main__":

	######## Import data ########

	# import results parameters
	matrix_parameters_infected, matrix_parameters_death, matrix_parameters_recovery, matrix_parameters_vaccination = import_results('infected_parameters_results.csv', 'death_parameters_results.csv', 'recovery_parameters_results.csv')
	n_parameters, n_days = np.shape(matrix_parameters_infected)

	
	# import data from governament 
	gov_data = parameter_importer(pathandfile('gov_data_fin.csv'))

	data_infections = column_extractor(gov_data, 'percentage_new_case')[65:65+n_days]
	data_death = column_extractor(gov_data, 'percentage_new_death')[65:65+n_days]
	
	plot_results_parameters("results_parameters.pdf", matrix_parameters_infected, matrix_parameters_death, matrix_parameters_recovery, matrix_parameters_vaccination)
