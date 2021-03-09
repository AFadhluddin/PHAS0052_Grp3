
import numpy as np
from numpy import random 
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix
import pandas as pd
from tqdm import tqdm


def save_results_parameters(matrix_parameters, matrix_averages_infected, matrix_averages_death, matrix_averages_recovery, non_spread_array):
	"""
	Saves the simulation results in csv files 
	Inputs: 
	matrix_parameters    Matrix of parameters 
	matrix_infected      Matrix of infected (n_simulation X n_day)
	matrix_death         Matrix of death (n_simulation X n_day)
	matrix_recovery      Matrix of recovery (n_simulation X n_day)
	"""
	# convert to dataframe
	df_paramters = pd.DataFrame(data=matrix_parameters)
	df_matrix_infected = pd.DataFrame(data=matrix_averages_infected)
	df_matrix_death = pd.DataFrame(data=matrix_averages_death)
	df_matrix_recovery = pd.DataFrame(data=matrix_averages_recovery)
	df_non_spread_array = pd.DataFrame(data=non_spread_array)

	# save 
	df_paramters.to_csv('parameters_matrix.csv')
	df_matrix_infected.to_csv("infected_parameters_results.csv")
	df_matrix_death.to_csv("death_parameters_results.csv")
	df_matrix_recovery.to_csv("recovery_parameters_results.csv")
	df_non_spread_array.to_csv("non_spread_array.csv")
	
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
	df_matrix_death.to_csv("death_results.csv")
	df_matrix_recovery.to_csv("recovery_results.csv")	
	#df_matrix_vaccination.to_csv("vaccination_results.csv")
	
	return