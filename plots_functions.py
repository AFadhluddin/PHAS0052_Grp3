
import numpy as np
from numpy import random 
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix
import pandas as pd
from tqdm import tqdm

def cumulative(matrix):
	
	n_simulations, n_days = np.shape(matrix) 
	cumulative_matrix = np.zeros((matrix.shape[0], matrix.shape[1] + 1))
	for i in range(n_simulations):
		for j in range(n_days):
			cumulative_matrix[i,j + 1] = matrix[i,j] + cumulative_matrix[i,j]
	return cumulative_matrix
	
def plot_results(name_file, matrix_infected, matrix_death, matrix_recovery, matrix_vaccination, plot_lockdowns = False):
	"""
	Plot the simulation results and save the plots
	Inputs: 
	matrix_infected      Matrix of infected (n_simulation X n_day)
	matrix_death         Matrix of death (n_simulation X n_day)
	matrix_recovery      Matrix of recovery (n_simulation X n_day)
	"""
	n_simulations, n_days = np.shape(matrix_infected) 

	fig = plt.figure(figsize = (14, 8))

	################## Plot day data ##################
	
	#### Infections subplots ####
	ax1 = fig.add_subplot(241)
	ax1.set_title('Infections')
	for i in range(n_simulations):
		ax1.plot(matrix_infected[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax1.plot(np.mean(matrix_infected, axis = 0), color = 'b') # plot the average

	#### Death subplots ####
	ax2 = fig.add_subplot(242)
	ax2.set_title('Deaths')
	for i in range(n_simulations):
		ax2.plot(matrix_death[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax2.plot(np.mean(matrix_death, axis = 0), color = 'r') # plot the average

	#### Infections subplots ####
	ax3 = fig.add_subplot(243)
	ax3.set_title('Recoveries')
	for i in range(n_simulations):
		ax3.plot(matrix_recovery[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax3.plot(np.mean(matrix_recovery , axis = 0), color = 'g') # plot the average

	#### plot vaccination ####
	ax4 = fig.add_subplot(244)
	ax4.set_title('Vaccinations')
	for i in range(n_simulations):
		ax4.plot(matrix_vaccination[i], color='black', linewidth=0.5,alpha = 0.5)
	ax3.plot(np.mean(matrix_vaccination , axis = 0), color = 'black') # plot the average

	################## Plot cumulative data ##################
	
	cumulative_infections = cumulative(matrix_infected)
	cumulative_deaths = cumulative(matrix_death)
	comulative_recoveries = cumulative(matrix_recovery)
	comulative_vaccinations = cumulative(matrix_vaccination)

	current_infected = cumulative_infections -cumulative_deaths - comulative_recoveries

	#### Infections subplots ####
	ax5 = fig.add_subplot(245)
	for i in range(n_simulations):
		ax5.plot(cumulative_infections[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax5.plot(np.mean(cumulative_infections , axis = 0), color = 'b') # plot the average
	ax5.set_title('Comulative Infections')

	#### Death subplots ####
	ax6 = fig.add_subplot(246)
	for i in range(n_simulations):
		ax6.plot(cumulative_deaths[i], color='grey', linewidth=0.5,alpha = 0.5)
	ax6.plot(np.mean(cumulative_deaths , axis = 0), color = 'r') # plot the average
	ax6.set_title('Comulative Deaths')

	#### Infections subplots ####
	ax7 = fig.add_subplot(247)
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
	
def plot_results_parameters(name_file, matrix_averages_infected, matrix_averages_death, matrix_averages_recovery, matrix_averages_vaccination, plot_lockdowns = False):
	"""
	Plot the simulation results and save the plots
	Inputs: 
	matrix_infected      Matrix of infected (n_simulation X n_day)
	matrix_death         Matrix of death (n_simulation X n_day)
	matrix_recovery      Matrix of recovery (n_simulation X n_day)
	"""

	n_parameters_combinations, n_days = np.shape(matrix_averages_infected) 

	fig = plt.figure(figsize = (14, 8))
	
	################## Plot day data ##################

	######## Infections subplots ########
	ax1 = fig.add_subplot(241)
	ax1.set_title('Infections')
	for i in range(n_parameters_combinations):
		ax1.plot(matrix_averages_infected[i], color='b', linewidth=1,alpha = 0.8)
	# ax1.plot(data_infections, color='m', label='data')
	# ax1.legend()

	######## Death subplots ########
	ax2 = fig.add_subplot(242)
	ax2.set_title('Deaths')
	for i in range(n_parameters_combinations):
		ax2.plot(matrix_averages_death[i], color='r', linewidth=1,alpha = 0.8)
	# ax2.plot(n_nodes, color='m', label='data')
	# ax2.legend()

	######## Infections subplots ########
	ax3 = fig.add_subplot(243)
	ax3.set_title('Recoveries')
	for i in range(n_parameters_combinations):
		ax3.plot(matrix_averages_recovery[i], color='g', linewidth=1,alpha = 0.8)

	######## Infections subplots ########
	ax4 = fig.add_subplot(244)
	ax4.set_title('Vaccinations')
	for i in range(n_parameters_combinations):
		ax4.plot(matrix_averages_vaccination[i], color='black', linewidth=1,alpha = 0.8)
	
	################## Plot cumulative data ##################

	cumulative_infections = cumulative(matrix_averages_infected)
	cumulative_deaths = cumulative(matrix_averages_death)
	comulative_recoveries = cumulative(matrix_averages_recovery)
	comulative_vaccinations = cumulative(matrix_averages_vaccination)

	current_infected = cumulative_infections -cumulative_deaths - comulative_recoveries

	ax5 = fig.add_subplot(245)
	ax5.set_title('Comulative Infections')
	for i in range(n_parameters_combinations):
		ax5.plot(cumulative_infections[i], color='b', linewidth=1,alpha = 0.8)

	######## Death subplots ########
	ax6 = fig.add_subplot(246)
	ax6.set_title('Comulative Deaths')
	for i in range(n_parameters_combinations):
		ax6.plot(cumulative_deaths[i], color='r', linewidth=1,alpha = 0.8)

	######## Infections subplots ########
	ax7 = fig.add_subplot(247)
	ax7.set_title('Comulative Recoveries')
	for i in range(n_parameters_combinations):
		ax7.plot(comulative_recoveries[i], color='g', linewidth=1,alpha = 0.8)

	######## Infections subplots ########
	ax8 = fig.add_subplot(248)
	ax8.set_title('Current Infected')
	for i in range(n_parameters_combinations):
		ax8.plot(current_infected[i], color='black', linewidth=1,alpha = 0.8)

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

	plt.savefig('results_parameters.pdf')
	return

