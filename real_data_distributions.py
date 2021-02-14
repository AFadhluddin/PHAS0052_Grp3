import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import networkx as nx
from networkx import convert_matrix
import pathlib
from data_importing_tool import *
import pandas as pd

########### Imports the data frames ###########

#path = 'C:{}/'.format(pathlib.Path(__file__).parent.absolute())
file_name = 'parameters_by_age.csv'
#parameter_by_age = parameter_importer(path, file_name)
#parameter_by_age = csv_to_dataframe(path)

parameter_by_age = pd.read_csv(file_name)
parameter_by_age['percentage_population'] = parameter_by_age['populationUK']/parameter_by_age['populationUK'].sum()


########### Generation of the Node ###########

def set_age():
	"""
	Calulate the age of the person from the real-data distribution 
	Return: Age_band    band age of the node (i.e. number of row in the df)
	""" 
	age_array = column_extractor(parameter_by_age, 'percentage_population')
	age_band = 0
	probability_span = 0
	random_number = np.random.rand()
	for i in range(len(age_array)):
		if random_number > probability_span and random_number < (probability_span + age_array[i]):
			age_band = i
		probability_span += age_array[i]
	return age_band 

def set_job(age_band):

	job = 'worker'

	# set students
	if age_band < 4:
		job = 'student'

	# set retired 
	elif age_band > 15:
		job = 'retired'
	
	else:
		# set unemplyed
		working_fractions = 1 - column_extractor(parameter_by_age, 'unemployment_rate')
		if np.random.rand() > working_fractions[age_band]:
			job = 'unemployed'
		
		# set essential workers
		essential_workers_fraction = column_extractor(parameter_by_age, 'unemployment_rate')
		if job == 'worker':
			if np.random.rand() < essential_workers_fraction[age_band]:
				job = 'essential_worker'

	return job

########### Generation of the Network ###########

def set_family_size():
	"""
	Calulate the size of a family from the real-data distribution 
	Return: family_size    size of the family 
	""" 
	family_size_array = np.array([8197000, 9609000, 4287000, 3881000, 1254000, 597000])/27824000
	family_size = 0
	probability_span = 0
	random_number = np.random.rand()
	for i in range(len(family_size_array)):
		if random_number > probability_span and random_number < (probability_span + family_size_array[i]):
			family_size = i+1
		probability_span += family_size_array[i]
	return family_size 


########### Main loop ###########


def initial_infect(n_infected, nodes_list):
	"""
	Randomly infects n_infected nodes
	Inputs:
	n_infected      Number of infected
	nodes_list		List of nodes of the graph
	Return: Updated node_list
	"""
	for i in range(n_infected):
		nodes_list[np.random.randint(0, len(nodes_list))].infect()
	return nodes_list


def death_probability(age, status, days_from_infection):
	"""
	calulate the death probability
	Inputs:
	age                Age of the node
	status             Status of the node
	days_from_infection Days since the infection
	Return: Probability of death
	"""
	prob = 0 
	if status == 'infected':
		prob = 0.01#*np.random.normal(8, 3)
	return prob

def heal_probability(age, status, days_from_infection):
	"""
	calulate the healing probability
	Inputs:
	age                Age of the node
	status             Status of the node
	days_from_infection Days since the infection
	Return: Probability of healing
	"""
	prob = 0 
	if status == 'infected':
		prob = days_from_infection/20
	return prob

def contagious_probability(age, status, days_from_infection):
	"""
	calulate the probability of becoming contageus
	Inputs:
	age                Age of the node
	status             Status of the node
	days_from_infection Days since the infection
	Return: Probability of death
	"""
	prob = 0 
	if status == 'infected':
		prob = 0.1#*np.random.normal(5, 7)
	return prob



def family_size_distribution():
	"""
	Calulate the size of a family from the real-data distribution
	Return:      family_size     size of the family
	"""

	# calulate the family size by the real data distribution 
	family_size = np.random.normal(4, 2)
	if family_size < 1:
		family_size = 1
	return int(family_size)



def calulate_infectivity(age, day_from_infection):
    """
    Calculate the infectivity of the person
    """
    infectivity = 3*age*day_from_infection # this function has to be look into
    return infectivity

