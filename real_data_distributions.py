import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import networkx as nx
from networkx import convert_matrix
import pathlib
import pandas as pd
import os
import glob

from data_importing_tool import *

########### Assumptions #########
# a person which die it shows symptoms


########### Imports the data frames ###########

#path = 'C:{}/'.format(pathlib.Path(__file__).parent.absolute())
#parameter_by_age = parameter_importer(path, file_name)
#parameter_by_age = csv_to_dataframe(path)

# create a dataframe for parameter where the index is age
parameter_age = parameter_importer(pathandfile('parameters_age.csv'))

# this is to form a new column 'percentage_population' that shows the percentage of the population in certain age range, in case you need it
parameter_age['percentage_population'] = parameter_age['population_UK']/parameter_age['population_UK'].sum()


# create a dataframe for parameter where the index is day
parameter_day = parameter_importer(pathandfile('parameters_day.csv'))


#exctract vectors

age_array = column_extractor(parameter_age, 'percentage_population')
working_fractions = column_extractor(parameter_age, 'employment_rate') 
essential_workers_fraction = column_extractor(parameter_age, 'proportion_of_key_workers') 


probability_day_death = column_extractor(parameter_day, 'Probability_of_Death')
probability_day_death = probability_day_death/np.sum(probability_day_death)

probability_day_incubation = column_extractor(parameter_day, 'Incubation_Period')
probability_day_incubation = probability_day_incubation/np.sum(probability_day_incubation)

death_probability_age_array = column_extractor(parameter_age, 'infection_fatality_ratio')

infectivity_age = column_extractor(parameter_day, 'Probability_of_infecting')

########### Generation of the Node ###########

def set_age():
	"""
	Calulate the age of the person from the real-data distribution 
	Return: Age_band    band age of the node (i.e. number of row in the df)
	""" 
	age_band = 0
	probability_span = 0
	random_number = np.random.rand()
	for i in range(len(age_array)):
		if random_number > probability_span and random_number < (probability_span + age_array[i]):
			age_band = i
		probability_span += age_array[i]
	return age_band 

def set_job(age_band):
	"""
	Set the job of the node from the real data
	Inputs:
	age_band       Age band of the node
	Outputs: job of the node
	"""

	job = 'worker'

	# set students
	if age_band < 4:
		job = 'student'

	# set retired 
	elif age_band > 15:
		job = 'retired'
	
	else:
		# set unemplyed
		if np.random.rand() > working_fractions[age_band]:
			job = 'unemployed'
		
		# set essential workers
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
	actually_infected = 0 
	while actually_infected < n_infected:
		random_index = np.random.randint(0, len(nodes_list))
		if nodes_list[random_index].status == 'healthy':
			nodes_list[random_index].infect()
			actually_infected +=1
	return nodes_list


def death_probability_age(age):
	"""
	Calulate the overall probability of death
	Inputs:
	age                Age of the node
	Return: Probability of death
	"""
	prob = death_probability_age_array[age]
	return prob

def hosp_probability_age(age):
	"""
	Calulate the overall probability of hospitalisation
	Inputs:
	age                Age of the node
	Return: Probability of hospitalisation
	"""
	prob = 0.1
	return prob


def contagious_probability_age():
	"""
	Calulate the probability of becoming contageus
	Return: Probability of show symptoms
	"""
	prob = 0.5#*np.random.normal(5, 7)
	return prob

def day_of_death():
	"""
	Returns the day of death from the day of infection
	"""
	day = 0
	probability_span = 0
	random_number = np.random.rand()
	for i in range(len(probability_day_death)):
		if random_number >= probability_span and random_number < (probability_span + probability_day_death[i]):
			day = i
			break
		probability_span += probability_day_death[i]
	return day

def day_of_first_symptoms():
	"""
	Returns the day of the first symptoms from the day of infection
	"""
	day = 0
	probability_span = 0
	random_number = np.random.rand()
	for i in range(len(probability_day_incubation)):
		if random_number >= probability_span and random_number < (probability_span + probability_day_incubation[i]):
			day = i
			break
		probability_span += probability_day_incubation[i]
	return day 

def infectivity_factor(day_from_infection):
    """
    Calculate the infectivity of the person
    """
    if day_from_infection < len(infectivity_age):
    	infectivity = infectivity_age[day_from_infection]
    else:
    	infectivity = 0 
    return infectivity


