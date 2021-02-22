import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import networkx as nx
from networkx import convert_matrix
import pathlib
import pandas as pd

from data_importing_tool import *

########### Assumptions #########
# a person which die it shows symptoms


########### Imports the data frames ###########

#path = 'C:{}/'.format(pathlib.Path(__file__).parent.absolute())
#parameter_by_age = parameter_importer(path, file_name)
#parameter_by_age = csv_to_dataframe(path)



# create a dataframe for parameter where the index is age
parameter_age = parameter_importer('parameters_age.csv')

# this is to form a new column 'percentage_population' that shows the percentage of the population in certain age range, in case you need it
parameter_age['percentage_population'] = parameter_age['population_UK']/parameter_age['population_UK'].sum()


# create a dataframe for parameter where the index is day
parameter_day = parameter_importer('parameters_day.csv')



#parameter_age = pd.read_csv('parameters_age.csv')
#parameter_age['percentage_population'] = parameter_age['population_UK']/parameter_age['population_UK'].sum()

#parameter_day = pd.read_csv('parameters_day.csv')

########### Generation of the Node ###########

def set_age():
	"""
	Calulate the age of the person from the real-data distribution 
	Return: Age_band    band age of the node (i.e. number of row in the df)
	""" 
	age_array = column_extractor(parameter_age, 'percentage_population')
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
		working_fractions = 1 - column_extractor(parameter_age, 'unemployment_rate')
		if np.random.rand() > working_fractions[age_band]:
			job = 'unemployed'
		
		# set essential workers
		essential_workers_fraction = column_extractor(parameter_age, 'unemployment_rate')
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


def death_probability_age(age):
	"""
	Calulate the overall probability of death
	Inputs:
	age                Age of the node
	status             Status of the node
	days_from_infection Days since the infection
	Return: Probability of death
	"""
	probability_age = column_extractor(parameter_age, 'infection_fatality_ratio')
	prob = probability_age[age]
	return prob

def hosp_probability_age(age):
	"""
	Calulate the overall probability of hospitalisation
	Inputs:
	age                Age of the node
	status             Status of the node
	days_from_infection Days since the infection
	Return: Probability of death
	"""
	prob = 0.1
	return prob


def contagious_probability_age(age):
	"""
	Calulate the probability of becoming contageus
	Inputs:
	age                Age of the node
	status             Status of the node
	days_from_infection Days since the infection
	Return: Probability of death
	"""
	prob = 0.5#*np.random.normal(5, 7)
	return prob

def day_of_death():
	"""
	Returns the day of death from the day of infection
	"""
	probability_day = column_extractor(parameter_day, 'Probability_of_Death')
	probability_day = probability_day/np.sum(probability_day)
	day = 0
	probability_span = 0
	random_number = np.random.rand()
	for i in range(len(probability_day)):
		if random_number >= probability_span and random_number < (probability_span + probability_day[i]):
			day = i
		probability_span += probability_day[i]
	return day

def day_of_first_symptoms():
	"""
	Returns the day of the first symptoms from the day of infection
	"""
	probability_day = column_extractor(parameter_day, 'Incubation_Period')
	probability_day = probability_day/np.sum(probability_day)
	day = 0
	probability_span = 0
	random_number = np.random.rand()
	for i in range(len(probability_day)):
		if random_number >= probability_span and random_number < (probability_span + probability_day[i]):
			day = i
		probability_span += probability_day[i]
	return day 

def infectivity_factor(day_from_infection):
    """
    Calculate the infectivity of the person
    """
    infectivity_age = column_extractor(parameter_day, 'Probability_of_infecting')
    if day_from_infection < len(infectivity_age):
    	infectivity = infectivity_age[day_from_infection]
    else:
    	infectivity = 0 
    return infectivity


