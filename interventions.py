import numpy as np
from numpy import random 
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix
import pandas as pd



def lockdowns(array_weights, day, days_lockdown_start, days_lockdown_end, day_school_close, day_school_open, original_array_weights):
	"""
	Check if there is a lockdown
	Inputs:
	array_weights   Array of the weights of the graph
	day             Day of the simulation
	Output: array_weights
	"""
	updeated_array_weights = array_weights
	change = False
	# check if the day is an important day 
	if day in days_lockdown_start:
		updeated_array_weights[1] = 0
		updeated_array_weights[4] = 0
		change = True

	if day in days_lockdown_end:
		updeated_array_weights[1] = original_array_weights[1]
		updeated_array_weights[4] = original_array_weights[4]
		change = True

	if day in day_school_close:
		updeated_array_weights[3] = 0
		change = True

	if day in day_school_open:
		updeated_array_weights[3] = updeated_array_weights[3]
		change = True

	return  updeated_array_weights, change

def vaccination(node_list, n_vaccination):
	"""
	Vaccinate n_vaccination number of nodes
	Inputs: 
	node_list        List of nodes
	n_vaccination    Number of daily vaccination
	Return: node_list
	"""
	actualy_vaccinated = 0
	while actualy_vaccinated < n_vaccination:
		random_index = np.random.randint(0, len(node_list))
		if node_list[random_index].vaccinated == False:
			node_list[random_index].vaccinate()
			actualy_vaccinated +=1
	return node_list, actualy_vaccinated

def vaccinations_array(n_days):
	"""
	Generate the array of vaccination
	Inputs:
	n_days         Number of days in the simulation
	Return: vaccinations_number_array
	"""
	vaccinations_number_array = np.zeros(n_days)
	vaccinations_number_array[-int(n_days/2):] = 1
	return vaccinations_number_array


