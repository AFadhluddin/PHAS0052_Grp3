# This is a test

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import networkx as nx
from networkx import convert_matrix

################### Constants ###################
# emplyment_fractions
# essential_workers_fraction
# immune_fraction_healing 
# immune_fraction_vac

def calulate_infectivity(age, day_from_infection):
	"""
	Calculate the infectivity of the person
	"""
	infectivity = 3*age*day_from_infection # this function has to be look into
	return infectivity

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

def age_distribution():
	"""
	Calulate the age of the person from the real-data distribution
	Return: Age_band    band age of the node (eg. 2 between 21 and 30 years old)
	"""
	# calulate the age by the real data distribution 
	age = np.random.normal(45, 20)
	# calulate the age band
	age_band = int(age/10 - (int(age/10) - age/10))
	if age_band > 9: # for >90 of age are all the same 
		age_band = 9
	return age_band

class Node:
	"""
	A node represents a person.
	"""

	def __init__(self, age, status='healthy'):
		"""
		Method which cunstruct the object. It sets the intial parameters.
		Inputs:
		age        Age band of the node
		"""

		# fraction of eployed by age band
		emplyment_fractions = np.array([0., 0., 0.5, 0.5, 0.8, 0.8, 0.8, 0.2, 0.2, 0.1,])
		essential_workers_fraction = 0.2

		self.age = age

		# Usally at the begining everyone is healthy, set inital conditions
		self.status = 'healthy'
		self.contagious = False
		self.days_from_infection = 0
		self.immune = False
		self.vaccinated = False

		# randomly decide if the person works or not
		self.job = 'worker'
		if np.random.rand() > emplyment_fractions[self.age]:
			self.job = 'unemplyed'

		# check if it is a student 
		if self.age < 2:
			self.job = 'student'

		# randomly decide if it is an essential worker
		if np.random.rand() < essential_workers_fraction:
			self.job = 'essential_worker'

	############### Methods of Updating Status ###############

	def infect(self):
		"""
		Infect the node if it's not immune
		"""
		if self.immune == False:
			self.status = 'infected'
		else:
			pass

	def kill(self):
		"""
		Kill the node
		"""
		self.status = 'dead'
		self.contagious = False

	def heal(self):
		"""
		Heal the node
		"""
		self.status = 'healthy'

		# everything set back to normal 
		self.tested = False
		self.contagious = False
		self.day_from_infection = 0

		# the node can become immune
		immune_fraction_healing = 0.8 # fraction of immune people after healing
		if random.rand() <= immune_fraction_healing:
			self.immune = True

	############### Action on the node ###############


	def update_days_from_infection(self):
		"""
		Update the days from infection
		"""
		if self.status == 'infected':
			self.days_from_infection += 1

	def set_contagious(self):
		"""
		function which sets the node as contagious
		"""
		self.contagious = True

	def test(self):
		"""
		Test the node
		"""
		self.test = True

	def vaccinate(self):
		"""
		Vaccinate the node
		"""
		self.vaccinated = True
		immune_fraction_vac = 0.9 # fraction of immune people after vacination
		if random.rand() <= immune_fraction_vac:
			self.immune = True

	############### Returns Properties of the Node ###############

	def return_infectivity(self):
		"""
		Calulate the infectivity of the node
		Return: number between 0 and 1, measuring the probability of infecting other linked nodes
		"""
		infectivity = 0

		if self.contagious == False:
			infectivity = 0
			return infectivity

		# case of being infected
		else:
			infectivity = calulate_infectivity(self.age, self.day_from_infection)
			return infectivity

	def print_node(self):
		"""
		Print the node age, job, and status
		"""
		print("the person is {0} years old, {1} and {2}".format(self.age*10, self.job, self.status))

def generate_nodes(number_nodes):
	"""
	Generates the nodes of the simulation
	Input:
	number_nodes       Number of nodes of the simulation
	Return:
	nodes_list         List of the nodes
	family_graph       Matrix representing the faliy subgraph
	"""
	# initalise the node_list and family_graph
	nodes_list = []
	family_graph = np.zeros((number_nodes, number_nodes))

	nodes_remaining = number_nodes
	nodes_done = 0
	while nodes_remaining != 0: # while it is possible to generate nodes

		# calulate the size of the family 
		family_size = family_size_distribution()
		if family_size > nodes_remaining: # if the family is larger of the remaining nodes
			family_size = nodes_remaining # set the family size as the remaining nodes


		for i in range(int(family_size)):
			# create the nodes
			age = age_distribution()
			nodes_list.append(Node(age))
			# create the family subgraph 
			for j in range(int(family_size)):
				family_graph[nodes_done+i,nodes_done+j] = 1

		nodes_remaining -= family_size # update the remaning nodes
		nodes_done += family_size # update the done nodes
		np.fill_diagonal(family_graph,0)

    # Networkx graph
	family_graph_nx = nx.convert_matrix.from_numpy_matrix(family_graph)
	return nodes_list, family_graph_nx



if __name__ == '__main__':

	number_nodes = 100
	nodes_list, family_graph_nx = generate_nodes(number_nodes)

    # Plotting the graph
	plt.figure(1)
	family_plot = nx.draw(family_graph_nx, with_labels=True, node_color='green')

    # Generation of random scale-free network using BA Model
	G_BA = nx.barabasi_albert_graph(number_nodes,2)

    # Plotting the graph
	plt.figure(2)
	BA_plot =nx.draw(G_BA, with_labels=True, node_color='red')

    # Composition of the graph together
    F = nx.compose(G_BA, family_graph_nx)

    #Ploting the graph
    plt.figure(3)
	joined_plot = nx.draw(F, with_labels=True)

	plt.show()
	for node_e in nodes_list:
		node_e.print_node()

