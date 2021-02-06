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
            self.job = 'unemployed'

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

