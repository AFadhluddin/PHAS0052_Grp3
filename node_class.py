import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import networkx as nx
from networkx import convert_matrix

from real_data_distributions import *

################### Constants ###################
# emplyment_fractions
# essential_workers_fraction
# immune_fraction_healing 
# immune_fraction_vac

class Node:
    """
    A node represents a person.
    """

    def __init__(self, status='healthy'):
        """
        Method which cunstruct the object. It sets the intial parameters.
        Inputs:
        age        Age band of the node
        """

        self.age = set_age()

        # Usally at the begining everyone is healthy, set inital conditions
        self.status = 'healthy'
        self.contagious = False
        self.day_from_infection = 0
        self.immune = False
        self.vaccinated = False

        # set the job of the node
        self.job = set_job(self.age)

        self.day_first_symptoms = -1
        self.day_of_death = -1
        self.day_of_heal = -1
        


    ############### Methods of Updating Status ###############

    def infect(self):
        """
        Infect the node if it's not immune
        """
        if self.immune == False:
            self.status = 'infected' # set as infected

            probability_contagues = contagious_probability_age()
            if probability_contagues > np.random.rand():
                self.day_first_symptoms = day_of_first_symptoms()
                self.day_of_heal = self.day_first_symptoms + 14

                probability_death = death_probability_age(self.age)/probability_contagues # Bias Theorem for conditional probability 
                if probability_death > np.random.rand():
                    self.day_of_death = day_of_death()
                else:
                    self.day_of_death = -1
            else:
                self.day_first_symptoms = -1 # imposible to be met
                self.day_of_heal = 14
                self.day_of_death = -1
        else:
            pass

    def kill(self):
        """
        Kill the node
        """
        self.status = 'dead'
        self.contagious = False
        self.day_first_symptoms = -1
        self.day_of_heal = -1 
        self.day_of_death = -1

    def heal(self):
        """
        Heal the node
        """
        self.status = 'healthy'

        # everything set back to normal 
        self.tested = False
        self.contagious = False
        self.day_from_infection = 0
        self.day_first_symptoms = -1
        self.day_of_heal = -1 

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
            self.day_from_infection += 1

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

        if self.contagious == True:
            infectivity = infectivity_factor(self.day_from_infection)
        
        return infectivity

    def print_node(self):
        """
        Print the node age, job, and status
        """
        print("the person is {0} years old, {1} and {2}".format(self.age*5, self.job, self.status))

