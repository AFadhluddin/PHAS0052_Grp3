import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx import convert_matrix
from Node_Test import *


class graph_distributions:
    """
    Class for the graph distrubtion depending on the nature of the graph
    """

    def __init__(self, graph_type):
        """
        Initialised the value of the graph_type
        Input: 
        graph_type (str) will take allowed values of the sub graphs
        """
        self.graph_type = graph_type

    def age_distribution(self):

        if self.graph_type == 'family' or self.graph_type == 'random':
            # calulate the age by the real data distribution 
            age = np.random.normal(45, 20)
            # calulate the age band
            age_band = int(age/10 - (int(age/10) - age/10))
            if age_band > 9: # for >90 of age are all the same 
                age_band = 9
            return age_band
    
        elif self.graph_type == 'workers' or self.graph_type == 'essential_workers' or self.graph_type == 'unemployed':
            # calulate the age by the real data distribution 
            age = np.random.choice(range(16,75))
            # calulate the age band
            age_band = int(age/10 - (int(age/10) - age/10))
            if age_band > 9: # for >90 of age are all the same 
                age_band = 9
            return age_band
    
        elif self.graph_type == 'student':
            # calulate the age by the real data distribution 
            age = np.random.choice(range(3,23))
            # calulate the age band
            age_band = int(age/10 - (int(age/10) - age/10))
            if age_band > 9: # for >90 of age are all the same 
                age_band = 9
            return age_band



    def size_distribution(self):
        """
        Calulate the size of a family from the real-data distribution
        Return:      family_size     size of the family
        """

        # calulate the family size by the real data distribution 
        if self.graph_type == 'family':
            family_size = np.random.normal(4, 2)
            if family_size < 1:
                family_size = 1
            return int(family_size)

    
        # calulate the workers size by the real data distribution 
        elif self.graph_type == "workers":
            workers_size = np.random.normal(4, 2)
            if workers_size < 1:
                workers_size = 1
            return int(workers_size)


        # calulate the essential worker size by the real data distribution 
        elif self.graph_type == 'essential_workers':
            essential_workers_size = np.random.normal(10, 2)
            if essential_workers_size < 1:
                essential_workers_size = 1
            return int(essential_workers_size)

        
        # calulate the student size by the real data distribution 
        elif self.graph_type == 'student':
            student_size = np.random.normal(8, 2)
            if student_size < 1:
                studnet_size = 1
            return int(student_size)



   




