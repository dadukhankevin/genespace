# environments.py
from genespace.layers import Layer
from genespace.individual import Individual
from genespace.genepool import GenePool
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import torch
import enum

class BackpropMode(enum.Enum):
    NONE = 1
    RANDOM_GRADIENT = 2
    GRADIENT_DESCENT = 3

class Environment:
    def __init__(self, layers: list[Layer], genepool: GenePool, pbf_function: Callable):
        self.layers: list[Layer] = layers
        self.individuals: list[Individual] = []
        self.max_individuals: int = 0
        self.compiled = False
        self.start_population: int = 2
        self.early_stop: float = float('inf')
        self.batch_size: int = 10
        self.genepool: GenePool = genepool
        self.pbf_function: Callable = pbf_function
        self.fitness_history: list[float] = []
        self.population_history: list[int] = []
    
    def compile(self, start_population: int, max_individuals: int, individuals: list[Individual] = [], early_stop: float = float('inf')):
        assert start_population >= 2, "Start population must be at least 2"

        self.individuals = individuals
        self.start_population = start_population
        self.max_individuals = max_individuals
        self.early_stop = early_stop
        self.compiled = True
        self.fitness_history = []
        self.population_history = []

        for layer in self.layers:
            layer.initialize(self)
    
    def batch_fitness(self):
        individuals_for_measurement = [individual for individual in self.individuals if individual.modified]
        
        if not individuals_for_measurement:
            return
        
        batch_size = self.batch_size
        for i in range(0, len(individuals_for_measurement), batch_size):
            batch = individuals_for_measurement[i:i+batch_size]
            batch_genes = torch.tensor([ind.genes for ind in batch], dtype=torch.float32).to(self.genepool.gsp.device)
            
            phenotypes = self.genepool.gsp.forward(batch_genes).detach()
            batch_fitnesses = self.pbf_function(phenotypes)
            
            for individual, fitness in zip(batch, batch_fitnesses):
                individual.fitness = float(fitness)  # Convert fitness to float
                individual.modified = False
    
    def batch_fitness_for_random_gradient(self, phenotypes):
        batch_fitnesses = self.pbf_function(phenotypes.detach())
        return batch_fitnesses
    
    def sort_individuals(self):
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
    
    def evolve(self, generations=100, backprop_mode: BackpropMode = BackpropMode.GRADIENT_DESCENT, backprop_every_n=1, epochs=1, selection_percent=.5, batch_size=32):
        assert self.compiled, "Environment must be compiled before evolving"
        self.batch_size = batch_size
        for i in range(generations):
            for layer in self.layers:
                while len(self.individuals) < self.start_population:
                    self.individuals.append(self.genepool.create_individual())
                new_individuals = layer.execute()
                if new_individuals:
                    self.individuals.extend(new_individuals)
                    self.batch_fitness()
                    self.sort_individuals()
                    self.individuals = self.individuals[:self.max_individuals]
                if self.individuals and self.individuals[0].fitness > self.early_stop:
                    print(f"Early stopping at generation {i} with fitness {self.individuals[0].fitness}")
                    return self.individuals
            
            # Train the GRN every backprop_every_n generations
            if backprop_mode == BackpropMode.GRADIENT_DESCENT:
                if i % backprop_every_n == 0:
                    for _ in range(epochs):
                        loss = self.genepool.gsp.backprop_network(self.individuals, selection_percent=selection_percent, batch_size=self.batch_size)
                        print(f"Loss: {loss}")
                        
            elif backprop_mode == BackpropMode.RANDOM_GRADIENT:
                if i % backprop_every_n == 0:
                    self.genepool.gsp.apply_random_gradient(self.individuals, n_gradients=1, pbf_function=self.batch_fitness_for_random_gradient, selection_percent=selection_percent, batch_size=self.batch_size)
            
            self.fitness_history.append(self.individuals[0].fitness)
            self.population_history.append(len(self.individuals))
            
            print(f"Generation: {i}")
            print("Max fitness: ", float(self.individuals[0].fitness))
            print("Population size: ", len(self.individuals))
     
         
    
        return self.individuals
    
    def plot(self):
        plt.plot(self.fitness_history)
        plt.show()
        
